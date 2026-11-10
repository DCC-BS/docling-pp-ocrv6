"""A faster driver for RapidOCR's PyTorch engine, with RapidOCR's own results.

It reuses the ``RapidOCR`` object docling builds (its loaded det/cls/rec torch modules, configs and
post-processing) and reproduces ``RapidOCR.__call__`` step by step, with two changes:

* Preprocessing on the GPU: images go to the GPU as uint8 and are normalised there through a
  256-entry lookup table computed with RapidOCR's own numpy formula, so every value is bit-identical
  to the CPU path. Rec/cls padding and the CTC argmax/max also run on the GPU, and only the
  (lines x steps) indices and probabilities come back instead of the full softmax.
* Batching: all OCR rectangles of a page batch go through one call. The det forwards are queued
  back to back before the first result is read, cls runs on all text lines at once (its input is
  padded to a fixed width, so batching does not change the maths), and rec keeps RapidOCR's exact
  line grouping (per image, sorted by aspect ratio, chunks of ``rec_batch_num``, each padded to its
  own width) but runs all chunks of the same padded width as one forward.

Everything outside these steps (resizes with cv2, detection post-processing, crop warping, score
filtering) is RapidOCR's own code, called in the same order. Written against rapidocr 3.10;
:func:`supported` turns it off for configurations it does not reproduce.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
from rapidocr.utils.output import RapidOCROutput
from rapidocr.utils.process_img import apply_vertical_padding, map_boxes_to_original

#: det forwards queued before their results are read back
_DET_QUEUE = 8
#: text lines per cls forward (its input has a fixed size)
_CLS_CHUNK = 512
#: rec: lines per forward times their padded width, at most
_REC_PIXELS = 24000
_RGB = 3

#: One recognition input: image index, line index, the line resized to the rec height.
type RecLine = tuple[int, int, np.ndarray]


def supported(reader: Any) -> bool:  # noqa: ANN401
    """Whether ``reader`` (a ``rapidocr.RapidOCR``) runs a configuration this driver reproduces."""
    try:
        from rapidocr.ch_ppocr_rec.main import RTL_LANGS  # noqa: PLC0415
        from rapidocr.utils.model_resolver import normalize_lang  # noqa: PLC0415
        from rapidocr.utils.typings import EngineType  # noqa: PLC0415

        cfg = reader.cfg
        return bool(
            normalize_lang(cfg.Rec.lang_type) not in RTL_LANGS
            and reader.use_det
            and reader.use_cls
            and reader.use_rec
            and not reader.return_word_box
            and all(cfg[k].engine_type == EngineType.TORCH for k in ("Det", "Cls", "Rec"))
            and next(iter(cfg.Rec.rec_img_shape)) == _RGB
        )
    except Exception:  # noqa: BLE001
        return False


@dataclass
class _Image:
    """One input image on its way through det, cls and rec."""

    ori: np.ndarray
    img: np.ndarray
    record: dict[str, Any]
    resized: np.ndarray | None
    boxes: np.ndarray | None = None
    lines: list[np.ndarray] = field(default_factory=list)
    txts: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


class FastRapidOcr:
    """Runs a ``rapidocr.RapidOCR`` reader's det, cls and rec models on a list of images."""

    def __init__(self, reader: Any, *, batch: bool = True) -> None:  # noqa: ANN401
        self.r = reader
        self.batch = batch
        self.det = reader._load_det_model()  # noqa: SLF001
        self.cls = reader._load_cls_model()  # noqa: SLF001
        self.rec = reader._load_rec_model()  # noqa: SLF001
        self.dev = self.det.session.device
        self.det_net = self.det.session.predictor
        self.cls_net = self.cls.session.predictor
        self.rec_net = self.rec.session.predictor

        # Det normalisation table, built with RapidOCR's own DetPreProcess.normalize so every
        # value matches the CPU path exactly: index = channel * 256 + pixel value.
        pre = self.det.get_preprocess()
        v = np.repeat(np.arange(256, dtype=np.uint8).reshape(256, 1, 1), 3, axis=2)
        det_lut = pre.normalize(v).astype(np.float32).reshape(256, 3).T.copy()
        self.det_lut = torch.from_numpy(det_lut.reshape(-1)).to(self.dev)
        self.ch_off = torch.tensor([0, 256, 512], device=self.dev, dtype=torch.int64)

        # Rec and cls share one formula: astype(float32) / 255, -= 0.5, /= 0.5 (float32 throughout).
        a = np.arange(256, dtype=np.uint8).astype("float32") / 255
        a -= 0.5
        a /= 0.5
        self.rc_lut = torch.from_numpy(a.astype(np.float32)).to(self.dev)

    # ---------------------------------------------------------------- helpers
    def _h2d(self, a: np.ndarray) -> torch.Tensor:
        """Host to device copy that does not wait for the kernels already queued.

        A copy from pageable memory blocks the host until the stream reaches it; staging the
        array in pinned memory lets the CPU go on preparing the next batch.
        """
        if self.dev.type != "cuda":
            return torch.from_numpy(a)
        return torch.from_numpy(a).pin_memory().to(self.dev, non_blocking=True)

    def _det_input(self, resized: np.ndarray) -> torch.Tensor:
        """uint8 (H, W, 3) -> normalised (1, 3, H, W), as ``DetPreProcess.normalize`` and ``permute``."""
        t = self._h2d(np.ascontiguousarray(resized))
        x = self.det_lut[t.long() + self.ch_off]  # H, W, 3
        return x.permute(2, 0, 1).unsqueeze(0).contiguous()

    def _rc_input(self, u8: np.ndarray, widths: np.ndarray) -> torch.Tensor:
        """uint8 (N, H, W, 3) batch with valid widths -> normalised (N, 3, H, W), zero padded."""
        t = self._h2d(u8)
        x = self.rc_lut[t.long()]
        w = self._h2d(widths)
        cols = torch.arange(u8.shape[2], device=self.dev)
        mask = cols.view(1, 1, -1, 1) < w.view(-1, 1, 1, 1)
        x = torch.where(mask, x, torch.zeros((), device=self.dev, dtype=x.dtype))
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _resize_width(img: np.ndarray, height: int, max_width: int) -> int:
        """RapidOCR's width for a line resized to ``height``, capped at ``max_width``."""
        h, w = img.shape[:2]
        return min(max_width, math.ceil(height * (w / float(h))))

    # ---------------------------------------------------------------- entry point
    def __call__(self, images: list[np.ndarray]) -> list[RapidOCROutput]:
        """RapidOCR's result for every image (all at once, or one by one with ``batch=False``)."""
        if not self.batch:
            return [self._run([im])[0] for im in images]
        return self._run(images)

    def _run(self, images: list[np.ndarray]) -> list[RapidOCROutput]:
        items = [self._prepare(im) for im in images]
        with torch.no_grad():
            self._detect(items)
            self._classify(items)
            self._recognise(items)
        return [RapidOCROutput() if it.boxes is None else self._final(it, it.boxes) for it in items]

    def _prepare(self, im: np.ndarray) -> _Image:
        r = self.r
        ori = r.load_img(im)
        img, record = r.preprocess_img(ori)
        if r.cfg.Global.use_vertical_padding:
            img, record = apply_vertical_padding(img, record, r.width_height_ratio, r.min_height)
        else:
            record["padding_1"] = {"top": 0, "left": 0}
        return _Image(ori=ori, img=img, record=record, resized=self.det.get_preprocess().resize(img))

    def _detect(self, items: list[_Image]) -> None:
        """Det forwards queued back to back, then RapidOCR's post-processing and crops per image."""
        preds: list[np.ndarray | None] = [None] * len(items)
        pending: list[tuple[int, torch.Tensor]] = []
        for i, it in enumerate(items):
            if it.resized is None:
                continue
            pending.append((i, self.det_net(self._det_input(it.resized))))
            if len(pending) >= _DET_QUEUE:
                for j, p in pending:
                    preds[j] = p.cpu().numpy()
                pending = []
        for j, p in pending:
            preds[j] = p.cpu().numpy()
        for it, pred in zip(items, preds, strict=True):
            if pred is None:
                continue
            boxes, _ = self.det.postprocess_op(pred, it.img.shape[:2])
            if len(boxes) < 1:
                continue
            it.boxes = self.det.sorted_boxes(boxes)
            it.lines = list(self.r.crop_text_regions(it.img, it.boxes))

    def _classify(self, items: list[_Image]) -> None:
        """Angle classification of all lines at once (fixed-size input: grouping does not change the maths)."""
        flat = [(it, k) for it in items for k in range(len(it.lines))]
        _, c_h, c_w = self.cls.cls_image_shape
        for s in range(0, len(flat), _CLS_CHUNK):
            part = flat[s : s + _CLS_CHUNK]
            u8 = np.zeros((len(part), c_h, c_w, 3), dtype=np.uint8)
            widths = np.zeros(len(part), dtype=np.int64)
            for q, (it, k) in enumerate(part):
                rw = self._resize_width(it.lines[k], c_h, c_w)
                u8[q, :, :rw] = cv2.resize(it.lines[k], (rw, c_h), interpolation=cv2.INTER_LINEAR)
                widths[q] = rw
            prob = self.cls_net(self._rc_input(u8, widths)).cpu().numpy()
            for (it, k), (label, score) in zip(part, self.cls.postprocess_op(prob), strict=True):
                if "180" in label and score > self.cls.cls_thresh:
                    it.lines[k] = cv2.rotate(it.lines[k], cv2.ROTATE_180)

    def _rec_chunks(self, items: list[_Image]) -> list[tuple[int, list[RecLine]]]:
        """RapidOCR's rec chunks (per image, sorted by aspect ratio) with their padded width.

        With ``batch``, all chunks of the same width are merged into one job.
        """
        img_h = self.rec.rec_image_shape[1]
        base_ratio = self.rec.rec_image_shape[2] / img_h
        bnum = self.rec.rec_batch_num
        chunks: list[tuple[int, list[RecLine]]] = []
        by_width: dict[int, list[RecLine]] = defaultdict(list)
        for i, it in enumerate(items):
            order = np.argsort(np.array([im.shape[1] / float(im.shape[0]) for im in it.lines]))
            for b in range(0, len(it.lines), bnum):
                idxs = order[b : b + bnum]
                max_wh = max([base_ratio] + [it.lines[k].shape[1] * 1.0 / it.lines[k].shape[0] for k in idxs])
                width = int(img_h * max_wh)
                chunk = []
                for k in idxs:
                    rw = self._resize_width(it.lines[k], img_h, width)
                    chunk.append((i, int(k), cv2.resize(it.lines[k], (rw, img_h))))
                if self.batch:
                    by_width[width].extend(chunk)
                else:
                    chunks.append((width, chunk))
        return list(by_width.items()) if self.batch else chunks

    def _recognise(self, items: list[_Image]) -> None:
        """Rec forwards per padded width, CTC argmax and max on the GPU, RapidOCR's decoder."""
        for it in items:
            it.txts = [""] * len(it.lines)
            it.scores = [0.0] * len(it.lines)
        img_h = self.rec.rec_image_shape[1]
        outs = []
        for width, lines in self._rec_chunks(items):
            per = max(1, min(len(lines), _REC_PIXELS // max(width, 1)))
            for s in range(0, len(lines), per):
                part = lines[s : s + per]
                u8 = np.zeros((len(part), img_h, width, 3), dtype=np.uint8)
                widths = np.zeros(len(part), dtype=np.int64)
                for q, (_, _, im) in enumerate(part):
                    u8[q, :, : im.shape[1]] = im
                    widths[q] = im.shape[1]
                mx = self.rec_net(self._rc_input(u8, widths)).max(dim=2)
                outs.append((part, mx.indices, mx.values))
        decode = self.rec.postprocess_op.decode
        for part, idx, prob in outs:
            res, _ = decode(idx.cpu().numpy(), prob.cpu().numpy(), False, remove_duplicate=True)  # noqa: FBT003
            for (i, k, _), (txt, score) in zip(part, res, strict=True):
                items[i].txts[k] = txt
                items[i].scores[k] = score

    def _final(self, it: _Image, boxes: np.ndarray) -> RapidOCROutput:
        """RapidOCR's final step: boxes back to the input image, empty and low-score lines dropped."""
        ori_h, ori_w = it.ori.shape[:2]
        boxes = map_boxes_to_original(boxes, it.record, ori_h, ori_w)
        keep = [i for i, v in enumerate(it.txts) if v.strip() and it.scores[i] >= self.r.text_score]
        if not keep:
            return RapidOCROutput()
        # rapidocr annotates these fields as one-element tuples
        fields: dict[str, Any] = {
            "boxes": np.array([boxes[i] for i in keep]),
            "txts": tuple(it.txts[i] for i in keep),
            "scores": tuple(it.scores[i] for i in keep),
            "word_results": (),
        }
        return RapidOCROutput(**fields)
