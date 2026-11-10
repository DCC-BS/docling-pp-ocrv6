"""A faster driver for RapidOCR's PyTorch engine, with RapidOCR's own results.

It reuses the ``RapidOCR`` object docling builds (its loaded det/cls/rec torch modules, configs and
post-processing) and reproduces ``RapidOCR.__call__`` step by step, with two changes:

* Preprocessing on the GPU: images go to the GPU as uint8 and are normalised there through a
  256-entry lookup table computed with RapidOCR's own numpy formula, so every value is bit-identical
  to the CPU path. Rec/cls padding and the CTC argmax/max also run on the GPU, and only the
  (lines x steps) indices and probabilities come back instead of the full softmax.
* Queued GPU work: all OCR rectangles of a page batch go through one call, and the det, cls and
  rec forwards of all of them are queued back to back before their results are read. The batches
  stay RapidOCR's own (per image, sorted by aspect ratio, ``cls_batch_num`` / ``rec_batch_num``
  lines, each padded to its own width): on the GPU a convolution's result can depend on the batch
  size (TF32), so merging batches would change the text.

Everything outside these steps (resizes with cv2, detection post-processing, crop warping, score
filtering) is RapidOCR's own code, called in the same order. Written against rapidocr 3.10;
:func:`supported` turns it off for configurations it does not reproduce.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

import cv2
import numpy as np
import torch
from rapidocr.utils.output import RapidOCROutput
from rapidocr.utils.process_img import apply_vertical_padding, map_boxes_to_original

from docling_pp_ocrv6._det_post import postprocess as det_postprocess

#: det forwards queued before their results are read back
_DET_QUEUE = 8
_RGB = 3
#: The rapidocr release whose ``RapidOCR.__call__`` this driver reproduces
_RAPIDOCR_VERSION = "3.10."


def supported(reader: Any) -> bool:  # noqa: ANN401
    """Whether ``reader`` (a ``rapidocr.RapidOCR``) runs a configuration this driver reproduces."""
    try:
        from rapidocr.ch_ppocr_rec.main import RTL_LANGS  # noqa: PLC0415
        from rapidocr.utils.model_resolver import normalize_lang  # noqa: PLC0415
        from rapidocr.utils.typings import EngineType  # noqa: PLC0415

        if not version("rapidocr").startswith(_RAPIDOCR_VERSION):
            return False  # its pipeline may differ from the one reproduced here
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


class _PinnedStaging:
    """Pinned host buffers for the uploads of one driver call, kept and reused by the next call.

    ``Tensor.pin_memory()`` allocates page-locked memory for every upload (0.5 to 5 ms of CPU time
    each); copying into a buffer allocated once costs microseconds. A call's uploads take
    consecutive slices; the next call reuses them after the last upload of the previous one has
    finished (an event per stream that uploaded).
    """

    _ALIGN = 256

    def __init__(self, chunk: int = 16 << 20) -> None:
        self._chunk = chunk
        self._bufs: list[torch.Tensor] = []
        self._cur = 0
        self._off = 0
        self._done: dict[torch.cuda.Stream, torch.cuda.Event] = {}

    def reset(self) -> None:
        """Wait until the previous call's uploads have left the buffers, then reuse them."""
        for event in self._done.values():
            event.synchronize()
        self._done.clear()
        self._cur = self._off = 0

    def upload(self, a: np.ndarray, dev: torch.device) -> torch.Tensor:
        """Copy ``a`` into pinned memory and start its asynchronous copy to ``dev``."""
        n = a.nbytes
        while True:
            if self._cur == len(self._bufs):
                self._bufs.append(torch.empty(max(self._chunk, n), dtype=torch.uint8, pin_memory=True))
            buf = self._bufs[self._cur]
            start = -(-self._off // self._ALIGN) * self._ALIGN
            if start + n <= buf.numel():
                break
            self._cur, self._off = self._cur + 1, 0
        self._off = start + n
        host = buf[start : start + n]
        np.copyto(host.numpy().view(a.dtype).reshape(a.shape), a)
        dtype = torch.from_numpy(a[:0]).dtype
        out = host.view(dtype).view(a.shape).to(dev, non_blocking=True)
        stream = torch.cuda.current_stream(dev)
        event = self._done.get(stream) or torch.cuda.Event()
        event.record(stream)
        self._done[stream] = event
        return out


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

    def __init__(self, reader: Any) -> None:  # noqa: ANN401
        self.r = reader
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
        self._local = threading.local()  # one set of staging buffers per calling thread

    # ---------------------------------------------------------------- helpers
    def _h2d(self, a: np.ndarray) -> torch.Tensor:
        """Host to device copy that does not wait for the kernels already queued.

        A copy from pageable memory blocks the host until the stream reaches it; staging the
        array in pinned memory lets the CPU go on preparing the next batch.
        """
        staging = self._staging()
        if staging is None:
            return torch.from_numpy(a).to(self.dev)
        return staging.upload(a, self.dev)

    def _staging(self) -> _PinnedStaging | None:
        """This thread's pinned upload buffers (None off CUDA)."""
        if self.dev.type != "cuda":
            return None
        staging = getattr(self._local, "staging", None)
        if staging is None:
            staging = self._local.staging = _PinnedStaging()
        return staging

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
        """RapidOCR's result for every image; the GPU work of all images is queued back to back."""
        staging = self._staging()
        if staging is not None:
            staging.reset()
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
            boxes, _ = det_postprocess(self.det.postprocess_op, pred, it.img.shape[:2])
            if len(boxes) < 1:
                continue
            it.boxes = self.det.sorted_boxes(boxes)
            it.lines = list(self.r.crop_text_regions(it.img, it.boxes))

    @staticmethod
    def _chunks(lines: list[np.ndarray], size: int) -> list[np.ndarray]:
        """RapidOCR's batches of one image's lines: sorted by aspect ratio, ``size`` lines each."""
        order = np.argsort(np.array([im.shape[1] / float(im.shape[0]) for im in lines]))
        return [order[b : b + size] for b in range(0, len(lines), size)]

    def _classify(self, items: list[_Image]) -> None:
        """Angle classification in RapidOCR's own batches; all forwards queued before the first read-back.

        The batches are RapidOCR's (per image, ``cls_batch_num`` lines): on the GPU a convolution's
        result can depend on the batch size, so other batches could change the angle of a line.
        """
        _, c_h, c_w = self.cls.cls_image_shape
        queued = []
        for it in items:
            for idxs in self._chunks(it.lines, self.cls.cls_batch_num):
                u8 = np.zeros((len(idxs), c_h, c_w, 3), dtype=np.uint8)
                widths = np.zeros(len(idxs), dtype=np.int64)
                for q, k in enumerate(idxs):
                    rw = self._resize_width(it.lines[k], c_h, c_w)
                    u8[q, :, :rw] = cv2.resize(it.lines[k], (rw, c_h), interpolation=cv2.INTER_LINEAR)
                    widths[q] = rw
                queued.append((it, idxs, self.cls_net(self._rc_input(u8, widths))))
        for it, idxs, prob in queued:
            for k, (label, score) in zip(idxs, self.cls.postprocess_op(prob.cpu().numpy()), strict=True):
                if "180" in label and score > self.cls.cls_thresh:
                    it.lines[k] = cv2.rotate(it.lines[k], cv2.ROTATE_180)

    def _recognise(self, items: list[_Image]) -> None:
        """Recognition in RapidOCR's own batches, forwards queued; CTC argmax and max on the GPU, RapidOCR's decoder."""
        img_h = self.rec.rec_image_shape[1]
        base_ratio = self.rec.rec_image_shape[2] / img_h
        queued = []
        for it in items:
            it.txts = [""] * len(it.lines)
            it.scores = [0.0] * len(it.lines)
            for idxs in self._chunks(it.lines, self.rec.rec_batch_num):
                max_wh = max([base_ratio] + [it.lines[k].shape[1] * 1.0 / it.lines[k].shape[0] for k in idxs])
                width = int(img_h * max_wh)
                u8 = np.zeros((len(idxs), img_h, width, 3), dtype=np.uint8)
                widths = np.zeros(len(idxs), dtype=np.int64)
                for q, k in enumerate(idxs):
                    rw = self._resize_width(it.lines[k], img_h, width)
                    u8[q, :, :rw] = cv2.resize(it.lines[k], (rw, img_h))
                    widths[q] = rw
                mx = self.rec_net(self._rc_input(u8, widths)).max(dim=2)
                queued.append((it, idxs, mx.indices, mx.values))
        decode = self.rec.postprocess_op.decode
        for it, idxs, idx, prob in queued:
            res, _ = decode(idx.cpu().numpy(), prob.cpu().numpy(), False, remove_duplicate=True)  # noqa: FBT003
            for k, (txt, score) in zip(idxs, res, strict=True):
                it.txts[k] = txt
                it.scores[k] = score

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
