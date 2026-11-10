"""A faster driver for RapidOCR's PyTorch engine, with RapidOCR's own results.

It reuses the ``RapidOCR`` object docling builds (its loaded det/cls/rec torch modules, configs and
post-processing) and reproduces ``RapidOCR.__call__`` step by step, with these changes:

* Preprocessing on the GPU: images go to the GPU as uint8 and are normalised there through a
  256-entry lookup table computed with RapidOCR's own numpy formula, so every value is bit-identical
  to the CPU path. Rec/cls padding and the CTC argmax/max also run on the GPU, and only the
  (lines x steps) indices and probabilities come back instead of the full softmax.
* Queued GPU work: all OCR rectangles of a page batch go through one call. The det forwards of all
  of them are queued first; each image's detection post-processing on the CPU then overlaps the
  GPU work of the next ones, and its cls and rec batches are queued as soon as its crops are ready.
  Inputs and results go through pinned buffers reused across calls, one wait per batch. The
  batches stay RapidOCR's own (per image, sorted by aspect ratio, ``cls_batch_num`` /
  ``rec_batch_num`` lines, each padded to its own width): on the GPU a convolution's result can
  depend on the batch size (TF32), so merging batches would change the text.
* CUDA graphs: a forward (with its normalisation and CTC max) of an input shape seen before is
  captured once as a CUDA graph and replayed, which saves the launch overhead of its thousand small
  kernels. A replay runs exactly the kernels of the eager forward of the same shapes, so its result
  is the eager result bit for bit (see :class:`_Graphs`).

Everything outside these steps (resizes with cv2, detection post-processing, crop warping, score
filtering) is RapidOCR's own code, called in the same order. Written against rapidocr 3.10;
:func:`supported` turns it off for configurations it does not reproduce.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from rapidocr.utils.output import RapidOCROutput
from rapidocr.utils.process_img import apply_vertical_padding, map_boxes_to_original

from docling_pp_ocrv6._det_post import postprocess as det_postprocess
from docling_pp_ocrv6._fx import inline

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    #: a forward: device tensors in, device tensors out
    Fn = Callable[..., tuple[torch.Tensor, ...]]

logger = logging.getLogger(__name__)

_RGB = 3
#: The rapidocr release whose ``RapidOCR.__call__`` this driver reproduces
_RAPIDOCR_VERSION = "3.10."
#: CUDA graphs kept per model, least recently used dropped first (each costs about 2 MB of device
#: memory besides the shared pool). Rec has one shape per (lines, padded width); det stays eager:
#: its large inputs keep the GPU busy, and its shapes vary per page.
_GRAPHS = {"cls": 8, "rec": 160}
#: largest input (uint8 elements) that is captured: 6 rec lines 640 px wide. Wider batches are rare
#: and long enough to hide their launches, and their buffers would grow the graphs' memory pool.
_GRAPH_MAX_INPUT = 6 * 48 * 640 * 3
#: calls of an input shape before it is captured: a shape seen once runs eagerly and costs no capture
_CAPTURE_AT = 2


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


class _PinnedArena:
    """Pinned host buffers for the inputs and outputs of one driver call, reused by the next call.

    ``torch.empty(..., pin_memory=True)`` allocates page-locked memory every time (0.5 to 5 ms of
    CPU time each); a slice of a buffer allocated once costs microseconds. A call takes consecutive
    slices; :meth:`reset` hands them out again once the previous call's last GPU work has finished.
    Buffers are never freed while in use, so none is released during a CUDA graph capture.
    """

    _ALIGN = 256

    def __init__(self, chunk: int = 16 << 20) -> None:
        self._chunk = chunk
        self._bufs: list[torch.Tensor] = []
        self._cur = 0
        self._off = 0
        self.done: torch.cuda.Event | None = None  # recorded after the previous call's last copy

    def reset(self) -> None:
        """Reuse the buffers, after the previous call's GPU work on them has finished."""
        if self.done is not None:
            self.done.synchronize()
            self.done = None
        self._cur = self._off = 0

    def take(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """An uninitialised pinned tensor of ``shape`` and ``dtype``, valid until the next :meth:`reset`."""
        n = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        while True:
            if self._cur == len(self._bufs):
                self._bufs.append(torch.empty(max(self._chunk, n), dtype=torch.uint8, pin_memory=True))
            buf = self._bufs[self._cur]
            start = -(-self._off // self._ALIGN) * self._ALIGN
            if start + n <= buf.numel():
                break
            self._cur, self._off = self._cur + 1, 0
        self._off = start + n
        return buf[start : start + n].view(dtype).view(tuple(shape))


class _Graphs:
    """CUDA graphs of one batched forward, one per input shape: captured lazily, replayed, least recently used dropped.

    Exactness: a graph replays the kernels its capture recorded, and the capture records what the
    eager call of the same shapes launches: every kernel choice here depends on shapes only (cuDNN's
    heuristic pick is cached per shape by the first eager call, which comes before any capture;
    cuBLAS and the elementwise kernels choose by shape and alignment, and the caching allocator
    aligns every buffer). So a replay gives the eager result bit for bit on any GPU, without
    touching global torch flags. Batches keep RapidOCR's shapes, so eager is RapidOCR's own forward.

    A shape runs eagerly until its ``_CAPTURE_AT``-th call; then it is captured on ``stream``, a
    stream used for nothing else (a pinned buffer released during a capture on a stream it was used
    on would record its event into the graph), and replayed on the current stream. The graphs share
    ``pool``: a replay may overwrite another graph's outputs, so the outputs of a call must be
    copied (queued on the current stream) before the next call. Capture uses ``thread_local`` mode,
    so docling's other models keep running on the GPU from other threads. Inputs of more than
    ``max_input`` elements stay eager.
    """

    def __init__(self, fn: Fn, capacity: int, *, stream: Any, pool: Any, max_input: int) -> None:  # noqa: ANN401
        self.fn = fn
        self.capacity = capacity
        self.stream = stream
        self.pool = pool
        self.max_input = max_input
        self.graphs: OrderedDict[tuple, tuple[Any, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = OrderedDict()
        self.calls: Counter[tuple] = Counter()
        self.warm = False
        self.enabled = True

    def __call__(self, host: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
        """The outputs for host inputs (pinned), queued on the current stream; valid until the next call."""
        key = tuple((tuple(t.shape), t.dtype) for t in host)
        entry = self.graphs.get(key)
        if entry is not None:
            self.graphs.move_to_end(key)
            graph, inputs, outputs = entry
            for dst, src in zip(inputs, host, strict=True):
                dst.copy_(src, non_blocking=True)
            graph.replay()
            return outputs
        args = tuple(t.to(device, non_blocking=True) for t in host)
        if not self.enabled or host[0].numel() > self.max_input:
            return self.fn(*args)
        self.calls[key] += 1
        if self.calls[key] < _CAPTURE_AT:
            if len(self.calls) > 16 * self.capacity:  # bounded memory of the shapes seen once
                self.calls.clear()
            return self.fn(*args)
        del self.calls[key]
        try:
            graph, inputs, outputs = self._capture(args, torch.cuda.current_stream(device))
        except Exception:  # noqa: BLE001 - a failed capture leaves the forward to eager mode, as RapidOCR runs it
            logger.warning("rapidocr-fast: CUDA graph capture failed, running eagerly", exc_info=True)
            self.enabled = False
            self._drop(len(self.graphs))
            return self.fn(*args)
        self.graphs[key] = (graph, inputs, outputs)
        self._drop(len(self.graphs) - self.capacity)
        graph.replay()
        return outputs

    def _capture(self, args: tuple[torch.Tensor, ...], current: Any) -> tuple[Any, tuple, tuple]:  # noqa: ANN401
        inputs = tuple(a.clone() for a in args)  # static inputs, outside the graph pool
        self.stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self.stream):
            if not self.warm:  # first use of the stream (cuBLAS workspace) outside a capture, one line
                self.fn(*(t[:1] for t in inputs))
                self.warm = True
            graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
            try:
                outputs = self.fn(*inputs)
            finally:
                graph.capture_end()
        current.wait_stream(self.stream)
        return graph, inputs, outputs

    def _drop(self, n: int) -> None:
        """Drop the ``n`` least recently used graphs, once no replay of theirs is in flight."""
        if n <= 0:
            return
        torch.cuda.current_stream().synchronize()
        for _ in range(n):
            self.graphs.popitem(last=False)


@dataclass
class _Pending:
    """Outputs of a queued forward, copied to (pinned) host memory once ``event`` has completed."""

    outputs: tuple[torch.Tensor, ...]
    event: torch.cuda.Event | None = None

    def result(self) -> tuple[np.ndarray, ...]:
        if self.event is not None:
            self.event.synchronize()
        return tuple(o.numpy() for o in self.outputs)


@dataclass
class _Image:
    """One input image on its way through det, cls and rec."""

    ori: np.ndarray
    img: np.ndarray
    record: dict[str, Any]
    resized: np.ndarray | None
    det: _Pending | None = None
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
        # The networks as flat fx graphs: same kernels, less Python per forward (eager if untraceable)
        self.det_net = inline(self.det.session.predictor)
        self.cls_net = inline(self.cls.session.predictor)
        self.rec_net = inline(self.rec.session.predictor)

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
        self._local = threading.local()  # one set of pinned buffers per calling thread

        self.cuda = self.dev.type == "cuda"
        self.graphs: dict[str, _Graphs] = {}
        # a graph's buffers serve one call at a time: callers in other threads queue in turn
        self._lock = threading.Lock()
        if self.cuda:
            # cls and rec graphs share one capture stream and one memory pool. Everything runs on
            # the current stream, whose cached memory docling's other models share: work on a
            # stream of its own would cache blocks of its own (gigabytes for det).
            stream, pool = torch.cuda.Stream(self.dev), torch.cuda.graph_pool_handle()
            for name, fn in (("cls", self._cls_fn), ("rec", self._rec_fn)):
                self.graphs[name] = _Graphs(fn, _GRAPHS[name], stream=stream, pool=pool, max_input=_GRAPH_MAX_INPUT)
        self.fns: dict[str, Fn] = {"det": self._det_fn, "cls": self._cls_fn, "rec": self._rec_fn}

    # ---------------------------------------------------------------- forwards
    def _det_norm(self, t: torch.Tensor) -> torch.Tensor:
        """uint8 (H, W, 3) -> normalised (1, 3, H, W), as ``DetPreProcess.normalize`` and ``permute``."""
        x = self.det_lut[t.long() + self.ch_off]  # H, W, 3
        return x.permute(2, 0, 1).unsqueeze(0).contiguous()

    def _rc_norm(self, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """uint8 (N, H, W, 3) batch with valid widths -> normalised (N, 3, H, W), zero padded."""
        x = self.rc_lut[t.long()]
        cols = torch.arange(t.shape[2], device=t.device)
        mask = cols.view(1, 1, -1, 1) < w.view(-1, 1, 1, 1)
        x = torch.where(mask, x, torch.zeros((), device=t.device, dtype=x.dtype))
        return x.permute(0, 3, 1, 2).contiguous()

    def _det_fn(self, t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.det_net(self._det_norm(t)),)

    def _cls_fn(self, t: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.cls_net(self._rc_norm(t, w)),)

    def _rec_fn(self, t: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, ...]:
        mx = self.rec_net(self._rc_norm(t, w)).max(dim=2)
        return mx.indices, mx.values

    def _host(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """A host buffer; on CUDA a slice of this thread's pinned buffers.

        Pinned, copies to and from the GPU do not wait for queued work; reused, they cost no
        page-locked allocation.
        """
        if not self.cuda:
            return torch.empty(tuple(shape), dtype=dtype)
        return self._arena().take(shape, dtype)

    def _arena(self) -> _PinnedArena:
        arena = getattr(self._local, "arena", None)
        if arena is None:
            arena = self._local.arena = _PinnedArena()
        return arena

    def _launch(self, model: str, host: tuple[torch.Tensor, ...]) -> _Pending:
        """Queue ``model``'s forward on host inputs and the copy of its outputs back to the host."""
        if not self.cuda:
            return _Pending(tuple(o.cpu() for o in self.fns[model](*(t.to(self.dev) for t in host))))
        graphs = self.graphs.get(model)
        with self._lock:
            if graphs is not None:
                outputs = graphs(host, self.dev)
            else:
                outputs = self.fns[model](*(t.to(self.dev, non_blocking=True) for t in host))
            back = tuple(self._host(o.shape, o.dtype) for o in outputs)
            for dst, src in zip(back, outputs, strict=True):
                dst.copy_(src, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        return _Pending(back, event)

    # The input helpers of the tests: normalised tensors from numpy arrays.
    def _det_input(self, resized: np.ndarray) -> torch.Tensor:
        return self._det_norm(torch.from_numpy(np.ascontiguousarray(resized)).to(self.dev))

    def _rc_input(self, u8: np.ndarray, widths: np.ndarray) -> torch.Tensor:
        return self._rc_norm(torch.from_numpy(u8).to(self.dev), torch.from_numpy(widths).to(self.dev))

    @staticmethod
    def _resize_width(img: np.ndarray, height: int, max_width: int) -> int:
        """RapidOCR's width for a line resized to ``height``, capped at ``max_width``."""
        h, w = img.shape[:2]
        return min(max_width, math.ceil(height * (w / float(h))))

    # ---------------------------------------------------------------- entry point
    def __call__(self, images: list[np.ndarray]) -> list[RapidOCROutput]:
        """RapidOCR's result for every image; the GPU work of all images is queued back to back."""
        if self.cuda:
            self._arena().reset()
        with torch.no_grad():
            items = [self._prepare(im) for im in images]
            cls = [self._detect(it) for it in items]
            rec = [self._classify(it, batches) for it, batches in zip(items, cls, strict=True)]
            for it, batches in zip(items, rec, strict=True):
                self._recognise(it, batches)
            if self.cuda:  # the buffers are free for the next call once the stream reaches here
                done = torch.cuda.Event()
                done.record()
                self._arena().done = done
        return [RapidOCROutput() if it.boxes is None else self._final(it, it.boxes) for it in items]

    def _prepare(self, im: np.ndarray) -> _Image:
        """RapidOCR's preprocessing and det resize on the CPU; the det forward queued."""
        r = self.r
        ori = r.load_img(im)
        img, record = r.preprocess_img(ori)
        if r.cfg.Global.use_vertical_padding:
            img, record = apply_vertical_padding(img, record, r.width_height_ratio, r.min_height)
        else:
            record["padding_1"] = {"top": 0, "left": 0}
        it = _Image(ori=ori, img=img, record=record, resized=self.det.get_preprocess().resize(img))
        if it.resized is not None:
            buf = self._host(it.resized.shape, torch.uint8)
            buf.numpy()[...] = it.resized
            it.det = self._launch("det", (buf,))
        return it

    def _detect(self, it: _Image) -> list[tuple[np.ndarray, _Pending]]:
        """RapidOCR's detection post-processing and crops of one image; its cls batches queued."""
        if it.det is None:
            return []
        (pred,) = it.det.result()
        it.det = None
        boxes, _ = det_postprocess(self.det.postprocess_op, pred, it.img.shape[:2])
        if len(boxes) < 1:
            return []
        it.boxes = self.det.sorted_boxes(boxes)
        it.lines = list(self.r.crop_text_regions(it.img, it.boxes))
        _, c_h, c_w = self.cls.cls_image_shape
        queued = []
        for idxs in self._chunks(it.lines, self.cls.cls_batch_num):
            u8 = self._host((len(idxs), c_h, c_w, 3), torch.uint8)
            widths = self._host((len(idxs),), torch.int64)
            arr, wid = u8.numpy(), widths.numpy()
            for q, k in enumerate(idxs):
                rw = self._resize_width(it.lines[k], c_h, c_w)
                arr[q, :, :rw] = cv2.resize(it.lines[k], (rw, c_h), interpolation=cv2.INTER_LINEAR)
                wid[q] = rw
            queued.append((idxs, self._launch("cls", (u8, widths))))
        return queued

    @staticmethod
    def _chunks(lines: list[np.ndarray], size: int) -> list[np.ndarray]:
        """RapidOCR's batches of one image's lines: sorted by aspect ratio, ``size`` lines each."""
        order = np.argsort(np.array([im.shape[1] / float(im.shape[0]) for im in lines]))
        return [order[b : b + size] for b in range(0, len(lines), size)]

    def _classify(self, it: _Image, batches: list[tuple[np.ndarray, _Pending]]) -> list[tuple[np.ndarray, _Pending]]:
        """Angle classification results of one image applied; its rec batches queued.

        The batches are RapidOCR's (per image, ``cls_batch_num`` lines): on the GPU a convolution's
        result can depend on the batch size, so other batches could change the angle of a line.
        """
        for idxs, pending in batches:
            (prob,) = pending.result()
            for k, (label, score) in zip(idxs, self.cls.postprocess_op(prob), strict=True):
                if "180" in label and score > self.cls.cls_thresh:
                    it.lines[k] = cv2.rotate(it.lines[k], cv2.ROTATE_180)
        img_h = self.rec.rec_image_shape[1]
        base_ratio = self.rec.rec_image_shape[2] / img_h
        queued = []
        for idxs in self._chunks(it.lines, self.rec.rec_batch_num):
            max_wh = max([base_ratio] + [it.lines[k].shape[1] * 1.0 / it.lines[k].shape[0] for k in idxs])
            width = int(img_h * max_wh)
            u8 = self._host((len(idxs), img_h, width, 3), torch.uint8)
            widths = self._host((len(idxs),), torch.int64)
            arr, wid = u8.numpy(), widths.numpy()
            for q, k in enumerate(idxs):
                rw = self._resize_width(it.lines[k], img_h, width)
                arr[q, :, :rw] = cv2.resize(it.lines[k], (rw, img_h))
                wid[q] = rw
            queued.append((idxs, self._launch("rec", (u8, widths))))
        return queued

    def _recognise(self, it: _Image, batches: list[tuple[np.ndarray, _Pending]]) -> None:
        """RapidOCR's CTC decoder on the GPU's argmax and max of each rec batch of one image."""
        it.txts = [""] * len(it.lines)
        it.scores = [0.0] * len(it.lines)
        decode = self.rec.postprocess_op.decode
        for idxs, pending in batches:
            idx, prob = pending.result()
            res, _ = decode(idx, prob, False, remove_duplicate=True)  # noqa: FBT003
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
