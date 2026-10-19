"""ONNX Runtime model wrappers (numpy + cv2 + onnxruntime only; no torch, no ultralytics).

* `YoloOnnx`: a YOLO11 detector exported by ultralytics (raw head output (1, 4 + nc, N)). Pre-processing is
  ultralytics' LetterBox (resize INTER_LINEAR, pad 114, centred); `rect=True` = its minimum-rectangle mode (pad only
  to a multiple of 32), which is what ultralytics' `.pt` predict does and needs a dynamic-shape export. Post-processing
  = conf filter + class-aware greedy NMS (IoU 0.7) + scale back to image pixels, as ultralytics' predict.
* `Recognizer`: the WP10b number recognizer (`recog.onnx`: image (N,3,32,W) RGB 0-1 -> logp (T,N,11) CTC log-probs).
* `RosterScorer`: P(roster number | crop) from the CTC sequence likelihood (numpy port of students/recog/common.py,
  which uses torch's ctc_loss): P(n) = P("n") + P("0n") for n < 10; P(other) = the rest.
* `vote_boxed`: the WP10b track vote (copy of students/recog/common.vote_boxed).
"""
import cv2
import numpy as np

BLANK = 0
CHARS = "0123456789"


class Runner:
    """One model on one backend: `onnxruntime` (CPU) or `openvino` (device CPU / GPU = Intel iGPU / AUTO).
    OpenVINO reads the same .onnx file. If OpenVINO (or the device) is unavailable, it warns and falls back to
    ONNX Runtime CPU, so a wrong config never stops the race-day program. `static_shape` (N,C,H,W) fixes a dynamic
    input for OpenVINO GPU (iGPU plugins want static shapes). `self.desc` says what actually runs."""

    def __init__(self, path, backend="onnxruntime", device="CPU", threads=1, spinning=False, static_shape=None,
                 log=print):
        self.path = str(path)
        self.backend = backend
        if backend == "openvino":
            try:
                self._init_ov(device, threads, static_shape)
                return
            except Exception as e:  # noqa: BLE001 (any OpenVINO / driver failure -> fall back)
                log("WARNING: OpenVINO %s failed for %s (%s: %s); falling back to ONNX Runtime CPU"
                    % (device, self.path, type(e).__name__, str(e)[:200]))
        self.backend = "onnxruntime"
        self.sess = make_session(self.path, threads, spinning)
        inp = self.sess.get_inputs()[0]
        self.input, self.shape = inp.name, list(inp.shape)
        self.desc = "onnxruntime CPU, %d thread(s)" % threads
        self.run = lambda x: self.sess.run(None, {self.input: x})[0]

    def _init_ov(self, device, threads, static_shape):
        import openvino as ov
        core = ov.Core()
        if device.split(":")[0].split(".")[0] not in ("AUTO", "MULTI", "HETERO") and \
                not any(d.startswith(device.split(".")[0]) for d in core.available_devices):
            raise RuntimeError("device %s not in %s" % (device, core.available_devices))
        model = core.read_model(self.path)
        if static_shape is not None:
            model.reshape({model.inputs[0].get_any_name(): list(static_shape)})
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if device.startswith("CPU"):
            cfg.update({"INFERENCE_NUM_THREADS": int(threads), "NUM_STREAMS": 1})
        cm = core.compile_model(model, device, cfg)
        req = cm.create_infer_request()
        out = cm.output(0)
        ps = cm.inputs[0].get_partial_shape()
        self.shape = [d.get_length() if d.is_static else "dyn" for d in ps]
        self.desc = "openvino %s%s" % (device, (", %d thread(s)" % threads) if device.startswith("CPU") else "")
        self.run = lambda x: req.infer({0: x})[out]


def make_session(path, threads, spinning=False):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if not spinning:
        # idle worker threads sleep instead of busy-waiting: at 10 fps the detector is idle most of the time, and
        # spinning threads would burn a laptop core (and inflate the CPU measurement) for nothing
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


# ------------------------------------------------------------------------------------------------ detector
def letterbox(img, new_shape, rect=False, stride=32):
    """ultralytics LetterBox(new_shape, auto=rect, center=True). Returns (image, gain, (left, top))."""
    h, w = img.shape[:2]
    nh_, nw_ = new_shape
    r = min(nh_ / h, nw_ / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = nw_ - new_unpad[0], nh_ - new_unpad[1]
    if rect:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2, dh / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)


def nms(boxes, scores, iou):
    """Greedy NMS (torchvision semantics: suppress IoU > iou). boxes (N,4) xyxy, sorted inside. Returns kept indices."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        r = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[r]) - np.maximum(x1[i], x1[r]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[r]) - np.maximum(y1[i], y1[r]), 0, None)
        inter = iw * ih
        ov = inter / (area[i] + area[r] - inter + 1e-9)
        order = r[ov <= iou]
    return np.array(keep, dtype=int)


class YoloOnnx:
    def __init__(self, path, imgsz, threads=1, rect=False, spinning=False, classes=None, backend="onnxruntime",
                 device="CPU", log=print):
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
        gpu = backend == "openvino" and not device.startswith("CPU")
        static = (1, 3) + self.imgsz if gpu else None
        self.runner = Runner(path, backend, device, threads, spinning, static_shape=static, log=log)
        shp = self.runner.shape[2:]
        fixed = all(isinstance(d, int) and d > 0 for d in shp)
        if fixed:
            self.imgsz = tuple(int(d) for d in shp)
            rect = False  # a fixed-shape model cannot take the minimum rectangle
        self.rect = rect
        self.classes = classes  # None = all classes
        self.desc = self.runner.desc + (", rect" if rect else ", %dx%d" % self.imgsz)

    def __call__(self, img, conf=0.25, iou=0.7, max_det=300):
        """img BGR uint8 -> (boxes (K,4) xyxy in img pixels, scores (K,), classes (K,))"""
        H, W = img.shape[:2]
        lb, r, (left, top) = letterbox(img, self.imgsz, self.rect)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        y = self.runner.run(np.ascontiguousarray(x))[0].T  # (N, 4 + nc)
        sc = y[:, 4:]
        cls = sc.argmax(1)
        s = sc[np.arange(len(y)), cls]
        k = s > conf
        if self.classes is not None:
            k &= np.isin(cls, self.classes)
        b, s, cls = y[k, :4], s[k], cls[k]
        if not len(s):
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        xyxy = np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]
        off = xyxy + (cls * 7680.0)[:, None]  # class-aware NMS (ultralytics max_wh offset)
        keep = nms(off, s, iou)[:max_det]
        xyxy, s, cls = xyxy[keep], s[keep], cls[keep]
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - left) / r).clip(0, W)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - top) / r).clip(0, H)
        return xyxy.astype(np.float32), s.astype(np.float32), cls


# ------------------------------------------------------------------------------------------------ recognizer
IMG_H, IMG_W = 32, 96


def to_input(img, h=IMG_H, w=IMG_W):
    """students/recog/common.to_input: BGR crop -> (h, w, 3) RGB, aspect kept, padded with the border mean."""
    ih, iw = img.shape[:2]
    nw = max(1, min(w, int(round(iw * h / max(ih, 1)))))
    r = cv2.resize(img, (nw, h), interpolation=cv2.INTER_AREA if ih > h else cv2.INTER_LINEAR)
    border = np.concatenate([r[:, :1].reshape(-1, 3), r[:, -1:].reshape(-1, 3)], 0).mean(0)
    out = np.empty((h, w, 3), np.uint8)
    out[:] = border.astype(np.uint8)
    x_off = (w - nw) // 2
    out[:, x_off:x_off + nw] = r
    return out[:, :, ::-1]


def plain(img, box):
    """students/recog/augment.plain: number box + 10% context, then to_input."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    H, W = img.shape[:2]
    a, b = int(max(0, np.floor(x1 - 0.1 * bw))), int(max(0, np.floor(y1 - 0.1 * bh)))
    c, d = int(min(W, np.ceil(x2 + 0.1 * bw))), int(min(H, np.ceil(y2 + 0.1 * bh)))
    return to_input(img[b:d, a:c])


class Recognizer:
    """Batches of number boxes. On OpenVINO GPU the batch is fixed (static_batch), so inputs are padded to it."""

    def __init__(self, path, threads=1, spinning=False, backend="onnxruntime", device="CPU", log=print,
                 static_batch=8):
        gpu = backend == "openvino" and not device.startswith("CPU")
        self.static_batch = static_batch if gpu else None
        static = (static_batch, 3, IMG_H, IMG_W) if gpu else None
        self.runner = Runner(path, backend, device, threads, spinning, static_shape=static, log=log)
        if self.runner.backend != "openvino":
            self.static_batch = None
        self.desc = self.runner.desc

    def __call__(self, crops):
        """list of (32, 96, 3) RGB uint8 -> logp (T, N, 11)"""
        x = np.stack(crops).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        if self.static_batch is None:
            return self.runner.run(np.ascontiguousarray(x))
        outs, n = [], len(x)
        for i in range(0, n, self.static_batch):
            chunk = x[i:i + self.static_batch]
            k = len(chunk)
            if k < self.static_batch:
                chunk = np.concatenate([chunk, np.zeros((self.static_batch - k,) + chunk.shape[1:], np.float32)])
            outs.append(self.runner.run(np.ascontiguousarray(chunk))[:, :k])
        return np.concatenate(outs, 1)


def _logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):  # log(0) = -inf is the intended "impossible path"
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)


class RosterScorer:
    """numpy CTC forward over every roster variant ("n", and "0n" for n < 10). __call__(logp (T,B,C)) ->
    (P (B, n_numbers), p_other (B,)). Matches torch.nn.functional.ctc_loss(reduction='none') to float precision."""

    def __init__(self, numbers):
        self.numbers = [int(n) for n in numbers]
        var, owner = [], []
        for j, n in enumerate(self.numbers):
            for v in ([str(n), "0" + str(n)] if n < 10 else [str(n)]):
                var.append(v)
                owner.append(j)
        self.var, self.owner = var, np.array(owner)
        L = max(len(v) for v in var)
        S = 2 * L + 1
        ext = np.zeros((len(var), S), dtype=int)       # extended label: blank, c1, blank, c2, ..., blank (padded)
        valid = np.zeros((len(var), S), dtype=bool)
        skip = np.zeros((len(var), S), dtype=bool)     # s may come from s-2 (non-blank, differs from s-2)
        self.final = []
        for i, v in enumerate(var):
            lab = [CHARS.index(ch) + 1 for ch in v]
            e = [BLANK]
            for c in lab:
                e += [c, BLANK]
            ext[i, :len(e)] = e
            valid[i, :len(e)] = True
            for s in range(2, len(e)):
                skip[i, s] = e[s] != BLANK and e[s] != e[s - 2]
            self.final.append((len(e) - 1, len(e) - 2))
        self.ext, self.valid, self.skip = ext, valid, skip
        self.fin = np.array(self.final)

    def __call__(self, logp):
        T, B, C = logp.shape
        V, S = self.ext.shape
        lp = logp[:, :, self.ext]                               # (T, B, V, S)
        neg = -np.inf
        alpha = np.full((B, V, S), neg)
        alpha[:, :, 0] = lp[0, :, :, 0]
        alpha[:, :, 1] = np.where(self.valid[:, 1], lp[0, :, :, 1], neg)
        for t in range(1, T):
            a1 = np.concatenate([np.full((B, V, 1), neg), alpha[:, :, :-1]], 2)
            a2 = np.concatenate([np.full((B, V, 2), neg), alpha[:, :, :-2]], 2)
            a2 = np.where(self.skip[None], a2, neg)
            stack = np.stack([alpha, a1, a2], 0)
            alpha = _logsumexp(stack, 0) + lp[t]
            alpha = np.where(self.valid[None], alpha, neg)
        vi = np.arange(V)
        ll = np.logaddexp(alpha[:, vi, self.fin[:, 0]], alpha[:, vi, self.fin[:, 1]])  # (B, V)
        pv = np.exp(ll)
        P = np.zeros((B, len(self.numbers)))
        np.add.at(P, (slice(None), self.owner), pv)
        P = P.clip(0, 1)
        return P, (1 - P.sum(1)).clip(0, 1)


def vote_boxed(P, has_box, p_support=0.5, score="share"):
    """students/recog/common.vote_boxed (WP10b). Returns (best index, score, runner-up score, n_boxed, support, scores
    over the roster); best = -1 when no crop has a box."""
    P = np.asarray(P, dtype=np.float64)
    hb = np.asarray(has_box, dtype=bool)
    if len(P) == 0 or not hb.any():
        return -1, 0.0, 0.0, 0, 0, None
    Q = P[hb]
    s = Q.mean(0)
    if score == "share":
        s = Q.sum(0) / max(float(Q.max(1).sum()), 1e-9)
    o = np.argsort(-s, kind="stable")
    j = int(o[0])
    sup = int(((Q.argmax(1) == j) & (Q[:, j] >= p_support)).sum())
    return j, float(s[j]), float(s[o[1]]) if len(o) > 1 else 0.0, int(hb.sum()), sup, s


def enlarge(b, pad, W, H):
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    return [max(0, x1 - pad * bw), max(0, y1 - pad * bh), min(W, x2 + pad * bw), min(H, y2 + pad * bh)]


def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

