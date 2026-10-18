"""Local VLM number reader (Qwen-VL via transformers).

dtype: Reader(dtype=...) or env LAPVISION_VLM_DTYPE = fp16 | bf16 | fp32 | auto (default auto).
auto = bfloat16 on GPUs with native bf16 (Ampere+, compute capability >= 8), float16 on older ones
(Turing / RTX 2080 Ti), float32 on CPU. Note: torch.cuda.is_bf16_supported() returns True on Turing because
it counts emulation, so we ask for native support explicitly.
Session-2 check (LESSONS.md, WP0): fp16 on the 2080 Ti reproduces the stored bf16 reads.
"""
import os
import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = ("This is a muddy off-road Baja SAE race car. Read its race number "
          "(painted on the rear firewall panel or on the white side plate). "
          "Answer with digits only, or 'none' if no number is legible.")

DTYPES = {"fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
          "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
          "fp32": torch.float32, "float32": torch.float32}


def native_bf16(device="cuda"):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older torch
        return torch.cuda.get_device_capability()[0] >= 8


def resolve_dtype(dtype=None, device="cuda"):
    """dtype: torch.dtype, a name from DTYPES, 'auto' or None (-> env LAPVISION_VLM_DTYPE, default auto)."""
    if dtype is None:
        dtype = os.environ.get("LAPVISION_VLM_DTYPE", "auto")
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype = str(dtype).lower()
    if dtype == "auto":
        if not str(device).startswith("cuda"):
            return torch.float32
        return torch.bfloat16 if native_bf16(device) else torch.float16
    return DTYPES[dtype]


class Reader:
    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", device="cuda", dtype=None, attn=None):
        """attn: transformers attn_implementation ('sdpa', 'eager'); None = library default (sdpa)."""
        self.device = device
        self.dtype = resolve_dtype(dtype, device)
        self.proc = AutoProcessor.from_pretrained(model_id)
        kw = dict(dtype=self.dtype, device_map=device)
        if attn:
            kw["attn_implementation"] = attn
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()

    def _inputs(self, paths, prompt, max_side):
        msgs, ims = [], []
        for p in paths:
            im = Image.open(p).convert("RGB")
            s = max_side / max(im.size)
            if s < 1:
                im = im.resize((int(im.width * s), int(im.height * s)))
            ims.append(im)
            m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            msgs.append(self.proc.apply_chat_template(m, add_generation_prompt=True))
        self.proc.tokenizer.padding_side = "left"
        return ims, self.proc(text=msgs, images=ims, return_tensors="pt", padding=True).to(self.device)

    @torch.inference_mode()
    def generate(self, paths, prompt=PROMPT, max_side=640, max_new_tokens=8):
        """Batch generate. Returns (list of raw texts, list of resized PIL images as fed to the model)."""
        ims, inp = self._inputs(paths, prompt, max_side)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        txt = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return txt, ims

    def read(self, paths, max_side=640):
        """Batch read. Returns list of (raw_text, number_or_None)."""
        txt, _ = self.generate(paths, PROMPT, max_side, 8)
        res = []
        for t in txt:
            m = re.search(r"\d{1,3}", t)
            res.append((t.strip(), int(m.group()) if m else None))
        return res
