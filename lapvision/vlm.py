"""Local VLM number reader (Qwen-VL via transformers)."""
import re
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = ("This is a muddy off-road Baja SAE race car. Read its race number "
          "(painted on the rear firewall panel or on the white side plate). "
          "Answer with digits only, or 'none' if no number is legible.")


class Reader:
    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map=device).eval()

    @torch.inference_mode()
    def read(self, paths, max_side=640):
        """Batch read. Returns list of (raw_text, number_or_None)."""
        msgs, ims = [], []
        for p in paths:
            im = Image.open(p).convert("RGB")
            s = max_side / max(im.size)
            if s < 1:
                im = im.resize((int(im.width * s), int(im.height * s)))
            ims.append(im)
            m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            msgs.append(self.proc.apply_chat_template(m, add_generation_prompt=True))
        self.proc.tokenizer.padding_side = "left"
        inp = self.proc(text=msgs, images=ims, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**inp, max_new_tokens=8, do_sample=False)
        txt = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        res = []
        for t in txt:
            m = re.search(r"\d{1,3}", t)
            res.append((t.strip(), int(m.group()) if m else None))
        return res
