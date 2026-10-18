# PARSeq model code adapted from baudm/parseq (Scene Text Recognition Model Hub), strhub/models/parseq/{model,modules}.py
# and strhub/models/parseq/system.py (permutation masks). Copyright 2022 Darwin Bautista.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Changes (WP10b): digits-only token set (EOS, 0-9, BOS, PAD) sliced from the pretrained 94-char model, tokenizer-free
# fixed-length decoding (ONNX-friendly, no early exit), and a CTC-shaped output wrapper so the PARSeq reader is a
# drop-in for the WP10 CRNN interface: (N,3,32,W) RGB in [0,1] -> (T,N,11) log-probs, class 0 = blank, d+1 = digit d.
"""PARSeq-tiny for the number recognizer (WP10b). Weights: weights/parseq/parseq_tiny-e7a21b54.pt (baudm/parseq v1.0.0
release, Apache-2.0; trained on real scene-text datasets, 6.0M params, 32x128 input, 94 chars)."""
import math
from functools import partial
from itertools import permutations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import transformer
from timm.models.vision_transformer import PatchEmbed, VisionTransformer

ROOT = Path(__file__).resolve().parents[2]
PRETRAINED = ROOT / "weights" / "parseq" / "parseq_tiny-e7a21b54.pt"
EOS, BOS, PAD = 0, 11, 12          # digits d -> d + 1 (same layout as the CRNN's CTC classes with EOS in the blank slot)
NUM_TOKENS = 13


def init_weights(module: nn.Module, name: str = "", exclude=()):
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def named_apply(fn, module, name="", depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class DecoderLayer(nn.Module):
    """Transformer decoder layer with two-stream attention (XLNet), pre-LN."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward_stream(self, tgt, tgt_norm, tgt_kv, memory, tgt_mask, tgt_key_padding_mask):
        tgt2, _ = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask,
                                 need_weights=False)
        tgt = tgt + self.dropout1(tgt2)
        tgt2, _ = self.cross_attn(self.norm1(tgt), memory, memory, need_weights=False)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        return tgt + self.dropout3(tgt2)

    def forward(self, query, content, memory, query_mask=None, content_mask=None, content_key_padding_mask=None,
                update_content: bool = True):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm, memory, content_mask,
                                          content_key_padding_mask)
        return query, content


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, content, memory, query_mask=None, content_mask=None, content_key_padding_mask=None):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(query, content, memory, query_mask, content_mask, content_key_padding_mask,
                                 update_content=not last)
        return self.norm(query)


class Encoder(VisionTransformer):
    def __init__(self, img_size=(32, 128), patch_size=(4, 8), embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0):
        super().__init__(img_size, patch_size, 3, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=True, embed_layer=PatchEmbed, num_classes=0, global_pool="",
                         class_token=False)

    def forward(self, x):
        return self.forward_features(x)


class TokenEmbedding(nn.Module):
    def __init__(self, charset_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class PARSeq(nn.Module):
    """PARSeq-tiny (embed 192, enc depth 12 / 3 heads, dec depth 1 / 6 heads), digits-only tokens."""

    def __init__(self, max_label_length=4, embed_dim=192, enc_num_heads=3, enc_depth=12, dec_num_heads=6,
                 dec_depth=1, mlp_ratio=4, dropout=0.1, refine_iters=1, img_w=128):
        super().__init__()
        self.max_label_length = max_label_length
        self.refine_iters = refine_iters
        self.img_w = img_w
        self.encoder = Encoder((32, img_w), (4, 8), embed_dim, enc_depth, enc_num_heads, mlp_ratio)
        layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * mlp_ratio, dropout)
        self.decoder = Decoder(layer, dec_depth, nn.LayerNorm(embed_dim))
        self.head = nn.Linear(embed_dim, NUM_TOKENS - 2)
        self.text_embed = TokenEmbedding(NUM_TOKENS, embed_dim)
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        named_apply(partial(init_weights, exclude=["encoder"]), self)
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

    def encode(self, img):
        return self.encoder(img)

    def decode(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, tgt_query=None, tgt_query_mask=None):
        N, L = tgt.shape
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self, images: Tensor) -> Tensor:
        """AR decoding over max_label_length + 1 steps (no early exit) + refine_iters cloze refinements.
        images: normalised (N,3,32,128). Returns logits (N, max_label_length + 1, 11)."""
        bs = images.shape[0]
        num_steps = self.max_label_length + 1
        memory = self.encode(images)
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)
        tgt_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=images.device), 1)
        toks = [torch.full((bs, 1), BOS, dtype=torch.long, device=images.device)]
        logits = []
        for i in range(num_steps):
            j = i + 1
            tgt_in = torch.cat(toks, 1)
            out = self.decode(tgt_in, memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                              tgt_query_mask=tgt_mask[i:j, :j])
            p_i = self.head(out)
            logits.append(p_i)
            if j < num_steps:
                toks.append(p_i[:, 0].argmax(-1, keepdim=True))
        logits = torch.cat(logits, dim=1)
        if self.refine_iters:
            query_mask = tgt_mask & ~torch.triu(torch.ones_like(tgt_mask), 2)   # cloze mask
            bos = torch.full((bs, 1), BOS, dtype=torch.long, device=images.device)
            for _ in range(self.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = (tgt_in == EOS).int().cumsum(-1) > 0
                out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, pos_queries, query_mask[:, :tgt_in.shape[1]])
                logits = self.head(out)
        return logits


def load_pretrained(model: PARSeq, path=PRETRAINED):
    """94-char pretrained weights -> digits-only model: head rows [EOS, '0'..'9'], embedding rows [EOS, '0'..'9', BOS,
    PAD] (BOS = 95, PAD = 96 in the original), first max_label_length + 1 position queries."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd["head.weight"] = sd["head.weight"][:11]
    sd["head.bias"] = sd["head.bias"][:11]
    e = sd["text_embed.embedding.weight"]
    sd["text_embed.embedding.weight"] = torch.cat([e[:11], e[95:97]], 0)
    sd["pos_queries"] = sd["pos_queries"][:, :model.max_label_length + 1]
    gw = model.img_w // 8
    if gw != 16:                                   # narrower input (img_w 64): resample the 8x16 position grid to 8xgw
        pe = sd["encoder.pos_embed"].reshape(1, 8, 16, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(8, gw), mode="bicubic", align_corners=False)
        sd["encoder.pos_embed"] = pe.permute(0, 2, 3, 1).reshape(1, 8 * gw, -1)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


# ---------------------------------------------------------------- permutation language modelling (training)

class PermGen:
    """gen_tgt_perms / generate_attn_masks of strhub PARSeq system (perm_num 6, forward + mirrored)."""

    def __init__(self, perm_num=6, perm_forward=True, perm_mirrored=True, seed=0):
        self.rng = np.random.default_rng(seed)
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward, self.perm_mirrored = perm_forward, perm_mirrored

    def perms(self, tgt):
        dev = tgt.device
        max_num_chars = tgt.shape[1] - 2
        if max_num_chars <= 1:
            return torch.arange(max_num_chars + 2, device=dev).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=dev)] if self.perm_forward else []
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        if max_num_chars < 5:
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=dev)[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool) and num_gen_perms > len(perms):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=dev) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            comp = perms.flip(-1)
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=dev)
        return perms

    @staticmethod
    def masks(perm):
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), dtype=torch.bool, device=perm.device)
        for i in range(sz):
            mask[perm[i], perm[i + 1:]] = True
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=perm.device)] = True
        query_mask = mask[1:, :-1]
        return content_mask, query_mask


def encode_labels(texts, device="cpu"):
    rows = [[BOS] + [int(c) + 1 for c in t] + [EOS] for t in texts]
    L = max(len(r) for r in rows)
    return torch.tensor([r + [PAD] * (L - len(r)) for r in rows], dtype=torch.long, device=device)


def plm_loss(model: PARSeq, images, texts, permgen: PermGen):
    tgt = encode_labels(texts, images.device)
    memory = model.encode(images)
    tgt_perms = permgen.perms(tgt)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    tgt_padding_mask = (tgt_in == PAD) | (tgt_in == EOS)
    loss, numel = 0, 0
    n = (tgt_out != PAD).sum().item()
    for i, perm in enumerate(tgt_perms):
        tgt_mask, query_mask = permgen.masks(perm)
        out = model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
        logits = model.head(out).flatten(end_dim=1)
        loss = loss + n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=PAD)
        numel += n
        if i == 1:
            tgt_out = torch.where(tgt_out == EOS, PAD, tgt_out)
            n = (tgt_out != PAD).sum().item()
    return loss / numel


# ---------------------------------------------------------------- CTC-shaped wrapper (WP10 interface)

def positions_to_ctc(probs, eps=1e-6):
    """probs (N, S, 11) per-position distributions over [EOS, 0-9] -> CTC-style log-probs (2S, N, 11).
    Position k becomes a frame with P(blank) = P(sequence ended at or before k) (cumulative EOS, assuming
    independence) and P(digit d) = (1 - ended) * p_k(d) / (1 - p_k(EOS)); a certain-blank separator frame follows
    each position so repeated digits ("11") decode. For a confident model the CTC likelihood of "ab" is ~
    p1(a) p2(b) p3(EOS), so greedy decoding, RosterScorer and the soft vote work unchanged."""
    e = probs[..., 0]
    ended = 1 - torch.exp(torch.cumsum(torch.log((1 - e).clamp_min(eps)), dim=1))   # cumprod (ONNX has no cumprod)
    dig = probs[..., 1:] / (1 - e).clamp_min(eps).unsqueeze(-1) * (1 - ended).unsqueeze(-1)
    frame = torch.cat([ended.unsqueeze(-1), dig], -1)
    sep = torch.zeros_like(frame)
    sep[..., 0] = 1
    seq = torch.stack([frame, sep], 2).flatten(1, 2)            # N, 2S, 11
    return torch.log(seq.clamp_min(eps)).permute(1, 0, 2).contiguous()


class ParseqCTC(nn.Module):
    """(N,3,32,W) RGB in [0,1] (the WP10 to_input / plain() crop, W = 96) -> resize to 32x128 -> PARSeq ->
    CTC-shaped (T,N,11) log-probs."""

    def __init__(self, **kw):
        super().__init__()
        self.net = PARSeq(**kw)

    def prep(self, x):
        x = F.interpolate(x, size=(32, self.net.img_w), mode="bilinear", align_corners=False)
        return (x - 0.5) / 0.5

    def forward(self, x):
        logits = self.net(self.prep(x))
        return positions_to_ctc(logits.softmax(-1))
