"""WP2b: zero-shot CLIP shot tags -> extra columns in work/<id>/shots.csv + shot_tags.done

usage: python teacher/shot_tags.py [--workers 2] [--force] [id ...]
       python teacher/shot_tags.py --sheet <id> [n=50] [seed=0]   contact sheet of n random shots with their tags (QA)
       python teacher/shot_tags.py --eval teacher/qa_labels/shot_tags_tune.csv   accuracy vs hand labels (cached emb)

Per shot, 1-3 frames (keyframes inside the shot when available, else an accurate seek; transition shots get 1 frame
at their middle), CLIP ViT-B/32 on the CPU, center 16:9 -> square crop. Each tag has several prompts; the frame score
is the softmax over tags of the max prompt similarity, shot score = mean over its frames.
Columns added: tag (argmax), tag_conf (its probability), p_<tag> for every tag.
Tags are for QA stratification and dataset balancing only (never to drop data).
"""
import csv
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, cli_ids, ffmpeg_frames, frame_at, keyframe_times, run_stage, select_videos, wdir  # noqa: E402

CLIP_W = next((p for p in (ROOT / "lapvision" / "weights" / "clip" / "ViT-B-32.pt", ROOT / "lapvision" / "ViT-B-32.pt")
               if p.exists()), ROOT / "lapvision" / "weights" / "clip" / "ViT-B-32.pt")
# Prompt sets tuned on 150 hand-labelled shots (teacher/qa_labels/shot_tags_tune.csv): 75% -> 79%. Notes: a generic
# "full screen broadcast graphic with text and logos" prompt pulled studio shots (desk with a big logo + TV) into
# graphics; sponsor commercials / promo videos count as graphics_fullscreen; mean-of-prompts pooling was worse than max.
TAGS = {
    "track_wide": [
        "a wide shot of an off-road dirt race track with small buggies far away",
        "a wide view of a dirt racing course with spectators and tents in the distance",
        "a landscape view of a muddy off-road track and a crowd far away",
    ],
    "track_closeup": [
        "a close-up of an off-road buggy driving on a dirt track",
        "a telephoto shot of a single off-road race car jumping over dirt",
        "a small off-road race car with a number plate driving through mud",
    ],
    "onboard": [
        "an onboard camera view from inside an off-road buggy, the roll cage and the driver's helmet in front",
        "a POV action camera footage from the cockpit of a race car with steel tubes framing the view",
        "a dirt track seen from inside an off-road buggy, a helmet at the bottom of the frame",
    ],
    "pits_paddock": [
        "off-road buggies parked in a pit area with team members working on them",
        "mechanics repairing a race car under a tent in the paddock",
        "a row of parked off-road race cars and people standing around them",
    ],
    "drone_aerial": [
        "an aerial drone photo looking down at a dirt race track",
        "a bird's eye view from high above of an off-road course and fields",
    ],
    "studio_interview": [
        "two people talking at a desk in a studio with microphones",
        "a person being interviewed holding a microphone",
        "commentators wearing headsets in a broadcast booth",
        "presenters sitting behind a white desk with a big logo in a TV studio",
        "a talk show with people at a table and a television screen on the wall behind them",
        "a speaker on a stage at a conference",
        "people sitting at a table talking",
        "a group of people seated at a long table",
    ],
    "graphics_fullscreen": [
        "a flat 2D infographic with text and logos",
        "a results table with names and numbers on a screen",
        "a sponsor advertisement with a big company logo",
        "a title card with a logo on a plain background",
        "a cinematic commercial advertisement",
        "a computer generated 3D animation",
        "a corporate promotional video with white text over the image",
    ],
    "crowd_other": [
        "a crowd of spectators standing behind a fence",
        "a group of people posing for a photo",
        "an award ceremony on a stage",
        "people walking outdoors at an event",
    ],
}
TAG_NAMES = list(TAGS)
TEMP = 100.0
_model = None


def model():
    global _model
    if _model is None:
        import clip
        import torch
        torch.set_num_threads(int(os.environ.get("TAG_THREADS", "4")))
        m, pre = clip.load(str(CLIP_W), device="cpu")
        m.eval()
        with torch.no_grad():
            tf = []
            for t in TAG_NAMES:
                e = m.encode_text(clip.tokenize(TAGS[t]))
                tf.append(e / e.norm(dim=-1, keepdim=True))
        _model = (m, pre, tf)
    return _model


def embed(frames):
    """L2-normalised CLIP image embeddings (n, 512) float32."""
    import torch
    from PIL import Image
    m, _, _ = model()
    pre = model()[1]
    x = torch.stack([pre(Image.fromarray(f)) for f in frames])
    with torch.no_grad():
        im = m.encode_image(x).float()
        im = im / im.norm(dim=-1, keepdim=True)
    return im.numpy()


def score(emb):
    """Per-frame tag probabilities from embeddings: softmax over tags of the best prompt similarity."""
    import torch
    tf = model()[2]
    im = torch.from_numpy(emb).float()
    sims = torch.stack([(im @ t.float().T).max(1).values for t in tf], 1)
    return torch.softmax(TEMP * sims, 1).numpy()


def read_shots(vid):
    with open(wdir(vid) / "shots.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    base = ["shot_id", "t0", "t1", "dur", "cut_in", "is_transition"]
    return [{k: r[k] for k in base} for r in rows]


def sample_times(shots, kts):
    """Per shot: up to 3 keyframe times inside the shot (0.3 s margin), else the shot middle (seek)."""
    kts = np.asarray(kts)
    plan = []
    for s in shots:
        t0, t1 = float(s["t0"]), float(s["t1"])
        if int(s["is_transition"]):
            plan.append([("seek", (t0 + t1) / 2)])
            continue
        inside = kts[(kts >= t0 + 0.3) & (kts <= t1 - 0.3)]
        if len(inside) >= 3:
            pick = inside[np.linspace(0, len(inside) - 1, 5)[1:4].round().astype(int)]
        else:
            pick = inside
        if len(pick) == 0:
            plan.append([("seek", (t0 + t1) / 2)])
        else:
            plan.append([("key", float(t)) for t in pick])
    return plan


def frames_for(path, plan, size=(398, 224)):
    """Decode all keyframes once (fast), seek for the rest."""
    need_key = sorted({t for p in plan for k, t in p if k == "key"})
    got = {}
    if need_key:
        want = set(need_key)
        for t, f in ffmpeg_frames(path, size=size, pix="rgb24", threads=2, keyframes_only=True):
            if t in want:
                got[t] = f.copy()
    out = []
    for p in plan:
        fs = []
        for k, t in p:
            f = got.get(t) if k == "key" else frame_at(path, t, size, "rgb24")
            if f is not None:
                fs.append(f)
        out.append(fs)
    return out


def shot_embeddings(row, shots):
    """CLIP embeddings of 1-3 frames per shot, cached in work/<id>/shot_clip.npz (keyed by the shot boundaries)."""
    cache = wdir(row["id"]) / "shot_clip.npz"
    key = np.array([[float(s["t0"]), float(s["t1"])] for s in shots], np.float32)
    if cache.exists():
        z = np.load(cache)
        if z["key"].shape == key.shape and np.allclose(z["key"], key):
            return z["emb"].astype(np.float32), z["shot"]
    path = ROOT / row["path"]
    plan = sample_times(shots, keyframe_times(path))
    frames = frames_for(path, plan)
    flat = [f for fs in frames for f in fs]
    shot_idx = np.array([i for i, fs in enumerate(frames) for _ in fs], np.int32)
    emb = np.concatenate([embed(flat[i:i + 64]) for i in range(0, len(flat), 64)]) if flat else np.zeros((0, 512))
    np.savez_compressed(cache, emb=emb.astype(np.float16), shot=shot_idx, key=key)
    return emb.astype(np.float32), shot_idx


def process(row):
    vid = row["id"]
    shots = read_shots(vid)
    emb, shot_idx = shot_embeddings(row, shots)
    probs = score(emb) if len(emb) else np.zeros((0, len(TAG_NAMES)))
    for i, s in enumerate(shots):
        sel = shot_idx == i
        if sel.any():
            p = probs[sel].mean(0)
        else:
            p = np.full(len(TAG_NAMES), 1 / len(TAG_NAMES))
        j = int(np.argmax(p))
        s["tag"], s["tag_conf"] = TAG_NAMES[j], round(float(p[j]), 3)
        for t, v in zip(TAG_NAMES, p):
            s[f"p_{t}"] = round(float(v), 3)
    fields = list(shots[0].keys())
    tmp = wdir(vid) / "shots.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(shots)
    os.replace(tmp, wdir(vid) / "shots.csv")
    hours = {}
    for s in shots:
        hours[s["tag"]] = hours.get(s["tag"], 0) + float(s["dur"]) / 3600
    return " ".join(f"{t}={h:.2f}h" for t, h in sorted(hours.items(), key=lambda x: -x[1]))


def sheet(vid, n=50, seed=0):
    """Contact sheet: n random non-transition shots, 1 frame each, labelled with id, tag and conf."""
    import cv2
    row = select_videos([vid])[0]
    with open(wdir(vid) / "shots.csv", newline="", encoding="utf-8") as f:
        shots = [s for s in csv.DictReader(f) if s["is_transition"] == "0"]
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(shots), size=min(n, len(shots)), replace=False))
    tiles = []
    for i in pick:
        s = shots[i]
        t = (float(s["t0"]) + float(s["t1"])) / 2
        f = frame_at(ROOT / row["path"], t, (320, 180), "bgr24")
        f = f.copy() if f is not None else np.zeros((180, 320, 3), np.uint8)
        txt = f"{s['shot_id']} {s['tag']} {float(s['tag_conf']):.2f}"
        cv2.rectangle(f, (0, 0), (320, 18), (0, 0, 0), -1)
        cv2.putText(f, txt, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(f)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
    out = wdir(vid) / "qa"
    out.mkdir(exist_ok=True)
    for k in range(0, len(rows), 5):
        p = out / f"tags_s{seed}_{k // 5}.jpg"
        cv2.imwrite(str(p), np.vstack(rows[k:k + 5]), [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(p)


ABBR = {"W": "track_wide", "C": "track_closeup", "O": "onboard", "P": "pits_paddock", "D": "drone_aerial",
        "S": "studio_interview", "G": "graphics_fullscreen", "X": "crowd_other"}


def evaluate(label_csv):
    """Accuracy + confusion of the current prompts on hand-labelled shots (id,shot_id,truth with ABBR letters),
    from the cached embeddings (no decoding)."""
    with open(label_csv, newline="", encoding="utf-8") as f:
        lab = list(csv.DictReader(f))
    cache = {}
    rows = []
    for r in lab:
        vid = r["id"]
        if vid not in cache:
            z = np.load(wdir(vid) / "shot_clip.npz")
            cache[vid] = (z["emb"].astype(np.float32), z["shot"])
        emb, sh = cache[vid]
        sel = sh == int(r["shot_id"])
        p = score(emb[sel]).mean(0)
        rows.append((ABBR[r["truth"]], TAG_NAMES[int(np.argmax(p))], vid))
    acc = np.mean([t == q for t, q, _ in rows])
    print(f"accuracy {acc:.1%} on {len(rows)} shots")
    short = {v: k for k, v in ABBR.items()}
    print("truth/pred " +" ".join(short[t] for t in TAG_NAMES))
    for t in TAG_NAMES:
        cnt = [sum(1 for a, b, _ in rows if a == t and b == q) for q in TAG_NAMES]
        if sum(cnt):
            print(f"{short[t]:>10} " + " ".join(f"{c:1d}" for c in cnt))
    per = {}
    for t, q, v in rows:
        per.setdefault(v, []).append(t == q)
    print(" ".join(f"{v}:{np.mean(x):.0%}" for v, x in per.items()))
    return acc


if __name__ == "__main__":
    if sys.argv[1:2] == ["--eval"]:
        evaluate(sys.argv[2])
        sys.exit()
    if sys.argv[1:2] == ["--sheet"]:
        a = sys.argv[2:]
        sheet(a[0], *(int(x) for x in a[1:]))
    else:
        workers, force, ids, _ = cli_ids(sys.argv[1:])
        run_stage("shot_tags", process, select_videos(ids), workers=workers, force=force, requires=("shots",))
