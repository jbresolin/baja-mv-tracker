"""WP5 (CPU): per-track identity labels, negative frames, QA batches and the WP2b/WP4/WP5 summary.

  python teacher/labels.py [id ...] [--force]        -> work/<id>/labels.parquet, negatives.parquet, labels.done
  python teacher/labels.py qa [EV ...] [--n 150] [--floor 6] [--seed 0]   -> out/qa/<EV>_teacher.jsonl
  python teacher/labels.py summary                   -> data/wp45_summary.csv, data/wp45_height.csv, agreement tables

Per car track (every buggy track of crop_tracks.parquet; identity only for readable ones):
  VLM consensus   idlib.vote over the track's plain reads (reads.parquet), snapped to rosters/<EV>.csv: majority, kept
                  when vote share >= 0.5 and >= 3 votes (share = votes / crops read).
  overlay id      from overlay_ids.parquet (WP2b). An overlay interval is assigned to the 'sizeable' tracks of a shot
                  (not screen_fixed / in_overlay / nested, h_px_max >= SIZE_PX, >= MIN_OVL_S s inside the interval) only
                  when at most `n_numbers` of them are on screen at the same time (1 for a lower third: sequential
                  fragments of the same car are all labelled; 2 for an SPP heat box). ovl_n_tracks records how many
                  tracks shared the assignment.
  final label     both agree -> 'both-agree'; disagree -> 'conflict' (label null, QA); only VLM -> 'vlm'; only a
                  lower-third overlay -> 'overlay'; heat (2 candidates): VLM consensus in the pair -> 'both-agree',
                  outside -> 'conflict', no consensus but >= 2 raw votes for exactly one of the pair -> 'heat-vlm-weak',
                  else no label (candidates kept in ovl_numbers).
  number boxes    grounded.parquet boxes whose label equals the consensus, per crop (crop pixels) and mapped back to
                  source-frame pixels (number_boxes_src).
Car check (teacher/verify.py): car_verified = yes / no / unsure from the mean yes-probability of 1-2 crops (>= 0.7 yes,
  <= 0.3 no; a 'no' on crops < 60 px becomes unsure); screen_fixed and ego_residue tracks are 'no'; None = not checked
  (median box < 30 px). Ego residue / screen_fixed tracks never get an identity label (source not_car_flag).
  is_car = 'no' for screen_fixed / ego residue, 'yes' when the track has a roster-number consensus or an overlay
  agreement (the yes/no check misses cars surrounded by marshals), else car_verified ('unchecked' below 30 px).
  Flags come from the current tracks.parquet (in_overlay is soft and carried, not used to drop tracks).
Negatives: 4 fps frames with no buggy detection (conf >= NEG_CONF) in the frame and +-1 s, outside transition shots and
  >= 1 s from a shot boundary; at most one per NEG_SPACING s per shot; total NEG_RATE x frames-with-cars per video,
  split evenly over shot tags (leftover redistributed), so no tag dominates.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, cli_ids, run_stage, select_videos, wdir  # noqa: E402
import idlib  # noqa: E402
import verify  # noqa: E402

VERIFY_NO_MIN_H = 60
SIZE_PX = 60
MIN_OVL_S = 1.0
EDGE_MAX = 0.8
AMBIG = 0.3
NEG_CONF = 0.10
NEG_WIN = 4          # frames (+-1 s at 4 fps)
NEG_SPACING = 10.0   # s
NEG_RATE = 0.10
NEG_MIN = 200
FPS = 4.0
HBINS = [0, 80, 100, 130, 170, 230, 330, 2000]
PROP_SHARE = 2 / 3


def nn(x):
    return None if x is None or (isinstance(x, float) and x != x) else x


def jdump(x):
    return json.dumps(x, ensure_ascii=False)


def track_votes(R, roster):
    out = {}
    for tid, g in R.sort_values("frame_idx").groupby("track_id"):
        nums = [None if pd.isna(x) else int(x) for x in g.num]
        v = idlib.vote(nums, roster)
        v["all_votes"] = dict(Counter(str(x) for x in nums))
        out[int(tid)] = v
    return out


def concurrency(iv):
    ev = sorted([(a, 1) for a, b in iv] + [(b, -1) for a, b in iv], key=lambda x: (x[0], x[1]))
    c = m = 0
    for _, s in ev:
        c += s
        m = max(m, c)
    return m


def assign_overlays(ct, O, shots):
    """-> {track_id: dict(numbers, mode, n_tracks, match, ovl_t0, ovl_t1, ambiguous)}

    Candidates: not screen_fixed / ego residue / verified 'no' / nested, h_px_max >= SIZE_PX, >= MIN_OVL_S inside the
    overlay interval, and not glued to a frame edge (edge_touch < EDGE_MAX: in DIN shots the cars waiting in the queue
    at the side of the picture are what the lower third does NOT announce). Per overlay row and shot, assigned only if
    at most n_numbers candidates are on screen at once. A track overlapped by several rows takes the row with the
    largest overlap; it is 'ambiguous' when a row with other numbers overlaps it >= AMBIG x as long."""
    out = {}
    if O is None or not len(O):
        return out
    sf = ct.sf_effective.to_numpy(bool) | ct.ego_residue.to_numpy(bool)
    notcar = (ct.car_verified == "no").to_numpy()
    ok = ct[~sf & ~notcar]  # in_overlay is a soft flag (WP3): not excluded
    ok = ok[(ok.nested_in < 0) & (ok.h_px_max >= SIZE_PX) & (ok.edge_touch < EDGE_MAX)]
    hits = {}
    for o in O.itertuples():
        nums = [int(x) for x in o.numbers]
        cap = 1 if o.mode == "single" else 2
        cand = ok[(ok.t1 >= o.t0) & (ok.t0 <= o.t1)].copy()
        cand["inside"] = np.minimum(cand.t1, o.t1) - np.maximum(cand.t0, o.t0)
        cand = cand[cand.inside >= MIN_OVL_S]
        for sid, g in cand.groupby("shot_id"):
            if concurrency(list(zip(g.t0, g.t1))) > cap:
                continue
            for r in g.itertuples():
                hits.setdefault(int(r.track_id), []).append(
                    (float(r.inside), dict(numbers=nums, mode=o.mode, n_tracks=len(g), match=float(o.match),
                                           ovl_t0=float(o.t0), ovl_t1=float(o.t1), ambiguous=False)))
    for tid, h in hits.items():
        h.sort(key=lambda x: -x[0])
        best = h[0][1]
        other = [x for x in h[1:] if x[1]["numbers"] != best["numbers"]]
        if other and other[0][0] >= AMBIG * h[0][0]:
            best["ambiguous"] = True
        out[tid] = best
    return out


def final_label(v, ov):
    if ov is None or ov.get("ambiguous"):
        if v and v["consensus"]:
            return v["number"], "vlm"
        return None, "none"
    nums = ov["numbers"]
    if ov["mode"] == "single" or len(nums) == 1:
        n = nums[0]
        if v and v["consensus"]:
            return (n, "both-agree") if v["number"] == n else (None, "conflict")
        return n, "overlay"
    # heat
    if v and v["consensus"]:
        return (v["number"], "both-agree") if v["number"] in nums else (None, "conflict")
    if v:
        av = {int(k): c for k, c in v["all_votes"].items() if k != "None"}
        a, b = nums
        if av.get(a, 0) >= 2 and av.get(b, 0) == 0:
            return a, "heat-vlm-weak"
        if av.get(b, 0) >= 2 and av.get(a, 0) == 0:
            return b, "heat-vlm-weak"
    return None, "none"


def negatives(d, shots, W=1280):
    D = pd.read_parquet(d / "dets_raw.parquet", columns=["frame_idx", "cls", "conf"])
    nf = int(D.frame_idx.max()) + 1 if len(D) else 0
    car = np.zeros(nf + 2 * NEG_WIN + 1, bool)
    fr = D.frame_idx[(D.cls == "buggy") & (D.conf >= NEG_CONF)].to_numpy()
    car[fr + NEG_WIN] = True
    # any car within +-NEG_WIN frames
    cs = np.concatenate([[0], np.cumsum(car)])
    idx = np.arange(nf)
    near = cs[idx + 2 * NEG_WIN + 1] - cs[idx]
    free = near == 0
    # frames with kept car tracks (for the target size)
    dd = pd.read_parquet(d / "dets.parquet", columns=["frame_idx", "cls", "track_id"])
    pos_frames = dd.frame_idx[(dd.cls == "buggy") & (dd.track_id >= 0)].nunique()
    rows = []
    sh = shots[shots.is_transition == 0]
    for s in sh.itertuples():
        f0, f1 = int(np.ceil((s.t0 + 1.0) * FPS)), int(np.floor((s.t1 - 1.0) * FPS))
        last = -1e9
        for f in range(max(f0, 0), min(f1, nf - 1) + 1, 8):
            if free[f] and f / FPS - last >= NEG_SPACING:
                rows.append((f, f / FPS, int(s.shot_id), s.tag))
                last = f / FPS
    N = pd.DataFrame(rows, columns=["frame_idx", "pts", "shot_id", "tag"])
    target = max(NEG_MIN, int(NEG_RATE * pos_frames))
    if len(N) <= target:
        return N.assign(reason="no_car_pm1s"), len(N), pos_frames
    avail = N.tag.value_counts().to_dict()
    alloc = {t: 0 for t in avail}
    left = target
    while left > 0:
        open_ = [t for t in avail if alloc[t] < avail[t]]
        if not open_:
            break
        share = max(1, left // len(open_))
        for t in open_:
            add = min(share, avail[t] - alloc[t], left)
            alloc[t] += add
            left -= add
            if left <= 0:
                break
    parts = [g.sample(alloc[t], random_state=0) for t, g in N.groupby("tag")]
    out = pd.concat(parts).sort_values("frame_idx").reset_index(drop=True)
    return out.assign(reason="no_car_pm1s"), len(N), pos_frames


def process(row):
    vid, ev = row["id"], row["event"]
    d = wdir(vid)
    roster = idlib.load_roster(ev)
    shots = pd.read_csv(d / "shots.csv")
    ct = pd.read_parquet(d / "crop_tracks.parquet")
    # current WP3 flags (track.py --reflag may have run after crops.py; track ids are stable)
    tr = pd.read_parquet(d / "tracks.parquet").set_index("track_id")
    for c in ("screen_fixed", "in_overlay", "long_static", "small", "nested_in", "edge_touch"):
        ct[c] = ct.track_id.map(tr[c]).to_numpy()
    ct["in_overlay"] = ct.in_overlay.map(lambda v: bool(v) if v is not None and v == v else False)
    ct["h_rel_median"] = ct.track_id.map(tr.h_rel_median)
    ego, _ = verify.ego_residue(d, row["height"])
    ct["ego_residue"] = ct.track_id.isin(ego)
    VF = pd.read_parquet(d / "verify.parquet") if (d / "verify.parquet").exists() else None
    vv = {}
    if VF is not None and len(VF):
        for tid, g in VF.groupby("track_id"):
            vd, sc = verify.verdict(list(g.score))
            if vd == "no" and g.h_px.max() < VERIFY_NO_MIN_H:
                vd = "unsure"   # small crops: 'no' is unreliable (validation recall 40% below 60 px)
            vv[int(tid)] = (vd, sc, float(g.h_px.max()))
    # WP8 recheck: 3 more crops for confident-detector 'no' tracks and for screen_fixed tracks flagged only by the
    # background-moving rule; majority over all crops of the track (tie -> unsure)
    V2 = pd.read_parquet(d / "verify2.parquet") if (d / "verify2.parquet").exists() else None
    sf_vlm = set()
    ct["rechecked"] = False
    if V2 is not None and len(V2):
        first = VF.groupby("track_id").score.apply(list).to_dict() if VF is not None else {}
        for tid, g in V2.groupby("track_id"):
            sc_all = (first.get(tid, []) if not bool(g.screen_fixed.iloc[0]) else []) + list(g.score)
            vv[int(tid)] = (verify.majority(sc_all), round(float(np.mean(sc_all)), 3), float(g.h_px.max()))
            if bool(g.screen_fixed.iloc[0]):
                sf_vlm.add(int(tid))
        ct["rechecked"] = ct.track_id.isin(set(V2.track_id))
    ct["sf_vlm_decides"] = ct.track_id.isin(sf_vlm)
    ct["car_verified"] = [vv[t][0] if t in vv else ("no" if (s_ or e_) else None)
                          for t, s_, e_ in zip(ct.track_id, ct.screen_fixed.astype(bool), ct.ego_residue)]
    # screen_fixed tracks whose verdict the VLM decides count as screen_fixed only if the VLM says no
    ct["sf_effective"] = ct.screen_fixed.astype(bool) & ~(ct.sf_vlm_decides & ct.car_verified.eq("yes"))
    ct["verify_score"] = ct.track_id.map(lambda t: vv[t][1] if t in vv else None)
    C = pd.read_parquet(d / "crops.parquet")
    R = pd.read_parquet(d / "reads.parquet") if (d / "reads.parquet").exists() else pd.DataFrame(
        columns=["path", "track_id", "frame_idx", "raw", "num"])
    R = R[R.path.isin(C.path)]
    G = pd.read_parquet(d / "grounded.parquet") if (d / "grounded.parquet").exists() else None
    O = pd.read_parquet(d / "overlay_ids.parquet") if (d / "overlay_ids.parquet").exists() else None
    V = track_votes(R, roster)
    OV = assign_overlays(ct, O, shots)
    tag = shots.set_index("shot_id").tag.to_dict()
    trans = shots.set_index("shot_id").is_transition.to_dict()
    # number boxes per crop
    cbox = {}
    if G is not None and len(G):
        for g in G.itertuples():
            k = json.loads(g.kept)
            if k:
                cbox[g.path] = k
    Cg = {tid: g.sort_values("frame_idx") for tid, g in C.groupby("track_id")}
    sharp = C.groupby("track_id").sharp.median()
    rows = []
    for r in ct.itertuples():
        tid = int(r.track_id)
        v = V.get(tid)
        ov = OV.get(tid)
        lab, src = final_label(v, ov) if r.readable or ov else (None, "none")
        if r.ego_residue or r.sf_effective:
            lab, src = None, "not_car_flag"
        nb, nbs = {}, {}
        if tid in Cg and lab is not None:
            for c in Cg[tid].itertuples():
                if c.path in cbox and v and v["number"] == lab:
                    nb[c.path] = cbox[c.path]
                    nbs[c.path] = [[round(c.cx1 + x1 / c.scale, 1), round(c.cy1 + y1 / c.scale, 1),
                                    round(c.cx1 + x2 / c.scale, 1), round(c.cy1 + y2 / c.scale, 1)]
                                   for x1, y1, x2, y2 in cbox[c.path]]
        rows.append(dict(
            video_id=vid, event=ev, track_id=tid, track_key=f"{vid}:{tid}", shot_id=int(r.shot_id),
            tag=tag.get(int(r.shot_id)), t0=r.t0, t1=r.t1, h_px_median=r.h_px_median, h_px_max=r.h_px_max,
            h_cand_max=r.h_cand_max, readable=bool(r.readable), reason=r.reason, n_crops=int(r.n_crops),
            n_read=int(v["n"]) if v else 0,
            vlm_top=v["top"] if v else None, vlm_share=v["share"] if v else 0.0, vlm_votes=v["votes"] if v else 0,
            vlm_runner=v["runner"] if v else None, vlm_number=v["number"] if v else None,
            vlm_consensus=bool(v["consensus"]) if v else False, votes=jdump(v["all_votes"]) if v else "{}",
            ovl_numbers=ov["numbers"] if ov else None, ovl_mode=ov["mode"] if ov else None,
            ovl_n_tracks=ov["n_tracks"] if ov else 0, ovl_match=ov["match"] if ov else None,
            ovl_ambiguous=bool(ov["ambiguous"]) if ov else False,
            label=lab, source=src, number_boxes=jdump(nb), number_boxes_src=jdump(nbs), n_boxes=sum(map(len, nb.values())),
            screen_fixed=bool(r.screen_fixed), in_overlay=bool(r.in_overlay), nested=bool(r.nested_in >= 0),
            ego_residue=bool(r.ego_residue), car_verified=r.car_verified, verify_score=r.verify_score,
            rechecked=bool(r.rechecked), sf_vlm_decides=bool(r.sf_vlm_decides), sf_effective=bool(r.sf_effective),
            small=bool(r.small), long_static=bool(r.long_static), transition=bool(trans.get(int(r.shot_id), 0)),
            sharp_median=float(sharp.get(tid, np.nan)), ctx=r.ctx))
    L = pd.DataFrame(rows)
    # is_car (for dataset A): the yes/no check, overruled by a roster-number consensus (only Baja cars carry one;
    # the check misses cars occluded by marshals / partial views) and by screen_fixed / ego residue
    flag = L.sf_effective | L.ego_residue
    L["is_car"] = np.where(flag, "no", np.where(L.vlm_consensus | L.source.eq("both-agree"), "yes",
                                                 L.car_verified.fillna("unchecked")))
    for c in ("vlm_top", "vlm_runner", "vlm_number", "label"):
        L[c] = L[c].astype("Int64")
    L.to_parquet(d / "labels.parquet", index=False)
    N, n_avail, pos_frames = negatives(d, shots)
    N.to_parquet(d / "negatives.parquet", index=False)
    rd = L[L.readable]
    both = L[L.vlm_consensus & L.ovl_numbers.notna() & (L.ovl_mode == "single") & ~L.ovl_ambiguous]
    info = dict(video=vid, car_tracks=len(L), readable=int(L.readable.sum()), read=int((L.n_read > 0).sum()),
                consensus=int(rd.vlm_consensus.sum()), overlay=int(L.ovl_numbers.notna().sum()),
                both_single=len(both), agree_single=int((both.source == "both-agree").sum()),
                labels=int(L.label.notna().sum()), sources={str(k): int(v) for k, v in L.source.value_counts().items()},
                boxes=int(L.n_boxes.sum()), negatives=len(N), neg_available=n_avail, pos_frames=int(pos_frames),
                verified={str(k): int(v) for k, v in L.car_verified.fillna("none").value_counts().items()},
                is_car={str(k): int(v) for k, v in L.is_car.value_counts().items()},
                ego_residue=int(L.ego_residue.sum()))
    (d / "labels.json").write_text(json.dumps(info, indent=1))
    return json.dumps(info)


# ---------------------------------------------------------------- QA batches

def allocate(sizes, n, floor):
    tot = sum(sizes.values())
    if n >= tot:
        return dict(sizes)
    alloc = {k: min(s, floor) for k, s in sizes.items()}
    left = n - sum(alloc.values())
    want = {k: s * n / tot for k, s in sizes.items()}
    while left > 0:
        k = max((k for k in sizes if alloc[k] < sizes[k]), key=lambda k: want[k] - alloc[k])
        alloc[k] += 1
        left -= 1
    return alloc


def qa(events, n=150, floor=6, seed=0):
    vids = select_videos()
    out_dir = ROOT / "out" / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    evs = events or sorted({v["event"] for v in vids})
    for ev in evs:
        items = []
        for v in [v for v in vids if v["event"] == ev]:
            d = wdir(v["id"])
            if not (d / "labels.done").exists():
                continue
            L = pd.read_parquet(d / "labels.parquet")
            L = L[L.readable & (L.n_read > 0)]
            C = pd.read_parquet(d / "crops.parquet")
            R = pd.read_parquet(d / "reads.parquet")[["path", "raw"]]
            C = C.merge(R, on="path").sort_values(["track_id", "frame_idx"])
            Cg = {t: g for t, g in C.groupby("track_id")}
            for r in L.itertuples():
                g = Cg.get(r.track_id)
                if g is None:
                    continue
                nb = json.loads(r.number_boxes)
                boxes = [nb.get(p) for p in g.path]
                ovl = [int(x) for x in r.ovl_numbers] if r.ovl_numbers is not None else None
                items.append(dict(
                    item_id=f"{v['id']}:{r.shot_id}:{r.track_id}", event=ev, video_id=v["id"],
                    shot_tag=r.tag, track_key=r.track_key,
                    proposed_number=None if pd.isna(r.vlm_number) else int(r.vlm_number),
                    vote_share=float(r.vlm_share) if not pd.isna(r.vlm_number) else 0.0,
                    n_votes=int(r.vlm_votes) if not pd.isna(r.vlm_number) else 0, n_crops=int(r.n_read),
                    overlay_number=None, crops=list(g.path), crop_reads=list(g.raw),
                    number_boxes=boxes if any(b for b in boxes) else None, car_box_context=r.ctx,
                    meta=dict(t=f"{int(r.t0 // 3600)}:{int(r.t0 // 60 % 60):02d}:{int(r.t0 % 60):02d}",
                              h_max=round(float(r.h_cand_max)), top=None if pd.isna(r.vlm_top) else int(r.vlm_top),
                              top_share=float(r.vlm_share), votes=r.votes,
                              hbin=str(pd.cut([r.h_cand_max], HBINS)[0]),
                              hidden=dict(overlay_numbers=ovl, overlay_mode=nn(r.ovl_mode), final_label=None if
                                          pd.isna(r.label) else int(r.label), source=r.source))))
        if not items:
            print(f"{ev}: no labelled videos")
            continue
        # strata = shot tag x (proposal / abstained); ~2/3 of the batch goes to tracks with a proposed number so
        # the precision estimate has n ~ 100, the rest to abstained tracks (coverage / missed reads); weights undo it
        strata = defaultdict(list)
        for it in items:
            strata[(it["shot_tag"], it["proposed_number"] is not None)].append(it)
        n_prop_pop = sum(len(v) for k, v in strata.items() if k[1])
        n_prop = min(n_prop_pop, int(round(n * PROP_SHARE)))
        n_abst = n - n_prop
        alloc = {}
        for flag, nn_, fl in ((True, n_prop, floor), (False, n_abst, max(2, floor // 2))):
            sub = {k: len(v) for k, v in strata.items() if k[1] == flag}
            if sub and nn_ > 0:
                alloc.update(allocate(sub, nn_, fl))
        rng = random.Random(seed)
        out = []
        for k in sorted(alloc, key=str):
            pick = rng.sample(strata[k], alloc[k])
            for it in pick:
                it["stratum"] = f"{k[0]}|{'read' if k[1] else 'noread'}"
                it["weight"] = round(len(strata[k]) / alloc[k], 4)
            out += pick
        rng.shuffle(out)
        dst = out_dir / f"{ev}_teacher.jsonl"
        with open(dst, "w", encoding="utf-8") as f:
            for it in out:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        nr = sum(it["proposed_number"] is None for it in out)
        print(f"{ev}: {len(out)} items ({nr} abstained) from {len(items)} tracks ({n_prop_pop} with a proposal) -> "
              f"{dst.relative_to(ROOT)}; strata {dict((f'{k[0]}|{int(k[1])}', (len(strata[k]), alloc[k])) for k in sorted(alloc, key=str))}")


# ---------------------------------------------------------------- summary

def summary():
    rows, hrows, agree = [], [], []
    for v in select_videos():
        d = wdir(v["id"])
        if not (d / "labels.done").exists():
            continue
        L = pd.read_parquet(d / "labels.parquet")
        g = json.loads((d / "gpu_time.json").read_text()) if (d / "gpu_time.json").exists() else {}
        oi = json.loads((d / "overlay_ids.json").read_text()) if (d / "overlay_ids.json").exists() else {}
        N = pd.read_parquet(d / "negatives.parquet")
        rd = L[L.readable]
        both = L[L.vlm_consensus & L.ovl_numbers.notna() & ~L.ovl_ambiguous]
        bs = both[both.ovl_mode == "single"]
        bh = both[both.ovl_mode == "heat"]
        rows.append(dict(video=v["id"], event=v["event"], car_tracks=len(L), readable=len(rd),
                         crops_read=int(rd.n_read.sum()), consensus=int(rd.vlm_consensus.sum()),
                         coverage=round(rd.vlm_consensus.mean(), 3) if len(rd) else None,
                         overlay_ids=oi.get("ids"), overlay_tracks=int(L.ovl_numbers.notna().sum()),
                         both_single=len(bs), agree_single=int((bs.source == "both-agree").sum()),
                         both_heat=len(bh), agree_heat=int((bh.source == "both-agree").sum()),
                         labels=int(L.label.notna().sum()), boxes_kept=int(L.n_boxes.sum()),
                         tracks_with_boxes=int((L.n_boxes > 0).sum()), negatives=len(N),
                         verified_yes=int((L.car_verified == "yes").sum()), verified_no=int((L.car_verified == "no").sum()),
                         verified_unsure=int((L.car_verified == "unsure").sum()),
                         is_car_yes=int((L.is_car == "yes").sum()), is_car_no=int((L.is_car == "no").sum()),
                         gpu_verify_min=round(g.get("verify_s", 0) / 60, 1),
                         gpu_plain_min=round(g.get("plain_s", 0) / 60, 1),
                         gpu_ground_min=round(g.get("ground_s", 0) / 60, 1),
                         gpu_ovl_min=round(oi.get("vlm_s", 0) / 60, 1)))
        C = pd.read_parquet(d / "crops.parquet")
        R = pd.read_parquet(d / "reads.parquet")
        roster = idlib.load_roster(v["event"])
        X = C.merge(R[["path", "num"]], on="path").merge(L[["track_id", "vlm_number", "label"]], on="track_id")
        X["hb"] = pd.cut(X.h_px, HBINS)
        X["in_roster"] = X.num.map(lambda x: (not pd.isna(x)) and int(x) in roster)
        X["eq_cons"] = X.num.notna() & X.vlm_number.notna() & (X.num == X.vlm_number)
        X["eq_label"] = X.num.notna() & X.label.notna() & (X.num == X.label)
        for hb, gg in X.groupby("hb", observed=True):
            hrows.append(dict(video=v["id"], event=v["event"], level="crop", hbin=str(hb), n=len(gg),
                              read_any=int(gg.num.notna().sum()), read_roster=int(gg.in_roster.sum()),
                              eq_consensus=int(gg.eq_cons.sum()), eq_label=int(gg.eq_label.sum())))
        rd = rd[rd.n_read > 0].assign(hb=pd.cut(rd.h_cand_max, HBINS))  # tracks that were read
        for hb, gg in rd.groupby("hb", observed=True):
            hrows.append(dict(video=v["id"], event=v["event"], level="track", hbin=str(hb), n=len(gg),
                              read_any=int((gg.vlm_top.notna()).sum()), read_roster=int(gg.vlm_top.notna().sum()),
                              eq_consensus=int(gg.vlm_consensus.sum()), eq_label=int(gg.label.notna().sum())))
        b = both.assign(hb=pd.cut(both.h_cand_max, HBINS))
        for r in b.itertuples():
            agree.append(dict(video=v["id"], event=v["event"], mode=r.ovl_mode, hbin=str(r.hb), tag=r.tag,
                              n_tracks=r.ovl_n_tracks, agree=r.source == "both-agree", vlm=r.vlm_number,
                              ovl=str(list(r.ovl_numbers)), share=r.vlm_share, track=r.track_id, t0=r.t0))
    S = pd.DataFrame(rows)
    H = pd.DataFrame(hrows)
    A = pd.DataFrame(agree)
    S.to_csv(DATA / "wp45_summary.csv", index=False)
    H.to_csv(DATA / "wp45_height.csv", index=False)
    A.to_csv(DATA / "wp45_agreement.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(S.to_string(index=False))
    if len(H):
        for lev in ("crop", "track"):
            h = H[H.level == lev].groupby("hbin", sort=False)[["n", "read_any", "read_roster", "eq_consensus",
                                                               "eq_label"]].sum()
            h = h.loc[[str(x) for x in pd.IntervalIndex.from_breaks(HBINS) if str(x) in h.index]]
            for c in ("read_any", "read_roster", "eq_consensus", "eq_label"):
                h[c + "%"] = (100 * h[c] / h.n).round(1)
            print(f"\nread rate vs box height ({lev} level; crop: box h of the crop, track: max candidate h)")
            print(h.to_string())
    if len(A):
        print("\nVLM consensus vs overlay identity")
        for keys in (["mode", "event"], ["mode", "hbin"], ["mode", "n_tracks"], ["mode", "tag"]):
            t = A.groupby(keys).agg(n=("agree", "size"), agree=("agree", "sum"))
            t["rate"] = (t.agree / t.n).round(3)
            print(t.to_string())
    return S


def main(argv):
    if argv and argv[0] == "qa":
        a = list(argv[1:])
        opt = {"--n": 150, "--floor": 6, "--seed": 0}
        for k in list(opt):
            if k in a:
                i = a.index(k)
                opt[k] = int(a[i + 1])
                del a[i:i + 2]
        qa([x for x in a if not x.startswith("--")], opt["--n"], opt["--floor"], opt["--seed"])
        return
    if argv and argv[0] == "summary":
        summary()
        return
    workers, force, ids, rest = cli_ids(argv)
    run_stage("labels", process, select_videos(ids or None), workers=workers, force=force,
              requires=("crops", "reads"))


if __name__ == "__main__":
    main(sys.argv[1:])
