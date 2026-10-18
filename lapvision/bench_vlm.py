"""Speed / accuracy of VLM variants. usage: bench_vlm.py <model_id> <cuda|cpu> <n_passes> [max_side] [threads]"""
import sys, time
import pandas as pd, torch
from collections import Counter
from vlm import Reader
import analyze

mid, dev, npass = sys.argv[1], sys.argv[2], int(sys.argv[3])
side = int(sys.argv[4]) if len(sys.argv) > 4 else 640
if len(sys.argv) > 5: torch.set_num_threads(int(sys.argv[5]))
ps = pd.read_csv("../out/passes_all.csv")
ps = ps[(ps.src == "crops_1129") & ps.gt_car.notna()]
rd = pd.read_csv("../out/crops_1129/reads.csv")
import passes as P
built = P.build(pd.read_csv("../out/crops_1129/tracks.csv"))
owner = {k: pid for pid, ks in zip(built.pass_id, built.tracks) for k in ks}
rd["pid"] = (rd.event + ":" + rd.track.astype(str)).map(owner)
ps = ps.sample(min(npass, len(ps)), random_state=0)
r = Reader(mid, dev)
r.read(list(rd.path[:2]), max_side=side)  # warm-up
ok = n = 0; t_tot = 0; ncrop = 0
for _, p in ps.iterrows():
    paths = list(rd[rd.pid == p.pass_id].path)
    t = time.time()
    res = []
    for i in range(0, len(paths), 8 if dev == "cuda" else 1):
        res += r.read(paths[i:i + (8 if dev == "cuda" else 1)], max_side=side)
    t_tot += time.time() - t; ncrop += len(paths)
    c = Counter(x for _, x in res if x in analyze.ROSTER)
    v = c.most_common(1)[0][0] if c and c.most_common(1)[0][1] >= 2 else None
    n += 1; ok += v == p.gt_car
print("%s %s side=%d passes=%d crops=%d  %.3f s/crop  acc(pass, any read)=%.3f" % (mid, dev, side, n, ncrop, t_tot / ncrop, ok / n))
if dev == "cuda": print("vram peak GB %.1f" % (torch.cuda.max_memory_allocated() / 1e9))
