"""Estimate clock offset between video and GT by matching foreground vehicle events."""
import sys, numpy as np, pandas as pd, gt, clock

def fg(tl, y2min=0.4, hmin=0.2, conf=0.3):
    v = tl[(tl.cls != "person") & (tl.conf > conf)]
    return v[(v.y2 > y2min * 1080) & ((v.y2 - v.y1) > hmin * 1080)]

def events(v, gap=2.5):
    t = np.sort(v.tod.unique()); ev = []
    if not len(t): return ev
    s = p = t[0]
    for x in t[1:]:
        if x - p > gap: ev.append((s, p)); s = x
        p = x
    ev.append((s, p)); return ev

if __name__ == "__main__":
    tl = pd.read_csv(sys.argv[1])
    ev = events(fg(tl))
    G = sorted([(gt.tod(x["ts"]), x["car"]) for x in gt.voltas()])
    g = np.array([x[0] for x in G if tl.tod.min() - 200 < x[0] < tl.tod.max()])
    st = np.array([a for a, b in ev])
    res = []
    for off in np.arange(-150, 150, 0.5):
        gg = g + off; gg = gg[(gg > tl.tod.min()) & (gg < tl.tod.max())]
        hit = sum(np.min(np.abs(st - x)) < 3 for x in gg)
        res.append((hit / max(1, len(gg)), hit, len(gg), off))
    res.sort(reverse=True)
    print(len(ev), "fg events;", "best:", res[:5])
    off = res[0][3]
    d = [st[np.argmin(np.abs(st - (x + off)))] - x for x in g if np.min(np.abs(st - (x + off))) < 6]
    print("residual offsets: median %.2f  p10 %.2f  p90 %.2f" % (np.median(d), np.percentile(d, 10), np.percentile(d, 90)))
