"""Write deploy/models/manifest.json for the ONNX set in deploy/models/ (car_n, car_s, panel, recog).

usage: python students/make_manifest.py --car-n cars_n640_v2 --car-s cars_s640 --panel num_n320_r2 --recog pq_spl_full
(run names under students/runs/ and students/runs/recog/; the .onnx files must already be in deploy/models/).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "deploy" / "models"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--car-n", default="cars_n640_v2")
    ap.add_argument("--car-s", default="cars_s640")
    ap.add_argument("--panel", default="num_n320_r2")
    ap.add_argument("--recog", default="pq_spl_full")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    tau = json.load(open(ROOT / "students" / "runs" / "recog" / "pq_spl_full" / "eval" / "panel_tau.json"))["chosen"]
    yolo_out = ("output0 (1, 5, N): rows 0-3 = box centre x, centre y, width, height in input pixels (after letterbox), "
                "row 4 = class score (sigmoid already applied); no NMS inside: filter by conf, then NMS IoU 0.7 "
                "(see students/bench_cpu.py postprocess)")
    letterbox = ("resize keeping aspect so the image fits, pad with 114 grey centred, BGR->RGB, CHW, float32 / 255 "
                 "(see students/bench_cpu.py letterbox)")
    m = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), note=a.note,
        models=dict(
            car_n=dict(file="car_n.onnx", sha256_16=sha(D / "car_n.onnx"), source_run="students/runs/%s" % a.car_n,
                       arch="YOLO11n", classes=["car"], input=dict(name="images", shape=[1, 3, 384, 640], preprocess=letterbox),
                       export=dict(tool="ultralytics 8.4.181 YOLO.export", format="onnx", imgsz=[384, 640], dynamic=False,
                                  simplify=True, nms=False, verified="WP11: matches the .pt within 0.001 px"),
                       output=yolo_out, cpu_ms_3900x_loaded=dict(threads_1=45, threads_4=21),
                       use="default laptop detector (real-time on one thread at 10 fps)"),
            car_s=dict(file="car_s.onnx", sha256_16=sha(D / "car_s.onnx"), source_run="students/runs/%s" % a.car_s,
                       arch="YOLO11s", classes=["car"], input=dict(name="images", shape=[1, 3, 384, 640], preprocess=letterbox),
                       export=dict(tool="ultralytics 8.4.181 YOLO.export", format="onnx", imgsz=[384, 640], dynamic=False,
                                  simplify=True, nms=False, verified="WP11: matches the .pt within 0.001 px"),
                       output=yolo_out, cpu_ms_3900x_loaded=dict(threads_1=126, threads_4=45),
                       use="better precision, needs 4 threads"),
            panel=dict(file="panel.onnx", sha256_16=sha(D / "panel.onnx"), source_run="students/runs/%s" % a.panel,
                       arch="YOLO11n", classes=["number"],
                       input=dict(name="images", shape=[1, 3, "H", "W"],
                                  preprocess="one car crop, letterboxed like ultralytics' rect mode: scale so the long "
                                             "side is 320, pad the short side only up to the next multiple of 32 "
                                             "(e.g. 320x256), grey 114, BGR->RGB, CHW, float32 / 255. A fixed 320x320 "
                                             "export padded to square gave the same box count as the .pt on only "
                                             "89-94% of crops; the dynamic export matches exactly"),
                       export=dict(tool="ultralytics 8.4.181 YOLO.export", format="onnx", imgsz=320, dynamic=True,
                                   simplify=True, nms=False, verified="students/verify_onnx.py: 500 26SE car crops, "
                                   "same box count on 100%, max coordinate difference 0.0001 px"),
                       output=yolo_out, cpu_ms_3900x_loaded=dict(threads_1=30, threads_4=19)),
            recog=dict(file="recog.onnx", sha256_16=sha(D / "recog.onnx"), source_run="students/runs/recog/%s" % a.recog,
                       arch="PARSeq-tiny (digits) with CTC-shaped output",
                       input=dict(name="image", shape=["N", 3, 32, "W"],
                                  preprocess="number box (+ the detector pad) enlarged by 10% per side "
                                             "(students/recog/augment.plain), resized to height 32 keeping aspect, "
                                             "width padded to 96 with the border mean, RGB float 0-1 "
                                             "(students/recog/common.to_input)"),
                       output="logp (T=10, N, 11): log-probs, class 0 = CTC blank, class d+1 = digit d. Roster snap: "
                              "students/recog/common.RosterScorer (P('n') + P('0n') per roster number from the CTC "
                              "sequence likelihood)",
                       export=dict(tool="students/recog/export_onnx.py", opset=17, dynamic=["N", "W"]),
                       cpu_ms_3900x_loaded=dict(threads_1=21, threads_4=12, threads_4_batch8_per_box=7)),
        ),
        pipeline=dict(
            car_detector=dict(model="car_n (or car_s)", fps=10, conf=0.15, classes=[0],
                              tracker="ultralytics bytetrack.yaml defaults (track_high 0.25, low 0.1, new 0.25, buffer 30 "
                                      "frames, match 0.8, fuse_score)",
                              note="26SE at 10 fps tracking (cars_n640_v2): pass recall 0.925, precision 0.971"),
            passes=dict(code="lapvision/passes.py build() (unchanged)", foreground="track max box height > 0.20 H and "
                        "max bottom > 0.40 H", min_frames=6, min_frames_note="frames, not seconds: 0.6 s at 10 fps",
                        merge_gap_s=3.0, static_gap_s=10.0, static_iou=0.2, edge_frac=0.02,
                        nested_box_dedupe="median intersection / smaller area > 0.6 on shared frames",
                        pass_time="t0 = first frame of the pass (car enters frame); tcross = centre-line crossing"),
            crops=dict(per_pass_max=30, selection="every track frame of the pass, evenly subsampled",
                       pad=0.05, note="cut from the full-resolution frame (lapvision/extract.py style)"),
            panel_detector=dict(conf=0.1, imgsz=320, box_pad_per_side=0.05, nms_iou=0.7),
            recognizer=dict(extra_pad_per_side=0.10, crop_posterior="max over the crop's boxes of the roster posterior"),
            vote=dict(code="students/recog/common.vote_boxed", score=tau.get("score", "share"), tau=tau["tau"],
                      min_support=tau["min_support"], p_support=0.5,
                      rule="ignore crops without a box; answer only if score >= tau and >= min_support boxed crops "
                           "vote for the winner with P >= 0.5; else 'unknown' (send to the human reviewer)",
                      source="picked on 25SU val by WP10b (students/runs/recog/pq_spl_full/eval/panel_tau.json)"),
            roster=dict(file="rosters/<EVENT>.csv (numero,equipe)", snapping="numbers outside the roster are never output"),
        ),
        training=dict(
            final="car_n / car_s / recog trained on all 5 stream events (19SU, 24SU, 25BR, 25SU, 26BR; no held-out split) "
                  "with the recipes of cars_n640_v2 / cars_s640 / pq_spl_full and a fixed epoch count = the best epoch of "
                  "the held-out run (same number of passes over the data; no early stopping; last.pt). panel = "
                  "num_n320_r2 (held-out run; its all-events version num_n320_r2_all was not adopted, see LESSONS)",
            regression="26SE e2e (students/e2e_26se.py) with these exact files: see benchmark_26se"),
        benchmark_26se=dict(
            teacher="YOLO-World + VLM: pass recall 0.933, precision 0.975, end-to-end 67.9%",
            car_n_chain="car_n + panel + recog (this set): pass recall 0.925, precision 0.967 (30 fps tracking); identity "
                        "no-abstain 84.4% right, end-to-end 77.4%; val rule 61% of matched passes at 100% (142/142), "
                        "end-to-end 56.4%",
            car_s_chain="car_s + panel + recog (this set): pass recall 0.929, precision 0.967; no-abstain 85.3%, "
                        "end-to-end 78.6%; val rule 60% at 99.3% (139/140), end-to-end 55.2%"),
    )
    (D / "manifest.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: v["file"] + " <- " + v["source_run"] for k, v in m["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
