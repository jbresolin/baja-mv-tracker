#!/bin/bash
# usage: bash students/after_train.sh <run>   (waits for train_det.py to finish incl. test eval, then 26SE eval on GPU, then CPU bench)
R=$1; cd /d/video_baja; export PATH="/c/Users/baja/AppData/Local/Microsoft/WinGet/Links:$PATH"; unset CUDA_VISIBLE_DEVICES
until grep -q '"split": "test"\|no .test. split\|Traceback' students/runs/$R.log; do sleep 30; done
grep -q Traceback students/runs/$R.log && { echo "TRAIN FAILED"; exit 1; }
sleep 20
.venv/Scripts/python.exe students/eval_26se.py --name $R --weights students/runs/$R/weights/best.pt > out/eval_$R.log 2>&1 || { echo "EVAL FAILED"; exit 1; }
mv out/eval_$R.log out/eval_$R/run.log
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe students/bench_cpu.py --weights students/runs/$R/weights/best.pt --imgsz 384x640 --priority high --rounds 3 --n 60 --formats onnx,openvino > students/runs/bench_$R.log 2>&1
echo ALL DONE
