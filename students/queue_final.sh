#!/bin/bash
# WP9 final models on all 5 events (fixed epochs = the chosen runs' best epoch, i.e. the same number of passes over the
# data; no early stopping; last.pt is the model), then the 26SE regression check (full eval with crops + e2e).
# usage: nohup bash students/queue_final.sh > students/runs/queue_final.log 2>&1 &
cd /d/video_baja; export PATH="/c/Users/baja/AppData/Local/Microsoft/WinGet/Links:$PATH"; unset CUDA_VISIBLE_DEVICES
PY=.venv/Scripts/python.exe
T="$PY students/train_det.py --patience 0 --workers 8"
echo "$(date) start"
$T --data datasets/numbers_crop_r2_all/data.yaml --model yolo11n.pt --name num_n320_r2_all --imgsz 320 --epochs 73 --batch 128 > students/runs/num_n320_r2_all.log 2>&1
echo "$(date) num_n320_r2_all: $(tail -1 students/runs/num_n320_r2_all/results.csv | cut -d, -f1,6-9)"
e2e () {  # $1 = car run; full 26SE eval with crops at native resolution, then the identity chain on the CPU
  R=$1; D=out/eval_${R}_e2e
  $PY students/eval_26se.py --name ${R}_e2e --weights students/runs/$R/weights/last.pt --crops --track-width 0 > out/eval_${R}_e2e.log 2>&1
  mv out/eval_${R}_e2e.log $D/run.log
  until [ -f students/runs/recog/pq_spl_all/recog.onnx ]; do sleep 60; done
  CUDA_VISIBLE_DEVICES=-1 $PY students/e2e_26se.py --eval $D --panel-weights students/runs/num_n320_r2_all/weights/last.pt --recog pq_spl_all > $D/e2e.log 2>&1
  CUDA_VISIBLE_DEVICES=-1 $PY students/e2e_26se.py --eval $D --panel-weights students/runs/num_n320_r2/weights/best.pt --recog pq_spl_full --tag cur > $D/e2e_cur.log 2>&1
  echo "$(date) $R: $(grep -E '^all' $D/run.log) | all-models: $(grep -E '^(no_abstain|val_rule)' $D/e2e.log | cut -c1-200 | tr '\n' ' ')"
}
$T --data datasets/cars_all/data.yaml --model yolo11n.pt --name cars_n640_v2_all --imgsz 640 --epochs 66 --batch 64 --washout 0.15 > students/runs/cars_n640_v2_all.log 2>&1
echo "$(date) cars_n640_v2_all trained"
e2e cars_n640_v2_all
$T --data datasets/cars_all/data.yaml --model yolo11s.pt --name cars_s640_all --imgsz 640 --epochs 55 --batch 32 > students/runs/cars_s640_all.log 2>&1
echo "$(date) cars_s640_all trained"
e2e cars_s640_all
echo "$(date) FINAL QUEUE DONE"
