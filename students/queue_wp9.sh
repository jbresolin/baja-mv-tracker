#!/bin/bash
# WP9 GPU queue after cars_s640 (one GPU job at a time): num_n320_r2 -> (panel_val r2 on CPU) + cars_n640_v2 -> its chain.
# usage: nohup bash students/queue_wp9.sh > students/runs/queue_wp9.log 2>&1 &
cd /d/video_baja; export PATH="/c/Users/baja/AppData/Local/Microsoft/WinGet/Links:$PATH"; unset CUDA_VISIBLE_DEVICES
PY=.venv/Scripts/python.exe
echo "$(date) waiting for the cars_s640 chain"
until grep -q "ALL DONE\|FAILED" students/runs/after_train_cars_s640.log 2>/dev/null; do sleep 60; done
echo "$(date) s640 chain: $(tail -1 students/runs/after_train_cars_s640.log)"
sleep 30
$PY students/train_det.py --data datasets/numbers_crop_r2/data.yaml --model yolo11n.pt --name num_n320_r2 --imgsz 320 \
    --epochs 150 --patience 30 --batch 128 --workers 8 > students/runs/num_n320_r2.log 2>&1
echo "$(date) num_n320_r2 done: $(grep -a '"split"' students/runs/num_n320_r2.log | tail -1)"
sleep 30
$PY students/train_det.py --data datasets/cars/data.yaml --model yolo11n.pt --name cars_n640_v2 --imgsz 640 \
    --epochs 100 --patience 20 --batch 64 --workers 8 --washout 0.15 > students/runs/cars_n640_v2.log 2>&1 &
TRAIN=$!
# panel-mode val/test for r2 on the CPU while the GPU trains cars_n640_v2
for c in 0.25 0.1; do
  CUDA_VISIBLE_DEVICES=-1 $PY students/panel_val.py --weights students/runs/num_n320_r2/weights/best.pt --split val \
      --imgsz 320 --conf $c --pads 0,0.05 --threads 4 > students/runs/num_n320_r2/panel_val_c$c.log 2>&1
  cp students/runs/num_n320_r2/panel_val.json students/runs/num_n320_r2/panel_val_c$c.json
done
echo "$(date) panel_val r2 done"
wait $TRAIN
echo "$(date) cars_n640_v2 trained: $(grep -a '"split"' students/runs/cars_n640_v2.log | tail -1)"
bash students/after_train.sh cars_n640_v2 > students/runs/after_train_cars_n640_v2.log 2>&1
echo "$(date) QUEUE DONE: $(tail -1 students/runs/after_train_cars_n640_v2.log)"
