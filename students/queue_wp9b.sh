#!/bin/bash
# WP9 queue part 2 (after queue_wp9.sh): panel boxes with num_n320_r2 (if not worse on val) -> pq_spl_full 26SE chain on
# the teacher's passes; then the fully student end-to-end for cars_n640 / cars_s640 / cars_n640_v2.
# usage: nohup bash students/queue_wp9b.sh > students/runs/queue_wp9b.log 2>&1 &
cd /d/video_baja; export PATH="/c/Users/baja/AppData/Local/Microsoft/WinGet/Links:$PATH"; unset CUDA_VISIBLE_DEVICES
PY=.venv/Scripts/python.exe
echo "$(date) waiting for queue_wp9"
until grep -q "QUEUE DONE" students/runs/queue_wp9.log 2>/dev/null; do sleep 60; done
# 1) panel model: r2 if its val track top-1 (conf 0.1, pad 0.05, same recognizer) >= num_n320's
PANEL=$($PY -c "
import json
a=json.load(open('students/runs/num_n320/panel_val_c0.1.json'))['per_pad']['0.05']['top1']
b=json.load(open('students/runs/num_n320_r2/panel_val_c0.1.json'))['per_pad']['0.05']['top1']
print('num_n320_r2' if b >= a else 'num_n320')")
echo "$(date) panel model: $PANEL"
W=students/runs/$PANEL/weights/best.pt
if [ "$PANEL" = "num_n320_r2" ]; then
  CUDA_VISIBLE_DEVICES=-1 $PY students/panel_boxes.py --weights $W --out out/panel_boxes_$PANEL.csv
  CUDA_VISIBLE_DEVICES=-1 $PY students/panel_boxes.py --weights $W --all-crops --out out/panel_boxes_${PANEL}_all.csv
  CUDA_VISIBLE_DEVICES=-1 $PY students/recog/harness_26se.py --name pq_spl_full --mode panels --boxes out/panel_boxes_$PANEL.csv --vote boxed --tag _r2 > out/harness_pq_r2.log 2>&1
  CUDA_VISIBLE_DEVICES=-1 $PY students/recog/harness_26se.py --name pq_spl_full --mode panels --boxes out/panel_boxes_${PANEL}_all.csv --all-crops --vote boxed --tag _r2 > out/harness_pq_r2_all.log 2>&1
  echo "$(date) harness r2 done"
fi
# 2) fully student end-to-end
for R in cars_s640 cars_n640_v2 cars_n640; do
  D=out/eval_${R}_e2e
  mkdir -p $D && cp out/eval_$R/tl_*.csv $D/
  $PY students/eval_26se.py --name ${R}_e2e --weights students/runs/$R/weights/best.pt --stages track,metrics --crops --track-width 0 > $D/run.log 2>&1
  CUDA_VISIBLE_DEVICES=-1 $PY students/e2e_26se.py --eval $D --panel-weights $W --recog pq_spl_full > $D/e2e.log 2>&1
  echo "$(date) e2e $R: $(grep -E '^(no_abstain|val_rule)' $D/e2e.log | tr '\n' ' ')"
done
echo "$(date) QUEUE B DONE"
