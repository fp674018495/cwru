source activate omni-event
CUDA_VISIBLE_DEVICES=3 nohup python cwru/load_phm.py  >run.log 2>&1 &