CUDA_VISIBLE_DEVICES=0 python main.py \
    label_smoothing=0 \
    log_dir=exps/label_smoothing \

CUDA_VISIBLE_DEVICES=0 python main.py \
    label_smoothing=0.1 \
    log_dir=exps/label_smoothing \

CUDA_VISIBLE_DEVICES=0 python main.py \
    label_smoothing=0.05 \
    log_dir=exps/label_smoothing \
