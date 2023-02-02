CUDA_VISIBLE_DEVICES=0 python main.py \
    lr=4e-5 \
    scheduler=CosineAnnealingLR \
    eta_min=2e-5 \
    log_dir=exps/lr_scheduling \
    beam_width=10 \
    num_positives=1 \
    num_negatives=9 \

CUDA_VISIBLE_DEVICES=0 python main.py \
    lr=5e-5 \
    scheduler=CosineAnnealingLR \
    eta_min=2e-5 \
    log_dir=exps/lr_scheduling \