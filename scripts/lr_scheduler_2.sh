CUDA_VISIBLE_DEVICES=0 python main.py \
    lr=4e-5 \
    scheduler=CosineAnnealingLR \
    eta_min=1e-5 \
    log_dir=exps/lr_scheduling \

CUDA_VISIBLE_DEVICES=0 python main.py \
    lr=4e-5 \
    scheduler=CosineAnnealingLR \
    eta_min=2e-5 \
    log_dir=exps/lr_scheduling \