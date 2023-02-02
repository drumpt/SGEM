CUDA_VISIBLE_DEVICES=0 python main.py \
    scheduler=CosineAnnealingLR \
    eta_min=2e-6 \
    log_dir=exps/lr_scheduling \

CUDA_VISIBLE_DEVICES=0 python main.py \
    scheduler=CosineAnnealingLR \
    eta_min=5e-6 \
    log_dir=exps/lr_scheduling \

CUDA_VISIBLE_DEVICES=0 python main.py \
    scheduler=CosineAnnealingLR \
    eta_min=1e-5 \
    log_dir=exps/lr_scheduling \