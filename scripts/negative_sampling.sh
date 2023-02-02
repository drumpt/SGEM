CUDA_VISIBLE_DEVICES=0 python main.py \
    num_negatives=1 \
    log_dir=exps/negative_sampling \

CUDA_VISIBLE_DEVICES=0 python main.py \
    num_negatives=9 \
    log_dir=exps/negative_sampling \
