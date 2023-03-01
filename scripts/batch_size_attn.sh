# CUDA_VISIBLE_DEVICES=0 python main.py \
#     asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
#     dataset_name=chime \
#     dataset_dir=/home/server17/hdd/changhun_workspace/CHiME3 \
#     batch_size=1 \
#     lr=2e-5 \
#     'method=[original]' \
#     em_coef=1 \
#     use_memory_queue=false \
#     log_dir=exps/batch_size_attn \


CUDA_VISIBLE_DEVICES=0 python main.py \
    asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
    dataset_name=chime \
    dataset_dir=/home/server17/hdd/changhun_workspace/CHiME3 \
    batch_size=4 \
    lr=2e-5 \
    'method=[original]' \
    em_coef=1 \
    use_memory_queue=false \
    log_dir=exps/batch_size_attn \


CUDA_VISIBLE_DEVICES=0 python main.py \
    asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
    dataset_name=chime \
    dataset_dir=/home/server17/hdd/changhun_workspace/CHiME3 \
    batch_size=16 \
    lr=2e-5 \
    'method=[original]' \
    em_coef=1 \
    use_memory_queue=false \
    log_dir=exps/batch_size_attn \