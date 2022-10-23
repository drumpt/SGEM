CUDA_VISIBLE_DEVICES=0 python main.py \
                asr=speechbrain/asr-crdnn-rnnlm-librispeech \
                dataset_name=librispeech \
                dataset_dir=/home/server08/hdd0/changhun_workspace/LibriSpeech \
                extra_noise=0.01 \
                method=original \
                log_dir=exps/exps_attn_libri_original \

CUDA_VISIBLE_DEVICES=0 python main.py \
                asr=speechbrain/asr-crdnn-rnnlm-librispeech \
                dataset_name=librispeech \
                dataset_dir=/home/server08/hdd0/changhun_workspace/LibriSpeech \
                extra_noise=0.01 \
                method=em_uncertainty \
                log_dir=exps/exps_attn_libri_em_uncertainty \

CUDA_VISIBLE_DEVICES=0 python main.py \
                asr=speechbrain/asr-crdnn-rnnlm-librispeech \
                dataset_name=librispeech \
                dataset_dir=/home/server08/hdd0/changhun_workspace/LibriSpeech \
                extra_noise=0.01 \
                method=em_sparse \
                log_dir=exps/exps_attn_libri_em_sparse \

CUDA_VISIBLE_DEVICES=0 python main.py \
                asr=speechbrain/asr-crdnn-rnnlm-librispeech \
                dataset_name=librispeech \
                dataset_dir=/home/server08/hdd0/changhun_workspace/LibriSpeech \
                extra_noise=0.01 \
                method=cr \
                log_dir=exps/exps_attn_libri_cr \

CUDA_VISIBLE_DEVICES=0 python main.py \
                asr=speechbrain/asr-crdnn-rnnlm-librispeech \
                dataset_name=chime \
                dataset_dir=/home/server08/hdd0/changhun_workspace/CHiME3 \
                method=original \
                log_dir=exps/exps_attn_chime_original \

CUDA_VISIBLE_DEVICES=0 python main.py \
                asr=speechbrain/asr-crdnn-rnnlm-librispeech \
                dataset_name=chime \
                dataset_dir=/home/server08/hdd0/changhun_workspace/CHiME3 \
                method=em_uncertainty \
                log_dir=exps/exps_attn_chime_em_uncertainty \

CUDA_VISIBLE_DEVICES=0 python main.py \
                asr=speechbrain/asr-crdnn-rnnlm-librispeech \
                dataset_name=chime \
                dataset_dir=/home/server08/hdd0/changhun_workspace/CHiME3 \
                method=em_sparse \
                log_dir=exps/exps_attn_chime_em_sparse \

CUDA_VISIBLE_DEVICES=0 python main.py \
                asr=speechbrain/asr-crdnn-rnnlm-librispeech \
                dataset_name=chime \
                dataset_dir=/home/server08/hdd0/changhun_workspace/CHiME3 \
                method=cr \
                log_dir=exps/exps_attn_chime_cr \