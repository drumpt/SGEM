CUDA_VISIBLE_DEVICES=2 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=librispeech \
                dataset_dir=/home/server08/hdd0/changhun_workspace/LibriSpeech \
                extra_noise=0.01 \
                method=original \
                log_dir=exps/exps_trans_libri_original \

CUDA_VISIBLE_DEVICES=2 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=librispeech \
                dataset_dir=/home/server08/hdd0/changhun_workspace/LibriSpeech \
                extra_noise=0.01 \
                method=em_uncertainty \
                log_dir=exps/exps_trans_libri_em_uncertainty \

CUDA_VISIBLE_DEVICES=2 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=librispeech \
                dataset_dir=/home/server08/hdd0/changhun_workspace/LibriSpeech \
                extra_noise=0.01 \
                method=em_sparse \
                log_dir=exps/exps_trans_libri_em_sparse \

CUDA_VISIBLE_DEVICES=2 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=librispeech \
                dataset_dir=/home/server08/hdd0/changhun_workspace/LibriSpeech \
                extra_noise=0.01 \
                method=cr \
                log_dir=exps/exps_trans_libri_cr \

CUDA_VISIBLE_DEVICES=2 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=chime \
                dataset_dir=/home/server08/hdd0/changhun_workspace/CHiME3 \
                method=original \
                log_dir=exps/exps_trans_chime_original \

CUDA_VISIBLE_DEVICES=2 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=chime \
                dataset_dir=/home/server08/hdd0/changhun_workspace/CHiME3 \
                method=em_uncertainty \
                log_dir=exps/exps_trans_chime_em_uncertainty \

CUDA_VISIBLE_DEVICES=2 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=chime \
                dataset_dir=/home/server08/hdd0/changhun_workspace/CHiME3 \
                method=em_sparse \
                log_dir=exps/exps_trans_chime_em_sparse \

CUDA_VISIBLE_DEVICES=2 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=chime \
                dataset_dir=/home/server08/hdd0/changhun_workspace/CHiME3 \
                method=cr \
                log_dir=exps/exps_trans_chime_cr \