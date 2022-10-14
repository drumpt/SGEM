CUDA_VISIBLE_DEVICES=3 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=commonvoice \
                dataset_dir=/home/server08/hdd0/changhun_workspace/cv-corpus-5.1-2020-06-22/en \
                extra_noise=0.01 \
                method=original \
                log_dir=exps/exps_trans_cv_original \

CUDA_VISIBLE_DEVICES=3 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=commonvoice \
                dataset_dir=/home/server08/hdd0/changhun_workspace/cv-corpus-5.1-2020-06-22/en \
                extra_noise=0.01 \
                method=em_uncertainty \
                log_dir=exps/exps_trans_cv_em_uncertainty \

CUDA_VISIBLE_DEVICES=3 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=commonvoice \
                dataset_dir=/home/server08/hdd0/changhun_workspace/cv-corpus-5.1-2020-06-22/en \
                extra_noise=0.01 \
                method=em_sparse \
                log_dir=exps/exps_trans_cv_em_sparse \

CUDA_VISIBLE_DEVICES=3 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=commonvoice \
                dataset_dir=/home/server08/hdd0/changhun_workspace/cv-corpus-5.1-2020-06-22/en \
                extra_noise=0.01 \
                method=cr \
                log_dir=exps/exps_trans_cv_cr \

CUDA_VISIBLE_DEVICES=3 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=ted \
                dataset_dir=/home/server08/hdd0/changhun_workspace/TEDLIUM_release2/test \
                method=original \
                log_dir=exps/exps_trans_td_original \

CUDA_VISIBLE_DEVICES=3 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=ted \
                dataset_dir=/home/server08/hdd0/changhun_workspace/TEDLIUM_release2/test \
                method=em_uncertainty \
                log_dir=exps/exps_trans_td_em_uncertainty \

CUDA_VISIBLE_DEVICES=3 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=ted \
                dataset_dir=/home/server08/hdd0/changhun_workspace/TEDLIUM_release2/test \
                method=em_sparse \
                log_dir=exps/exps_trans_td_em_sparse \

CUDA_VISIBLE_DEVICES=3 python main.py \
                asr=pretrained_models/stt_en_conformer_transducer_small.nemo \
                dataset_name=ted \
                dataset_dir=/home/server08/hdd0/changhun_workspace/TEDLIUM_release2/test \
                method=cr \
                log_dir=exps/exps_trans_td_cr \