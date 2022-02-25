#! /bin/bash

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 2.5 \
                --episodic \
                --em_coef 0.7 \
                --reweight \
                --log_dir feat_exps \
                --lr 2e-5 \
                --non_blank \
                --train_feature \
                --extra_noise 0 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 2 \
                --episodic \
                --em_coef 0.7 \
                --reweight \
                --lr 2e-5 \
                --non_blank \
                --log_dir feat_exps \
                --train_feature \
                --extra_noise 0 \
                

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 2.5 \
                --episodic \
                --em_coef 0.7 \
                --reweight \
                --lr 2e-5 \
                --non_blank \
                --train_feature \
                --log_dir feat_exps \
                --extra_noise 0 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 2 \
                --episodic \
                --em_coef 0.7 \
                --reweight \
                --lr 2e-5 \
                --non_blank \
                --log_dir feat_exps \
                --train_feature \
                --extra_noise 0 \