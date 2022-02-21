#! /bin/bash

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 2 \
                --episodic \
                --em_coef 0.5 \
                --reweight \
                --non_blank \


python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 1.5 \
                --episodic \
                --em_coef 0.5 \
                --reweight \
                --non_blank \


python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 1 \
                --episodic \
                --em_coef 0.5 \
                --reweight \
                --non_blank \

##
python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 2 \
                --episodic \
                --em_coef 0 \
                --reweight \
                --non_blank \


python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 1.5 \
                --episodic \
                --em_coef 0 \
                --reweight \
                --non_blank \


python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 1 \
                --episodic \
                --em_coef 0 \
                --reweight \
                --non_blank \
