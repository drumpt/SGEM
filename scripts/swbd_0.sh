#! /bin/bash

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name swbd \
                --dataset_dir /home/daniel094144/data/Switchboard \
                --temp 1.5 \
                --episodic \
                --non_blank \
                --em_coef 0.7 \
                --reweight \
                --extra_noise 0 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name swbd \
                --dataset_dir /home/daniel094144/data/Switchboard \
                --temp 1 \
                --episodic \
                --non_blank \
                --em_coef 0.7 \
                --reweight \
                --extra_noise 0 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name swbd \
                --dataset_dir /home/daniel094144/data/Switchboard \
                --temp 1.5 \
                --episodic \
                --non_blank \
                --em_coef 1 \
                --extra_noise 0 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name swbd \
                --dataset_dir /home/daniel094144/data/Switchboard \
                --temp 1 \
                --episodic \
                --non_blank \
                --em_coef 1 \
                --extra_noise 0 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name swbd \
                --dataset_dir /home/daniel094144/data/Switchboard \
                --temp 1.5 \
                --episodic \
                --non_blank \
                --em_coef 0 \
                --reweight \
                --extra_noise 0 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name swbd \
                --dataset_dir /home/daniel094144/data/Switchboard \
                --temp 1 \
                --episodic \
                --non_blank \
                --em_coef 0 \
                --reweight \
                --extra_noise 0 \


