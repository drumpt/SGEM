#! /bin/bash

# temp 2.5
python main.py --asr facebook/wav2vec2-large-960h-lv60 \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --lr 7e-5 \
                --temp 2.5 \
                --episodic \
                --non_blank \


# temp 2
python main.py --asr facebook/wav2vec2-large-960h-lv60 \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 2 \
                --lr 7e-5 \
                --episodic \
                --non_blank \


# non_blank == False
python main.py --asr facebook/wav2vec2-large-960h-lv60 \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 2.5 \
                --lr 7e-5 \
                --episodic \
