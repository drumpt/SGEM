#! /bin/bash

# temp 2.5
python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 1.5 \
                --episodic \
                --non_blank \


# temp 2
python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 1 \
                --episodic \
                --non_blank \


# non_blank == False
python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 1.5 \
                --episodic \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 1 \
                --episodic \
