#! /bin/bash

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name librispeech \
                --dataset_dir /home/daniel094144/data/LibriSpeech \
                --temp 2.5 \
                --episodic \
                --em_coef 0.7 \
                --reweight \
                --non_blank \
                --extra_noise 0.01 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name librispeech \
                --dataset_dir /home/daniel094144/data/LibriSpeech \
                --temp 2 \
                --episodic \
                --em_coef 0.7 \
                --reweight \
                --non_blank \
                --extra_noise 0.01 \
                

# python main.py --asr facebook/wav2vec2-base-960h \
#                 --steps 40 \
#                 --dataset_name librispeech \
#                 --dataset_dir /home/daniel094144/data/LibriSpeech \
#                 --temp 2.5 \
#                 --episodic \
#                 --em_coef 0.7 \
#                 --reweight \
#                 --non_blank \
#                 --extra_noise 0.005 \

# python main.py --asr facebook/wav2vec2-base-960h \
#                 --steps 40 \
#                 --dataset_name librispeech \
#                 --dataset_dir /home/daniel094144/data/LibriSpeech \
#                 --temp 2 \
#                 --episodic \
#                 --em_coef 0.7 \
#                 --reweight \
#                 --non_blank \
#                 --extra_noise 0.005 \