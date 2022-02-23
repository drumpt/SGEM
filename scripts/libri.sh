#! /bin/bash

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name librispeech \
                --dataset_dir /home/daniel094144/data/LibriSpeech \
                --temp 2 \
                --episodic \
                --non_blank \
                --extra_noise 0.01 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 40 \
                --dataset_name librispeech \
                --dataset_dir /home/daniel094144/data/LibriSpeech \
                --temp 2 \
                --episodic \
                --non_blank \
                --extra_noise 0.005 \

# python main.py --asr facebook/wav2vec2-base-960h \
#                 --steps 40 \
#                 --dataset_name librispeech \
#                 --dataset_dir /home/daniel094144/data/LibriSpeech \
#                 --temp 1.5 \
#                 --episodic \
#                 --non_blank \
#                 --extra_noise 0.005 \

# python main.py --asr facebook/wav2vec2-base-960h \
#                 --steps 40 \
#                 --dataset_name librispeech \
#                 --dataset_dir /home/daniel094144/data/LibriSpeech \
#                 --temp 1.5 \
#                 --episodic \
#                 --non_blank \
#                 --extra_noise 0.01 \

# # temp 2
# python main.py --asr facebook/wav2vec2-base-960h \
#                 --steps 40 \
#                 --dataset_name librispeech \
#                 --dataset_dir /home/daniel094144/data/LibriSpeech \
#                 --temp 1 \
#                 --episodic \
#                 --non_blank \
#                 --extra_noise 0 \

# python main.py --asr facebook/wav2vec2-base-960h \
#                 --steps 40 \
#                 --dataset_name librispeech \
#                 --dataset_dir /home/daniel094144/data/LibriSpeech \
#                 --temp 1 \
#                 --episodic \
#                 --non_blank \
#                 --extra_noise 0.005 \

# python main.py --asr facebook/wav2vec2-base-960h \
#                 --steps 40 \
#                 --dataset_name librispeech \
#                 --dataset_dir /home/daniel094144/data/LibriSpeech \
#                 --temp 1 \
#                 --episodic \
#                 --non_blank \
#                 --extra_noise 0.01 \

# # non_blank == False
# python main.py --asr facebook/wav2vec2-base-960h \
#                 --steps 40 \
#                 --dataset_name librispeech \
#                 --dataset_dir /home/daniel094144/data/LibriSpeech \
#                 --temp 1 \
#                 --episodic \
#                 --extra_noise 0 \

# python main.py --asr facebook/wav2vec2-base-960h \
#                 --steps 40 \
#                 --dataset_name librispeech \
#                 --dataset_dir /home/daniel094144/data/LibriSpeech \
#                 --temp 1 \
#                 --episodic \
#                 --extra_noise 0.005 \

# python main.py --asr facebook/wav2vec2-base-960h \
#                 --steps 40 \
#                 --dataset_name librispeech \
#                 --dataset_dir /home/daniel094144/data/LibriSpeech \
#                 --temp 1 \
#                 --episodic \
#                 --extra_noise 0.01 \