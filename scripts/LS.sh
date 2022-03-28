# LS + 0
python main.py --asr facebook/wav2vec2-base-960h \
                --steps 10 \
                --dataset_name librispeech \
                --dataset_dir /home/daniel094144/data/LibriSpeech \
                --temp 2.5 \
                --episodic \
                --em_coef 0.3 \
                --reweight \
                --log_dir exps \
                --lr 2e-5 \
                --non_blank \
                --train_feature \
                --extra_noise 0 \

# LS + 0.005
python main.py --asr facebook/wav2vec2-base-960h \
                --steps 10 \
                --dataset_name librispeech \
                --dataset_dir /home/daniel094144/data/LibriSpeech \
                --temp 2.5 \
                --episodic \
                --em_coef 0.3 \
                --reweight \
                --log_dir exps \
                --lr 2e-5 \
                --non_blank \
                --train_feature \
                --extra_noise 0.005 \

# LS + 0.01
python main.py --asr facebook/wav2vec2-base-960h \
                --steps 10 \
                --dataset_name librispeech \
                --dataset_dir /home/daniel094144/data/LibriSpeech \
                --temp 2.5 \
                --episodic \
                --em_coef 0.3 \
                --reweight \
                --log_dir exps \
                --lr 2e-5 \
                --non_blank \
                --train_feature \
                --extra_noise 0.01 \