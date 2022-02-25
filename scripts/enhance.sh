python main.py --asr facebook/wav2vec2-base-960h \
                --steps 1 \
                --dataset_name ted \
                --dataset_dir /home/daniel094144/data/TEDLIUM_release2/test \
                --temp 1 \
                --episodic \
                --enhance \
                --log_dir feat_exps \
                --extra_noise 0 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 1 \
                --dataset_name swbd \
                --dataset_dir /home/daniel094144/data/Switchboard \
                --temp 1 \
                --episodic \
                --enhance \
                --log_dir feat_exps \
                --extra_noise 0 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 1 \
                --dataset_name chime \
                --dataset_dir /home/daniel094144/data/CHiME3 \
                --temp 1 \
                --episodic \
                --enhance \
                --log_dir feat_exps \
                --extra_noise 0 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 1 \
                --dataset_name librispeech \
                --dataset_dir /home/daniel094144/data/LibriSpeech \
                --temp 1 \
                --episodic \
                --enhance \
                --log_dir feat_exps \
                --extra_noise 0.01 \

python main.py --asr facebook/wav2vec2-base-960h \
                --steps 1 \
                --dataset_name librispeech \
                --dataset_dir /home/daniel094144/data/LibriSpeech \
                --temp 1 \
                --episodic \
                --enhance \
                --log_dir feat_exps \
                --extra_noise 0.005 \

