python main.py \
                asr=facebook/wav2vec2-base-960h \
                steps=10 \
                dataset_name=chime \
                dataset_dir=/home/server08/hdd0/changhun_workspace/CHiME3 \
                temp=2.5 \
                em_coef=0.3 \
                reweight=Ture \
                log_dir=exps \
                lr=2e-5 \
                not_blank=True \
                'train_params=[all]' \
                extra_noise=0 \
                episodic=True\