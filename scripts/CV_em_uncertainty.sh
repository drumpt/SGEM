python main.py --asr facebook/wav2vec2-base-960h \
                    --steps 10 \
                    --dataset_name commonvoice \
                    --dataset_dir /home/server08/hdd0/changhun_workspace/cv-corpus-5.1-2020-06-22/en \
                    --temp 2.5 \
                    --episodic \
                    --non_blank \
                    --em_coef 0.3 \
                    --reweight \
                    --log_dir exps \
                    --lr 2e-5 \
                    --train_feature \
                    --extra_noise 0 \
                    --method em_uncertainty \