python main.py --asr facebook/wav2vec2-base-960h \
                --steps 10 \
                --dataset_name ted \
                --dataset_dir /home/server08/hdd0/changhun_workspace/TEDLIUM_release2/test \
                --temp 2.5 \
                --em_coef 0.3 \
                --reweight \
                --log_dir exps \
                --lr 2e-6 \
                --non_blank \
                --train_feature \
                --method cr \
                --teacher_student \