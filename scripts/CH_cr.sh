python main.py --asr facebook/wav2vec2-base-960h \
                --steps 10 \
                --dataset_name chime \
                --dataset_dir /home/server08/hdd0/changhun_workspace/CHiME3 \
                --temp 2.5 \
                --em_coef 0.3 \
                --reweight \
                --log_dir exps \
                --lr 5e-6 \
                --non_blank \
                --train_feature \
                --method cr \
                --episodic \
                --teacher_student \

# python main.py --asr facebook/wav2vec2-base-960h \
#                 --steps 10 \
#                 --dataset_name chime \
#                 --dataset_dir /home/server08/hdd0/changhun_workspace/CHiME3 \
#                 --temp 2.5 \
#                 --episodic \
#                 --em_coef 0.3 \
#                 --reweight \
#                 --log_dir exps \
#                 --lr 2e-6 \
#                 --non_blank \
#                 --train_all \
#                 --method cr \