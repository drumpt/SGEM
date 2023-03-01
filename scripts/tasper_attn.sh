CUDA_VISIBLE_DEVICES=0 python main.py \
    asr=speechbrain/asr-crdnn-rnnlm-librispeech \
    dataset_name=chime \
<<<<<<< HEAD
    dataset_dir=/home/server27/hdd/changhun_workspace/nfs-client/CHiME3 \
    batch_size=1 \
    lr=1e-6 \
    steps=10 \
=======
    dataset_dir=/home/server17/hdd/changhun_workspace/CHiME3 \
    batch_size=1 \
    lr=1e-5 \
    steps=1 \
>>>>>>> 2c1d37b7912e1637ac3cf544a23af6f58d59f650
    'train_params=[all]' \
    'method=[em_joint, original]' \
    em_coef=0 \
    use_memory_queue=true \
    selective_adaptation=false \
    teacher_student=false \
    stochastic_restoration=false \
    log_dir=exps/tasper_ctc \

CUDA_VISIBLE_DEVICES=0 python main.py \
    asr=speechbrain/asr-crdnn-rnnlm-librispeech \
    dataset_name=ted \
<<<<<<< HEAD
    dataset_dir=/home/server27/hdd/changhun_workspace/nfs-client/TEDLIUM_release2/test \
    batch_size=1 \
    lr=1e-6 \
    steps=10 \
=======
    dataset_dir=/home/server17/hdd/changhun_workspace/TEDLIUM_release2/test \
    batch_size=1 \
    lr=1e-5 \
    steps=1 \
>>>>>>> 2c1d37b7912e1637ac3cf544a23af6f58d59f650
    'train_params=[all]' \
    'method=[em_joint, original]' \
    em_coef=0 \
    use_memory_queue=true \
    selective_adaptation=false \
    teacher_student=false \
    stochastic_restoration=false \
    log_dir=exps/tasper_ctc \

CUDA_VISIBLE_DEVICES=0 python main.py \
    asr=speechbrain/asr-crdnn-rnnlm-librispeech \
    dataset_name=valentini \
<<<<<<< HEAD
    dataset_dir=/home/server27/hdd/changhun_workspace/nfs-client/Valentini \
    batch_size=1 \
    lr=1e-6 \
    steps=10 \
=======
    dataset_dir=/home/server17/hdd/changhun_workspace/Valentini \
    batch_size=1 \
    lr=1e-6 \
    steps=1 \
>>>>>>> 2c1d37b7912e1637ac3cf544a23af6f58d59f650
    'train_params=[all]' \
    'method=[em_joint, original]' \
    em_coef=0 \
    use_memory_queue=true \
    selective_adaptation=false \
    teacher_student=false \
    stochastic_restoration=false \
    log_dir=exps/tasper_ctc \

