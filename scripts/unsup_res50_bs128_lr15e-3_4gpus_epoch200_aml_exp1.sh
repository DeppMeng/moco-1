source activate moco-python3.6.3
python main_moco.py \
    -a resnet50 \
    --lr 0.015 \
    --batch-size 128 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --output-dir unsup_res50_bs128_lr15e-3_4gpus_epoch200 \
    ../data/imagenet/images/
