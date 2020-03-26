python -m torch.distributed.launch --nproc_per_node=4 main_moco_distributed_launch.py \
    -a resnet50 \
    --lr 0.015 \
    --batch-size 128 \
    --dist-url 'tcp://localhost:10001' \
    # --multiprocessing-distributed \
    # --world-size 1 \
    # --rank 0 \
    --output-dir unsup_res50_bs128_lr15e-3_4gpus_epoch200_dist_launch \
    --resume checkpoint_0020.pth.tar\
    data/imagenet/images/
