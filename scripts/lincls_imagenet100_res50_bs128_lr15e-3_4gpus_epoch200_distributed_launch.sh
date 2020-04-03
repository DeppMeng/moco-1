python -m torch.distributed.launch --nproc_per_node=4 main_lincls_dist_launch.py \
    -a resnet50 \
    --lr 30 \
    --batch-size 256 \
    --epochs 60 \
    --dist-url 'tcp://localhost:10001' \
    --pretrained output/unsup_imagenet100_res50_bs128_lr15e-3_4gpus_epoch200_dist_launch/checkpoint_current.pth.tar
    --output-dir unsup_imagenet100_res50_bs128_lr15e-3_4gpus_epoch200_dist_launch \
    --resume auto \
    --tsv-data \
    data/imagenet/images/
