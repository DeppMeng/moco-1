source activate moco-python3.6.3
python -m torch.distributed.launch --nproc_per_node=4 main_lincls_dist_launch.py \
    -a resnet50 \
    --lr 30 \
    --batch-size 256 \
    --epochs 60 \
    --pretrained output/unsup_imagenet100_res50_mocov2_bs128_lr15e-3_4gpus_epoch200_distributed_launch_aml_exp2/checkpoint_current.pth.tar \
    --output-dir unsup_imagenet100_res50_mocov2_bs128_lr15e-3_4gpus_epoch200_distributed_launch_aml_exp2 \
    --resume auto \
    --tsv-data \
    ../data/imagenet/images/
