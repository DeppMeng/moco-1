source activate moco-python3.6.3
python -m torch.distributed.launch --nproc_per_node=4 main_moco_distributed_launch.py \
    -a resnet50 \
    --lr 0.015 \
    --epochs 800 \
    --batch-size 128 \
    --output-dir imagenet100_res50_mocov2_bs128_lr15e-3_4gpus_epoch800_dist_launch_aml_exp3 \
    --resume auto \
    --tsv-data \
    --cos \
    --aug-plus \
    --mlp \
    ../data/imagenet100/images/
python -m torch.distributed.launch --nproc_per_node=4 main_lincls_dist_launch.py \
    -a resnet50 \
    --lr 30 \
    --batch-size 256 \
    --epochs 60 \
    --pretrained output/imagenet100_res50_mocov2_bs128_lr15e-3_4gpus_epoch800_dist_launch_aml_exp3/checkpoint_current.pth.tar \
    --output-dir imagenet100_res50_mocov2_bs128_lr15e-3_4gpus_epoch800_dist_launch_aml_exp3 \
    --resume auto \
    --tsv-data \
    --le-dataset imagenet \
    ../data/imagenet/images/
