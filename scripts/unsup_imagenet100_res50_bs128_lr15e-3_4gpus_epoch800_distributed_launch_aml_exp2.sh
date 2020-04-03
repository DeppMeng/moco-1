source activate moco-python3.6.3
python -m torch.distributed.launch --nproc_per_node=4 main_moco_distributed_launch.py \
    -a resnet50 \
    --lr 0.015 \
    --epochs 800 \
    --batch-size 128 \
    --dist-url 'tcp://localhost:10001' \
    --output-dir unsup_imagenet100_res50_bs128_lr15e-3_4gpus_epoch800_dist_launch_aml_exp2 \
    --resume auto \
    --tsv-data \
    --cos \
    ../data/imagenet100/images/
