### scala on deit_s for 100 epoch

python -m torch.distributed.launch --nproc_per_node=4 --use_env main_scala.py \
    --batch-size 256 --epochs 100 --data-path IMAGENET_LOCATION \
    --aa rand-m1-mstd0.5-inc1 --no-repeated-aug --lr 2e-3 --warmup-epochs 3 \
    --teacher-model deit_small_patch16_224 --distillation-type hard \
    --teacher-path https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
    --model deit_small_distilled_patch16_224_scala \
    --smallest_ratio 0.25 --largest_ratio 1.0 --granularity 0.0625 --distill_type soft \
    --transfer_type progressive --token_type dist_token --ce_coefficient 1.0 \
    --full_warm_epoch 10 --output_dir log/deit_small_distilled_scala;



### scala on deit_b for 300 epoch

python -m torch.distributed.launch --nproc_per_node=4 --use_env main_scala.py \
    --batch-size 128 --epochs 300 --data-path IMAGENET_LOCATION \
    --aa rand-m4-mstd0.5-inc1 --no-repeated-aug --lr 1e-3 --warmup-epochs 5 \
    --teacher-model deit_base_patch16_224 --distillation-type hard \
    --teacher-path https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth \
    --model deit_base_distilled_patch16_224_scala \
    --smallest_ratio 0.25 --largest_ratio 1.0 --discrete_ratio 0.0625 --distill_type soft \
    --transfer_type progressive --token_type dist_token --ce_coefficient 1.0 \
    --full_warm_epoch 10 --output_dir log/deit_base_distilled_scala;