### eval scala on deit_s at width ratio of 0.5

python -m torch.distributed.launch --nproc_per_node=4 --use_env main_scala.py --eval \
    --data-path IMAGENET_LOCATION \
    --model deit_small_distilled_patch16_224_scala \
    --batch-size 256 --eval_ratio 0.5 \
    --resume MODEL_PATH;