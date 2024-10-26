export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# 将图像转换成.npy文件
torchrun --nnodes=1 \
    --nproc_per_node=1 extract_features.py \
    --image-size 512 \
    --model DiT-XL/2 \
    --data-path /data/xujialiu/fast-DiT/dataset_artifact_severity \
    --features-path dataset_artifact_severity_features_512

# train
nohup accelerate launch \
    --gpu_ids 0 \
    --mixed_precision bf16 train.py \
    --image-size 512 \
    --global-batch-size=32 \
    --num-classes 6\
    --model DiT-XL/2 \
    --feature-path dataset_artifact_severity_features_512 \
    --epochs 100000 \
    --ckpt-every 1000 \
    --log-every 100 > test_dit_artifact.log 2>&1 &

accelerate launch \
    --gpu_ids 1 \
    --mixed_precision bf16 sample.py \
    --model DiT-XL/2 \
    --ckpt /data/xujialiu/fast-DiT/results/002-DiT-XL-2/checkpoints/0055000.pt \
    --image-size 512 \
    --num-classes 6 \
    --num-epochs 1\
    --batch-size 10 \
    --category "0,1,2,3,4,5" \
    --save-dir gen_STDR/OCTA_55000 \
    --num-sampling-steps 100


accelerate launch \
    --gpu_ids 1 \
    --mixed_precision bf16 sample.py \
    --model DiT-XL/2 \
    --ckpt /data/xujialiu/fast-DiT/results/037-DiT-XL-2/checkpoints/0050000.pt \
    --image-size 512 \
    --num-classes 2 \
    --num-epochs 1\
    --batch-size 10 \
    --category "0,1" \
    --save-dir gen_STDR/OCTA_512_2 \
    --num-sampling-steps 100