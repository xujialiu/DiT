torchrun --nnodes=1 \
    --nproc_per_node=1 fast-DiT/extract_features.py \
    --model DiT-XL/2 \
    --data-path test_DiT_dataset \
    --features-path test_DiT_dataset_features

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# 原始
nohup accelerate launch \
    --gpu_ids 2 \
    --mixed_precision bf16 train.py \
    --global-batch-size=32 \
    --num-classes 2\
    --model DiT-XL/2 \
    --feature-path test_DiT_dataset_features \
    --epochs 100000 > test_dit_2.log 2>&1 &


nohup accelerate launch \
    --gpu_ids 2 \
    --mixed_precision bf16 train.py \
    --global-batch-size=96 \
    --num-classes 2\
    --model DiT-XL/2 \
    --feature-path test_DiT_dataset_features \
    --epochs 100000 > test_dit_2.log 2>&1 &

accelerate launch \
    --gpu_ids 2 \
    --mixed_precision bf16 train.py \
    --global-batch-size=256 \
    --num-classes 2\
    --model DiT-XL/2 \
    --feature-path test_DiT_dataset_features \
    --epochs 100000

accelerate launch \
    --gpu_ids 2 \
    --mixed_precision bf16 sample1.py \
    --model DiT-XL/2 \
    --ckpt /home/xujia/fast-DiT/results/001-DiT-XL-2/checkpoints/0300000.pt \
    --image-size 256 \
    --num-classes 2 \
    --num-imgs 10\
    --category 0 \
    --save-dir gen_STDR/500 \
    --num-sampling-steps 500

nohup accelerate launch \
    --gpu_ids 2 \
    --mixed_precision bf16 train.py \
    --global-batch-size=32 \
    --num-classes 2\
    --model DiT-XL/2 \
    --feature-path test_DiT_dataset_features \
    --epochs 100000 > test_dit_2.log 2>&1 &



#------
torchrun --nnodes=1 \
    --nproc_per_node=1 extract_features.py \
    --image-size 512 \
    --model DiT-XL/2 \
    --data-path dataset_scp \
    --features-path dataset_scp_features_512



accelerate launch \
    --gpu_ids 0 \
    --mixed_precision bf16 train.py \
    --ckpt /data/xujialiu/fast-DiT/results/037-DiT-XL-2/checkpoints/0050000.pt\
    --global-batch-size=32 \
    --image-size 512 \
    --num-classes 2\
    --model DiT-XL/2 \
    --feature-path /data/xujialiu/fast-DiT/dataset_scp_features_512 \
    --epochs 100000 \
    --ckpt-every 1000 > test_dit_5.log 2>&1 &

# 生成
accelerate launch \
    --gpu_ids 0 \
    --mixed_precision bf16 sample1.py \
    --model DiT-XL/2 \
    --ckpt /data/xujialiu/fast-DiT/results/037-DiT-XL-2/checkpoints/0050000.pt \
    --image-size 512 \
    --num-classes 2 \
    --num-imgs 10\
    --category 0 \
    --save-dir gen_STDR/512 \
    --num-sampling-steps 500


accelerate launch \
    --gpu_ids 0 \
    --mixed_precision bf16 sample1.py \
    --model DiT-XL/2 \
    --ckpt /data/xujialiu/fast-DiT/results/037-DiT-XL-2/checkpoints/0050000.pt \
    --image-size 512 \
    --num-classes 2 \
    --num-imgs 10\
    --category 1 \
    --save-dir gen_STDR/512 \
    --num-sampling-steps 500

# trial new sample.py
accelerate launch \
    --gpu_ids 0 \
    --mixed_precision bf16 sample.py \
    --model DiT-XL/2 \
    --ckpt /data/xujialiu/fast-DiT/results/037-DiT-XL-2/checkpoints/0050000.pt \
    --image-size 512 \
    --num-classes 2 \
    --num-epochs 10\
    --batch-size 2 \
    --category "0,1" \
    --save-dir gen_STDR/512_test \
    --num-sampling-steps 100