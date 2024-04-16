# for some server
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=3,5,6,7
python -m torch.distributed.launch --nproc_per_node=4  --master_port='29511' tools/train.py --cfg experiments_yaml/hrnet_head.yaml --modelDir output --logDir logs
# python -m torch.distributed.launch --nproc_per_node=4  --master_port='29511' tools/train.py --cfg experiments_yaml/oeformer.yaml --modelDir output --logDir logs
