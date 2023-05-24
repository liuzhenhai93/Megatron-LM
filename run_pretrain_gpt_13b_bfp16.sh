unset NCCL_DEBUG
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_CUDA_ARCH_LIST=8.0
if [ $PADDLE_TRAINER_ID -gt '7' ]
then 
    echo "id grather then 7, exit "
    exit
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK"

rm checkpoints/ -rf
CHECKPOINT_PATH=checkpoints/gpt2_13b
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=./my-gpt2_text/my-gpt2-enwiki_text_document

rm -rf ./my-gpt2_text/*indexmap*

#source /root/paddlejob/workspace/env_run/gpt_benchmark/env/gpt_benchmark/bin/activate

GPT_ARGS="--num-layers 10 \
          --seed 1234 \
          --hidden-size 8192 \
          --attention-dropout 0.1 \
          --hidden-dropout 0.1 \
          --num-attention-heads 64 \
          --seq-length 32768 \
          --max-position-embeddings 32768 \
          --micro-batch-size 1 \
          --global-batch-size 8 \
          --lr 0.00005 \
          --min-lr 0.00001 \
          --lr-decay-iters 360000 \
          --lr-decay-style cosine \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --lr-warmup-fraction .01 \
          --no-bias-gelu-fusion \
          --no-bias-dropout-fusion \
          --train-iters 300 \
          --recompute-granularity full \
          --recompute-method uniform \
          --use-sparse-attn \
          --bf16 
          "
          #--bf16 \

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000000 \
             --eval-interval 100000 \
             --eval-iters 1"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun $DISTRIBUTED_ARGS \
       ./pretrain_gpt.py \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 1 \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       2>&1 |tee temp.log

