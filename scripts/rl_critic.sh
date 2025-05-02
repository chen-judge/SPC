TOTAL_BATCH_SIZE=64
MICRO_TRAIN_BATCH_SIZE=1
MICRO_EVAL_BATCH_SIZE=1
NUM_GPUS=$NODE_NUM
SAVE_STEPS=100
EVAL_STEPS=100
GRADIENT_CHECKPOINTING=True
BF16=True
LEARNING_RATE=2e-6
MODEL_MAX_LENGTH=4096
GRADIENT_ACCUMULATION_STEPS=$[$TOTAL_BATCH_SIZE/$MICRO_TRAIN_BATCH_SIZE/$NUM_GPUS]

# SFT model (please download it from Hugging Face)
MODEL_NAME_OR_PATH=checkpoints/SPC-Critic-0

DATA_DIR=data
TRAIN_DATA_PATH=data/train/data_round2_rl_critic.json
SFT_DATA_PATH=data/train/data_round0_sft_critic.json

OUTPUT_DIR=saved_models/test
LOGS_PATH=${OUTPUT_DIR}/logs
mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_PATH

# distributed setting
MASTER_ADDR=$CHIEF_IP
MASTER_PORT=6000
TMP_DIR=${OUTPUT_DIR}/tmp
mkdir -p $TMP_DIR

echo $NODE_IP_LIST > ${TMP_DIR}/env.txt

# generate hostfile and pssh.hosts
sed "s/:/ slots=/g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/hostfile
sed "s/:.//g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/pssh.hosts


deepspeed --hostfile ${TMP_DIR}/hostfile --master_addr ${MASTER_ADDR} --master_port=${MASTER_PORT} src/offline_rl.py \
    --output_dir ${OUTPUT_DIR} \
    --do_train True \
    --data_paths ${TRAIN_DATA_PATH}  ${SFT_DATA_PATH} \
    --model_type qwen \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --remove_unused_columns False \
    --report_to tensorboard \
    --overwrite_output_dir True \
    --per_device_train_batch_size ${MICRO_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${MICRO_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_train_epochs 3 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_steps ${SAVE_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --evaluation_strategy no \
    --warmup_steps 10 \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --bf16 ${BF16} \
    --lm_kl_coeff 0.1 \
    --lm_sft_coeff 0.15 \
    --deepspeed config/ds_config.json

