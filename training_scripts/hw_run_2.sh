# (TO MODIFY) Slurm arguments
node=c001
partition=yuxinchen-own
mem=24G
jobname=lora

# Other arguments
export MODEL_NAME="emilianJR/chilloutmix_NiPrunedFp32Fix"
export INSTANCE_DIR="./data/harry_winston"
export OUTPUT_DIR="./exps/output_hw"

srun -w ${node} --gres=gpu:1 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --scale_lr \
  --learning_rate_unet=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --placeholder_tokens="<s1>|<s2>" \
  --use_template="style"\
  --save_steps=100 \
  --max_train_steps_ti=1000 \
  --max_train_steps_tuning=1000 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001\
  --continue_inversion \
  --continue_inversion_lr=1e-4 \
  --device="cuda" \
  --lora_rank=1 \
#  --use_face_segmentation_condition\
