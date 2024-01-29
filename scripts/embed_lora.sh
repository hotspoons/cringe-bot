#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )
BASE_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
#LORA_CHECKPOINT_PATH=mixtral-toxic-23-yo-rich-finetune/checkpoint-700
OUTDIR=cringe-bot

if [[ -z "$LORA_CHECKPOINT_PATH" ]]; then
  echo "Please set the env var LORA_CHECKPOINT_PATH, and try again"
  exit 1
fi

python $SCRIPT_DIR/merge_peft_adapters.py --base_model_name_or_path $BASE_MODEL --peft_model_path $LORA_CHECKPOINT_PATH --output_dir $OUTDIR --device cpu