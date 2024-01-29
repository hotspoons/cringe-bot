#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )
MODEL_NAME=cringe-bot-8x7b-instruct
MODEL_FOLDER=$SCRIPT_DIR/merged
OUTPUT_FOLDER=$SCRIPT_DIR/cringe-bot-gguf
BASE_MODEL_PATH=/root/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/125c431e2ff41a156b9f9076f744d2f35dd6e67a/
LLAMA_CPP_PATH=/opt/dev/llama.cpp
NUM_PROCESSORS=24
QUANT=q6_k
set -e

# Copy metadata from base model since what came out of the QLoRA embedding process was missing something
#cd $MODEL_FOLDER
#mkdir old_meta
#mv *.json old_meta
#cp $BASE_MODEL_PATH/*.json .
#cp $BASE_MODEL_PATH/*.model .

mkdir $OUTPUT_FOLDER || EXIT_CODE=$?
cd $SCRIPT_DIR

# Convert safetensors to an unquantized GGUF file
python $LLAMA_CPP_PATH/convert.py $MODEL_FOLDER --outtype f16 --concurrency $NUM_PROCESSORS --outfile $OUTPUT_FOLDER/out.gguf
# Quantize the GGUF file
$LLAMA_CPP_PATH/quantize $OUTPUT_FOLDER/out.gguf $OUTPUT_FOLDER/$MODEL_NAME-$QUANT.gguf $QUANT