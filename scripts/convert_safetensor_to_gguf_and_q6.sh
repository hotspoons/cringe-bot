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

if [[ ! -f  "$LLAMA_CPP_PATH/convert.py" || ! -f  "$LLAMA_CPP_PATH/quantize"  || ! -f  "$BASE_MODEL_PATH" ]]; then
  echo "Please ensure llama.cpp is compiled and available on the path $LLAMA_CPP_PATH (or set the env var LLAMA_CPP_PATH)"
  echo "and that the model is downloaded to $BASE_MODEL_PATH (or set the env var BASE_MODEL_PATH) and try again"
  exit 1
fi

mkdir $OUTPUT_FOLDER || EXIT_CODE=$?
cd $SCRIPT_DIR

# Convert safetensors to an unquantized GGUF file
python $LLAMA_CPP_PATH/convert.py $MODEL_FOLDER --outtype f16 --concurrency $NUM_PROCESSORS --outfile $OUTPUT_FOLDER/out.gguf
# Quantize the GGUF file
$LLAMA_CPP_PATH/quantize $OUTPUT_FOLDER/out.gguf $OUTPUT_FOLDER/$MODEL_NAME-$QUANT.gguf $QUANT