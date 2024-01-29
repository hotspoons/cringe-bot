#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )
cd $SCRIPT_DIR
pwd
rm -rf cringe-bot-gguf/*
rm -rf finetune/*
rm -rf merged/*
rm -rf results/*