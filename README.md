# CringeBot 

This is a CLI tool to perform a QLoRA fine tuning of large language models with your old AOL Instant Messenger chat logs!
I tried to design sufficient flexibility into the tool so other types of chat source data can be adapted in the future
by implementing the `SplitDataset` interface and providing the new implementation to the `Trainer` class.

## Usage

This should work with any model that can be loaded with `AutoModelForCausalLM.from_pretrained`. 

```shell
python3 app.py --model 'mistralai/Mixtral-8x7B-Instruct-v0.1' --chats-location '/path/to/my/aim/chats/root' \
  --screen-names 'MyAIMScreenName,MyOtherAIMScreenName' --peft-folder 'results'
```

or (with env vars)

```shell
export MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
export CHATS_LOCATION="/path/to/my/aim/chats/root"
export SCREEN_NAMES="MyAIMScreenName,MyOtherAIMScreenName"
export PEFT_FOLDER="finetune"
# Additional parameters as an example
export TRAINING_TEMPLATE="<s> [INST] {{user}} [/INST] \n {{assistant}}"
export ACTIONS="train"
python3 app.py
```

Or can you can provide a `.env` file (an example is in `.env.example`) instead of setting environment variables.

You can optionally pass a comma separated list of actions using the `--actions` CLI arg or `ACTIONS` env var that will be
executed in the order you provide. Supported actions:
 - train (fine tune the underlying model using QLoRA adapaters)
 - merge (merge the adapters into the base model)

## Configuration
 
To see all runtime arguments, run `python3 app.py --help`. 

```
usage: app.py [-h] [-L PARAMETER_MAP_LOCATION] -m MODEL -c CHATS_LOCATION -s SCREEN_NAMES -f PEFT_FOLDER [-M MERGED_FOLDER] [-q] [-w WANDB_KEY] [-u WANDB_URL] [-d DEVICE_MAP] [-D MERGE_DEVICE_MAP] [-C MERGE_CHECKPOINT_ID] [-p PORT] [-P PROJECT_NAME] [-x EXTENDED_PARAMETERS] [--no-quantization] [-i INSTRUCT_PROMPT]
              [-t TRAINING_TEMPLATE] [-l LOG_LEVEL] [-X MAX_DATA_LENGTH] [-F EVAL_FRACTION] [-a ACTIONS]

optional arguments:
  -h, --help            show this help message and exit
  -L PARAMETER_MAP_LOCATION, --parameter-map-location PARAMETER_MAP_LOCATION
                        Location of parameter map JSON file (defaults to parameter-map.json in current directory)
  -m MODEL, --model MODEL
                        The Hugging Face model (or local path) to use as the base model to fine tune
  -c CHATS_LOCATION, --chats-location CHATS_LOCATION
                        The folder on the file system that contains AOL Instant Messenger chat logs. This will be searched recursively for htm/html files
  -s SCREEN_NAMES, --screen-names SCREEN_NAMES
                        A comma-separated list of screen names to use as the assistant role
  -f PEFT_FOLDER, --peft-folder PEFT_FOLDER
                        Folder for where PEFT/LoRA weights will be stored, must be writable
  -M MERGED_FOLDER, --merged-folder MERGED_FOLDER
                        Provide a destination folder for the LoRA weights merged into the base model. Defaults to ${current_folder}/merged
  -q, --quantized-merge
                        Quantize the merged model
  -w WANDB_KEY, --wandb-key WANDB_KEY
                        Weights and Biases authentication key, will override local wandb config
  -u WANDB_URL, --wandb-url WANDB_URL
                        Weights and Biases authentication URL, if not using main service
  -d DEVICE_MAP, --device-map DEVICE_MAP
                        PyTorch device map, defaults to 'auto'. A complex device map may be provided as JSON
  -D MERGE_DEVICE_MAP, --merge-device-map MERGE_DEVICE_MAP
                        PyTorch device map for model merge step, defaults to 'cpu'. A complex device map may be provided as JSON
  -C MERGE_CHECKPOINT_ID, --merge-checkpoint-id MERGE_CHECKPOINT_ID
                        Checkpoint ID you wish to merge (e.g. checkpoint-150); if omitted, the last checkpoint will be used
  -p PORT, --port PORT  API server port for hosting API and UI to observe progress. Defaults to zero (disabled)
  -P PROJECT_NAME, --project-name PROJECT_NAME
                        Project name for Weights and Biases; will be auto-generated if omitted
  -x EXTENDED_PARAMETERS, --extended-parameters EXTENDED_PARAMETERS
                        Optional JSON map for parameters to override QLoRA, BnB, SFT, and training hyperparameters. A sparse map will be deep merged, so only provide values you wish to override. Default values are in parameter-map.json
  --no-quantization     If you can load the model without quantization, set this flag
  -i INSTRUCT_PROMPT, --instruct-prompt INSTRUCT_PROMPT
                        Extra instruct prompt prepended to each training data point (optional)
  -t TRAINING_TEMPLATE, --training-template TRAINING_TEMPLATE
                        The training template in Jinja2 format that will have the values "user", "assistant", and "prompt" available
  -l LOG_LEVEL, --log-level LOG_LEVEL
                        Log level (CRITICAL, ERROR, WARNING, INFO, DEBUG)
  -X MAX_DATA_LENGTH, --max-data-length MAX_DATA_LENGTH
                        Maximum data length; conversations exceeding this will be split (defaults to 1024)
  -F EVAL_FRACTION, --eval-fraction EVAL_FRACTION
                        Fraction (1/$this_number) of data points to add to evaluation set, defaults to 10 (e.g. one-tenth)
  -a ACTIONS, --actions ACTIONS
                        Comma separated list of action(s) to perform. Options are "FINE_TUNE", "MERGE", "QUANTIZE", "SERVE", default is TRAIN
```

## Parameters and hyperparameters

Parameters for qlora, tokenizer, model, training, and sft are all configured in `parameter-map.json`. To change
these values, you can either directly modify that file; provide a different version of the file on the
`--parameter-map-location` argument/`PARAMETER_MAP_LOCATION` env var; or you can override individual values in a sparse 
dictionary by passing a JSON dictionary to `--extended-parameters`. 

As an example, to change the training learning rate and epochs using extended parameters: 
```shell
python3 app.py --extended-parameters '{"training":{"learning_rate": 2e-4, "num_train_epochs": 3}}'
``` 
