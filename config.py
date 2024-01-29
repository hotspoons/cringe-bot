import argparse, os, json, sys, logging
from deepmerge import always_merger
import traceback
from dotenv import load_dotenv

class Config():

    def __init__(self):
        load_dotenv()
        self.override_params = {}
        self.parameter_map = None
        self.parse_args()
        self.logger = logging.getLogger()
        self.logger.setLevel(self.config.log_level)
        self.logger.info("Application configured")

    def envar_or_req(self, key, req=True, default=None):
        if os.environ.get(key):
            return {'default': os.environ.get(key)}
        elif default != None:
            return {'default': default}
        else:
            return {'required': req}
        
    def get_parameter_map(self):
        if not self.parameter_map:
            parameter_map_file = self.config.parameter_map_location
            if parameter_map_file == '':
                parameter_map_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'parameter-map.json')
            with open(parameter_map_file) as f:
                self.parameter_map =json.load(f)
        return always_merger.merge(self.parameter_map, self.override_params)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-L', '--parameter-map-location', help='Location of parameter map JSON file (defaults to parameter-map.json in current directory)', 
                            **self.envar_or_req('PARAMETER_MAP_LOCATION', False, ''))
        parser.add_argument('-m', '--model', help='The Hugging Face model (or local path) to use as the base model to fine tune', 
                            **self.envar_or_req('MODEL', True))
        parser.add_argument('-c', '--chats-location', help='The folder on the file system that contains AOL Instant Messenger ' + 
                            'chat logs. This will be searched recursively for htm/html files', **self.envar_or_req('CHATS_LOCATION', True))
        parser.add_argument('-s', '--screen-names', help='A comma-separated list of screen names to use as the assistant role', 
                            **self.envar_or_req('SCREEN_NAMES', True))
        parser.add_argument('-f', '--peft-folder', help='Folder for where PEFT/LoRA weights will be stored, must be writable', 
                            **self.envar_or_req('PEFT_FOLDER', True))
        parser.add_argument('-M', '--merged-folder', help='Provide a destination folder for the LoRA weights merged into the base model. Defaults to ${current_folder}/merged', 
                            **self.envar_or_req('MERGED_FOLDER', False, 'merged'))
        parser.add_argument('-q', '--quantized-merge', help='Quantize the merged model', action='store_true', **self.envar_or_req('QUANTIZED_MERGE', True, False)) 
        parser.add_argument('-w', '--wandb-key', help='Weights and Biases authentication key, will override local wandb config', 
                            **self.envar_or_req('WANDB_KEY', False))
        parser.add_argument('-u', '--wandb-url', help='Weights and Biases authentication URL, if not using main service', 
                            **self.envar_or_req('WANDB_URL', False))
        parser.add_argument('-d', '--device-map', help='PyTorch device map, defaults to \'auto\'. A complex device map may be provided as JSON', 
                            **self.envar_or_req('DEVICE_MAP', False, 'auto'))
        parser.add_argument('-D', '--merge-device-map', help='PyTorch device map for model merge step, defaults to \'cpu\'. A complex device map may be provided as JSON', 
                            **self.envar_or_req('MERGE_DEVICE_MAP', False, 'cpu'))
        parser.add_argument('-C', '--merge-checkpoint-id', help='Checkpoint ID you wish to merge (e.g. checkpoint-150); if omitted, the last checkpoint will be used',
                            **self.envar_or_req('MERGE_CHECKPOINT_ID', False, ''))
        parser.add_argument('-p', '--port', help='API server port for hosting API and UI to observe progress. Defaults to zero (disabled)', 
                            type=int, **self.envar_or_req('PORT', False, 0))
        parser.add_argument('-P', '--project-name', help='Project name for Weights and Biases; will be auto-generated if omitted', 
                            **self.envar_or_req('PROJECT_NAME', False))
        parser.add_argument('-x', '--extended-parameters', help='Optional JSON map for parameters to override QLoRA, BnB, SFT, and training hyperparameters. ' +
                            'A sparse map will be deep merged, so only provide values you wish to override. Default values are in parameter-map.json', 
                            **self.envar_or_req('EXTENDED_PARAMETERS', False, '{}'))
        parser.add_argument('--no-quantization', help='If you can load the model without quantization, set this flag', action='store_true', **self.envar_or_req('NO_QUANTIZATION', True, False))
        parser.add_argument('-i', '--instruct-prompt', help='Extra instruct prompt prepended to each training data point (optional)', 
                            **self.envar_or_req('INSTRUCT_PROMPT', False, ''))
        parser.add_argument('-t', '--training-template', help='The training template in Jinja2 format that will have the values "user", "assistant", and "prompt" available', 
                            **self.envar_or_req('TRAINING_TEMPLATE', False, '<s> [INST] {{prompt}}{{user}} [/INST] \n {{assistant}}'))
        parser.add_argument('-l', '--log-level', help='Log level (CRITICAL, ERROR, WARNING, INFO, DEBUG)', 
                            **self.envar_or_req('LOG_LEVEL', False, 'INFO'))
        parser.add_argument('-X', '--max-data-length', help='Maximum data length; conversations exceeding this will be split (defaults to 1024)', 
                            type=int, **self.envar_or_req('MAX_DATA_LENGTH', False, 1024))
        parser.add_argument('-F', '--eval-fraction', help='Fraction (1/$this_number) of data points to add to evaluation set, defaults to 10 (e.g. one-tenth)', 
                            type=int, **self.envar_or_req('EVAL_FRACTION', False, 10))
        parser.add_argument('-a', '--actions', help='Comma separated list of action(s) to perform. Options are "FINE_TUNE", "MERGE", "QUANTIZE", "SERVE", default is TRAIN', 
                            **self.envar_or_req('ACTIONS', False, "TRAIN"))
        
        try:
            self.config = parser.parse_args()
            self.override_params = json.loads(self.config.extended_parameters)
        except:
            traceback.print_exc()
            parser.print_help()
            sys.exit(0)
