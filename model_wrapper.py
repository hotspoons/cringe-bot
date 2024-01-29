
import json
import torch
from config import Config
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

class ModelWrapper():
    def __init__(self, config: Config):
        self.config = config.config
        self.parameter_map = config.get_parameter_map()
        self.model = None
        self.tokenizer = None
    def get_model(self):
        if not self.model:
            model_args = {}
            model_args['pretrained_model_name_or_path'] = self.config.model
            model_args['device_map'] = self.get_device_map()
            
            if not self.config.no_quantization:
                model_args['quantization_config'] = self.get_bnb_config()
            model_args = {**model_args, **self.parameter_map['model']}
            self.model = AutoModelForCausalLM.from_pretrained(**model_args)
        return self.model
    
    def get_tokenizer(self):
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, 
                                                  **self.parameter_map['tokenizer'])
        return self.tokenizer

    def get_device_map(self):
        if '"' in self.config.device_map:
            return json.loads(self.config.device_map)            
        return self.config.device_map

    def get_bnb_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    