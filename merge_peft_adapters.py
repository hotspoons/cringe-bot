
import json, os, glob
import torch
from config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Adapted from https://gist.github.com/TheBloke/d31d289d3198c24e0ca68aaf37a19032

class MergePeftAdapters:
    def __init__(self, config: Config) -> None:
        self.config = config.config
        self.logger = config.logger
        self.parameter_map = config.get_parameter_map()

    def get_device_map(self):
        device_map = self.config.merge_device_map if self.config.merge_device_map else self.config.device_map
        if '"' in device_map:
            device_map = json.loads(self.config.device_map)
        if isinstance(device_map, str):
            if device_map == 'auto':
                return { 'device_map': 'auto' }
            else:
                return { 'device_map': { "": device_map} }
        else:
            return { 'device_map': device_map }
            

    def get_model(self):
        self.logger.info(self.get_device_map())
        return AutoModelForCausalLM.from_pretrained(
            self.config.model,
            offload_folder="offload/",
            return_dict=True,
            torch_dtype=torch.float16,
            **self.get_device_map()
        )
    
    def get_lora_snapshot(self):
        snapshots_folder = self.parameter_map['training']['output_dir']
        if self.config.merge_checkpoint_id == '':
            return self.config.peft_folder
        else:
            snapshot = snapshots_folder + '/' + self.config.merge_checkpoint_id
            self.logger.info(f"Using snapshot {snapshot}")
            return snapshot

    def merge(self):
        args = self.config

        self.logger.info(f"Loading base model: {args.model}")
        base_model = self.get_model()
        snapshot_or_final = self.get_lora_snapshot()
        self.logger.info(f"Loading PEFT: {snapshot_or_final}")
        model = PeftModel.from_pretrained(
                base_model, 
                self.snapshot_or_final, 
                offload_folder="offload/",
                **self.get_device_map())
        self.logger.info(f"Running merge_and_unload")
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model.save_pretrained(f"{args.merged_folder}")
        tokenizer.save_pretrained(f"{args.merged_folder}")
        self.logger.info(f"Model saved to {args.merged_folder}")