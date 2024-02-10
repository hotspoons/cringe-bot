from peft import LoraConfig
from transformers import (
    TrainingArguments
)
from trl import SFTTrainer
import wandb, os, re
from config import Config

from model_wrapper import ModelWrapper
from split_dataset import SplitDataset

class Trainer():
    def __init__(self, model_wrapper: ModelWrapper, split_dataset: SplitDataset, config: Config):
        self.config = config.config
        self.parameter_map = config.get_parameter_map() 
        self.model_wrapper = model_wrapper
        self.split_dataset = split_dataset
        self.observe()

    def observe(self):
        kwargs = {}
        if self.config.wandb_key:
            kwargs['key'] = self.config.wandb_key
        if self.config.wandb_url:
            kwargs['host'] = self.config.wandb_url
        if self.config.project_name:
            os.environ["WANDB_PROJECT"] = self.config.project_name
        else:
            os.environ["WANDB_PROJECT"] = f"{re.compile('[^a-zA-Z0-9_]').sub('-', self.config.model)}-finetune"

        wandb.login(**kwargs)

    def train(self):
        base_model = self.model_wrapper.get_model()
        base_model.config.use_cache = False
        base_model.config.pretraining_tp = 1
        lora_config = LoraConfig(**self.parameter_map['qlora'])
        training_args = TrainingArguments(**self.parameter_map['training'])
        self.trainer = SFTTrainer(
            **{
                'model': self.model_wrapper.get_model(),
                'train_dataset': self.split_dataset.get_training(),
                'eval_dataset': self.split_dataset.get_eval(),
                'peft_config': lora_config,
                'dataset_text_field': "train_data",
                'max_seq_length': self.config.max_data_length,
                'tokenizer': self.model_wrapper.get_tokenizer(),
                'args': training_args,
                'packing': False
            }
        )
        self.trainer.train()
        self.trainer.model.save_pretrained(self.config.peft_folder)