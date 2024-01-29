from flask import Flask, render_template
import waitress
from huggingface_hub import notebook_login

from aim_log_scraper import AIMSplitDataset, AimLogScraper
from config import Config
from merge_peft_adapters import MergePeftAdapters
from model_wrapper import ModelWrapper
from split_dataset import SplitDataset
from trainer import Trainer

"""
Cleaned up and generalized from 
https://www.datacamp.com/tutorial/mistral-7b-tutorial and 
https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model-0f39647b20fe
which seem to be copies of each other
And made to suit more use cases
"""

class Api():
    def __init__(self, config: Config, trainer: Trainer = None, model_wrapper: ModelWrapper = None, split_dataset: SplitDataset = None, merger: MergePeftAdapters = None) -> None:
        self.logger = config.logger
        self.config = config.config
        if self.config.port != 0:
            if trainer != None:
                self.trainer = trainer
            if model_wrapper != None:
                self.model_wrapper = model_wrapper
            if split_dataset != None:
                self.split_dataset = split_dataset
            if merger != None:
                self.merger = merger
            self.app.run(host="0.0.0.0", port=self.config.port)
            self.app = Flask(__name__)
            self.register_routes()
        else:
            self.logger.info("REST API is disabled, not starting")
        
    def register_routes(self):
            self.app.add_url_rule('/', 'index', self.index)
    def index(self):
         return render_template("index.html")

if __name__ == "__main__":
   config = Config()
   api = Api(config)
   actions = config.config.actions.split(',')
   for action in actions:
       if action.lower() == 'train':
           dataset = AIMSplitDataset(config)
           model_wrapper = ModelWrapper(config)
           aim_log_scraper = AimLogScraper(config)
           trainer = Trainer(model_wrapper, dataset, config)
           api.split_dataset = dataset
           api.model_wrapper = model_wrapper
           api.trainer = trainer
           trainer.train()
       elif action.lower() == 'merge':
           merger = MergePeftAdapters(config)
           api.merger = merger
           merger.merge()