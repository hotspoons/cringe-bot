import os
import re
import sys
import json
import pandas
from html2text import HTML2Text
from datasets import Dataset
from jinja2 import Template

from config import Config
from split_dataset import SplitDataset



class AIMSplitDataset(SplitDataset):

    def __init__(self, config: Config) -> None:
        self.aim_log_scraper = AimLogScraper(config)
        self.config = config.config
        self.max_data_length = self.config.max_data_length
        self.prompt = self.config.instruct_prompt
        self.prompt_padding = len(self.prompt)
        self.train_dataset = []
        self.eval_dataset = []
        self.build_dataset()

    def split_convo(self, conversation):
        convos = []
        asst = conversation['assistant']
        user = conversation['user']
        assistant_one, assistant_two = asst[:len(asst)//2], asst[len(asst)//2:]
        user_one, user_two = asst[:len(user)//2], asst[len(user)//2:]
        convo_one = {
            'assistant': assistant_one,
            'user': user_one, 
            'id': conversation['id'] + '_0'
        }
        convo_two = {
            'assistant': assistant_two,
            'user': user_two, 
            'id': conversation['id'] + '_1'
        }
        if len(assistant_one + user_one) + self.prompt_padding  > self.max_data_length:
            convos.extend(self.split_convo(convo_one))
        else:
            convos.append(convo_one)
        if len(assistant_two + user_two) + self.prompt_padding  > self.max_data_length:
            convos.extend(self.split_convo(convo_two))
        else:
            convos.append(convo_two)
        return convos

    def build_dataset(self):
        output = self.aim_log_scraper.convert_htmls()
        conversations = self.aim_log_scraper.to_convo_format(output)
        i = 0
        train_dataset_raw = []
        eval_dataset_raw = []
        for conversation in conversations:
            if len(conversation['assistant'] + conversation['user']) + self.prompt_padding > self.max_data_length:
                train_dataset_raw.extend(self.split_convo(conversation))
            else:
                if i % self.config.eval_fraction == 0:
                    eval_dataset_raw.append(conversation)
                else:
                    train_dataset_raw.append(conversation)
            i+=1
        for item in train_dataset_raw:
            item['train_data'] = self.format_training_text(item)
        for item in eval_dataset_raw:
            item['train_data'] = self.format_training_text(item)
        self.eval_dataset = Dataset.from_pandas(pandas.DataFrame(data=eval_dataset_raw))
        self.train_dataset = Dataset.from_pandas(pandas.DataFrame(data=train_dataset_raw))
       

    def format_training_text(self, convo):
        prompt = self.config.instruct_prompt + ' ' if self.config.instruct_prompt != '' else ''
        template = Template(self.config.training_template)
        return template.render({
            'prompt': prompt,
            'user': convo['user'],
            'assistant': convo['assistant']
        })
    
    def get_training(self) -> Dataset:
        return self.train_dataset
    
    def get_eval(self) -> Dataset:
        return self.eval_dataset

USER='user'
ASSISTANT='assistant'

class AimLogScraper():

    def __init__(self, config: Config = None):
        self.html_converter = HTML2Text()
        self.html_converter.unicode_snob = 1
        self.html_converter.ignore_emphasis = 1
        self.html_converter.ignore_images = 1
        self.html_converter.ignore_links = 1
        self.html_converter.body_width = 0
        if config != None:
            self.config = config.config

    def convert_htmls(self, path = None, screennames = None):
        output = ""
        if path == None:
            path = self.config.chats_location
        if screennames == None:
            screennames = self.config.screen_names
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".htm") or file.endswith(".html"):
                    with open(os.path.join(root, file), "r") as f:
                        output += self.filter(self.html_converter.handle(f.read()), screennames)
        return output

    def to_convo_format(self, input):
        turns = 0
        id = 0
        last_role = ""
        list_of_convos = []
        convo_struct = {
            'user': '',
            'assistant': '',
            'id': str(id)
        }
        list_of_convos.append(convo_struct)
        for line in input.split("\n"):
            if line.startswith('assistant :'):
                role = ASSISTANT
            elif line.startswith('user :'):
                role = USER
            else:
                role = USER if last_role == "" else last_role
                
            if last_role == "": #init
                last_role = role
            if turns == 1 and last_role != role:
                id += 1
                convo_struct = {
                    'user': '',
                    'assistant': '',
                    'id': str(id)
                }
                list_of_convos.append(convo_struct)
                turns = 0
            elif last_role != role:
                turns += 1
            if convo_struct[role] != '':
                convo_struct[role] += '\n'
            convo_struct[role] += line.replace(f"{role} :", '').strip()
            last_role = role
        return list_of_convos

    def filter(self, text, screennames):
        lines = text.splitlines()
        screennames = screennames.split(",")
        filtered = []
        for line in lines:
            if "is idle at " in line or "is no longer idle at " in line or \
                " returned at " in line or "is away at " in line or \
                "Session concluded at " in line or \
                "Auto response from " in line or "* * *" in line or \
                "wants to directly connect " in line or \
                "is now directly connected " in line or \
                "Your screen name " in line or \
                "direct connection is closed " in line or \
                " signed off at " in line or \
                " signed on at " in line :
                continue
            line = re.sub(r'\((\d{1,2}:\d{2}:\d{2}\s*(AM|PM))\)', '', line)
            for screenname in screennames:
                line = line.replace(f"{screenname} :", "assistant___")
            line = re.sub(r"^([^:]*):", 'user :', line, 0, re.MULTILINE)
            line = line.replace("assistant___", "assistant :")
            if line.isspace() == True or line == "":
                continue
            filtered.append(line)
        return "\n".join(filtered)

# arg 1 is the path to where you wish to scan for AIM chat logs,
# arg 2 is a comma separated list of AIM screennames to use as the assistant role
if __name__ == "__main__":
   aim_log_scraper = AimLogScraper()
   if len(sys.argv) < 3:
       print("Usage: python aim_log_scraper /path/to/aim/chats MyFirstScreenname,MySecondScreenname")
       exit(0)
   output = aim_log_scraper.convert_htmls(sys.argv[1], sys.argv[2])
   convos = aim_log_scraper.to_convo_format(output)
   for convo in convos:
       print(json.dumps(convo))
