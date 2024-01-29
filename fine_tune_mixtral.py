from datasets import load_dataset
import datasets
from accelerate import FullyShardedDataParallelPlugin, Accelerator
import pandas as pd
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import wandb, os
import torch
import transformers
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from aim_log_scraper import convert_htmls, to_convo_format

max_length = 1024

train_dataset_raw = []
eval_dataset_raw = []

prompt = ""

prompt_padding = len(prompt)

output = convert_htmls("/media/import/media/backups/old_drives/80GBMaxtor_files_from_2006/want to keep/AIM Logs", "elesford")
conversations = to_convo_format(output)

def split_convo(conversation):
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
    if len(assistant_one + user_one) + prompt_padding  > max_length:
        convos.extend(split_convo(convo_one))
    else:
        convos.append(convo_one)
    if len(assistant_two + user_two) + prompt_padding  > max_length:
        convos.extend(split_convo(convo_two))
    else:
        convos.append(convo_two)
    return convos

i = 0
for conversation in conversations:
    if len(conversation['assistant'] + conversation['user']) + prompt_padding > max_length:
        train_dataset_raw.extend(split_convo(conversation))
    else:
        if i % 9 == 0:
            eval_dataset_raw.append(conversation)
        else:
            train_dataset_raw.append(conversation)
    i+=1

eval_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=eval_dataset_raw))
train_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=train_dataset_raw))

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    sync_module_states=True
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin, device_placement=True)


wandb.login()

wandb_project = "toxic-23-yo-rich-finetune-mixtral-8x7b"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project


print(train_dataset)
print(eval_dataset)


base_model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_eos_token=True,
    add_bos_token=True, 
)

def tokenize(prompt):
    result = tokenizer(prompt)
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = f" [INST] {prompt} [/INST]\nUser: {data_point['user']}\nAssistant: {data_point['assistant']} "
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

untokenized_text = tokenizer.decode(tokenized_train_dataset[1]['input_ids']) 
print(untokenized_text)

 

# redefine the tokenize function and tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,  
    add_bos_token=True,  
)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

print(tokenized_train_dataset[4]['input_ids'])
untokenized_text = tokenizer.decode(tokenized_train_dataset[1]['input_ids']) 
print(untokenized_text)


eval_prompt = f" [INST] {prompt} [/INST] \nUser: Hey Rich! What's going on tonight?\nLet's go get some drinks at the Monkey Barrel!!\nAssistant: "

#model = accelerator.prepare_model(model, device_placement=True)
# Re-init the tokenizer so it doesn't add padding or eos token
eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
)

device = "cuda"
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=128)[0], skip_special_tokens=True))



model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



print(model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "w1",
        "w2",
        "w3",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Apply the accelerator. You can comment this out to remove the accelerator.
#model = accelerator.prepare_model(model)

print(model)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")




project = "toxic-23-yo-rich-finetune"
base_model_name = "mixtral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        #resume_from_checkpoint="mixtral-toxic-23-yo-rich-finetune/checkpoint-150",
        gradient_accumulation_steps=4,
        max_steps=700,
        learning_rate=2.5e-5, 
        logging_steps=25,
        fp16=True, 
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=49,               # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train(
    #resume_from_checkpoint="mixtral-toxic-23-yo-rich-finetune/checkpoint-150"
    )
