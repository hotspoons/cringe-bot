import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#base_model_id = "mistralai/Mixtral-8x7B-v0.1"
base_model_id = "ch "
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mixtral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
)

from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "mixtral-toxic-23-yo-rich-finetune/checkpoint-650")

eval_prompt = f"""<s> [INST] Listen Rich, you are just a little too clingy. Can we just be friends? [/INST] """

model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=120)[0], skip_special_tokens=True))
