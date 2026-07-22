import os
import gc
import torch
from unsloth import FastLanguageModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = None
tokenizer = None
MODO_ATUAL = None


def carregar_modelo_no_modo(modo_precisao="Q4"):

    global model, tokenizer, MODO_ATUAL

    MODO_ATUAL = modo_precisao.upper()

    if MODO_ATUAL == "Q4":
        load_in_4bit = True

    else:
        load_in_4bit = False


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-Coder-7B-Instruct",
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
    )


    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )


    return model, tokenizer


model, tokenizer = carregar_modelo_no_modo("Q4")
