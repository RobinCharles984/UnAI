import os
import gc
import torch
from unsloth import FastLanguageModel

# ============================================================
# Configuração CUDA
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # coloque 0 se você possui apenas uma GPU

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# Variáveis globais
# ============================================================

model = None
tokenizer = None
MODO_ATUAL = None

MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LENGTH = 4096

# Detecta automaticamente BF16
SUPPORTS_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

DTYPE = torch.bfloat16 if SUPPORTS_BF16 else torch.float16

# ============================================================
# Carregamento
# ============================================================

def carregar_modelo_no_modo(modo_precisao="Q4"):
    global model, tokenizer, MODO_ATUAL

    if model is not None:
        print(f"Descarregando modo {MODO_ATUAL}...")

        del model
        del tokenizer

        gc.collect()
        torch.cuda.empty_cache()

    MODO_ATUAL = modo_precisao.upper()

    if MODO_ATUAL == "Q4":

        load_in_4bit = True
        print("Modo Q4")

    elif MODO_ATUAL == "Q8":

        load_in_4bit = False
        print("Modo Q8")

    else:

        load_in_4bit = False
        print("Modo FP16/BF16")

    model, tokenizer = FastLanguageModel.from_pretrained(

        model_name=MODEL_NAME,

        max_seq_length=MAX_SEQ_LENGTH,

        dtype=DTYPE,

        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(

        model,

        r=32,

        lora_alpha=32,

        lora_dropout=0,

        bias="none",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],

        use_gradient_checkpointing="unsloth",

        random_state=3407,

        use_rslora=False,

        loftq_config=None,
    )

    print()

    print("=" * 60)

    print("GPU :", torch.cuda.get_device_name(0))

    print("Precisão :", DTYPE)

    print("Modo :", MODO_ATUAL)

    print("=" * 60)

    return model, tokenizer


carregar_modelo_no_modo("Q4")
