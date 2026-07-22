import os
import torch
#pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#pip install --no-deps xformers "triton>=3.0.0" bitsandbytes accelerate peft trl unsloth_zoo

from unsloth import FastLanguageModel
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Variáveis globais para podermos resetar o modelo de qualquer lugar
model = None
tokenizer = None
MODO_ATUAL = None

def carregar_modelo_no_modo(modo_precisao):
    global model, tokenizer, MODO_ATUAL

    # 1. Se já existir um modelo na memória, deleta e limpa a GPU
    if model is not None:
        print(f"Descarregando o modo {MODO_ATUAL} da memória de vídeo...")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache() # Esvazia o lixo acumulado na VRAM

    MODO_ATUAL = modo_precisao.upper()
    max_seq_length = 4096
    model_name = "unsloth/Qwen2.5-Coder-7B-Instruct"

    # 2. Configura a quantização dinamicamente baseado no comando
    if MODO_ATUAL == "Q4":
        load_in_4bit = True
        print("Carregando Qwen em Modo Q4 (Ideal para PCs Comuns)")
    elif MODO_ATUAL == "Q8":
        load_in_4bit = False # O Unsloth gerencia o carregamento de precisões maiores
        print("Carregando Qwen em Modo Q8 (Equilíbrio Técnico)")
    else:
        load_in_4bit = False
        print("Carregando Qwen em Modo NATIVA (FP16 - Força Máxima)")

    # 3. Carrega o novo modelo solicitado pelo usuário
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = torch.bfloat16,
        load_in_4bit = load_in_4bit,
    )

    # 4. Re-acopla as camadas de Fine-Tuning (LoRA) para garantir que ele continue treinável
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407
    )
    print(f"Concluído! O sistema agora está operando em modo: {MODO_ATUAL}\n")

# Inicializa o modelo por padrão no modo mais leve (Q4) para economizar recursos
carregar_modelo_no_modo("Q4")