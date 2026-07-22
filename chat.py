from model import model, tokenizer, MODO_ATUAL, carregar_modelo_no_modo
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import json
from datetime import datetime

# Aplica o padrão de mensagens oficial da Qwen
tokenizer = get_chat_template(tokenizer, chat_template = "qwen-2.5")

# Memória temporária que simula o banco de dados local armazenando as correções do usuário
historico_dataset_usuario = []

def salvar_historico(contexto, prompt_usuario, resposta_ia):
    
    # ============================
    # SALVAR TXT (LEITURA HUMANA)
    # ============================

    with open("historico_unity.txt", "a", encoding="utf-8") as f:

        f.write("\n" + "=" * 80 + "\n")
        f.write("NOVA INTERAÇÃO\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"DATA: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")

        f.write("CONTEXTO:\n")
        f.write(contexto)
        f.write("\n\n")

        f.write("USUÁRIO:\n")
        f.write(prompt_usuario)
        f.write("\n\n")

        f.write("ASSISTENTE:\n")
        f.write(resposta_ia)
        f.write("\n")


    # ============================
    # SALVAR JSONL (DATASET)
    # ============================

    conversa_json = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é um assistente especialista em Unity 3D "
                    "usando C#."
                    f" Contexto do projeto: {contexto}"
                )
            },
            {
                "role": "user",
                "content": prompt_usuario
            },
            {
                "role": "assistant",
                "content": resposta_ia
            }
        ],

        "metadata": {
            "data": datetime.now().isoformat(),
            "origem": "assistente_unity_local"
        }
    }


    with open("historico_unity.jsonl", "a", encoding="utf-8") as f:

        f.write(
            json.dumps(
                conversa_json,
                ensure_ascii=False
            )
            + "\n"
        )


    print("Histórico salvo em TXT e JSONL!")

def interagir_e_coletar(contexto_unity, prompt_usuario, resposta_correta_final):
    """
    Simula o agente Unity conversando e salvando o feedback.
    resposta_correta_final: O código C# definitivo que o usuário usou após a interação.
    """

    # Mensagem estruturada em formato Conversacional (ShareGPT/HuggingFace)
    conversa = [
        {"role": "system", "content": f"Você é um assistente especialista em Unity 3D. Contexto atual do projeto: {contexto_unity}"},
        {"role": "user", "content": prompt_usuario},
        {"role": "assistant", "content": resposta_correta_final} # Treina focado no acerto final
    ]

    # Converte para o texto formatado com tokens especiais (como <|im_start|>)
    texto_formatado = tokenizer.apply_chat_template(conversa, tokenize=False, add_generation_prompt=False)

    # Guarda no dataset local do computador do usuário
    historico_dataset_usuario.append({"text": texto_formatado})
    print("Interação capturada e adicionada ao Dataset Local do Usuário!")

# --- SIMULAÇÃO DE USO DO PLUGIN NA UNITY ---
# Exemplo 1: Usuário pedindo para mover o player
interagir_e_coletar(
    contexto_unity = "Cena 3D, Player com Rigidbody",
    prompt_usuario = "Como faço para o meu personagem pular usando C#?",
    resposta_correta_final = "void Update() { if(Input.GetButtonDown(\"Jump\")) { rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse); } }"
)

# Lista na memória simulando o banco de dados local do usuário
#historico_dataset_usuario = []

def processar_prompt_usuario(prompt, contexto_unity=""):
    """
    Processa prompts do usuário: intercepta comandos do sistema (/modo)
    ou chama o Qwen 2.5 para GERAR respostas reais de C# na Unity.
    """
    global model, tokenizer

    # 1. Garante que o template de conversa da Qwen2.5 esteja ativo
    tokenizer = get_chat_template(tokenizer, chat_template = "qwen-2.5")

    texto = prompt.strip()

    # 2. INTERCEPTOR DE COMANDOS: Troca de quantização dinamicamente
    if texto.startswith("/modo"):
        partes = texto.split()
        if len(partes) > 1 and partes[1].upper() in ["Q4", "Q8", "NATIVA"]:
            novo_modo = partes[1].upper()
            print(f"\nComando detectado! Alterando infraestrutura para {novo_modo}...")
            carregar_modelo_no_modo(novo_modo)
            return f"Sistema alterado com sucesso para o modo {novo_modo}."
        else:
            return "Erro: Use '/modo Q4', '/modo Q8' ou '/modo NATIVA'."

    # 3. GERAÇÃO REAL DE CÓDIGO VIA LLM
    print(f"\nIA gerando resposta no modo {MODO_ATUAL}...")

    # Prepara o modelo para inferência rápida (otimização do Unsloth)
    FastLanguageModel.for_inference(model)

    # Monta a estrutura da conversa no formato do Qwen
    conversa = [
        {"role": "system", "content": f"Você é um assistente especialista em desenvolvimento de jogos na Unity 3D usando C#. Contexto do projeto: {contexto_unity}"},
        {"role": "user", "content": texto}
    ]

    # Converte o texto em Tokens e gera a máscara de atenção
    inputs = tokenizer.apply_chat_template(
        conversa,
        tokenize = True,
        add_generation_prompt = True,
        return_dict = True,
        return_tensors = "pt"
    ).to("cuda")

    # Garante que o pad_token esteja definido explicitamente
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Otimiza os parâmetros do modelo para evitar o aviso de max_length
    model.config.max_length = None

    # Gera a resposta passando explicitamente a máscara e os tokens
    outputs = model.generate(
        **inputs,
        max_new_tokens = 1024,
        pad_token_id = tokenizer.pad_token_id,
        use_cache = True,
        temperature = 0.2
    )

    tamanho_prompt_entrada = inputs["input_ids"].shape[1]

    # Decodifica apenas os novos tokens gerados pela IA
    resposta_codigo_real = tokenizer.decode(outputs[0][tamanho_prompt_entrada:], skip_special_tokens=True)

    # Imprime a resposta real gerada pela IA na tela
    print("\nResposta da IA:")
    salvar_historico(
        contexto_unity,
        texto,
        resposta_codigo_real
    )


    # 4. SALVA NO DATASET LOCAL PARA O PRÓXIMO TREINAMENTO
    conversa_completa = [
        {"role": "system", "content": f"Especialista Unity. Contexto: {contexto_unity}"},
        {"role": "user", "content": texto},
        {"role": "assistant", "content": resposta_codigo_real}
    ]

    texto_formatado = tokenizer.apply_chat_template(conversa_completa, tokenize=False, add_generation_prompt=False)
    historico_dataset_usuario.append({"text": texto_formatado})
    print("\nInteração real capturada e salva no dataset local com sucesso!")

    return resposta_codigo_real

# --- TESTANDO O CHAT REAL ---

# 1. Teste de pergunta real para a LLM (Em modo Q4)
processar_prompt_usuario("""Meu personagem Unity usa Rigidbody. Crie um sistema de pulo com:
- detecção de chão usando Raycast;
- variável de força do pulo no Inspector;
- coyote time;
- bloqueio de pulo duplo;
- código organizado seguindo boas práticas.
""")

processar_prompt_usuario("Crie um sistema de câmera em terceira pessoa na Unity usando C#. A câmera deve seguir o jogador suavemente, permitir rotação com mouse, evitar atravessar paredes usando SphereCast e ter ajuste de distância pelo Inspector.")

processar_prompt_usuario("""Analise este código Unity C# e explique por que o personagem está tremendo quando anda. Depois corrija o código.

public class Player : MonoBehaviour
{
    public float speed = 5;

    void Update()
    {
        float x = Input.GetAxis("Horizontal");
        float z = Input.GetAxis("Vertical");

        transform.position += new Vector3(x,0,z) * speed * Time.deltaTime;
    }
}
""")

processar_prompt_usuario("""Preciso criar um sistema de inventário na Unity usando C#. Quero:
- itens empilháveis;
- ScriptableObjects para criar novos itens;
- salvar e carregar inventário;
- eventos quando um item é adicionado ou removido.

Crie uma arquitetura escalável.
""")

# 2. Mudando a precisão pelo prompt
processar_prompt_usuario("/modo Q8")

# 3. Teste de pergunta real agora rodando em Q8
processar_prompt_usuario("""Meu jogo Unity está rodando a 30 FPS com 500 inimigos na cena. Explique como otimizar usando:
- Object Pooling;
- Jobs System;
- Burst Compiler;
- redução de chamadas no Update.

Mostre exemplos em C#.

""")

processar_prompt_usuario("""Crie um shader para Unity URP usando Shader Graph ou HLSL que faça um material de água com:
- transparência;
- movimento de ondas;
- reflexo simples;
- controle de intensidade pelo Inspector.
""")

processar_prompt_usuario("""Estou criando um multiplayer cooperativo na Unity. Explique uma arquitetura usando Netcode for GameObjects. Crie um exemplo de sincronização de movimento de jogadores e explique como evitar problemas de latência.""")

processar_prompt_usuario("""Tenho um projeto Unity 3D de sobrevivência. O jogador coleta recursos, constrói bases e enfrenta inimigos. Quero uma arquitetura completa de código usando:
- ScriptableObjects;
- eventos;
- sistemas desacoplados;
- save/load;
- gerenciamento de estados.

Explique a estrutura de pastas e gere os principais scripts.
""")
