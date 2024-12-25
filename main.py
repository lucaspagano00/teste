import os
import io
import json
import re
import pandas as pd
import numpy as np
import chardet
from typing import Optional

from fastapi import FastAPI, UploadFile, Body
from pydantic import BaseModel

# LangChain e OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentType, tool, initialize_agent
from langchain.memory import ConversationBufferMemory

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

############################################
# CONFIGURAÇÕES GERAIS
############################################

# Em produção, a API key deve vir de variáveis de ambiente ou serviço de secret
os.environ["OPENAI_API_KEY"] = "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"
GPT_MODEL_NAME = "gpt-4"

app = FastAPI(title="Agente de IA Autônomo com LangChain")

# Variáveis globais (em produção, prefira armazenar em um gerenciador de estado)
DATAFRAME_GLOBAL = None
TARGET_COLUMN = None
BEST_ACCURACY = 0.0

# Memória de conversa (LangChain)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

############################################
# 1. DEFINIR TOOLS (FERRAMENTAS)
############################################

@tool("mostrar_estatisticas", return_direct=True)
def mostrar_estatisticas_tool(params: dict) -> str:
    """
    Retorna estatísticas descritivas do DataFrame Global em formato JSON.
    """
    global DATAFRAME_GLOBAL
    if DATAFRAME_GLOBAL is None:
        return "Nenhum DataFrame carregado no momento."
    desc = DATAFRAME_GLOBAL.describe(include='all').fillna("").to_dict()
    return json.dumps(desc, indent=2)

@tool("limpar_e_engineering", return_direct=True)
def limpar_e_engineering_tool(instructions: str) -> str:
    """
    Aplica transformações no DataFrame global com base em instruções em JSON.
    Exemplo:
    {
      "drop_columns": ["colA", "colB"],
      "fill_na": {"columns": ["colC"], "strategy": "median"},
      "create_features": [
        {"name": "nova_col", "formula": "DATAFRAME_GLOBAL['colX'] * 2"}
      ]
    }
    """
    global DATAFRAME_GLOBAL
    if DATAFRAME_GLOBAL is None:
        return "Nenhum DataFrame carregado."

    # Tentar fazer parsing do JSON de instruções
    try:
        parsed_instructions = json.loads(instructions)
    except json.JSONDecodeError as e:
        return f"Erro ao interpretar JSON: {e}"

    # 1. Drop columns
    drop_cols = parsed_instructions.get("drop_columns", [])
    for col in drop_cols:
        if col in DATAFRAME_GLOBAL.columns:
            DATAFRAME_GLOBAL.drop(columns=[col], inplace=True)

    # 2. Fill NA
    fill_info = parsed_instructions.get("fill_na", {})
    if fill_info:
        cols_fill = fill_info.get("columns", [])
        strategy = fill_info.get("strategy", "mean")
        for c in cols_fill:
            if c in DATAFRAME_GLOBAL.columns:
                if strategy == "mean":
                    val = DATAFRAME_GLOBAL[c].mean()
                    DATAFRAME_GLOBAL[c].fillna(val, inplace=True)
                elif strategy == "median":
                    val = DATAFRAME_GLOBAL[c].median()
                    DATAFRAME_GLOBAL[c].fillna(val, inplace=True)
                elif strategy == "mode":
                    val = DATAFRAME_GLOBAL[c].mode().iloc[0]
                    DATAFRAME_GLOBAL[c].fillna(val, inplace=True)

    # 3. Create new features
    new_feats = parsed_instructions.get("create_features", [])
    for feat in new_feats:
        feat_name = feat.get("name")
        formula = feat.get("formula")
        if feat_name and formula:
            try:
                # Substituir 'df' por 'DATAFRAME_GLOBAL' se necessário
                code_line = formula.replace("df", "DATAFRAME_GLOBAL")
                exec(code_line, globals(), locals())
            except Exception as ex:
                return f"Erro ao criar feature '{feat_name}': {ex}"

    return "Transformações aplicadas com sucesso."

@tool("treinar_e_avaliar", return_direct=True)
def treinar_e_avaliar_tool(params: str) -> str:
    """
    Treina um RandomForest e retorna a acurácia em formato JSON.
    params (JSON) pode conter hiperparâmetros, e.g.:
    { "n_estimators": 100, "random_state": 42 }
    """
    global DATAFRAME_GLOBAL, TARGET_COLUMN
    if DATAFRAME_GLOBAL is None:
        return "Nenhum DataFrame carregado."
    if not TARGET_COLUMN or TARGET_COLUMN not in DATAFRAME_GLOBAL.columns:
        return f"Coluna alvo '{TARGET_COLUMN}' inexistente ou não definida."

    # Parse hiperparâmetros
    try:
        parsed_params = json.loads(params) if params else {}
    except json.JSONDecodeError:
        parsed_params = {}

    df = DATAFRAME_GLOBAL.copy()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Remover colunas não numéricas
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        return json.dumps({"accuracy": 0.0, "info": "Nenhuma feature numérica disponível."})

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators = parsed_params.get("n_estimators", 100)
    random_state = parsed_params.get("random_state", 42)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)

    return json.dumps({"accuracy": acc})

############################################
# 2. CRIAR O AGENTE LANGCHAIN
############################################

tools = [
    mostrar_estatisticas_tool,
    limpar_e_engineering_tool,
    treinar_e_avaliar_tool
]

llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4",
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

############################################
# 3. DEFINIR ROTAS FASTAPI
############################################

class ChatRequest(BaseModel):
    query: str

app = FastAPI()

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile, target_column: str):
    global DATAFRAME_GLOBAL, TARGET_COLUMN, BEST_ACCURACY
    content = await file.read()

    # Detectar a codificação
    detected_encoding = chardet.detect(content)
    encoding = detected_encoding.get("encoding", "utf-8")

    try:
        df = pd.read_csv(io.BytesIO(content), encoding=encoding)

        # Tentar converter colunas para tipos apropriados
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass

        DATAFRAME_GLOBAL = df
        TARGET_COLUMN = target_column
        BEST_ACCURACY = 0.0
        return {
            "message": "CSV carregado com sucesso",
            "columns": list(df.columns),
            "shape": df.shape
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/autonomous_loop/")
def autonomous_loop(max_iterations: int = 5, min_improvement: float = 0.01):
    """
    Executa o loop de melhorias de forma autônoma.
    - max_iterations: quantas vezes tentar
    - min_improvement: melhora mínima para continuar
    """
    global DATAFRAME_GLOBAL, TARGET_COLUMN, BEST_ACCURACY

    if DATAFRAME_GLOBAL is None:
        return {"error": "Nenhum DataFrame carregado."}
    if not TARGET_COLUMN or TARGET_COLUMN not in DATAFRAME_GLOBAL.columns:
        return {
            "error": f"Coluna alvo inválida ou não definida: {TARGET_COLUMN}",
            "available_columns": list(DATAFRAME_GLOBAL.columns)  # Adicionar colunas disponíveis
        }

    # Avaliação inicial
    initial_eval_str = agent.run("treinar_e_avaliar {}")
    try:
        init_eval_json = json.loads(initial_eval_str)
        BEST_ACCURACY = init_eval_json.get("accuracy", 0.0)
    except:
        BEST_ACCURACY = 0.0

    results = []
    results.append({
        "iteration": 0,
        "accuracy": BEST_ACCURACY,
        "description": "Acurácia inicial"
    })

    for i in range(1, max_iterations + 1):
        # Obter estatísticas
        stats = agent.run("mostrar_estatisticas _")

        prompt = (
            "Você é um engenheiro de dados autônomo. "
            "Analise essas estatísticas e aplique as ferramentas disponíveis (limpeza, feature engineering). "
            "Depois, chame a ferramenta 'treinar_e_avaliar' para me dar a acurácia final. "
            "Eu preciso apenas da acurácia no final em formato JSON ou texto que contenha 'accuracy'. "
            f"Estatísticas:\n{stats}\n\n"
        )

        response = agent.run(prompt)

        # Buscar a acurácia mencionada
        match = re.search(r"\"accuracy\"\s*:\s*([\d\.]+)", response)
        if match:
            new_accuracy = float(match.group(1))
        else:
            new_accuracy = 0.0

        improvement = new_accuracy - BEST_ACCURACY
        results.append({
            "iteration": i,
            "accuracy": new_accuracy,
            "description": f"Melhora de {improvement:.4f}"
        })

        if new_accuracy > BEST_ACCURACY:
            BEST_ACCURACY = new_accuracy
            if improvement < min_improvement:
                results[-1]["description"] += " -> Melhoria insuficiente, encerrando."
                break
        else:
            results[-1]["description"] += " -> Não melhorou, encerrando."
            break

    return {
        "best_accuracy": BEST_ACCURACY,
        "iterations_result": results
    }

@app.post("/chat/")
def chat_interaction(request: ChatRequest):
    """
    Permite ao usuário conversar com o agente em linguagem natural
    sobre o que foi feito, colunas removidas, acurácia etc.
    """
    query = request.query
    response = agent.run(query)
    return {"response": response}
