import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain e IA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# Carregar .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("ERRO: Variável GROQ_API_KEY não encontrada no arquivo .env")

app = FastAPI(title="RAGPY API")

# Configuração de Caminhos (Padrão Docker)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "../docs")
DATA_DIR = os.path.join(BASE_DIR, "../data")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

client = Groq(api_key=GROQ_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = None

def carregar_ou_criar_banco():
    """Carrega o banco existente ou cria um novo a partir dos PDFs"""
    global vector_db
    
    # Se já existir dados na pasta /data, carregar de lá
    if os.path.exists(os.path.join(DATA_DIR, "chroma.sqlite3")):
        print("--- Carregando banco de dados persistente ---")
        vector_db = Chroma(persist_directory=DATA_DIR, embedding_function=embeddings)
    else:
        print("--- Criando novo banco de dados a partir dos PDFs ---")
        loader = DirectoryLoader(DOCS_DIR, glob="./*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        
        if docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            vector_db = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings, 
                persist_directory=DATA_DIR
            )
        else:
            print("Aviso: Nenhum PDF encontrado em /docs.")

carregar_ou_criar_banco()

# --- MODELOS DE DADOS ---
class QueryData(BaseModel):
    prompt: str
    setor: str = "geral"

# OTIMIZE O PROMPT DA FORMA COMO DESEJAR, MAS GARANTA QUE A IA ENTENDA QUE O TEXTO EXTRAÍDO DOS PDFS É O 
# CONTEXTO PARA RESPONDER AS PERGUNTAS. SEJA CLARO QUE A RESPOSTA DEVE SER BASEADA APENAS NESSE CONTEXTO.
system_message = "Você é um assistente inteligente para otimizar processos em um sistema interno que responde perguntas baseado em documentos fornecidos. Seja conciso e preciso."

# --- ENDPOINTS ---
@app.get("/arquivos")
async def listar_arquivos(setor: str = "geral"):
    caminho_setor = DOCS_DIR if setor == "geral" else os.path.join(DOCS_DIR, setor)
    
    if not os.path.exists(caminho_setor):
        return {"arquivos": []}

    try:
        if setor == "geral":
            arquivos = []
            for root, dirs, files in os.walk(DOCS_DIR):
                for file in files:
                    if file.endswith(".pdf"):
                        arquivos.append(file)
            return {"arquivos": list(set(arquivos))}
        else:
            arquivos = [f for f in os.listdir(caminho_setor) if f.endswith(".pdf")]
            return {"arquivos": arquivos}
    except Exception:
        return {"arquivos": []} 

@app.post("/upload")
async def upload_pdf(setor: str, file: UploadFile = File(...)):
    setor_dir = os.path.join(DOCS_DIR, setor)
    os.makedirs(setor_dir, exist_ok=True)
    
    file_path = os.path.join(setor_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        carregar_ou_criar_banco_por_setor()
        return {"ok": True, "message": f"Arquivo {file.filename} salvo em {setor}!"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def carregar_ou_criar_banco_por_setor():
    global vector_db
    loader = DirectoryLoader(DOCS_DIR, glob="./**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    if docs:
        for doc in docs:
            path_parts = doc.metadata['source'].split(os.sep)
            doc.metadata['setor'] = path_parts[-2] if len(path_parts) > 2 else "geral"

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=DATA_DIR
        )
        return True
    return False

@app.post("/perguntar")
async def perguntar(data: QueryData):
    if vector_db is None:
        return {"ok": False, "error": "Banco de dados não inicializado."}

    search_filter = {"setor": data.setor} if data.setor != "geral" else None
    
    docs = vector_db.similarity_search(data.prompt, k=4, filter=search_filter)
    
    # -- DEBUG OPCIONAL --
    #print(f"DEBUG: Buscando por '{data.prompt}' no setor '{data.setor}'")
    #print(f"DEBUG: Blocos de texto encontrados: {len(docs)}")

    if not docs:
        return {"ok": True, "output": "Não encontrei informações específicas nos documentos deste setor."}

    contexto = "\n\n".join([doc.page_content for doc in docs])

    system_message = f"""
    Você é um assistente técnico. Abaixo está o conteúdo extraído dos documentos PDF:
    {contexto}
    
    Responde apenas com base no conteúdo acima. Se não souber, diz que a informação não consta nos documentos.
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": data.prompt}
            ]
        )
        return {"ok": True, "output": completion.choices[0].message.content}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/recarregar")
async def recarregar():
    try:
        carregar_ou_criar_banco()
        return {"ok": True, "mensagem": "Banco de dados atualizado!"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# --- SERVIR O FRONT-END ---
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)