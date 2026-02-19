# CHATBOT RAG - Inteligência Documental por Setores

Este é um sistema de **RAG (Retrieval-Augmented Generation)** de nível empresarial, projetado para permitir que usuários interajam com documentos PDF específicos de diferentes setores de forma isolada e segura.

O projeto utiliza **FastAPI**, **LangChain**, **ChromaDB** e a API do **Groq (Llama 3)** para fornecer respostas precisas com base no contexto documental fornecido.

## Funcionalidades

* **Busca Contextual (RAG):** Respostas geradas estritamente com base nos documentos carregados.
* **Organização por Setores:** Pastas isoladas que garantem que a IA não misture informações de departamentos diferentes.
* **Upload e Indexação em Tempo Real:** Suba um PDF e comece a perguntar imediatamente.
* **Gestão de Memória:** Visualização de quais documentos já foram processados pelo motor de IA.
* **🐳 Dockerizado:** Ambiente pronto para rodar com apenas um comando.

---

## Arquitetura do Sistema

O fluxo de dados segue o padrão moderno de IA generativa:
1. **Ingestão:** PDFs são lidos e divididos em pequenos pedaços (chunks).
2. **Embeddings:** O modelo `sentence-transformers` converte texto em vetores matemáticos.
3. **Vector Store:** O `ChromaDB` armazena esses vetores com metadados de setor.
4. **Retrieval:** Ao perguntar, o sistema busca os 4 trechos mais relevantes do setor selecionado.
5. **Augmentation:** O contexto é enviado ao modelo `Llama-3-8b` via Groq para gerar a resposta final.

## Como Rodar o Projeto

### Pré-requisitos
* **Docker** e **Docker Compose** instalados.
* Uma **API KEY do Groq** (Obtenha em [console.groq.com](https://console.groq.com)).

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/luizadorr/chatbot-rag.git](https://github.com/luizadorr/chatbot-rag.git)
   cd chatbot-rag


2. **Configure as Variáveis de Ambiente:**
Crie um arquivo `.env` na raiz do projeto:
```env
GROQ_API_KEY=sua_chave_aqui

```


3. **Suba os Containers:**
```bash
docker-compose up --build

```


4. **Acesse no Navegador:**
Abra [http://localhost:8000]()

---

## Tecnologias Utilizadas

* **Backend:** FastAPI (Python 3.10+)
* **IA Framework:** LangChain & LangChain-Chroma
* **LLM:** Groq (Llama 3 8B/70B)
* **Embeddings:** HuggingFace (All-MiniLM-L6-v2)
* **Vector Database:** ChromaDB
* **Frontend:** Bootstrap 5 

---

## Estrutura de Pastas

* `/app/main.py`: Lógica principal da API e motor RAG.
* `/app/static/`: Interface Web (HTML/CSS/JS).
* `/docs/`: Volumes mapeados onde os PDFs são salvos por setor.
* `/data/`: Persistência do banco de dados vetorial.

