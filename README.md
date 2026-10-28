# GitHub Repository Analyzer

A Streamlit app that indexes a GitHub repository and lets you ask questions about its code. It parses source files into code chunks, creates embeddings, stores them in Qdrant, and uses retrieval agents to answer questions with repository context.

## Features

- Clone a public GitHub repository or select a repository already cloned under `repos/`.
- Parse and chunk supported source files using tree-sitter.
- Generate code embeddings with the configured sentence-transformers model.
- Store and retrieve repository-scoped vectors in Qdrant.
- Ask questions through an orchestrator that can plan sub-questions, retrieve code, and verify answers.
- Adjust model, retrieval, vector database, and logging settings in `config.yaml`.

## Requirements

- Python 3.10 or newer
- A Groq API key for the configured LLM provider
- Qdrant, either as a local server or embedded storage

The embedding model is downloaded when first used. PyTorch and sentence-transformers can require substantial disk space and memory.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Groq API key:

   ```dotenv
   GROQ_API_KEY=your_groq_api_key
   ```

4. Configure Qdrant in `config.yaml`:

   - **Qdrant server (default):** leave `path: null` and run a local Qdrant instance at the configured URL. For example, with Docker:

     ```bash
     docker run -p 6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
     ```

   - **Embedded mode:** set `vector_db.path` to a local directory such as `qdrant_storage`. No Qdrant server is needed. Do not open the same embedded storage directory from multiple processes at once.

## Run the app

From the project root, start Streamlit:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. Enter a GitHub repository URL and select **Ingest & Index**. Once ingestion completes, enter a question about the active repository and select **Ask Agent**.

The app also supports switching to repositories already cloned under `repos/`. Clone and embedding operations require network access. GitHub authentication may be needed for private repositories.

## Configuration

`config.yaml` is the main runtime configuration. It currently defines:

- `llm`: provider, model, temperature, and retry count.
- `embedding`: embedding model and batch size.
- `vector_db`: Qdrant collection, vector dimension, server URL, or embedded path.
- `retrieval`: candidate and final result limits, per-file limits, and context budget.
- `orchestrator`: planner/verifier toggles, sub-question and revision limits, and route budgets.
- `logging`: log level, file rotation, and console output.

The configured vector size must match the embedding model's output dimension. If you change embedding models or dimensions, use a compatible collection; existing vectors may need to be reindexed.

## Project layout

```text
app.py                  Streamlit user interface and ingestion flow
config.yaml             Runtime settings
config_loader.py        Defaults and YAML configuration loader
agents/                 Orchestrator and retrieval agents
src/                    Repository cloning, parsing, embeddings, Qdrant, logging
tools/                  Retrieval tools used by agents
repos/                  Cloned repositories (generated/local data)
qdrant_storage/         Optional embedded Qdrant data
logs/                   Application logs
test.py                 Interactive command-line ingestion and Q&A pipeline
```

## Command-line pipeline

An interactive CLI pipeline is also available:

```bash
python test.py
```

It prompts for a GitHub repository URL and proceeds through cloning, indexing, and question answering.

## Troubleshooting

- **LLM authentication errors:** confirm `GROQ_API_KEY` is set in `.env` and restart Streamlit.
- **Cannot connect to Qdrant:** start the Qdrant server at `vector_db.url`, or set `vector_db.path` for embedded mode.
- **Vector dimension mismatch:** align `vector_db.vector_size` with the selected embedding model and rebuild/reindex the collection as needed.
- **Slow first startup:** the embedding model may need to download and load before ingestion can begin.

