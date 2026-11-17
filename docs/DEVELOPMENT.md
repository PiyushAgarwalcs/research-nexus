# Local Development, Testing, and Deployment Guide

This guide describes how to set up the development environment, run automated tests, build Docker containers, and contribute to **ResearchNexus**.

---

## 1. Prerequisites & Environment Setup

### 1.1 Requirements
- **Python**: 3.10 or 3.11
- **C/C++ Build Tools**: Required for `tiktoken` and `pymupdf` compilation on some platforms
- **Hardware**: CPU supported; NVIDIA GPU with CUDA 11.8+ recommended for accelerated cross-encoder reranking

### 1.2 Virtual Environment Installation

**Windows (PowerShell)**:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Linux / macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 1.3 Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your secrets:
```env
OPENAI_API_KEY=your_openai_api_key_here
DENSE_WEIGHT=0.7
SPARSE_WEIGHT=0.3
RRF_K=60
RETRIEVAL_DEPTH=60
RERANKER_CANDIDATES=20
FINAL_K=5
CONFIDENCE_THRESHOLD=0.75
DEFAULT_RERANKER_MODEL=finetuned
```

---

## 2. Running Automated Tests

ResearchNexus includes a comprehensive pytest suite covering all components:
- API routing & direct handler execution
- Document loader, metadata extraction & deduplication
- Chunking strategies
- Dense, BM25, and RRF retrieval logic
- Cross-encoder reranking, model loading & dynamic switching
- Evidence stance analysis & citation verification
- Evaluation metrics and judge execution

### Run all tests:
```bash
pytest -v
```

### Run specific test modules:
```bash
pytest tests/test_retriever.py -v
pytest tests/test_loader.py -v
pytest tests/test_generator.py -v
pytest tests/test_api.py -v
```

### Run with test coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

---

## 3. Running the Applications

### 3.1 FastAPI Server
Start the development server with hot-reload:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
- API Base: `http://localhost:8000`
- Interactive Swagger Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 3.2 Streamlit Research Intelligence Interface
Launch the multi-page dashboard:
```bash
streamlit run streamlit_app.py --server.port 8501
```
Navigate to `http://localhost:8501`.

### 3.3 Command-Line Interface (CLI)
Query the knowledge base directly from the terminal:
```bash
python main.py --query "What is the efficacy of mRNA vaccines against SARS-CoV-2 variants?"
```

---

## 4. Docker Deployment

The application includes multi-stage containerization with `Dockerfile` and `docker-compose.yml`.

### 4.1 Build and Run with Docker Compose
```bash
docker compose up --build
```
This starts:
- FastAPI service on `http://localhost:8000`
- Streamlit application on `http://localhost:8501`

### 4.2 Check Container Status & Logs
```bash
docker compose ps
docker compose logs -f api
docker compose logs -f streamlit
```

---

## 5. Coding Standards & Git Workflow

1. **Format & Lint**: Run `black .` and `flake8` prior to committing.
2. **Never Commit Large Binaries**: Weights (`*.safetensors`, `*.pt`, `*.bin`) are excluded via `.gitignore`.
3. **Never Commit Secrets**: Ensure `.env` is never staged.
4. **Resumable State**: Keep `PROJECT_PROGRESS.md`, `PROJECT_HANDOFF.md`, and `docs/IMPLEMENTATION_CHECKLIST.md` updated after every major feature milestone.
