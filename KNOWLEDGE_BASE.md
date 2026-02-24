# XU University Knowledge Base

This file is now a lightweight index to avoid duplicated documentation maintenance.

## Canonical docs

- Developer usage and architecture: [README.md](README.md)
- Deployment/operations workflow: [DEPLOYMENT.md](DEPLOYMENT.md)
- Persistent storage snapshot: [PERSISTENT_STORAGE_SUMMARY.md](PERSISTENT_STORAGE_SUMMARY.md)

## Current stack (summary)

- Retrieval backend: LangChain + ChromaDB
- Embeddings: local HashingVectorizer adapter (offline)
- Storage artifacts:
  - `student_service/.knowledge_base_index.pkl`
  - `student_service/.knowledge_base_index.chroma/`
  - `student_service/.knowledge_base_index.txt`

## Quick commands

```bash
python -m student_service.rebuild_kb
python -m student_service.ask "What bachelor programs are offered?"
python test/run_batch_queries.py --questions-file test/fixtures/questions.txt --output test/results/results_$(date +%Y%m%d_%H%M%S).jsonl
```

Core packages:
- `requests`
- `beautifulsoup4`
- `pypdf`
- `scikit-learn`
- `chromadb`
- `langchain-core`
- `langchain-chroma`
