# Persistent Storage Summary

This document is intentionally brief. Canonical operational details are maintained in:

- [README.md](README.md)
- [DEPLOYMENT.md](DEPLOYMENT.md)

## Snapshot

- Backend: LangChain + ChromaDB
- Embeddings: local HashingVectorizer adapter
- Rebuild mode: manual (`python -m student_service.rebuild_kb`)

## Storage artifacts

- `student_service/.knowledge_base_index.pkl`
- `student_service/.knowledge_base_index.chroma/`
- `student_service/.knowledge_base_index.txt`

## Maintainer checklist

1. After retrieval/ranking code changes, run rebuild.
2. Validate with CLI and at least one batch test file.
3. Keep deep details in README/DEPLOYMENT only.
