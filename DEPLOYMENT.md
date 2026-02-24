# Knowledge Base Deployment Guide

## Overview

The XU University knowledge base uses **persistent storage** with **manual rebuild control**.

Current retrieval stack:
- **LangChain + ChromaDB** vector retrieval
- Local `HashingVectorizer` embedding adapter (offline)
- Manual rebuild and persistent index artifacts on disk

## Deployment Workflow

### Initial Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build knowledge base index (one-time)
python -m student_service.rebuild_kb

# 3. Verify index artifacts
ls -lh student_service/.knowledge_base_index.pkl
ls -ld student_service/.knowledge_base_index.chroma

# 4. Test query
python -m student_service.ask "What bachelor programs are offered?"
```

### Production Deployment

```bash
# In your CI/CD pipeline or deployment script
python -m student_service.rebuild_kb

# Start application
adk run student_service
```

## Maintenance

### When to Rebuild

Rebuild when:
1. Website content changes (programs, fees, admissions docs)
2. New deployment environment is provisioned
3. Retrieval/ranking logic changes in `rag.py`
4. Index artifacts are missing/corrupted

### How to Rebuild

```bash
python -m student_service.rebuild_kb
```

### Fast Cache Rebuild (Developer)

If network crawling is unstable and a text export already exists:

```python
from student_service.rag import WebsiteRAG

rag = WebsiteRAG()
stats = rag.rebuild_index_from_cache(verbose=True)
print(stats)
```

## Monitoring

```python
from student_service.rag import WebsiteRAG
import time

rag = WebsiteRAG()

if rag.index_path.exists():
    size_mb = rag.index_path.stat().st_size / (1024 * 1024)
    age_hours = (time.time() - rag.index_path.stat().st_mtime) / 3600
    print(f"Index file: {rag.index_path}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Age: {age_hours:.1f} hours")
else:
    print("Index metadata missing - run rebuild command")

print(f"Vector DB dir exists: {rag.vector_db_path.exists()}")
```

## File Structure

```text
student_service/
├── rag.py                          # RAG implementation
├── rebuild_kb.py                   # Rebuild script
├── agent.py                        # ADK agent integration
├── ask.py                          # CLI interface
├── .knowledge_base_index.pkl       # Persistent metadata/chunks (gitignored)
├── .knowledge_base_index.chroma/   # Persistent vector DB (gitignored)
├── .knowledge_base_index.txt       # Collected text export (gitignored)
└── .env                            # Config (if needed)
```

## Configuration

### Custom index location

```python
from student_service.rag import WebsiteRAG

rag = WebsiteRAG(index_path="/opt/data/xu_index.pkl")
```

### Network timeout tuning

```python
from student_service.rag import WebsiteRAG

rag = WebsiteRAG(timeout_seconds=30.0)
rag.rebuild_index()
```

## Performance (Typical)

| Operation | Typical | Notes |
|---|---|---|
| Full rebuild | ~50-90s | network/site dependent |
| Cache rebuild | ~1-5s | from `.knowledge_base_index.txt` |
| First load | usually sub-second | metadata + vector DB init |
| Query | usually sub-second | depends on candidate count |

## Troubleshooting

### Error: Knowledge base index not found

Cause: index artifacts missing.

Fix:

```bash
python -m student_service.rebuild_kb
```

### Build hangs or times out

1. Check network access to `xu-university.com`
2. Increase `timeout_seconds` in `WebsiteRAG`
3. Use `rebuild_index_from_cache()` when text export is available

### Corrupted index artifacts

```bash
rm -f student_service/.knowledge_base_index.pkl
rm -rf student_service/.knowledge_base_index.chroma
rm -f student_service/.knowledge_base_index.txt
python -m student_service.rebuild_kb
```

## CI/CD Example

```yaml
- name: Build Knowledge Base
  run: |
    python -m student_service.rebuild_kb
    ls -lh student_service/.knowledge_base_index.pkl
    ls -ld student_service/.knowledge_base_index.chroma
```
