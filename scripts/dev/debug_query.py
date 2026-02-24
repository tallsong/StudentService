#!/usr/bin/env python3
"""Debug test - see raw Chroma results before ranking."""
import sys
sys.path.insert(0, '/Users/cedar/code/StudentService')

from student_service.rag import WebsiteRAG

rag = WebsiteRAG()

# Monkey-patch to see raw results
original_query = rag.query

def debug_query(question, top_k=5):
    print(f"\n📋 Querying: {question}")
    
    rag._ensure_index()
    collection = rag._collection
    
    result = collection.query(
        query_embeddings=rag._embed_texts([question]),
        n_results=10,
        include=["documents", "metadatas", "distances"],
    )
    
    metadatas_raw = result.get("metadatas", [[]])
    distances_raw = result.get("distances", [[]])
    
    rows_meta = metadatas_raw[0] if metadatas_raw else []
    rows_dist = distances_raw[0] if distances_raw else []
    
    print(f"\nRaw Chroma results (top 10):")
    for i, (metadata, dist) in enumerate(zip(rows_meta, rows_dist)):
        url = str(metadata.get("source_url", ""))
        score = 1.0 / (1.0 + float(dist))
        print(f"  {i+1}. {url}")
        print(f"     Distance: {dist}, Score: {score:.4f}")
    
    # Call original
    return original_query(question, top_k)

rag.query = debug_query

r = rag.query("how much is the fee of Industry 4.0 MSc program")
print("\n\nFinal ranked results:")
for i, src in enumerate(r.get('sources', [])[:3]):
    print(f"  {i+1}. {src['source_url']}")
