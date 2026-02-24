#!/usr/bin/env python3
"""Quick test of vector DB re-ranking."""
import sys
sys.path.insert(0, '/Users/cedar/code/StudentService')

# Import rag directly to avoid agent dependency
from student_service.rag import WebsiteRAG

print("Initializing RAG...")
rag = WebsiteRAG()

test_queries = [
    'how much is the fee of Industry 4.0 MSc program',
    'Industry 4.0 MSc fee',
    'Industry 4.0 master program',
]

for query in test_queries:
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print('='*70)
    
    r = rag.query(query)
    s = r.get('sources', [])
    
    print(f"Top 3 sources:")
    for i, src in enumerate(s[:3]):
        url = src['source_url']
        score = src.get('score')
        is_master = '/master-programs/' in url
        is_industry = 'industry-4' in url.lower()
        
        tag = ""
        if is_master and is_industry:
            tag = "[✓✓ MASTER + INDUSTRY 4.0]"
        elif is_master:
            tag = "[✓ Master]"
        elif is_industry:
            tag = "[✓ Industry 4.0]"
        
        print(f"  {i+1}. {url}")
        print(f"     Score: {score} {tag}")

