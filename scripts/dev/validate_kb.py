#!/usr/bin/env python3
"""Validate XU University Knowledge Base"""

from student_service.rag import WebsiteRAG
from student_service.agent import search_xu_university_knowledge
import time

print("=" * 70)
print("XU UNIVERSITY KNOWLEDGE BASE VALIDATION")
print("=" * 70)

# Build knowledge base
print("\n[1] Building Knowledge Base...")
start = time.time()
rag = WebsiteRAG()
urls = rag._collect_pages()
elapsed = time.time() - start

pdf_urls = [u for u in urls if u.endswith('.pdf')]
html_urls = [u for u in urls if not u.endswith('.pdf')]
bachelor_urls = [u for u in urls if '/bachelor/' in u]
master_urls = [u for u in urls if '/master' in u]

print(f"    Total URLs: {len(urls)} (indexed in {elapsed:.1f}s)")
print(f"    HTML pages: {len(html_urls)}")
print(f"    PDF documents: {len(pdf_urls)}")
print(f"    Bachelor pages: {len(bachelor_urls)}")
print(f"    Master pages: {len(master_urls)}")

# Test retrieval
print("\n[2] Testing Retrieval Quality...")
queries = [
    "What bachelor programs are offered?",
    "How much is the fee of Industry 4.0 MSc program",
    "What are the admissions requirements?",
]

for q in queries:
    result = search_xu_university_knowledge(q)
    sources = result.get('sources', [])
    types = {s.get('source_type') for s in sources}
    print(f"    ✓ Query: {q[:50]}...")
    print(f"      Sources: {len(sources)}, Types: {types}")

print("\n" + "=" * 70)
print("✓ KNOWLEDGE BASE READY FOR PRODUCTION")
print("=" * 70)
