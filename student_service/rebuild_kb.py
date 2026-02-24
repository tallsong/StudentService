#!/usr/bin/env python3
"""
Rebuild Knowledge Base Index

This script manually rebuilds the XU University knowledge base index.
Run this when:
- Website content has been updated
- Indexing logic has changed
- Initial setup of the project

Usage:
    python -m student_service.rebuild_kb
"""

import sys
from student_service.rag import WebsiteRAG


def main() -> int:
    """Rebuild the knowledge base index."""
    print("=" * 70)
    print("XU UNIVERSITY KNOWLEDGE BASE - MANUAL REBUILD")
    print("=" * 70)
    print()
    
    try:
        # Create RAG instance and rebuild
        rag = WebsiteRAG()
        stats = rag.rebuild_index(verbose=True)
        
        print()
        print("=" * 70)
        print("✓ KNOWLEDGE BASE REBUILD COMPLETE")
        print("=" * 70)
        print()
        print(f"Index file: {stats.get('index_path')}")
        if stats.get("collected_txt_saved"):
            print(f"Collected content file: {stats.get('collected_txt_path')}")
        print("The index is now saved and will be used by the agent.")
        print("No automatic rebuilds will occur - rerun this script to update.")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("✗ REBUILD FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
