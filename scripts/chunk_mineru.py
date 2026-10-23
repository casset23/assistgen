"""Chunk local MinerU output without loading an embedding or OCR model."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend/deepseek_agent/llm_backend'))
from app.services.book_rag.mineru_book_rag_service import MinerUBookRAGService

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Directory containing full.md or content_list_v2.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target, result = MinerUBookRAGService().save_chunks_from_mineru_result(args.source, args.output)
    print(f"{result['stats']['chunk_count']} chunks -> {target}")
