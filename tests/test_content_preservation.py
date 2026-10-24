import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend/deepseek_agent/llm_backend'))
from app.services.book_rag.mineru_book_rag_service import MinerUBookRAGService


class ContentPreservationTests(unittest.TestCase):
    def test_complete_paragraph_starting_with_continuation_character_survives(self):
        result = MinerUBookRAGService().build_chunks_from_mineru_result(
            str(Path(__file__).resolve().parents[1] / 'examples/mineru-demo'))
        content = '\n'.join(c['text'] for c in result['chunks'])
        self.assertIn('并发限制控制同时处理的请求数量', content)
        self.assertIn('等待队列需要设置边界', content)
