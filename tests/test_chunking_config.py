import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend/deepseek_agent/llm_backend'))
from app.services.book_rag.mineru_book_rag_service import MinerUBookRAGService

DEFAULT = dict(target_chunk_length=1100, max_chunk_length=1500,
               chunk_soft_limit=1620, chunk_overlap=120, min_chunk_length=320)


class ChunkingConfigTests(unittest.TestCase):
    def config(self, **changes):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'config.json')
            p.write_text(json.dumps({**DEFAULT, **changes}), encoding='utf-8')
            return MinerUBookRAGService(str(p))

    def test_valid_config(self):
        self.assertEqual(self.config()._config, DEFAULT)

    def test_invalid_config_fails_before_chunking(self):
        for values in [dict(target_chunk_length=0), dict(chunk_overlap=-1),
                       dict(chunk_overlap=1100), dict(min_chunk_length=1200),
                       dict(max_chunk_length=1000), dict(chunk_soft_limit=1400),
                       dict(chunk_overlap=True), dict(max_chunk_length=1500.5)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(**values)
