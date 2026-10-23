"""Synthetic fixture: Markdown chunking must not require PDF/OCR model imports."""
import builtins
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend/deepseek_agent/llm_backend"))


class MarkdownChunksTest(unittest.TestCase):
    def test_markdown_path_without_ocr_dependencies(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'fitz', 'rapidocr_onnxruntime', 'PyPDF2', 'torch'}:
                raise ImportError('PDF/OCR dependencies unavailable')
            return original(name, *args, **kwargs)
        builtins.__import__ = guarded
        try:
            from app.services.book_rag.mineru_book_rag_service import MinerUBookRAGService
            with tempfile.TemporaryDirectory() as d:
                Path(d, 'full.md').write_text('# 第一章 缓存基础\n\n缓存将重复请求的数据暂存在内存中，减少对后端存储的访问。' * 25, encoding='utf-8')
                service = MinerUBookRAGService()
                result = service.build_chunks_from_mineru_result(d, book_name='Synthetic demo')
                self.assertGreater(result['stats']['chunk_count'], 0)
                self.assertTrue(all(c['text'].strip() for c in result['chunks']))
                out, _ = service.save_chunks_from_mineru_result(d)
                self.assertEqual(json.loads(out.read_text(encoding='utf-8'))['book_name'], Path(d).name)
        finally:
            builtins.__import__ = original


if __name__ == '__main__':
    unittest.main()
