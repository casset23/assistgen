from pathlib import Path
from app.services.book_rag.document_parser_service import DocumentParserService
from app.services.embedding_service import EmbeddingService
import json

PDF_PATH = str(Path(__file__).resolve().parent / "data" / '15281823.pdf')
OUTPUT_PATH = str(Path(__file__).resolve().parent / "data" / '15281823.ocr.first40.cleaned.json')
CHUNK_DEBUG_OUTPUT_PATH = str(Path(__file__).resolve().parent / "data" / '15281823.first40.chunks.debug.json')
MAX_PAGES = 40

parser_service = DocumentParserService(
    debug_ocr=True,
    ocr_min_text_length=30,
    ocr_zoom_matrix=(1.0, 1.0),
    ocr_failure_threshold=3,
)
output_file, result = parser_service.save_parsed_result(
    file_path=PDF_PATH,
    output_path=OUTPUT_PATH,
    max_pages=MAX_PAGES,
)

print("saved_to:", output_file)
print("max_pages:", MAX_PAGES if MAX_PAGES is not None else "all")
print("book_name:", result["book_name"])
print("total_pages:", result["total_pages"])
print("non_empty_pages:", result["non_empty_pages"])
print("ocr_pages:", result["ocr_pages"])
print("ocr_attempted_pages:", result["ocr_attempted_pages"])

for page in result["pages"]:
    print(f"page_{page['page']}_method:", page["extract_method"])
    print(f"page_{page['page']}_ocr_attempted:", page["ocr_attempted"])
    print(f"page_{page['page']}_direct_text_length:", page["direct_text_length"])
    print(f"page_{page['page']}_ocr_text_length:", page["ocr_text_length"])
    print(f"page_{page['page']}_raw_char_count:", page["raw_char_count"])
    print(f"page_{page['page']}_char_count:", page["char_count"])
    print(f"page_{page['page']}_raw_preview:")
    print(page["raw_text"][:500])
    print(f"page_{page['page']}_cleaned_preview:")
    print(page["text"][:500])
    print("-" * 40)

embedding_service = EmbeddingService()
chunk_records = embedding_service._build_chunks_from_parsed_result(result)

with open(CHUNK_DEBUG_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "file_path": PDF_PATH,
            "parsed_output_path": str(output_file),
            "chunk_count": len(chunk_records),
            "chunks": chunk_records,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("chunk_debug_saved_to:", CHUNK_DEBUG_OUTPUT_PATH)
print("chunk_count:", len(chunk_records))

for chunk in chunk_records[:12]:
    metadata = chunk["metadata"]
    print(f"chunk_{metadata['chunk_index']} pages {metadata['page_start']}-{metadata['page_end']} chars {metadata['char_count']}")
    print(chunk["text"][:900])
    print("=" * 60)
