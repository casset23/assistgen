from app.services.book_rag.document_parser_service import DocumentParserService

PDF_PATH = r"C:\Users\L\Desktop\15281823.pdf"
OUTPUT_PATH = r"C:\Users\L\Desktop\15281823.ocr.full.cleaned.v3.json"
MAX_PAGES = None

service = DocumentParserService(debug_ocr=True)
output_file, result = service.save_parsed_result(
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
