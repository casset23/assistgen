from app.services.book_rag.document_parser_service import DocumentParserService

PDF_PATH = r"C:\Users\L\Desktop\15281823.pdf"
OUTPUT_PATH = r"C:\Users\L\Desktop\15281823.ocr.json"

service = DocumentParserService(debug_ocr=True)
output_file, result = service.save_parsed_result(
    file_path=PDF_PATH,
    output_path=OUTPUT_PATH,
)

print("saved_to:", output_file)
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
    print(f"page_{page['page']}_char_count:", page["char_count"])
    print(f"page_{page['page']}_preview:")
    print(page["text"][:500])
    print("-" * 40)
