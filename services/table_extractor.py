import pdfplumber
from logger import get_logger

logger = get_logger("table_extractor")


def extract_tables_as_markdown(pdf_path: str) -> str:
    markdown_output = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if not tables:
                    continue

                markdown_output += f"### Tables from Page {page_num + 1}\n\n"
                for table_idx, table in enumerate(tables):
                    if not table or not table[0]:
                        continue

                    markdown_output += f"#### Table {table_idx + 1}\n\n"

                    for row_idx, row in enumerate(table):
                        cleaned_row = [" ".join(str(cell).split()) if cell is not None else "" for cell in row]
                        row_str = "| " + " | ".join(cleaned_row) + " |"
                        markdown_output += row_str + "\n"

                        if row_idx == 0:
                            separator = "| " + " | ".join(["---"] * len(cleaned_row)) + " |"
                            markdown_output += separator + "\n"

                    markdown_output += "\n"
    except Exception as e:
        logger.warning(f"Table extraction failed for {pdf_path}: {e}")

    return markdown_output
