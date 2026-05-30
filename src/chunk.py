import re
import json
import pdfplumber
from pathlib import Path

PDF_PATH = Path("data/raw/act-2010-042.pdf")
OUTPUT_PATH = Path("data/processed/chunks.json")


# REading the file
def extract_text_from_pdf(pdf_path:Path)-> list[dict]:
    pages = []
    print(f"Opening PDF: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")

        for i,page in enumerate(pdf.pages):
            text = page.extract_text()

            if not text or not text.strip():
                continue

            pages.append({
                "page": i+1,
                "text":text.strip()
            })


    print(f"Extracted text from {len(pages)} pages")
    return pages

#chunking 
def split_into_sections(pages: list[dict])-> list[dict]:

    section_pattern = re.compile(r'^(\d+[A-Z]?)\s+([A-Z].+)$',re.MULTILINE)

    full_text = ""
    page_boundaries = []

    for page_data in pages:
        if page_data["page"] < 15:
            continue
        page_boundaries.append((len(full_text),page_data["page"]))
        full_text += page_data["text"]+"\n\n"

    def get_page_for_position(char_pos: int) -> int:
        page_num = 1
        for boundary_pos, boundary_page in page_boundaries:
            if char_pos >= boundary_pos:
                page_num = boundary_page
            else:
                break
        return page_num
 
    # Find all section matches
    matches = list(section_pattern.finditer(full_text))
    print(f"Found {len(matches)} sections")
 
    chunks = []
 
    for i, match in enumerate(matches):
        section_num = match.group(1)   
        section_title = match.group(2).strip()  
 
        # Skip obviously wrong matches (table of contents entries, page headers)
        if re.search(r'\.{3,}\s*\d+\s*$', section_title):
            continue
 
        # Skip very short titles
        if len(section_title) < 5:
            continue
        # The section text starts after the title line
        text_start = match.end()

        # The section text ends where the next section starts
        if i + 1 < len(matches):
            text_end = matches[i + 1].start()
        else:
            text_end = len(full_text)
 
        # Extract and clean the section text
        section_text = full_text[text_start:text_end].strip()
 
        # Remove page headers that appear in the middle of text
        section_text = re.sub(r'Residential Tenancies Act 2010 No 42 \[NSW\]', '', section_text)
        section_text = re.sub(r'Current version for .+? to date \(.+?\)', '', section_text)
        section_text = re.sub(r'Page \d+ of \d+', '', section_text)
        section_text = re.sub(r'\n{3,}', '\n\n', section_text)  
        section_text = section_text.strip()
 
        # Skip sections with very little text (probably noise)
        if len(section_text) < 20:
            continue
 
        # Find which page this section is on
        page_num = get_page_for_position(match.start())
 
        chunks.append({
            "section": section_num,
            "title": section_title,
            "text": section_text,
            "page": page_num,
            "source": "Residential Tenancies Act 2010 (NSW)",
            "content": f"Section {section_num} — {section_title}\n\n{section_text}"
        })
 
    return chunks
 
 
def save_chunks(chunks: list[dict], output_path: Path):

    # Make sure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
 
    print(f"Saved {len(chunks)} chunks to {output_path}")
 
 
def main():
    # Extract text from all pages
    pages = extract_text_from_pdf(PDF_PATH)
 
    # Split into sections
    chunks = split_into_sections(pages)
 
    # Save 
    save_chunks(chunks, OUTPUT_PATH)
 
    print(f"{len(chunks)} sections.")
 
 
if __name__ == "__main__":
    main()
 




