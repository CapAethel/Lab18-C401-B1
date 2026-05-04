"""
Extract text from scanned/image-based PDFs using OpenAI Vision API.
Saves extracted text as .md files in data/ for the pipeline to consume.
"""

import base64
import io
import os
import time

import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")


def pdf_page_to_base64_jpeg(page, dpi=150):
    """Render PDF page as JPEG image and return base64 string."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    # Convert to JPEG for smaller size
    from PIL import Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def ocr_image_openai(b64_image: str, page_num: int, retries: int = 3) -> str:
    """Send image to OpenAI Vision for OCR with retries."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=120)

    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Hãy trích xuất TOÀN BỘ nội dung văn bản từ hình ảnh này. "
                                    "Giữ nguyên cấu trúc, số liệu, bảng biểu. "
                                    "Trả về dưới dạng markdown. "
                                    "Chỉ trả về nội dung đã trích xuất, không thêm gì khác."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        ],
                    }
                ],
                max_output_tokens=4096,
                reasoning={"effort": "minimal"},
            )
            return resp.output_text.strip()
        except Exception as e:
            print(f"\n    Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return f"[OCR FAILED for page {page_num + 1}]"


def extract_pdf(pdf_path: str, output_path: str):
    """Extract text from all pages of a PDF using OCR."""
    doc = fitz.open(pdf_path)
    all_text = []

    print(f"\nExtracting: {pdf_path} ({len(doc)} pages)")

    for i, page in enumerate(doc):
        print(f"  Page {i + 1}/{len(doc)}...", end=" ", flush=True)
        b64 = pdf_page_to_base64_jpeg(page)
        print(f"(img: {len(b64)//1024}KB)", end=" ", flush=True)
        text = ocr_image_openai(b64, i)
        all_text.append(f"<!-- Page {i + 1} -->\n\n{text}")
        print(f"{len(text)} chars extracted")
        time.sleep(0.5)

    full_text = "\n\n---\n\n".join(all_text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"  Saved to {output_path} ({len(full_text)} total chars)")
    return full_text


def main():
    pdfs = [
        ("data/BCTC.pdf", "data/BCTC.md"),
        ("data/Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf",
         "data/Nghi_dinh_13_2023.md"),
    ]

    for pdf_path, md_path in pdfs:
        if os.path.exists(md_path) and os.path.getsize(md_path) > 500:
            print(f"Skipping {pdf_path} — {md_path} already exists with content")
            continue
        if not os.path.exists(pdf_path):
            print(f"Skipping {pdf_path} — not found")
            continue
        # Remove stale partial files
        if os.path.exists(md_path):
            os.remove(md_path)
        extract_pdf(pdf_path, md_path)

    print("\nDone! PDF text extracted to data/*.md")


if __name__ == "__main__":
    main()
