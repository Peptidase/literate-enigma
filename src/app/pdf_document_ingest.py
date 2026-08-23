import pypdf
import pymupdf

import glob
from dotenv import load_dotenv
import os 
import requests
from semantic_text_splitter import TextSplitter
import base64


load_dotenv(".env")

model_path = os.getenv('llama_address')

embedding_port = os.getenv("llama_embedding_port")
vision_port = os.getenv("llama_vision_port")


embedding_base_url = f"http://{model_path}:{embedding_port}"
vision_base_url = f"http://{model_path}:{vision_port}"

print(f"Embedding Base URL: {embedding_base_url}")
print(f"Vision Base URL: {vision_base_url}")

corpus = "corpus.yaml"
pdf_directory = "/home/arif/Documents/literate-enigma/test_data/"

pdf_files = glob.glob(pdf_directory + "*.pdf")




splitter = TextSplitter((300,450))

documents = []
document_embeddings = {}

for i in pdf_files:
    
    pdf_reader = pypdf.PdfReader(i)
    documents.append(pdf_reader)
    document_embeddings[i] = []
    pages = pdf_reader.pages

    doc = pymupdf.open(i)
    for page_no, page in enumerate(doc):
        png_bytes = page.get_pixmap(dpi=100).tobytes("png")
        text = page.get_text("text")


        for chunk in splitter.chunks(text):
            
            r = requests.post(
            f"{embedding_base_url}/v1/embeddings",
            json={"model": "x", "input":chunk},
            timeout=120,)

            if r.status_code != 200:
                print(r.status_code, r.text)
                r.raise_for_status()


            r.raise_for_status()
            data = r.json()["data"]
            vectors = [d["embedding"] for d in data]
            document_embeddings[i].append(vectors)

        png_bytes = page.get_pixmap(dpi=150).tobytes("png")
        b64 = base64.b64encode(png_bytes).decode()

        r = requests.post(
            f"{vision_base_url}/v1/chat/completions",
            json={
                "model": "Qwen2.5-VL-3B-Instruct",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this figure. List any component labels, part numbers or pin names visible. If there arent any labels, just describe the figure and what it contains"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }],
                "max_tokens": 300,
                "temperature": 0,
            },
            timeout=600,
        )

        if r.status_code != 200:
            print(r.status_code, r.text)
            r.raise_for_status()

        caption = r.json()["choices"][0]["message"]["content"]

        print(caption)

print(len(document_embeddings))