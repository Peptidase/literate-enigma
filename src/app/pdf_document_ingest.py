import pymupdf
import glob
import os 
import requests
import base64
import tqdm

from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core import SimpleDirectoryReader, Document,VectorStoreIndex
from semantic_text_splitter import TextSplitter

from tokenizers import Tokenizer
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer




from dotenv import load_dotenv

load_dotenv(".env")

model_path = os.getenv('llama_address')
embedding_port = os.getenv("llama_embedding_port")
vision_port = os.getenv("llama_vision_port")

tok = Tokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
pre = TextSplitter.from_huggingface_tokenizer(tok, 160)


base_split = split_by_sentence_tokenizer()   # returns the callable


def safe_sentences(text: str) -> list[str]:
    out = []
    for s in base_split(text):        # llama_index default
        out.extend([s] if len(tok.encode(s).ids) <= 400 else pre.chunks(s))
    return out


embed_model = OpenAILikeEmbedding(
    model_name="bge-base-en-v1.5",        # must match what the router advertises
    api_base=f"http://{model_path}:6767/v1",
    api_key="not-used",                    # llama.cpp ignores it, the client requires it
    embed_batch_size=8,                    # LlamaIndex defaults to 10
) #similarity_model


similarity_model = SemanticSplitterNodeParser(
    embed_model=embed_model,
    buffer_size=1,                        # sentences grouped before embedding
    breakpoint_percentile_threshold=95,   # cut where distance exceeds this percentile
    sentence_splitter=safe_sentences
)

embedding_base_url = f"http://{model_path}:{embedding_port}"
vision_base_url = f"http://{model_path}:{vision_port}"

print(f"Embedding Base URL: {embedding_base_url}")
print(f"Vision Base URL: {vision_base_url}")

corpus = "corpus.yaml"
pdf_directory = "/home/arif/Documents/literate-enigma/test_data/"

pdf_files_path = glob.glob(pdf_directory + "*.pdf")[:3]


documents, image_pages = [], []
document_embeddings = {}

for path in tqdm.tqdm(pdf_files_path, desc="Processing PDFs", unit="file", dynamic_ncols=True):
    with pymupdf.open(path) as doc:
        for page_no, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                documents.append(Document(
                    text=text,
                    metadata={"source": path, "page": page_no},
                ))
            else:
                image_pages.append((path, page_no))



nodes = similarity_model.get_nodes_from_documents(documents)
capped = SentenceSplitter(chunk_size=480, chunk_overlap=0).get_nodes_from_documents(nodes)
print(max(len(n.get_content()) for n in capped))

index = VectorStoreIndex(capped, embed_model=embed_model)
index.storage_context.persist("../../test_data/vector_storage")

"""Splits into sentences, embeds every sentence, measures the distance between neighbours, 
cuts where distance spikes, groups the sentences between cuts into nodes, 
throws all those vectors away. 
Returns text chunks with embedding=None. 
The embeddings were used as a ruler."""




def embedding_request(chunk,doc_name):
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
    document_embeddings[doc_name].append(vectors)



def image_request(png_bytes,doc_name,page_no):
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





"""
        if vision:

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
"""
