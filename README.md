# literate-enigma
A simple airgapped document parsing tool using locally procured LLM insights to allow users to ask questions about uploaded documents. Features OCR and Vision based document process as well as a RAG document recovery storage tool. 

## Installation

Please prepare a `.env` file for the required details of where endpoints reside. The models used are shown below and are hosted using llama.cpp on one service serving to the same port. 

## Technical Details
Models:

1. QWEN Instruct-Model for user interface
2. bge-base-en-v1.5-gguf for embedding

    Note: These are run from llama.cpp, build from scratch and started with the following command
    ```
    llama-server --model ./<Model>.gguf --host <ip> --port <port> --no-ui
    ```
    This disables the web-ui, allowing for an interface only interactable from the python application itself.


### RAG Structure

We use document ingestion to take all the information from the test data and create embeddings using the aforementioned embedding model. 