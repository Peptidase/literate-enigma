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

    A more specific setup includes running both on seperate ports on a machine and using a lightweight reverse proxy to send requests to the correct model and having only one available endpoint. The code to run those models is below:

    Ensure you have the VLM projector alongside the Model itself. This will allow images to be processed alongside the typical text input for models.

    ```
    echo "starting embedding server (bge-base-en-v1.5) on 127.0.0.1:$EMBED_PORT"
    "$LLAMA_BIN" \
    -m "$EMBED_MODEL" \
    --embedding \
    --pooling cls \
    --ctx-size "$EMBED_CTX" \
    --host 127.0.0.1 --port "$EMBED_PORT" \
    --no-webui \
    > "$LOG_DIR/embedding.log" 2>&1 &
    EMBED_PID=$!
    echo "$EMBED_PID" > "$RUN_DIR/embedding.pid"

    echo "starting VLM server (Qwen2.5-VL-3B-Instruct) on 127.0.0.1:$VLM_PORT"
    "$LLAMA_BIN" \
    -m "$VLM_MODEL" \
    --mmproj "$VLM_MMPROJ" \
    --ctx-size "$VLM_CTX" \
    --host 127.0.0.1 --port "$VLM_PORT" \
    --no-webui \
    > "$LOG_DIR/vlm.log" 2>&1 &
    VLM_PID=$!
    echo "$VLM_PID" > "$RUN_DIR/vlm.pid"
```

### RAG Structure

We use document ingestion to take all the information from the test data and create embeddings using the aforementioned embedding model. We use a semantic chunker to split information, it performs best with variety of different document data inputs and hence allows us to get the best information from our documents. `SemanticChunker` is the most documented chunker and allows us to apply it to an airgapped system.

The key component in this is the self-hosted embedding model, we use llama_index to semantically parse the corpus for each sentence, when a sentence grows too much in semantic distance from the previous, that is a chunk. This is then turned back to text and chunked accordingly. This is then used alongside an embedding model and correctly indexed with page number and other relevant metadata.

### Known Issues

Sometimes, there can be limits on the amount of text to be sent to an embedding model, for these issues where your provider throws "input too large to proceed" error please decrease the `TextSplitter.from_huggingface_tokenizer(tok,160)` value to a lower number. 