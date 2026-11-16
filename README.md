# Local PDF Assistant

Ask questions about a text-based PDF with a chat model running in Ollama. LangChain connects the prompt to the chat model and the embedding model. The app keeps its small search index in memory, so it is rebuilt each time you start the program.

## What you will learn

1. **Load:** `pdftotext` extracts selectable text and keeps page breaks.
2. **Split:** `split_text` makes smaller, slightly overlapping chunks. A whole PDF may be too much text to send for every question.
3. **Embed:** `OllamaEmbeddings` turns each chunk into a list of numbers that represents its meaning. The app sends chunks in batches and updates a live terminal counter as vectors come back.
4. **Retrieve:** the app embeds your question and uses cosine similarity to find the closest chunks.
5. **Prompt and generate:** a `ChatPromptTemplate` passes those chunks and your question to `ChatOllama`. The model is asked to answer from the PDF and cite page numbers.

This first version uses a tiny in-memory vector index instead of a vector database. That keeps the retrieval steps visible while you are learning.

## Before you start

- Python 3.12 and the packages already available in this project's `.venv` (`langchain-core` and `langchain-ollama`).
- Ollama running on this computer.
- A chat model and an embedding model available in Ollama.
- Poppler's `pdftotext` command. On Ubuntu or Debian, install it with `sudo apt install poppler-utils` if it is missing.
- A PDF containing selectable text. Scanned PDFs need OCR, which this version does not include.

Check the model names Ollama has available with:

```bash
ollama list
```

The defaults are `llama3` for chat and `nomic-embed-text` for embeddings. If you do not have either, fetch it with `ollama pull MODEL_NAME`, or pass the model names you already have on the command line.

## Run it

From this folder, replace `./example.pdf` with your PDF:

```bash
.venv/bin/python app.py ./example.pdf
```

To choose models explicitly:

```bash
.venv/bin/python app.py ./example.pdf --chat-model llama3 --embedding-model nomic-embed-text
```

Ask a question at the `You:` prompt. Type `exit` to quit. The first run embeds every chunk, so it can take a while for a large PDF. A live counter shows embedding progress. Those vectors are not saved; the next run creates them again. The default embedding batch size is 64; lower it with `--embedding-batch-size` if a batch uses too much memory.

You can also set `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBEDDING_MODEL`, and `OLLAMA_BASE_URL` environment variables instead of passing those options. Use `--help` to see the chunk and retrieval options.

## Follow the code

- `read_pdf` runs `pdftotext` and labels text with its page number.
- `split_text` makes overlapping windows without cutting at every word boundary.
- `OllamaEmbeddings.embed_documents` creates the document vectors once at startup.
- `OllamaEmbeddings.embed_query` creates a vector for each question.
- `cosine_similarity` ranks the chunks by meaning similarity.
- `create_prompt_chain` uses LangChain's `prompt | model | parser` pattern (LCEL).
- `ask_loop` connects retrieval and generation for each question.

## Try these learning changes

1. Change `--top-k` and see how giving the model more or fewer chunks changes its answer.
2. Change `--chunk-size` and `--chunk-overlap`; notice when a fact gets split across chunks.
3. Print the retrieved pages and similarity scores before calling the chat model.
4. Add support for a folder of PDFs, then later save embeddings so startup is faster.

With the default Ollama URL, document extraction, embeddings, and chat inference happen locally. If you point `OLLAMA_BASE_URL` or `--base-url` at another machine, the PDF text and questions are sent to that Ollama server. The app does not call a hosted LLM service on its own.
