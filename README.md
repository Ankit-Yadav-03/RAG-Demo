# Document Q&A (minimal RAG)

Answers questions **from your own documents**, shows the retrieved passages, and says
**"Not in the documents"** instead of guessing.

## Run

```bash
pip install -r requirements.txt
python rag.py "What is the lifespan of a solar panel?"   # one question (offline, TF-IDF)

export GEMINI_API_KEY=your_key   # https://aistudio.google.com/apikey
python rag.py --embedder gemini --llm "How long do solar panels last?"
python rag.py                                            # interactive mode
python rag.py --docs path/to/your/folder "your question" # your own .txt/.md files
```

Sample documents are in `docs/` (solar energy, coffee, sleep).

## How it works

| Step | What happens |
|---|---|
| Chunk | Sentence-based: whole sentences packed to ~300 chars, 1-sentence overlap between chunks |
| Embed | Default: TF-IDF vectors (scikit-learn, no downloads). Optional: `--embedder gemini` for dense Google GenAI embeddings (`gemini-embedding-001`) |
| Retrieve | Cosine similarity, top `--k` (default 3) chunks |
| Answer | Default: the best-matching sentences from the retrieved chunks (extractive). Optional: `--llm` has Gemini write the answer from the passages only (`--model` to change, default `gemini-2.5-flash`) |
| Cite | Every answer prints the retrieved passages with file name, chunk id and score |
| Refuse | If the top score is below `--min-score`, it prints "Not in the documents" |

## Decisions and limits (read this)

- **TF-IDF is the default** so it runs anywhere with zero downloads. The cost: it matches
  *words*, not *meaning*. "How long do solar panels last?" retrieves the wrong passage because the
  document says "lifespan", not "last". "What is the lifespan..." works. Use `--embedder gemini` for semantic matching.
- **The not-in-documents check is a score threshold**, not understanding. Defaults (0.10 TF-IDF,
  0.55 gemini) were set by hand on the sample docs. Tune `--min-score` for your own documents.
  A question that shares a few words with a document can still pass the threshold.
- **Extractive answers are not fluent.** They return the best sentences, nothing more.
  `--llm` gives readable answers but needs `GEMINI_API_KEY`.
- **Not tested against the live API:** `--embedder gemini` and `--llm` (no API access in my build environment). The 0.55 gemini threshold is a guess; tune it.
  The TF-IDF path was tested end-to-end.
- Only `.txt` and `.md` files are supported. No PDFs.
- The index is rebuilt on every run. Fine for small folders, wasteful for large ones.