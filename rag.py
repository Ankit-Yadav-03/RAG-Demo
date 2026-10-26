"""Minimal RAG: answer questions from your own documents, with sources.

Pipeline: load docs -> chunk -> embed -> retrieve top-k -> answer -> cite.
If the best match scores below --min-score, it says the answer is not in the documents.
"""
import os
import argparse
import re
import sys
from dotenv import load_dotenv
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai
from google.genai import types


load_dotenv()

EXTS = {".txt", ".md"}


#   1. load + chunk  
def load_docs(folder):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"No .txt/.md files found in {folder}")
    return [(p.name, p.read_text(encoding="utf-8", errors="ignore")) for p in files]


def chunk_text(text, size=300, overlap=1):
    """Sentence-based chunking: pack whole sentences up to `size` chars; repeat the last
    `overlap` sentence(s) at the start of the next chunk so context isn't cut at the seam."""
    text = re.sub(r"^#+ .*$", "", text, flags=re.M)  # drop markdown heading lines
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if x.strip()]
    chunks, cur = [], []
    for sent in sents:
        if cur and len(" ".join(cur)) + len(sent) + 1 > size:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.append(sent)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_chunks(folder, size, overlap):
    out = []
    for name, text in load_docs(folder):
        for i, c in enumerate(chunk_text(text, size, overlap)):
            out.append({"source": name, "chunk_id": i, "text": c})
    return out


#   2. embed  
class TfidfEmbedder:
    """Offline, no downloads. Sparse lexical vectors, L2-normalised."""
    name = "tfidf"

    def __init__(self, texts):
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.vec.fit(texts)

    def encode(self, texts, query=False):
        return self.vec.transform(texts).toarray()


class GeminiEmbedder:
    """Dense semantic embeddings via Google GenAI. Needs GEMINI_API_KEY."""
    name = "gemini"

    def __init__(self, texts=None, model="gemini-embedding-001"):
        self.client = get_client()
        self.model = model

    def _embed(self, texts, task):
        out = []
        for i in range(0, len(texts), 100):  # API batch limit
            r = self.client.models.embed_content(
                model=self.model, contents=texts[i:i + 100],
                config=types.EmbedContentConfig(task_type=task))
            out += [e.values for e in r.embeddings]
        m = np.asarray(out, dtype=float)
        return m / np.linalg.norm(m, axis=1, keepdims=True)  # normalise -> dot = cosine

    def encode(self, texts, query=False):
        return self._embed(texts, "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT")


def get_client():
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set. Get a key at https://aistudio.google.com/apikey")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def make_embedder(kind, texts):
    return GeminiEmbedder(texts) if kind == "gemini" else TfidfEmbedder(texts)


#   3. retrieve  
def retrieve(query, chunks, matrix, embedder, k):
    q = embedder.encode([query], query=True)[0]
    scores = matrix @ q  # cosine, since both sides are normalised
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in top]


#   4. answer  
def extractive_answer(query, hits, embedder):
    """No LLM: pick the 2 sentences from the retrieved chunks most similar to the question."""
    sents = []
    for _, c in hits:
        text = re.sub(r"^#+ .*$", "", c["text"], flags=re.M)  # drop markdown headings
        sents += [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if len(x.strip()) > 20]
    sents = list(dict.fromkeys(sents))  # overlapping chunks repeat sentences
    if not sents:
        return hits[0][1]["text"]
    scores = embedder.encode(sents) @ embedder.encode([query], query=True)[0]
    top = np.argsort(-scores)[:2]
    keep = [i for i in top if scores[i] > 0 and scores[i] >= 0.5 * scores[top[0]]]  # no padding with weak matches
    return " ".join(sents[i] for i in sorted(keep)) or sents[top[0]]


def llm_answer(query, hits, model):
    ctx = "\n\n".join(f"[{i+1}] ({h['source']})\n{h['text']}" for i, (_, h) in enumerate(hits))
    r = get_client().models.generate_content(
        model=model,
        contents=f"Passages:\n{ctx}\n\nQuestion: {query}",
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=500,
            system_instruction=(
                "Answer ONLY from the numbered passages. Cite passages like [1]. "
                "If the passages do not contain the answer, reply exactly: "
                "'Not in the documents.' Never use outside knowledge.")))
    return (r.text or "").strip()


def ask(query, chunks, matrix, embedder, args):
    hits = retrieve(query, chunks, matrix, embedder, args.k)
    best = hits[0][0]
    print(f"\nQ: {query}")
    if best < args.min_score:
        print(f"A: Not in the documents. (best match score {best:.2f} < threshold {args.min_score})")
        return
    kept = [(s, c) for s, c in hits if s >= args.min_score]
    answer = llm_answer(query, kept, args.model) if args.llm else extractive_answer(query, kept, embedder)
    print(f"A: {answer}")
    if args.llm and "not in the documents" in answer.lower():
        return
    print("\nRetrieved passages:")
    for i, (s, c) in enumerate(kept, 1):
        snippet = c["text"].replace("\n", " ")
        print(f"  [{i}] {c['source']} (chunk {c['chunk_id']}, score {s:.2f})\n      {snippet[:300]}{'...' if len(snippet) > 300 else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    ap.add_argument("--docs", default="docs", help="Folder of .txt/.md files (default: docs)")
    ap.add_argument("--embedder", choices=["tfidf", "gemini"], default="tfidf")
    ap.add_argument("--k", type=int, default=3, help="Passages to retrieve")
    ap.add_argument("--min-score", type=float, default=None,
                    help="Below this top score -> 'not in the documents' (default 0.10 tfidf, 0.55 gemini)")
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=1, help="Sentences repeated between chunks")
    ap.add_argument("--llm", action="store_true", help="Write the answer with Gemini (needs GEMINI_API_KEY)")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model for --llm")
    args = ap.parse_args()
    if args.min_score is None:
        args.min_score = 0.55 if args.embedder == "gemini" else 0.10

    chunks = build_chunks(args.docs, args.chunk_size, args.overlap)
    embedder = make_embedder(args.embedder, [c["text"] for c in chunks])
    matrix = embedder.encode([c["text"] for c in chunks])
    print(f"Indexed {len(chunks)} chunks from {args.docs}/ using {embedder.name}")

    if args.question:
        ask(args.question, chunks, matrix, embedder, args)
    else:
        print("Ask a question (empty line to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if not q:
                break
            ask(q, chunks, matrix, embedder, args)


if __name__ == "__main__":
    main()