#!/usr/bin/env python3
"""
СЛОЙ 3 — ПАМЯТЬ как настоящая векторная RAG-сеть (а не string-match).
  embed (nomic-embed-text, 768d) -> векторный стор memory_store.json (растёт) ->
  косинусный поиск top-k. Многоузловая: рассуждение даёт несколько запросов,
  каждый ищется отдельно, хиты сливаются. Консолидация: пишем выученное обратно (с вектором).

Источники recall: ЭПИЗОДИЧЕСКАЯ память (векторный стор выученного) + СЕМАНТИЧЕСКАЯ
(параметрические знания модели) — сливаются. Это и есть RAG-сеть, а не один извлекатель.
"""
import json, os, math, urllib.request

OLLAMA_EMBED = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
STORE = os.path.join(os.path.dirname(__file__), "memory_store.json")
MAX_ITEMS = 1000


def embed(text):
    body = json.dumps({"model": EMBED_MODEL, "prompt": str(text)[:2000]}).encode()
    req = urllib.request.Request(OLLAMA_EMBED, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["embedding"]


def _load():
    try:
        with open(STORE) as f: return json.load(f)
    except Exception: return []
def _save(s):
    with open(STORE, "w") as f: json.dump(s[-MAX_ITEMS:], f, ensure_ascii=False)


def add(text):
    """Консолидация: вшить факт в векторный стор (с эмбеддингом), если его там ещё нет."""
    text = str(text).strip()
    if not text: return
    s = _load()
    if any(e.get("text") == text for e in s): return
    try: v = embed(text)
    except Exception: return
    s.append({"text": text, "vec": v}); _save(s)


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def search(query, k=4, floor=0.45):
    """Косинусный top-k по эпизодическому стору."""
    s = _load()
    if not s: return []
    try: qv = embed(query)
    except Exception: return []
    scored = sorted(((_cos(qv, e["vec"]), e["text"]) for e in s), reverse=True)
    return [t for sc, t in scored[:k] if sc >= floor]


def size(): return len(_load())
