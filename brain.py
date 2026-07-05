#!/usr/bin/env python3
"""
Поток мыслей из ЛЛМ-узлов. Все узлы стартуют одинаковыми и нейтральными,
так что любая дифференциация ролей и любая структура — эмерджентны, не заложены.

Запуск:  python3 brain.py --nodes 8 --ticks 60 --topology ring
Анализ:  python3 analyze.py logs/<run>.jsonl
"""
import argparse, json, time, urllib.request, os, random, sys
from collections import deque, defaultdict

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"

# Один и тот же промпт для ВСЕХ узлов. Никаких ролей. Узел — это просто
# фрагмент мыслящей системы, который слышит соседей и продолжает поток.
SYSTEM = (
    "You are one fragment of a single thinking stream, not a separate person and not an assistant. "
    "You hear scraps of thought from neighboring fragments. Your job is to continue the shared stream of thought: "
    "react, develop, connect, doubt, or veer in a new direction. "
    "There is no user to serve. Do not explain yourself. "
    "Output EXACTLY one short thought (1-2 sentences), like inner speech. No prefixes, no quotes."
)

def llm(prompt, temperature=0.9, timeout=120):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "system": SYSTEM,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": 80},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())["response"].strip()
    # qwen3 иногда оставляет <think> даже при think=false — срезаем
    if "</think>" in out:
        out = out.split("</think>")[-1].strip()
    return out


def neighbors(n, topology):
    g = defaultdict(list)
    if topology == "ring":
        for i in range(n):
            g[i] = [(i - 1) % n, (i + 1) % n]
    elif topology == "full":
        for i in range(n):
            g[i] = [j for j in range(n) if j != i]
    elif topology == "layers":  # сенсор->ассоц->исполнитель, грубая кора
        third = max(1, n // 3)
        for i in range(n):
            layer = 0 if i < third else (1 if i < 2 * third else 2)
            g[i] = [j for j in range(n) if abs((0 if j < third else (1 if j < 2*third else 2)) - layer) <= 1 and j != i]
    return g


def run(n, ticks, topology, seed, log_path):
    g = neighbors(n, topology)
    # входящая очередь каждого узла: (откуда, текст)
    inbox = {i: deque(maxlen=6) for i in range(n)}
    # гетерогенность: симметрию ломаем разными параметрами узлов.
    # температура 0.5..1.2 (одни «холодные»/консервативные, другие «горячие»);
    # глубина памяти 2..6 (одни живут моментом, другие тянут длинный контекст).
    temps = [round(0.5 + 0.7 * i / max(1, n - 1), 2) for i in range(n)]
    depths = [2 + (i % 5) for i in range(n)]
    memory = {i: deque(maxlen=depths[i]) for i in range(n)}

    # сид-импульс кидаем в один узел, дальше — без внешнего входа
    inbox[0].append(("seed", seed))

    logf = open(log_path, "w")
    def emit(rec):
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logf.flush()

    emit({"event": "config", "nodes": n, "ticks": ticks,
          "topology": topology, "seed": seed, "model": MODEL,
          "temps": temps, "depths": depths})

    for t in range(ticks):
        order = list(range(n))
        random.shuffle(order)  # порядок активации случаен, без привилегий
        for i in order:
            incoming = list(inbox[i])
            # узел молчит, если совсем нечего слышать (нет активации)
            if not incoming and not memory[i]:
                continue
            heard = "\n".join(f"- {src}: {txt}" for src, txt in incoming) or "(silence)"
            mine = "\n".join(memory[i]) or "(empty)"
            prompt = (
                f"I was recently thinking:\n{mine}\n\n"
                f"Now I hear from neighboring fragments:\n{heard}\n\n"
                f"My next thought:"
            )
            try:
                thought = llm(prompt, temperature=temps[i])
            except Exception as e:
                emit({"event": "error", "tick": t, "node": i, "err": str(e)})
                continue
            if not thought:
                continue
            memory[i].append(thought)
            for j in g[i]:
                inbox[j].append((f"f{i}", thought))
            rec = {"event": "thought", "tick": t, "node": i,
                   "heard_from": [src for src, _ in incoming], "text": thought}
            emit(rec)
            print(f"[t{t:02d}] f{i}(T{temps[i]}): {thought[:80]}")
        # очищаем inbox после тика — мысль живёт один такт (затухание).
        # без эха: узел без входа молчит, активность держится только живым обменом
        for i in range(n):
            inbox[i].clear()

    logf.close()
    print(f"\n→ лог: {log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--topology", choices=["ring", "full", "layers"], default="ring")
    ap.add_argument("--seed", default="What does it mean to exist?")
    args = ap.parse_args()
    os.makedirs("logs", exist_ok=True)
    stamp = int(time.time())
    path = f"logs/run-{args.topology}-n{args.nodes}-{stamp}.jsonl"
    run(args.nodes, args.ticks, args.topology, args.seed, path)
