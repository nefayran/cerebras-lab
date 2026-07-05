#!/usr/bin/env python3
"""ЗОНА ВЫИГРЫША: рекурсивное замощение (tile.py) против ОДНОГО думающего прохода той же 8b,
на больших раскладываемых делах. Попарный судья (позиция рандомизирована) + объём.
Запуск: being/venv/bin/python being/tile_vs_pure.py"""
import io, contextlib
import tile
from stream import llm, jget

TASKS = [
    ("onboarding", "Design a complete onboarding experience for a new mobile banking app: everything a "
     "thoughtful team would specify before building it."),
    ("coffee-plan", "Write a comprehensive launch plan for a specialty cold-brew coffee shop in Tokyo: "
     "audience, brand, menu, location, staffing, marketing, budget, timeline, risks."),
    ("finance-arch", "Design the architecture and full feature set for a personal finance tracking app."),
]

PURE = ("Produce the deliverable for the task fully and concretely. Think carefully, cover everything a "
        "thoughtful team would, then write the complete result.")
PAIR = ('Compare two answers to the same task on COVERAGE (how many distinct important aspects are addressed), '
        'concreteness, and usefulness. Which is better overall? Reply A, B, or tie. JSON {"winner":"A|B|tie"}.')


def run_tile(task):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doc = tile.run(task)
    return str(doc), buf.getvalue().count("(leaf)")


def pairwise(task, net_doc, base_doc, seed):
    swap = seed % 2 == 1
    a, b = (base_doc, net_doc) if swap else (net_doc, base_doc)
    w = str(jget(llm(PAIR, f"Task: {task}\n\nAnswer A:\n{a[:1800]}\n\nAnswer B:\n{b[:1800]}\nJSON:",
                     temp=0.2), "winner")[0] or "tie").strip().upper()[:1]
    if w == "A": return "base" if swap else "tile"
    if w == "B": return "tile" if swap else "base"
    return "tie"


import sys
def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"TILE (recursive) vs PURE (one thinking pass) — big deliverables, K={K} repeats\n")
    tally = {}
    for i, (name, task) in enumerate(TASKS):
        wins = {"tile": 0, "base": 0, "tie": 0}
        for k in range(K):
            net, leaves = run_tile(task)
            base = llm(PURE, f"Task: {task}\n\nDeliverable:", temp=0.5, max_tokens=4000, think=True)
            w = pairwise(task, net, base, i * K + k)
            wins[w] += 1
            print(f"  {name:13} r{k} tile {leaves}lf/{len(net.split()):4}w | pure {len(base.split()):4}w -> {w}")
        tally[name] = wins
        print(f"  == {name:13} TILE {wins['tile']}/{K}   pure {wins['base']}/{K}   tie {wins['tie']}/{K}\n")
    print("SUMMARY:")
    for name, w in tally.items():
        print(f"  {name:13} TILE {w['tile']}/{K}  pure {w['base']}/{K}  tie {w['tie']}/{K}")


if __name__ == "__main__":
    main()
