#!/usr/bin/env python3
"""ОБУЧЕНИЕ СЕТИ ОДНОРОДНЫХ LLM-УЗЛОВ. Гоняем батарею задач, считаем reward (эталон),
Хебб правит веса (горели вместе на успехе -> связь крепнет; на провале -> слабеет). Смотрим
ЭМЕРДЖЕНЦИЮ: как из почти однородного графа возникают устойчивые маршруты и специализация узлов.
Запуск: being/venv/bin/python being/learn_net.py [epochs]"""
import sys, os, re
import net

TASKS = [
    ("What is 15% of 240?", "36"),
    ("A tank holds 480 liters. It is 3/4 full. You drain 90 liters, then add back 1/3 of what remains. "
     "How many liters now?", "360"),
    ("Ann, Bob and Cara each own a different pet: a cat, a dog, or a fish. Ann does not own the cat. "
     "Bob owns the dog. Who owns the fish?", "ann"),
    ("Tom, Sue and Max each picked a different fruit: an apple, a pear, or a plum. Tom did not pick the "
     "pear or the plum. Sue picked the plum. Which fruit did Tom pick?", "apple"),
]


def score(ans, truth):
    return 1.0 if truth.lower() in str(ans).lower() else -1.0


def stats(W):
    n = net.N
    flat = [W[i][j] for i in range(n) for j in range(n) if i != j]
    strong = sum(1 for w in flat if w > 0.7)
    return sum(flat) / len(flat), strong, max(flat)


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    if os.path.exists(net.WPATH): os.remove(net.WPATH)     # учимся с НУЛЯ (детерминир. seed-граф)
    W = net.init_weights()
    n = net.N
    node_act = [0.0] * n          # суммарная активация узла по всем задачам (профиль специализации)
    print(f"LEARNING homogeneous LLM-node network — {epochs} epochs x {len(TASKS)} tasks\n")
    m0, s0, mx0 = stats(W)
    print(f"  seed graph: mean_w {m0:.3f}  strong_edges(>0.7) {s0}  max_w {mx0:.2f}\n")
    for ep in range(epochs):
        hits = 0
        for task, truth in TASKS:
            states, acts, hist = net.settle(task, W)
            ans = net.readout(task, states, acts)
            r = score(ans, truth); hits += int(r > 0)
            W = net.hebb(W, hist, r)
            for i in range(n): node_act[i] += sum(h[i] for h in hist)
            top = sorted(range(n), key=lambda i: acts[i], reverse=True)[:3]
            print(f"  ep{ep} r{r:+.0f} hot={['n%d'%i for i in top]}  | {task[:42]} -> {str(ans)[:34]!r}")
        m, s, mx = stats(W)
        print(f"  -- ep{ep}: hits {hits}/{len(TASKS)}   mean_w {m:.3f}  strong_edges {s}  max_w {mx:.2f}\n")
    import json
    with open(net.WPATH, "w") as f: json.dump(W, f)
    # ЭМЕРДЖЕНЦИЯ: профиль узлов + сильнейшие маршруты
    print("NODE specialization (total activation across all tasks):")
    for i in sorted(range(n), key=lambda i: node_act[i], reverse=True):
        bar = "#" * int(node_act[i] / max(node_act) * 30)
        print(f"  n{i}  {node_act[i]:6.1f}  {bar}")
    print("\nSTRONGEST routes (learned edges j -> i, weight):")
    edges = sorted(((W[i][j], j, i) for i in range(n) for j in range(n) if i != j), reverse=True)[:8]
    for w, j, i in edges:
        print(f"  n{j} -> n{i}   {w:.2f}")


if __name__ == "__main__":
    main()
