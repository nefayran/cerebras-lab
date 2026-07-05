#!/usr/bin/env python3
"""БЕНЧМАРК: мозг-коннектом vs ГОЛАЯ qwen3:8b на разных типах задач.
Голая = один вызов 8b «дай ответ». Мозг = connectome.run (вся структура).
Честность: чистим эпизодическую вектор-память (чтобы мерить СТРУКТУРУ, а не запомненный ответ).
Запуск: being/venv/bin/python being/bench.py"""
import io, re, os, json, contextlib
import connectome
from stream import llm

# (тип, вопрос, проверка, эталон)
TASKS = [
    ("arith",  "A shop had 3 crates of 24 apples each. They sold 17 apples on Monday and twice as many on Tuesday. On Wednesday they received 2 more crates of 24. How many apples does the shop have now?", "num", 69),
    ("arith",  "Sarah reads 18 pages per day for 5 days, then 25 pages per day for 3 days. Her book has 200 pages. How many pages are left to read?", "num", 35),
    ("arith",  "A tank holds 480 liters. It is 3/4 full. You drain 90 liters, then add back 1/3 of what remains. How many liters are in the tank now?", "num", 360),
    ("percent","What is 15% of 240?", "num", 36),
    ("trick",  "A farmer has 17 sheep. All but 9 die. How many sheep are left?", "num", 9),
    ("logic",  "Ann, Bob and Cara each own a different pet: a cat, a dog, or a fish. Ann does not own the cat. Bob owns the dog. Who owns the fish?", "kw", ["ann"]),
    ("time",   "A train leaves at 8:00 and the journey takes 35 minutes. What time does it arrive?", "kw", ["8:35"]),
    ("fact",   "Why is the sky blue?", "kw", ["rayleigh", "scatter"]),
    ("fermi",  "Roughly how many piano tuners work in the city of Chicago?", "range", (15, 400)),
]


def nums(s):
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", str(s).replace(",", ""))]
def check(kind, target, out):
    s = str(out).lower()
    if kind in ("num", "percent", "trick"):
        return any(abs(n - target) < 1e-6 for n in nums(out))
    if kind == "kw":
        return any(k in s for k in target)
    if kind == "range":
        lo, hi = target; return any(lo <= n <= hi for n in nums(out))
    return False


def baseline(task):
    return llm("You are a solver. Read the problem and output ONLY the final answer (a number if numeric, one short line otherwise).",
               f"Problem: {task}\nFinal answer:", temp=0.2, max_tokens=120)


def brain(task):
    with contextlib.redirect_stdout(io.StringIO()):       # тихо
        return connectome.run(task)


if __name__ == "__main__":
    store = os.path.join(os.path.dirname(__file__), "memory_store.json")   # чистим эпизодическую память
    if os.path.exists(store): os.rename(store, store + ".benchbak")

    print(f"{'type':8} {'truth':>8}  {'BARE 8b':9} {'BRAIN':9}  task")
    print("-" * 90)
    b_ok = r_ok = 0
    rows = []
    for kind, q, ck, tgt in TASKS:
        b = baseline(q); r = brain(q)
        bok = check(ck, tgt, b); rok = check(ck, tgt, r)
        b_ok += bok; r_ok += rok
        rows.append((kind, tgt, bok, rok, q[:40], str(b)[:30], str(r)[:30]))
        print(f"{kind:8} {str(tgt):>8}  {'OK ' if bok else 'xWRONG':9} {'OK ' if rok else 'xWRONG':9}  {q[:46]}")
        print(f"{'':28}bare: {str(b)[:46]!r}")
        print(f"{'':28}brain:{str(r)[:46]!r}")
    n = len(TASKS)
    print("-" * 90)
    print(f"TOTAL   BARE 8b: {b_ok}/{n}    BRAIN: {r_ok}/{n}")

    if os.path.exists(store + ".benchbak"): os.replace(store + ".benchbak", store)  # вернуть память
