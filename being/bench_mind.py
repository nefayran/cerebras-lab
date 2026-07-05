#!/usr/bin/env python3
"""СТЕНД mind.py: одна динамика без роутера на батарее разнотипных задач.
- Сходящиеся (есть эталон): повторяем R раз -> hit-rate, среднее число тактов, доля тишины.
- Генеративные: mind vs голая база, судья 0..1 -> кто полнее (мозг не должен быть хуже базы).
Запуск: being/venv/bin/python being/bench_mind.py [R]"""
import sys, io, re, contextlib, random
import mind
from stream import llm, jget

CONV = [
    ("arith-simple",   "What is 15% of 240?", "36"),
    ("arith-composite","A shop had 3 crates of 24 apples each. They sold 17 apples on Monday and twice as "
                       "many on Tuesday. On Wednesday they received 2 more crates of 24. How many apples now?", "69"),
    ("logic-pets",     "Ann, Bob and Cara each own a different pet: a cat, a dog, or a fish. Ann does not "
                       "own the cat. Bob owns the dog. Who owns the fish?", "ann"),
    ("logic-fruit",    "Tom, Sue and Max each picked a different fruit: an apple, a pear, or a plum. Tom did "
                       "not pick the pear or the plum. Sue picked the plum. Which fruit did Tom pick?", "apple"),
    ("clock",          "A train leaves at 8:00. The trip takes 95 minutes. What time does it arrive?", "9:35"),
]
GEN = [
    ("marketing", "Write a marketing plan to launch a new specialty cold-brew coffee shop in Tokyo. Budget is "
                  "5000 dollars over 3 months. Cover target audience, channels, budget split, timeline, metrics."),
    ("email",     "Write a short welcome email for new users of a budgeting app."),
    ("explain",   "Explain why the sky is blue, clearly and completely, for a curious adult."),
]
# Новые КЛАССЫ задач: дизайн, кодинг, проектирование (архитектура), планирование продукта
CLASSES = [
    ("coding",    "Write a Python function is_balanced(s) returning True iff the brackets ()[]{} in s are "
                  "correctly balanced and nested. Handle empty string and non-bracket chars. Include the code."),
    ("design-ui", "Design the home screen UI for a mobile expense-tracking app: name the sections, the visual "
                  "hierarchy, the primary actions, and one key interaction. Be concrete."),
    ("architecture","Design a system architecture for a URL shortener serving 10000 requests/sec: components, "
                  "data store choice, how reads/writes scale, and how short codes are generated without collision."),
    ("product",   "Design a 3-tier pricing plan for a small SaaS note-taking app: name the tiers, the price "
                  "points, which features gate each tier, and the target user for each."),
]


def run_capture(task):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doc = mind.run(task)
    tr = buf.getvalue()
    return str(doc), tr.count("\n  ·"), ("[silence" in tr)


def base(task):
    return llm("You are an expert. Answer the task directly, fully and concretely. No meta, no restating.",
               f"Task: {task}", temp=0.5, max_tokens=900)


def judge(task, out):
    s = jget(llm('Score 0..1: is this a COMPLETE, on-task, useful answer? (1=full real answer, 0=non-answer). '
                 'JSON {"score":0.0}.', f"Task: {task}\nAnswer: {out[:1600]}\nJSON:", temp=0.2), "score")[0]
    try: return max(0.0, min(1.0, float(s)))
    except Exception: return 0.5


PAIR = ('Compare two answers to the same task on completeness, concreteness, correctness and usefulness. '
        'Which is better? Reply "A", "B", or "tie" only. JSON {"winner":"A|B|tie","why":".."}.')
def pairwise(task, mind_out, base_out, seed=0):
    """Попарно: какой ответ лучше. Позицию рандомизируем (анти-bias), мапим обратно к mind/base."""
    swap = (seed % 2 == 1)
    a, b = (base_out, mind_out) if swap else (mind_out, base_out)
    w = jget(llm(PAIR, f"Task: {task}\n\nAnswer A:\n{a[:1500]}\n\nAnswer B:\n{b[:1500]}\nJSON:", temp=0.2), "winner")[0]
    w = str(w or "tie").strip().upper()[:1]
    if w == "A": return "base" if swap else "mind"
    if w == "B": return "mind" if swap else "base"
    return "tie"


def run_gen(label, tasks):
    print(f"\n{label} (mind vs bare base, pairwise A/B, position randomized):")
    for i, (name, task) in enumerate(tasks):
        doc, nt, s = run_capture(task)
        b = base(task)
        w = pairwise(task, doc, b, seed=i)
        flag = {"mind": "MIND wins", "base": "BASE wins", "tie": "tie"}[w]
        print(f"  {name:13} {len(doc.split()):4}w mind / {len(b.split()):4}w base, {nt:2}t  -> {flag}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "3"
    if arg == "classes":                       # только новые классы: дизайн/кодинг/архитектура/продукт
        print("BENCH mind.py — NEW CLASSES vs base\n")
        run_gen("CLASSES", CLASSES); return
    R = int(arg)
    print(f"BENCH mind.py — convergent x{R} repeats + generative vs base\n")
    print("CONVERGENT (one no-router dynamic, organs self-activate):")
    for name, task, want in CONV:
        hits, ticks, sil = 0, [], 0
        for _ in range(R):
            doc, nt, s = run_capture(task)
            if want.lower() in doc.lower(): hits += 1
            ticks.append(nt); sil += int(s)
        avg = sum(ticks) / len(ticks)
        print(f"  {name:16} hit {hits}/{R}   avg_ticks {avg:4.1f}   silence {sil}/{R}   want={want!r}")
    run_gen("GENERATIVE", GEN)


if __name__ == "__main__":
    main()
