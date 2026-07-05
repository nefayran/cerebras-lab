#!/usr/bin/env python3
"""Проверка: ОДНА динамика mind.py (без роутера) тянет и генеративные, и сходящиеся задачи.
Для сходящихся смотрим, всплыл ли верный ответ в осадке (VERIFY/WRITE сами вскинулись на числах/логике).
Запуск: being/venv/bin/python being/batch_mind.py"""
import io, contextlib
import mind

TASKS = [
    ("gen",  "Write a marketing plan to launch a new specialty cold-brew coffee shop in Tokyo. "
             "Budget is 5000 dollars over 3 months. Cover target audience, channels, budget split, "
             "timeline, and success metrics.", None),
    ("conv", "What is 15% of 240?", "36"),
    ("conv", "A tank holds 480 liters. It is 3/4 full. You drain 90 liters, then add back 1/3 of "
             "what remains. How many liters are in the tank now?", "360"),
    ("conv", "Ann, Bob and Cara each own a different pet: a cat, a dog, or a fish. Ann does not own "
             "the cat. Bob owns the dog. Who owns the fish?", "ann"),
]


def main():
    for kind, task, want in TASKS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doc = mind.run(task)
        trace = buf.getvalue()
        ticks = trace.count("\n  ·")
        silenced = "[silence" in trace
        head = task[:48].replace("\n", " ")
        if kind == "conv":
            ok = want.lower() in str(doc).lower()
            print(f"[{kind}] ticks={ticks:2} silence={silenced!s:5} want={want!r:6} -> {'HIT' if ok else 'MISS'}  | {head}")
        else:
            print(f"[{kind}] ticks={ticks:2} silence={silenced!s:5} words={len(str(doc).split()):4}            | {head}")


if __name__ == "__main__":
    main()
