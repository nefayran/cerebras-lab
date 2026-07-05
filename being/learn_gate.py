#!/usr/bin/env python3
"""ОБУЧЕНИЕ ГЕЙТА engage/passthrough из ИСХОДА. Мозг пробует оба действия на смешанных задачах:
сходящиеся (эталон) и открытые (судья по полноте ответа). Награда сдвигает политику. Должен выучить:
сходящиеся -> engage (органы дают верно), открытые -> passthrough (база даёт реальный ответ, машинерия — нет).
Запуск: being/venv/bin/python being/learn_gate.py [rounds]"""
import sys, io, re, contextlib, os
import connectome, control
from stream import llm, jget

TASKS = [
    ("conv", "A tank holds 480 liters. It is 3/4 full. You drain 90 liters, then add back 1/3 of what remains. How many liters are in the tank now?", 360),
    ("conv", "Ann, Bob and Cara each own a different pet: a cat, a dog, or a fish. Ann does not own the cat. Bob owns the dog. Who owns the fish?", "ann"),
    ("conv", "What is 15% of 240?", 36),
    ("gen", "Write a marketing plan to launch a cold-brew coffee shop in Tokyo, 5000 dollars over 3 months.", None),
    ("gen", "Write a short welcome email for new users of a budgeting app.", None),
]


def runq(task, action):
    with contextlib.redirect_stdout(io.StringIO()):
        return str(connectome.run(task, force_action=action))


def judge(task, out):
    s = jget(llm('Score 0..1: is this a COMPLETE, on-task, useful answer to the task? '
                 '(1 = full real answer, 0 = non-answer / restating / off-topic). JSON {"score":0.0}.',
                 f"Task: {task}\nAnswer: {out[:1500]}\nJSON:", temp=0.2), "score")[0]
    try: return max(0.0, min(1.0, float(s)))
    except Exception: return 0.5


def reward(kind, truth, task, out):
    if kind == "conv":
        s = out.lower()
        if isinstance(truth, str):
            return 1.0 if truth in s else -1.0
        nums = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
        return 1.0 if any(abs(float(n) - truth) < 1e-6 for n in nums) else -1.0
    return judge(task, out) * 2 - 1     # 0..1 -> -1..1


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    pol = os.path.join(os.path.dirname(__file__), "control_policy.json")
    if os.path.exists(pol): os.remove(pol)        # учимся с нуля
    print("LEARNING GATE engage/passthrough\n")
    for rnd in range(rounds):
        for kind, task, truth in TASKS:
            for action in ("engage", "passthrough"):
                out = runq(task, action)
                r = reward(kind, truth, task, out)
                control.update(control.features(task), action, r)
                print(f"  r{rnd} {kind:4} {action:11} reward {r:+.2f}  | {task[:38]} -> {out[:34]!r}")
        print()
    print("LEARNED CHOICES:")
    for _, t, _ in TASKS:
        a, _ = control.choose(t, explore=False)
        print(f"  {a:11} <- {t[:50]}")


if __name__ == "__main__":
    main()
