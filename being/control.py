#!/usr/bin/env python3
"""ВЫУЧЕННЫЙ ГЕЙТ engage/passthrough — интринсивная пластичность управления (не хардкод, не Сонет).
Мозг сам учится из ИСХОДА, когда включать машинерию (engage), а когда не мешать и дать базе ответить
одним проходом (passthrough). ПОЛ: никогда не хуже базы — где машинерия не помогает, отходим.
Контекстный бандит: признаки задачи -> предпочтение действия; награда сдвигает веса. Веса на диске."""
import json, os, re, random

PATH = os.path.join(os.path.dirname(__file__), "control_policy.json")
ACTIONS = ["engage", "passthrough"]   # engage: полная машинерия мозга; passthrough: один проход базы
LR = 0.3
EPS = 0.2


def features(problem):
    p = str(problem).lower()
    return {
        "bias": 1.0,
        "divergent": 1.0 if re.search(r"\b(write|plan|essay|describe|suggest|design|strateg|outline|brainstorm|ideas?|draft|story|letter|email|propose a)\b", p) else 0.0,
        "question": 1.0 if p.strip().endswith("?") else 0.0,
        "numbers": 1.0 if re.search(r"\d", p) else 0.0,
    }


def _load():
    try: return json.load(open(PATH))
    except Exception: return {}
def _save(w): json.dump(w, open(PATH, "w"), indent=1)


def choose(problem, explore=True):
    feats = features(problem); w = _load()
    if explore and random.random() < EPS:
        return random.choice(ACTIONS), feats
    sc = {a: sum(feats[f] * w.get(f, {}).get(a, 0.0) for f in feats) for a in ACTIONS}
    return max(ACTIONS, key=lambda a: sc[a]), feats


def update(feats, action, reward):
    w = _load()
    for f, v in feats.items():
        if v == 0: continue
        w.setdefault(f, {a: 0.0 for a in ACTIONS})
        w[f][action] = w[f].get(action, 0.0) + LR * reward * v
    _save(w)


if __name__ == "__main__":
    for t in ["What is 15% of 240?", "Write a marketing plan for a coffee shop.",
              "Who owns the fish?", "A tank holds 480 liters..."]:
        a, _ = choose(t, explore=False)
        print(f"{a:11} <- {t}")
