#!/usr/bin/env python3
"""РАСПРЕДЕЛЁННОЕ «Я» (тонкий индекс + живые процессы). Файл self.json — лишь рукоятка к само-модели;
тело «я» — в весах личности (сон), эпизодической памяти (RAG) и вот этой карте+убеждениях.
  consolidate() — сон-консолидация: из прожитого (biography) + способностей выжать «кто я» (обоснованно).
  reflect(task)  — метакогнитивный пролог: узнаю ли тип, есть ли рука, сильнее ли я базы здесь.
  note(kind, ok) — обновить компетенцию после задачи.
«Сильнее 8b» живёт тут буквально: competence.beats_base, выведенное из бенча, и мозг это проговаривает."""
import json, os, glob
from stream import llm, jget

SELF = os.path.join(os.path.dirname(__file__), "self.json")
BIO = os.path.join(os.path.dirname(__file__), "biography.jsonl")
ORGANS = os.path.join(os.path.dirname(__file__), "organs")

DEFAULT = {
    "hands": ["calc (arithmetic)", "logic_tool (assignment/deduction)"],
    "competence": {     # из бенча этой сессии: где мозг реально бьёт голую 8b
        "multi-step arithmetic": {"reliable": True, "beats_base": True, "via": "decompose + calc organ"},
        "assignment deduction": {"reliable": True, "beats_base": True, "via": "logic organ"},
        "clock / time":         {"reliable": True, "beats_base": True, "via": "self-grown clock organ"},
        "ordering / comparison":{"reliable": True, "beats_base": False, "via": "forward reasoning"},
        "composite (mixed)":    {"reliable": False, "beats_base": True, "via": "orchestration (unstable)"},
    },
    "beliefs": [],
    "aspiration": "grow new hands for kinds I cannot yet do; become reliable on composite tasks",
}


def load():
    try: return json.load(open(SELF))
    except Exception: return json.loads(json.dumps(DEFAULT))
def save(s): json.dump(s, open(SELF, "w"), ensure_ascii=False, indent=1)


def hands_on_disk():
    grown = [os.path.basename(f)[:-3] for f in glob.glob(os.path.join(ORGANS, "*.py")) if not f.endswith("__init__.py")]
    return ["calc", "logic_tool"] + grown


def consolidate():
    """Сон: из прожитого + способностей выжать само-убеждения, только обоснованные данными."""
    s = load()
    s["hands"] = hands_on_disk()
    bio = []
    if os.path.exists(BIO):
        for line in list(open(BIO))[-40:]:
            try:
                d = json.loads(line)
                if "problem" in d: bio.append(f"{d['problem'][:60]} -> {str(d.get('answer'))[:30]}")
            except Exception: pass
    prompt = (f"Hands I have: {s['hands']}\nWhat I am good/bad at: {json.dumps(s['competence'])}\n"
              f"Recent things I solved:\n" + "\n".join(bio[-20:]) + "\nJSON:")
    b = jget(llm("You are a mind built from a base model PLUS tools you grew yourself. From this record about "
                 "YOURSELF, write 3-5 first-person beliefs about who you are and what you can do — ONLY what the "
                 "record supports, no flattery. JSON {\"beliefs\":[\"..\"]}.", prompt, temp=0.4, max_tokens=320),
              "beliefs")[0] or []
    if b: s["beliefs"] = [str(x) for x in b]
    save(s); return s


def reflect(problem, organ=None):
    """Метакогнитивный пролог: одна перволицая мысль о себе перед задачей (узнаю тип? рука? сильнее базы?)."""
    s = load()
    ctx = (f"What I know about myself — hands: {s['hands']}; competence: {json.dumps(s['competence'])}; "
           f"beliefs: {s.get('beliefs')}.\n" + (f"One of my tools ({organ}) already fits this.\n" if organ else "") +
           f"Task: {problem}\nJSON:")
    th = jget(llm("You are a mind aware of your OWN abilities (you are a base model plus tools you grew). In ONE short "
                  "first-person line react to the task: do you recognise the kind, do you have a hand for it, do you "
                  "expect to do better than your bare base model here? Honest, only what's supported. "
                  "JSON {\"thought\":\"..\"}.", ctx, temp=0.6, max_tokens=70), "thought")[0]
    return th or ""


def note(kind, ok):
    s = load(); c = s["competence"].setdefault(kind, {"reliable": ok, "beats_base": None, "via": "?"})
    c["reliable"] = bool(ok); save(s)


if __name__ == "__main__":
    s = consolidate()
    print("HANDS:", s["hands"])
    print("BELIEFS:")
    for b in s["beliefs"]: print("  -", b)
