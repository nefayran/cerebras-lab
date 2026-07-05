#!/usr/bin/env python3
"""
МОЗГ КАК СЕТЬ ОБЛАСТЕЙ -> ПОТОК СОЗНАНИЯ. Не один умный промпт, а много однофункциональных
областей (каждая = отдельный вызов qwen3:8b с КРОШЕЧНЫМ промптом про СВОЮ функцию, без подсказок
как решать конкретную задачу). Сложность — в их связях: ДЕРЕВО (префронтальная кора дробит цель
на подцели -> рекурсия) и ПЕТЛИ (поясная кора ACC ловит ошибку -> назад в исполнение).

Аналогии областей:
  SALIENCE  (островок/ACC)      — первичная оценка ("большая задача")
  PERCEIVE  (зрит. кора+Вернике)— что тут БУКВАЛЬНО написано (факты), без выдумки
  DECIDE    (префронтальная)    — атомарно или дробить на подвопросы (ДЕРЕВО)
  OPERATE   (префронт./баз.ганг)— исполнить один шаг; арифметика -> выражение
  calc      (мозжечок)          — точная рука, детерминированная арифметика
  MONITOR   (ACC)               — ошибка/конфликт? -> ПЕТЛЯ назад в OPERATE
  COMBINE   (префронтальная)    — собрать результаты подцелей в ответ родителя
  SPEAK     (Брока/DMN)         — озвучить шаг внутренней речью -> строка ПОТОКА
  REWARD    (дофамин/стриатум)  — чувство в конце

Рабочий стол (global workspace) = notes (рабочая память) + поток озвучек.
Запуск: being/venv/bin/python being/stream.py
"""
import json, re, ast, operator, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

import os
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
MAX_DEPTH = 2
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge.json")


def load_kb():
    try:
        with open(KB_PATH) as f: return json.load(f)
    except Exception: return []
def save_kb(kb):
    with open(KB_PATH, "w") as f: json.dump(kb[-200:], f, ensure_ascii=False, indent=1)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
def _ev(n):
    if isinstance(n, ast.Constant): return n.value
    if isinstance(n, ast.BinOp): return _OPS[type(n.op)](_ev(n.left), _ev(n.right))
    if isinstance(n, ast.UnaryOp): return _OPS[type(n.op)](_ev(n.operand))
    raise ValueError
def calc(expr):
    """МОЗЖЕЧОК: точная арифметика (8b ненадёжен даже на одном вычитании)."""
    expr = str(expr).replace("×", "*").replace("÷", "/")
    m = re.search(r"[-+*/().\d\s]+", expr)
    v = _ev(ast.parse(m.group(0).strip(), mode="eval").body)
    return int(v) if isinstance(v, float) and v.is_integer() else (round(v, 4) if isinstance(v, float) else v)


def llm(system, user, temp=0.4, max_tokens=130, think=False):
    """БЕЗ ОГРАНИЧИТЕЛЕЙ: длина не лимитируется (num_predict=-1), таймаут вызова максимальный.
    max_tokens оставлен в сигнатуре для совместимости, но НЕ применяется."""
    body = json.dumps({"model": MODEL, "system": system, "prompt": user, "stream": False,
                       "think": think, "options": {"temperature": temp, "num_predict": -1}}).encode()
    try:
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.loads(r.read())["response"]
        return (out.split("</think>")[-1] if "</think>" in out else out).strip()
    except Exception as e:
        return f"(silent:{e})"
def jget(s, *keys):
    m = re.search(r"\{.*\}", s, re.S); d = {}
    if m:
        try: d = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except Exception: d = {}
    return [d.get(k) for k in keys]


# --- области: каждый промпт описывает ТОЛЬКО свою функцию, не как решать задачу ---
SALIENCE = ('You just glanced at a whole problem. One short first-person gut reaction. JSON {"thought":"..."}.')
# СЛОЙ 1 — сеть перцепторов: каждый узел ловит свой признак (как колонки коры), потом связывание
PERCEIVERS = {
    "entities":   'List ONLY the entities/objects named in this text. JSON {"facts":["..",".."]}.',
    "quantities": 'List ONLY the numbers in this text and what each one counts. JSON {"facts":["..",".."]}.',
    "relations":  'List ONLY the relations or actions between things in this text. JSON {"facts":["..",".."]}.',
    "goal":       'State ONLY what this text is ultimately asking for. JSON {"facts":["..",".."]}.',
}
RECALL   = ('What do you already know that helps answer this? Give concrete facts or reasonable estimates with rough '
            'numbers (e.g. a city size, a typical rate). JSON {"facts":["..",".."]}.')


class Workspace:
    """Общий рабочий стол (Global Workspace): все области ЧИТАЮТ и ПИШУТ сюда. Это и делает
    структуру коннектомом, а не конвейером — память/личность подмешиваются по ходу, есть рекуррентность."""
    def __init__(self, facts):
        self.facts = list(facts); self.notes = []
    def remember(self, new):
        have = {str(x).strip().lower() for x in self.facts}
        added = []
        for f in new:
            k = str(f).strip().lower()
            if k and k not in have:
                have.add(k); self.facts.append(str(f).strip()); added.append(str(f).strip())
        return added
    def factstr(self): return "; ".join(str(f) for f in self.facts)


def recall_into(ws, goal):
    """Top-down связь ACC->ПАМЯТЬ: вспомнить/прикинуть под текущую подцель и влить в рабочий стол."""
    return ws.remember(jget(llm(RECALL, f"Question: {goal}\nJSON:"), "facts")[0] or [])


def perceive(problem):
    """Сеть параллельных перцепторов -> объединённый перцепт (связывание)."""
    def one(item):
        name, prompt = item
        return jget(llm(prompt, f'Text: "{problem}"\nJSON:', temp=0.2), "facts")[0] or []
    facts, seen = [], set()
    with ThreadPoolExecutor(max_workers=len(PERCEIVERS)) as ex:
        for fl in ex.map(one, PERCEIVERS.items()):
            for f in fl:
                k = str(f).strip().lower()
                if k and k not in seen:
                    seen.add(k); facts.append(str(f).strip())
    return facts
SPLIT    = ('Break this question into the 2-4 smaller questions you would ask YOURSELF, in order, to work it out. '
            'If it truly cannot be broken down, give []. JSON {"subs":["..",".."]}.')
OPERATE  = ('Answer this one small question from the facts. If arithmetic, give the expression; else the result. '
            'JSON {"expr":".. or \\"\\"","result":".. or \\"\\""}.')
MONITOR  = ('Does this result answer the question and fit the facts? If not, say briefly what to fix. '
            'JSON {"ok":true/false,"fix":".."}.')
COMBINE  = ('You answered these sub-questions. Combine their results to answer the parent. If arithmetic, give the '
            'expression; else the result. JSON {"expr":".. or \\"\\"","result":".. or \\"\\""}.')
SPEAK    = ('Say this step of your thinking as ONE short first-person thought (inner voice). JSON {"thought":"..."}.')
REWARD   = ('You finished with the answer. One short first-person feeling. JSON {"thought":"..."}.')


def factstr(facts): return "; ".join(str(f) for f in (facts or []))
def say(text, indent): print(f"{indent}{text}")
def voice(moment, indent, temp=0.85):
    """Голос рождается узлом SPEAK, а не зашит. moment — лишь когнитивный повод, не текст реплики."""
    th = jget(llm(SPEAK, f"{moment}\nJSON:", temp=temp, max_tokens=40), "thought")[0]
    if th and str(th).strip() not in ("", "None"): say(str(th), indent)
    return th


def resolve_atomic(goal, ws, indent):
    """Один шаг + петля ACC: исполнить -> проверить -> если конфликт: рекрутировать ПАМЯТЬ (top-down) и назад."""
    ctx = ("Notes:\n" + "\n".join(f"- {q} = {v}" for q, v in ws.notes) + "\n\n") if ws.notes else ""
    hint, val, expr = "", None, ""
    for attempt in range(3):
        expr, result = jget(llm(OPERATE, f"{ctx}Facts: {ws.factstr()}\nQuestion: {goal}\n{hint}JSON:"),
                            "expr", "result")
        if expr and re.search(r"\d", str(expr)):
            try: val = calc(expr)
            except Exception: val = result or expr
        else:
            val = result if result not in (None, "") else expr
        ok, fix = jget(llm(MONITOR, f"Question: {goal}\nResult: {val}\nFacts: {ws.factstr()}\nJSON:"), "ok", "fix")
        if ok is not False or not fix:
            break
        voice(f"You sense something is off here: {fix}", indent)   # ПЕТЛЯ ACC, голос рождается
        added = recall_into(ws, goal)             # top-down: ACC рекрутирует ПАМЯТЬ
        if added: voice(f"You reach into memory and recall: {factstr(added)[:150]}", indent)
        hint = f"On reflection: {fix}\n"
    return val, expr


def solve(goal, ws, depth=0, indent="  "):
    subs = [str(s) for s in (jget(llm(SPLIT, f"Question: {goal}\nFacts: {ws.factstr()}\nJSON:"), "subs")[0] or [])]

    if depth >= MAX_DEPTH or len(subs) < 2:
        val, expr = resolve_atomic(goal, ws, indent)
        th = jget(llm(SPEAK, f"Question: {goal}\nFacts: {ws.factstr()}\nAnswer: {val}\nJSON:", temp=0.7), "thought")[0]
        say(f"{th}", indent)
        if expr and re.search(r"[-+*/]", str(expr)): say(f"  ({expr} = {val})", indent)
        return val

    say(jget(llm(SPEAK, f"I need to break this down: {goal}\nJSON:", temp=0.7), "thought")[0], indent)
    results = []
    for i, sub in enumerate(subs):
        if i: voice("You finished that sub-thought and turn to what comes next.", indent)
        r = solve(sub, ws, depth + 1, indent + "    ")
        results.append((sub, r))
        ws.notes.append((sub[:50], r))
    summary = "\n".join(f"- {q} = {r}" for q, r in results)
    expr, result = jget(llm(COMBINE, f"Parent question: {goal}\nSub-answers:\n{summary}\nJSON:"), "expr", "result")
    if expr and re.search(r"\d", str(expr)):
        try: val = calc(expr)
        except Exception: val = result or expr
    else:
        val = result if result not in (None, "") else expr
    if val in (None, "", "None"):     # сборка пустая -> взять самый содержательный под-ответ
        val = max((str(r) for _, r in results if r not in (None, "", "None")), key=len, default="")
    voice("You now hold your sub-answers and gather them to reach the answer.", indent)
    if expr and re.search(r"[-+*/]", str(expr)): say(f"  ({expr} = {val})", indent)
    return val


def run(problem):
    print("=" * 72)
    print(f"TASK: {problem}\n\nSTREAM OF CONSCIOUSNESS:\n")
    say(jget(llm(SALIENCE, problem, temp=0.9, max_tokens=40), "thought")[0], "  ")
    kb = load_kb()
    perceived = perceive(problem)                       # СЛОЙ 1 — сеть перцепторов
    recalled = jget(llm(RECALL, f"Question: {problem}\nJSON:"), "facts")[0] or []
    known = [k for k in kb if any(w in k.lower() for w in re.findall(r"[a-z]{5,}", problem.lower()))]
    if recalled: voice(f"You bring to mind what you already know: {factstr(recalled)[:200]}", "  ")
    ws = Workspace(perceived + recalled + known)        # общий рабочий стол
    answer = solve(problem, ws, 0, "  ")
    say(jget(llm(REWARD, f"The answer is {answer}.", temp=0.9, max_tokens=40), "thought")[0], "  ")
    print(f"\n  => ANSWER: {answer}\n")
    kb.append(f"{problem.strip()} -> {answer}"); save_kb(kb)   # память растёт
    return answer


if __name__ == "__main__":
    P = ("A shop had 3 crates of 24 apples each. They sold 17 apples on Monday and twice as many on Tuesday. "
         "On Wednesday they received 2 more crates of 24. How many apples does the shop have now?")
    run(sys.argv[1] if len(sys.argv) > 1 else P)
