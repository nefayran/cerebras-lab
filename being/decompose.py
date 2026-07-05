#!/usr/bin/env python3
"""
МОЗГ КАК ДЕКОМПОЗИЦИЯ. Малые ЛЛМ = области мозга. Задача рекурсивно дробится на куски
настолько мелкие, что 8b берёт каждый уверенно; области решают кусочки; COMPOSER собирает
вверх. Сила — в дроблении+сборке, а не в уме узла. Узлов много (вызов на каждый кусок).

Тест: то же 8b на ЦЕЛОЙ задаче (baseline, один кусок) vs декомпозиция. Если дроблёное берёт
там, где целое ошибается — тезис («малые модели+структура») доказан.

Запуск: being/venv/bin/python being/decompose.py
"""
import json, re, ast, operator, urllib.request

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _ev(n):
    if isinstance(n, ast.Constant): return n.value
    if isinstance(n, ast.BinOp): return _OPS[type(n.op)](_ev(n.left), _ev(n.right))
    if isinstance(n, ast.UnaryOp): return _OPS[type(n.op)](_ev(n.operand))
    raise ValueError("bad expr")


def calc(expr):
    """Детерминированная область-КАЛЬКУЛЯТОР: точная арифметика, без 8b."""
    expr = expr.replace("×", "*").replace("÷", "/").strip()
    m = re.search(r"[-+*/().\d\s]+", expr)
    v = _ev(ast.parse(m.group(0).strip(), mode="eval").body)
    return int(v) if isinstance(v, float) and v.is_integer() else (round(v, 4) if isinstance(v, float) else v)

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
MAX_DEPTH = 2
CALLS = [0]


def llm(system, user, temp=0.3, max_tokens=200):
    CALLS[0] += 1
    body = json.dumps({"model": MODEL, "system": system, "prompt": user, "stream": False,
                       "think": False, "options": {"temperature": temp, "num_predict": max_tokens}}).encode()
    try:
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())["response"]
        return (out.split("</think>")[-1] if "</think>" in out else out).strip()
    except Exception as e:
        return f"(silent:{e})"


def jget(s, key):
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try: return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0))).get(key)
        except Exception: pass
    return None


DECOMPOSER = (
    "You are the DECOMPOSER region of a brain. Given a task, decide: is it ATOMIC (one tiny step a weak "
    "calculator can do in a single shot), or should it be SPLIT into smaller ordered sub-steps? Split so "
    "each sub-step is much simpler and self-contained; later steps may use 'the result of step k'. "
    "Reply ONLY JSON: {\"atomic\": true/false, \"subtasks\": [\"...\",\"...\"]} (subtasks empty if atomic).")

SOLVER = ("You are the SOLVER region. You DECIDE the calculation but do NOT compute it yourself. "
          "Using the known facts and the task, write the SINGLE arithmetic expression that answers this step, "
          "with concrete numbers (a calculator region will evaluate it). "
          "Reply ONLY JSON {\"expr\":\"...\"}, e.g. {\"expr\":\"72 - 51\"} or {\"expr\":\"18 * 5\"}. "
          "If the step is not a calculation, put the value in expr.")

COMPOSER = ("You are the COMPOSER region. Combine the sub-results into the final answer as a SINGLE arithmetic "
            "expression with their concrete numbers (a calculator evaluates it). Reply ONLY JSON {\"expr\":\"...\"}.")


def solver_value(ctx, task):
    raw = llm(SOLVER, f"{ctx}Task: {task}\nJSON:", temp=0.1)
    expr = jget(raw, "expr")
    if expr is not None:
        try: return str(calc(str(expr)))
        except Exception: return str(expr).strip()
    return raw.strip()[:30]


def _ctx(prior):
    return ("" if not prior else "Facts already known:\n" + "\n".join(f"- {q} = {r}" for q, r in prior) + "\n\n")


def solve(task, prior=None, depth=0, indent=""):
    """task — ЧИСТАЯ подзадача; prior — результаты соседей этого уровня (отдельный контекст, не в task)."""
    prior = prior or []
    print(f"{indent}• {task[:64]}")
    ctx = _ctx(prior)
    if depth >= MAX_DEPTH:
        ans = solver_value(ctx, task)
        print(f"{indent}  = {ans[:40]}"); return ans
    d = llm(DECOMPOSER, f"{ctx}Task: {task}")
    if jget(d, "atomic") or not (jget(d, "subtasks") or []):
        ans = solver_value(ctx, task)
        print(f"{indent}  = {ans[:40]}"); return ans
    results = []
    for st in jget(d, "subtasks"):
        r = solve(str(st), results, depth + 1, indent + "    ")   # st ЧИСТАЯ; prior=соседи передаётся аргументом
        results.append((str(st)[:48], r))
    summary = "\n".join(f"- {q} = {r}" for q, r in results)
    raw = llm(COMPOSER, f"Task: {task}\n\nSub-results in order:\n{summary}\n\nJSON:", temp=0.1)
    expr = jget(raw, "expr")
    final = str(calc(str(expr))) if expr is not None and re.search(r"\d", str(expr)) else str(expr or raw).strip()
    try:
        final = str(calc(str(expr)))
    except Exception:
        final = (results[-1][1] if results else raw).strip() if isinstance(final, str) else final
    print(f"{indent}  ⇒ compose = {str(final)[:40]}")
    return final


def baseline(task):
    return llm("You are a solver. Read the problem and output ONLY the final answer (a number if numeric).",
               f"Problem: {task}\nFinal answer:", max_tokens=120)


PROBLEMS = [
    ("A shop had 3 crates of 24 apples each. They sold 17 apples on Monday and twice as many on Tuesday. "
     "On Wednesday they received 2 more crates of 24. How many apples does the shop have now?", 69),
    ("Sarah reads 18 pages per day for 5 days, then 25 pages per day for 3 days. Her book has 200 pages. "
     "How many pages are left to read?", 35),
    ("A tank holds 480 liters. It is 3/4 full. You drain 90 liters, then add back 1/3 of what remains. "
     "How many liters are in the tank now?", 360),
]


def check(out, ans):
    nums = re.findall(r"-?\d+", out.replace(",", ""))
    return str(ans) in nums


if __name__ == "__main__":
    for task, ans in PROBLEMS:
        print("=" * 70)
        print(f"PROBLEM (true answer = {ans}): {task[:64]}...")
        CALLS[0] = 0
        b = baseline(task)
        bcalls = CALLS[0]
        print(f"\n[BASELINE — whole task, 1 region call] -> {b[:60]}  {'✓' if check(b, ans) else '✗ WRONG'}")
        print(f"\n[BRAIN — decomposition tree]:")
        CALLS[0] = 0
        br = solve(task)
        print(f"\n[BRAIN] -> {br[:60]}  {'✓' if check(br, ans) else '✗ WRONG'}  ({CALLS[0]} region calls vs {bcalls} baseline)")
        print()
