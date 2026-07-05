#!/usr/bin/env python3
"""САМО-ОТРАЩИВАНИЕ ОРГАНА. Когда мозг застрял на классе (autoloop пометил NEEDS_ORGAN),
он отращивает себе руку: собирает провальные примеры с эталоном -> Sonnet (claude -p) пишет
ЧИСТУЮ функцию-орган -> ВЕРИФИКАЦИЯ ИСПОЛНЕНИЕМ на эталоне в песочнице -> принять, только если
бьёт порог -> сохранить в organs/<class>.py. Отбор по эталону, не доверие: мусор не проходит.
Запуск: being/venv/bin/python being/grow_organ.py clock"""
import subprocess, re, os, sys, builtins as _b

CLAUDE = "/Users/VKAZERO1/.vscode/extensions/anthropic.claude-code-2.1.175-darwin-arm64/resources/native-binary/claude"
ORGANS = os.path.join(os.path.dirname(__file__), "organs")

# --- кодген-голова: Sonnet через claude -p ---
def ask_sonnet(examples):
    spec = "\n".join(f"INPUT: {q}\nCORRECT OUTPUT: {t}" for q, t in examples)
    prompt = (
        "Write a PURE Python function `def organ(text: str) -> str:` that DETERMINISTICALLY solves this class "
        "of problems. It receives the problem text and returns the answer as a string in the SAME format as "
        "CORRECT OUTPUT. Handle the GENERAL case, not only these examples. Use only the standard library "
        "(re, math, datetime, itertools, collections). No I/O, no network, no printing, no input(). "
        "Output ONLY the function inside ONE ```python code block.\n\nExamples:\n" + spec)
    r = subprocess.run([CLAUDE, "-p", prompt, "--model", "sonnet"], capture_output=True, text=True)
    m = re.search(r"```python\s*\n(.*?)```", r.stdout, re.S) or re.search(r"```\s*\n(.*?)```", r.stdout, re.S)
    return m.group(1) if m else None

# --- песочница ---
FORBID = re.compile(r"\b(open|eval|exec|compile|subprocess|socket|shutil|globals|locals|input|breakpoint)\b|os\.|sys\.|__")
ALLOWED = {"re", "math", "datetime", "itertools", "collections"}
def _safe_import(name, *a, **k):
    if name.split(".")[0] in ALLOWED: return __import__(name, *a, **k)
    raise ImportError(f"blocked import: {name}")
DANGER = {"open", "eval", "exec", "compile", "input", "breakpoint", "__import__", "memoryview", "quit", "exit", "help"}
SAFE = {k: getattr(_b, k) for k in dir(_b) if k not in DANGER}
SAFE["__import__"] = _safe_import

def load_organ(code):
    if FORBID.search(code): raise ValueError("forbidden token in generated code")
    ns = {}
    exec(compile(code, "<organ>", "exec"), {"__builtins__": SAFE}, ns)
    if "organ" not in ns or not callable(ns["organ"]): raise ValueError("no organ()")
    return ns["organ"]

def verify(organ, gen, level, k, kind):
    import autoloop
    ok = 0
    for _ in range(k):
        q, t = gen(level)
        try: r = organ(q)
        except Exception: r = None
        if r is not None and autoloop.check(kind, t, r): ok += 1
    return ok / k

def grow(class_name, level=None, n_examples=5, n_verify=10, thresh=0.85):
    import autoloop
    gen, start = autoloop.CLASSES[class_name]
    level = level or start
    examples = [gen(level) for _ in range(n_examples)]
    code = ask_sonnet(examples)
    if not code: return None, "sonnet returned no code", None
    try: organ = load_organ(code)
    except Exception as e: return None, f"load/sandbox failed: {e}", code
    acc = verify(organ, gen, level, n_verify, class_name)
    if acc >= thresh:
        os.makedirs(ORGANS, exist_ok=True)
        open(os.path.join(ORGANS, "__init__.py"), "a").close()
        with open(os.path.join(ORGANS, f"{class_name}.py"), "w") as f: f.write(code)
        return acc, "ACCEPTED (saved)", code
    return acc, "rejected (below threshold)", code


if __name__ == "__main__":
    cls = sys.argv[1] if len(sys.argv) > 1 else "clock"
    acc, msg, code = grow(cls)
    print(f"organ[{cls}]: acc={acc} -> {msg}")
    if code: print("\n--- generated organ ---\n" + code[:800])
