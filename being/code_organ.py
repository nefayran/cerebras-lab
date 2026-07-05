#!/usr/bin/env python3
"""ОРГАН-ИСПОЛНИТЕЛЬ: детерминированная рука для кода. Берёт код-кандидат + имя функции + тест-кейсы,
запускает в ОТДЕЛЬНОМ процессе с таймаутом, возвращает (passed, total, first_error). Это объективная
проверка правоты — факт, а не мнение судьи. Тесты задаём мы (определение задачи), код пишет мозг."""
import re, json, subprocess, sys, os, tempfile

PY = sys.executable
HARNESS = '''
{code}

import json
_cases = json.loads({cases!r})
_p, _err = 0, ""
for _args, _exp in _cases:
    try:
        _r = {fn}(*_args)
        if _r == _exp: _p += 1
        elif not _err: _err = "{fn}(%r) -> %r, expected %r" % (tuple(_args), _r, _exp)
    except Exception as _e:
        if not _err: _err = "{fn}(%r) raised %s: %s" % (tuple(_args), type(_e).__name__, _e)
print("RESULT", _p, len(_cases), _err)
'''


def extract_code(text):
    """Выдрать python-код из ответа (```python ... ``` или весь текст)."""
    m = re.search(r"```(?:python)?\s*(.+?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def run_tests(code, fn, cases, timeout=6):
    """(passed, total, first_error). cases: список [[args...], expected]."""
    src = HARNESS.format(code=code, cases=json.dumps(cases), fn=fn)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        out = subprocess.run([PY, path], capture_output=True, text=True, timeout=timeout)
        m = re.search(r"RESULT (\d+) (\d+) ?(.*)", out.stdout, re.S)
        if m:
            return int(m.group(1)), int(m.group(2)), (m.group(3).strip() or out.stderr[-300:])
        return 0, len(cases), (out.stderr[-300:] or "no RESULT")     # код не исполнился (синтаксис и т.п.)
    except subprocess.TimeoutExpired:
        return 0, len(cases), "timeout"
    finally:
        try: os.unlink(path)
        except Exception: pass


if __name__ == "__main__":
    code = "def add(a,b):\n    return a+b"
    print(run_tests(code, "add", [[[1, 2], 3], [[5, 5], 10]]))
