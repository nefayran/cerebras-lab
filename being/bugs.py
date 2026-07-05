#!/usr/bin/env python3
"""
РОЙ ЖУЧКОВ (bugs.py): программа-файл собирается роем. Каждый ЖУЧОК отвечает РОВНО за одну строку,
видит только строку СВЕРХУ, строку СНИЗУ и общую задачу — больше ничего. За один ПРОХОД каждый жучок
переписывает свою строку (соседей берёт из снимка прошлого прохода -> жучки работают параллельно).
Проходы идут, пока строки не перестанут меняться (рой устаканился). Потом пробуем исполнить.

Никакого глобального автора: каждый жучок локален (±1 строка + задача). Связность — эмерджентна.
Запуск: being/venv/bin/python being/bugs.py ["задача"]
"""
import sys, re
from concurrent.futures import ThreadPoolExecutor
from stream import llm, jget
import viz, code_organ

SWEEPS = 6
THINK = False   # массовые вызовы (посев, широкий проход) — быстрые; думают ТОЧЕЧНО (вестник/виновный/матка)

# Задача роя + ТЕСТЫ (ground truth): орган исполняет, провал = «боль», возвращается жучкам.
DEFAULT = {
    "task": ("Write a Python function fizzbuzz(n) that returns a LIST of strings for 1..n: 'Fizz' if "
             "divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if both, else the number as a string."),
    "fn": "fizzbuzz",
    "cases": [[[5], ["1", "2", "Fizz", "4", "Buzz"]], [[6], ["1", "2", "Fizz", "4", "Buzz", "Fizz"]],
              [[15], ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11",
                      "Fizz", "13", "14", "FizzBuzz"]]],
}

SEED = ("Write a correct Python program for the task, EXPANDED: ONE simple statement per line, NO one-liners, "
        "NO list/dict comprehensions, NO semicolons — write each step on its own line so it reads as many "
        "small lines. Output ONLY the code, no fences, no comments, no explanation.")
BUG = ("You are one BUG responsible for EXACTLY ONE line of a Python program. You see ONLY the line above "
       "you, the line below you, and the overall task. If YOUR current line is already correct and fits "
       "between its neighbours, output it UNCHANGED. Only fix it if it is wrong for this position. Keep it "
       "to ONE statement (do NOT cram the whole program into your line). PRESERVE the exact leading "
       "indentation appropriate here. Output ONLY your single line — no fences, no commentary, no extra lines.")
# ЖУЧОК-ВЕСТНИК: видит провал и весь файл, находит ВИНОВНУЮ строку и доносит адресную боль её жучку.
DIAGNOSE = ("A program fails a test. Given the NUMBERED lines and the FAILURE, find the ONE line MOST "
            "responsible for the wrong behaviour and say SPECIFICALLY what that line must do to fix it. "
            'JSON {"line": <index int>, "fix": "<specific instruction for that one line>"}.')


def messenger(task, lines, err):
    """Вестник локализует боль: (индекс виновной строки, конкретная инструкция) или (None, '')."""
    numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(lines))
    li, fix = jget(llm(DIAGNOSE, f"TASK: {task}\n\nPROGRAM:\n{numbered}\n\nFAILURE:\n{err}\n\nJSON:",
                       temp=0.3, max_tokens=200, think=True), "line", "fix")   # вестник ДУМАЕТ о потоке
    try: li = int(li)
    except Exception: li = None
    return (li if (li is not None and 0 <= li < len(lines)) else None), str(fix or "")


# МАТКА: когда рабочие застряли, выпускает нового жучка на НОВУЮ строку (вставка) или убирает лишнюю.
QUEEN = ("A program fails a test and local line-edits are stuck — it likely needs a STRUCTURAL change. "
         "Propose ONE move: INSERT a new line, or DELETE a line. For insert, give the new line WITH correct "
         "leading indentation and the index it goes BEFORE. "
         'JSON {"action":"insert" or "delete","line":<index int>,"content":"<new line with indentation>"}.')


def queen(task, lines, err):
    """Структурный ход: (action, index, content) или (None, None, None)."""
    numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(lines))
    act, li, content = jget(llm(QUEEN, f"TASK: {task}\n\nPROGRAM:\n{numbered}\n\nFAILURE:\n{err}\n\nJSON:",
                                temp=0.4, max_tokens=200, think=True), "action", "line", "content")   # матка думает
    act = str(act or "").strip().lower()
    try: li = int(li)
    except Exception: li = None
    if act not in ("insert", "delete") or li is None or not (0 <= li <= len(lines)):
        return None, None, None
    return act, li, str(content or "")


def seed_lines(task):
    code = llm(SEED, f"Task: {task}\n\nCode:", temp=0.4, max_tokens=500, think=THINK)
    code = re.sub(r"```(?:python)?|```", "", code)
    return [ln for ln in code.split("\n")]


def beetle(task, above, cur, below, fail="", think=False):
    """Один жучок переписывает свою строку, видя соседей, задачу И «боль» (провал теста, если есть)."""
    pain = (f"\n\nTHE WHOLE PROGRAM CURRENTLY FAILS A TEST:\n{fail}\nIf YOUR line is part of this bug, fix it; "
            f"otherwise keep your line unchanged.") if fail else ""
    ctx = (f"TASK: {task}\n\nLINE ABOVE YOU:\n{above}\n\nYOUR CURRENT LINE:\n{cur}\n\n"
           f"LINE BELOW YOU:\n{below}{pain}\n\nYour single line:")
    out = llm(BUG, ctx, temp=0.5, max_tokens=120, think=think)
    out = re.sub(r"```(?:python)?|```", "", out)
    first = next((l for l in out.split("\n") if l.strip()), cur)   # жучок владеет ОДНОЙ строкой
    indent = re.match(r"\s*", cur).group(0)                        # отступ — «слот», наследуем от строки
    return indent + first.strip()


def sweep(task, lines, fail=""):
    """Один проход роя: все жучки параллельно, соседей берут из СНИМКА прошлого состояния."""
    def one(i):
        above = lines[i - 1] if i > 0 else "(top of file)"
        below = lines[i + 1] if i < len(lines) - 1 else "(end of file)"
        return beetle(task, above, lines[i], below, fail)
    with ThreadPoolExecutor(max_workers=min(8, len(lines))) as ex:
        return list(ex.map(one, range(len(lines))))


def try_run(code):
    """Пробуем исполнить собранное (синтаксис + не падает на импорте). (ok, err)."""
    import subprocess, tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\nprint('OK_RUN')"); p = f.name
    try:
        r = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=6)
        return ("OK_RUN" in r.stdout), (r.stderr.strip()[-200:])
    except Exception as e:
        return False, str(e)
    finally:
        try: os.unlink(p)
        except Exception: pass


def run(spec, seed_text=None):
    task, fn, cases = spec["task"], spec["fn"], spec["cases"]
    print("=" * 72); print(f"TASK: {task}\n\nBUG SWARM + EXECUTION FEEDBACK (one bug/line, pain from failed tests):\n")
    viz.begin(); viz.emit("start", problem=task)
    lines = seed_text.split("\n") if seed_text else seed_lines(task)   # можно дать кривой посев
    viz.emit("seed", lines=lines)
    print("  seed:\n" + "\n".join(f"   {i:2} | {l}" for i, l in enumerate(lines)) + "\n")
    # ОТЖИГ: рабочее состояние МОЖЕТ временно просесть (ход вестника принимаем, даже если хуже —
    # доверяем диагнозу и ныряем в яму), но «лучшее за всё время» хранится как пол и возвращается.
    # Прочие правки/матка — только не-хуже-рабочего. Так колония переходит долину без риска итогу.
    cur_p, n, err = code_organ.run_tests("\n".join(lines), fn, cases)
    best_lines, best_p, stall = list(lines), cur_p, 0
    for s in range(SWEEPS):
        print(f"  sweep {s}: cur {cur_p}/{n}  best {best_p}/{n}" + ("" if cur_p == n else f"  pain: {err[:60]}"))
        viz.emit("test", sweep=s, passed=cur_p, total=n)
        if best_p == n:
            print("  [colony converged: all tests pass]"); break
        # ТОЧЕЧНО: вестник указал виновную строку -> правим ТОЛЬКО её (без широкого прохода по всем).
        li, fix = messenger(task, lines, err)
        changed = []
        if li is not None:
            print(f"        messenger -> line {li}: {fix[:66]}")
            viz.emit("message", sweep=s, line=li, fix=fix[:80])
            above = lines[li - 1] if li > 0 else "(top of file)"
            below = lines[li + 1] if li < len(lines) - 1 else "(end of file)"
            prop = beetle(task, above, lines[li], below, fix, think=True)
            if prop != lines[li]:
                cand = list(lines); cand[li] = prop
                cp, _, ce = code_organ.run_tests("\n".join(cand), fn, cases)
                keep = cp >= cur_p or stall == 0          # ОТЖИГ: первый раз доверяем диагнозу даже в яму
                viz.emit("bug", sweep=s, line=li, text=(prop if keep else lines[li])[:80], changed=keep)
                if keep:
                    lines[li], cur_p, err = prop, cp, ce; changed.append(li)
        if not changed:                                                 # рабочие никак -> МАТКА (структурный ход)
            act, k, content = queen(task, lines, err)
            if act:
                cand = list(lines)
                if act == "insert": cand.insert(k, content)
                else: cand.pop(k)
                cp, _, ce = code_organ.run_tests("\n".join(cand), fn, cases)
                if cp >= cur_p and cand != lines:
                    print(f"        QUEEN {act} @ {k}: {content[:56]!r} -> {cp}/{n}")
                    lines, cur_p, err = cand, cp, ce
                    viz.emit("queen", sweep=s, action=act, line=k, content=content[:80], lines=lines); changed = [k]
                else:
                    print(f"        QUEEN {act} @ {k} -> {cp}/{n} reject")
        if cur_p > best_p:                                              # новый рекорд -> запомнить пол
            best_lines, best_p, stall = list(lines), cur_p, 0
        else:
            stall += 1
        if stall >= 2 and cur_p < best_p:                              # яма не окупилась -> вернуться к лучшему
            print(f"        (anneal reset: cur {cur_p} < best {best_p}, restore best)")
            lines, cur_p, _ , err = best_lines[:], best_p, None, code_organ.run_tests("\n".join(best_lines), fn, cases)[2]
            stall = 0
        viz.emit("frame", sweep=s, lines=lines, changed=changed)
        if not changed and cur_p == best_p:
            print("  [colony stuck: no accepted change and at best]"); break
    lines = best_lines                                                  # итог = лучшее за всё время
    code = "\n".join(lines)
    p, n, _ = code_organ.run_tests(code, fn, cases)
    print(f"\n  => PROGRAM ({len(lines)} lines, tests {p}/{n}):\n\n{code}\n")
    viz.emit("answer", value=code[:4000], runs=(p == n)); viz.end()
    return code, p, n


if __name__ == "__main__":
    run(DEFAULT)
