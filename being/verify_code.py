#!/usr/bin/env python3
"""ОБЪЕКТИВНОЕ ПРЕВОСХОДСТВО НА КОДЕ: сеть с органом-исполнителем против одного прохода.
- PURE: один думающий проход пишет ВСЕ функции сразу, без обратной связи исполнения.
- BRAIN: каждую функцию пишет отдельный лист, ОРГАН исполняет тесты; падает -> ошибка возвращается
  узлу, он чинит (≤R попыток). Это и есть верификация-рука, которой у одного прохода нет.
Мера: сколько функций прошли тесты (факт, не судья).
Запуск: being/venv/bin/python being/verify_code.py
"""
import re
import code_organ
from stream import llm

R = 3   # попыток починки на функцию

# Определение задачи (спека+тесты — наши; код пишет модель). Функции нетривиальные, легко ошибиться.
SUITE = [
    ("is_valid_ipv4", "Return True iff s is a valid IPv4 address: four parts 0-255 separated by dots, NO "
     "leading zeros (e.g. '01' invalid), no empty parts, no trailing dot.",
     [[["1.1.1.1"], True], [["0.0.0.0"], True], [["255.255.255.255"], True], [["256.0.0.1"], False],
      [["1.1.1"], False], [["01.1.1.1"], False], [["1.1.1.1."], False], [["1..1.1"], False]]),
    ("eval_expr", "Evaluate an integer arithmetic expression string with + - * and parentheses, correct "
     "operator precedence. No division. Return the integer result.",
     [[["2+3*4"], 14], [["(2+3)*4"], 20], [["10-2-3"], 5], [["2*3+4*5"], 26], [["2*(3+4)*5"], 70],
      [["1+2+3"], 6]]),
    ("compress_ranges", "Given a sorted list of unique ints, return list of strings collapsing consecutive "
     "runs: a single n -> 'n', a run a..b -> 'a-b'.",
     [[[[1, 2, 3, 5, 7, 8, 9]], ["1-3", "5", "7-9"]], [[[1]], ["1"]], [[[]], []],
      [[[1, 3, 5]], ["1", "3", "5"]], [[[1, 2, 4, 5]], ["1-2", "4-5"]]]),
    ("title_case", "Title-case s: capitalize each word EXCEPT small words {a,an,the,of,and,in,on,to,for}, "
     "but ALWAYS capitalize the first word. Words are space-separated, input lowercase.",
     [[["the lord of the rings"], "The Lord of the Rings"], [["a tale of two cities"], "A Tale of Two Cities"],
      [["to be or not to be"], "To Be Or Not to Be"], [["of mice and men"], "Of Mice and Men"]]),
    ("add_binary", "Add two binary strings a and b, return their sum as a binary string (no leading zeros, "
     "except '0' itself).",
     [[["11", "1"], "100"], [["0", "0"], "0"], [["1010", "1011"], "10101"], [["1", "1"], "10"],
      [["110", "11"], "1001"]]),
    ("spreadsheet_col", "Convert a spreadsheet column label to its number: A=1, Z=26, AA=27, AZ=52, BA=53.",
     [[["A"], 1], [["Z"], 26], [["AA"], 27], [["AZ"], 52], [["BA"], 53], [["ZZ"], 702]]),
]

WRITE = ("Write a single correct Python function named `{fn}` for this spec. Output ONLY a ```python code "
         "block with the function (and any helpers). No explanation.\nSpec: {spec}")
REPAIR = ("Your function `{fn}` failed a test:\n{err}\nFix the bug. Output ONLY a ```python code block with "
          "the corrected function.\nSpec: {spec}")


SYS = "Write correct Python. Output ONLY one ```python code block."
def brain_solve(fn, spec, cases):
    """Лист пишет функцию, орган исполняет тесты, падает -> чинит по ошибке (≤R)."""
    code = code_organ.extract_code(llm(SYS, WRITE.format(fn=fn, spec=spec), temp=0.4, think=True))
    for _ in range(R):
        p, n, err = code_organ.run_tests(code, fn, cases)
        if p == n:
            return p, n
        code = code_organ.extract_code(
            llm(SYS, REPAIR.format(fn=fn, spec=spec, err=err), temp=0.5, think=True))
    p, n, _ = code_organ.run_tests(code, fn, cases)
    return p, n


def pure_solve_all():
    """Один проход пишет ВСЕ функции сразу, без исполнения."""
    spec = "\n".join(f"- {fn}: {s}" for fn, s, _ in SUITE)
    out = llm("Write correct Python. Output one ```python code block with ALL the functions.",
              f"Implement these functions:\n{spec}\n\nCode:", temp=0.4, think=True)
    return code_organ.extract_code(out)


def main():
    print("OBJECTIVE: brain (write+execute+repair) vs pure (one pass) — functions passing tests\n")
    pure_code = pure_solve_all()
    bp = bt = pp = pt = 0
    for fn, spec, cases in SUITE:
        b_p, b_n = brain_solve(fn, spec, cases)
        p_p, p_n, _ = code_organ.run_tests(pure_code, fn, cases)
        bp += (b_p == b_n); pp += (p_p == p_n); bt += 1; pt += 1
        print(f"  {fn:18} brain {b_p}/{b_n} {'OK' if b_p==b_n else 'FAIL':4} | "
              f"pure {p_p}/{p_n} {'OK' if p_p==p_n else 'FAIL'}")
    print(f"\n  FUNCTIONS FULLY CORRECT:  brain {bp}/{bt}   |   pure {pp}/{pt}")


if __name__ == "__main__":
    main()
