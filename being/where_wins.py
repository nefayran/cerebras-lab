#!/usr/bin/env python3
"""ГДЕ СЕТЬ БЬЁТ ЧИСТУЮ МОДЕЛЬ. Сравниваем на высоковариативных задачах:
  - net  = сеть думающих узлов (несколько цепочек + перекличка + readout-консенсус)
  - pure = ОДИН думающий проход той же 8b
по K попыток на задачу -> hit-rate. Где net стабильнее одиночного прохода — его зона.
Запуск: being/venv/bin/python being/where_wins.py [K]"""
import sys, re
import net
from stream import llm

TASKS = [
    ("apples-composite", "A shop had 3 crates of 24 apples each. They sold 17 apples on Monday and twice as "
     "many on Tuesday. On Wednesday they received 2 more crates of 24. How many apples now?", "69"),
    ("two-percents", "What is 15% of 240 plus 10% of 80?", "44"),
    ("pens-change", "A store sells pens at 3 for 2 dollars. Maria buys 18 pens and pays with a 20 dollar bill. "
     "How much change does she get, in dollars?", "8"),
    ("jobs-logic", "Pat, Quinn, Rosa and Sam each have a different job: doctor, lawyer, teacher, chef. Pat is "
     "not the doctor or lawyer. Quinn is the chef. Rosa is not the teacher. Sam is not the doctor. "
     "Who is the doctor?", "rosa"),
]

PURE = ("Solve the problem. Reason carefully, then give the final answer.")


def hit(ans, truth):
    return truth.lower() in str(ans).lower()


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"WHERE NET BEATS PURE — thinking nodes, K={K} trials each\n")
    for name, task, truth in TASKS:
        nh = ph = 0
        for _ in range(K):
            states, acts, _ = net.settle(task, net.init_weights())
            if hit(net.readout(task, states, acts), truth): nh += 1
            if hit(llm(PURE, f"Problem: {task}\nAnswer:", temp=0.6, max_tokens=2000, think=True), truth): ph += 1
        flag = "NET WINS" if nh > ph else ("pure wins" if ph > nh else "tie")
        print(f"  {name:18} net {nh}/{K}   pure {ph}/{K}   -> {flag}   want={truth!r}")


if __name__ == "__main__":
    main()
