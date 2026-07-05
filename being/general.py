#!/usr/bin/env python3
"""
ДОМЕН-АГНОСТИЧНЫЙ МОЗГ. Тезис: малые модели + структура решают РАЗНЫЕ классы задач одной
архитектурой — меняется только модуль домена, не мозг.

Разделение труда (вывод из всей лестницы): ЛЛМ силён в семантике (предложить ход), слаб в
надёжном применении правил -> правила отдаём СТРУКТУРЕ. Петля:
  ЛЛМ ПРЕДЛАГАЕТ ходы -> домен ПРОВЕРЯЕТ законность -> структура СИМУЛИРУЕТ прогресс каждого
  на копии -> argmax по прогрессу -> применяет -> помнит. Гейт/выбор общие, не про домен.

Домен даёт: state_text, goal_text, moves, move_help, valid, apply, passive, progress, done, succeeded.

Запуск: being/venv/bin/python being/general.py equation
        being/venv/bin/python being/general.py survival
"""
import sys, json, re, copy, urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"


def ollama(system, user, temp=0.5, max_tokens=120):
    body = json.dumps({"model": MODEL, "system": system, "prompt": user, "stream": False,
                       "think": False, "options": {"temperature": temp, "num_predict": max_tokens}}).encode()
    try:
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())["response"]
        return (out.split("</think>")[-1] if "</think>" in out else out).strip()
    except Exception as e:
        return f"(silent:{e})"


def jget_list(s, key):
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            v = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0))).get(key)
            if isinstance(v, list): return [str(x).strip() for x in v]
        except Exception:
            pass
    return []


# ---------- ДОМЕН 1: решить линейное уравнение ----------
class EquationDomain:
    """a*x+b = c*x+d -> изолировать x. Ходы структурные, домен применяет точно и проверяет."""
    name = "equation"
    def __init__(self, a=3, b=2, c=1, d=8):
        self.L = [a, b]; self.R = [c, d]
        self.xstar = (d - b) / (a - c)
    def _fmt(self, p): return f"{p[0]}*x + {p[1]}"
    def state_text(self): return f"{self._fmt(self.L)} = {self._fmt(self.R)}"
    def goal_text(self): return "transform the equation into the form '1*x + 0 = 0*x + <number>' (x isolated)."
    def moves(self): return ["collect_x_left", "collect_const_right", "divide_by_x_coeff"]
    def move_help(self):
        return ("collect_x_left: subtract the right side's x-term from both sides (kills x on the right). "
                "collect_const_right: subtract the left side's constant from both sides (kills the constant on the left). "
                "divide_by_x_coeff: divide both sides by the left x-coefficient (only when right has no x and left has no constant) -> isolates x.")
    def valid(self, m):
        if m == "collect_x_left": return (self.R[0] != 0, "no x on the right to move")
        if m == "collect_const_right": return (self.L[1] != 0, "no constant on the left to move")
        if m == "divide_by_x_coeff":
            ok = self.R[0] == 0 and self.L[1] == 0 and self.L[0] not in (0,) and self.R[1] % self.L[0] == 0
            return (ok, "can only divide when right has no x and left has no constant")
        return (False, "unknown move")
    def apply(self, m):
        if m == "collect_x_left": self.L[0] -= self.R[0]; self.R[0] = 0
        elif m == "collect_const_right": self.R[1] -= self.L[1]; self.L[1] = 0
        elif m == "divide_by_x_coeff":
            self.R[1] //= self.L[0]; self.L[0] = 1
    def passive(self): pass
    def progress(self):
        return -((self.R[0] != 0) + (self.L[1] != 0) + (self.L[0] != 1))
    def solved(self): return self.L == [1, 0] and self.R[0] == 0 and self.R[1] == self.xstar
    def done(self): return self.solved()
    def succeeded(self): return self.solved()
    def result(self): return f"x = {self.R[1]}" if self.solved() else self.state_text()


# ---------- ДОМЕН 2: выжить (один и тот же мозг) ----------
class SurvivalDomain:
    name = "survival"
    SRC = 9; DRIFT = 0.10
    def __init__(self): self.cold = 0.25; self.x = 6
    def _near(self): return self.x >= self.SRC - 1
    def state_text(self):
        return f"cold={self.cold:.2f} (1.0=dead), position x={self.x}, warmth source at x={self.SRC}, {'at_source' if self._near() else 'away'}"
    def goal_text(self): return "stay alive: keep cold below 0.9 every tick (warm yourself before it climbs)."
    def moves(self): return ["step_to_source", "warm", "wait"]
    def move_help(self):
        return ("step_to_source: move one tile toward the source (does not lower cold by itself). "
                "warm: lower cold a lot, but ONLY when at the source. wait: nothing.")
    def valid(self, m):
        if m == "warm": return (self._near(), "not at the source")
        return (True, "")
    def apply(self, m):
        if m == "step_to_source": self.x = min(self.SRC, self.x + 1)
        elif m == "warm" and self._near(): self.cold = max(0.0, self.cold - 0.6)
    def passive(self): self.cold = min(1.0, self.cold + self.DRIFT)
    def progress(self):
        return -self.cold - 0.08 * max(0, (self.SRC - 1) - self.x)   # тепло + приближение к источнику
    def done(self): return self.cold >= 0.9
    def succeeded(self): return self.cold < 0.9
    def result(self): return f"final cold={self.cold:.2f}, {'ALIVE' if self.succeeded() else 'DEAD'}"


DOMAINS = {"equation": EquationDomain, "survival": SurvivalDomain}


def brain(dom, budget):
    """Один и тот же мозг для любого домена."""
    history = []
    print(f"GOAL: {dom.goal_text()}\n")
    for step in range(budget):
        if dom.done():
            break
        state = dom.state_text()
        hist = "; ".join(history[-4:]) or "(nothing yet)"
        prop = ollama(
            "You are the PROPOSER of a mind solving a task. Suggest moves that advance toward the goal — "
            "you do NOT have to be sure; the body will verify and pick. "
            f"Goal: {dom.goal_text()}\nMoves available: {dom.moves()}\nWhat each does: {dom.move_help()}\n"
            "Reply ONLY JSON {\"candidates\":[\"move\",\"move\",...]} (1-3 move names from the list).",
            f"Current state: {state}\nRecent: {hist}\nPropose candidate moves: /no_think")
        cands = [m for m in jget_list(prop, "candidates") if m in dom.moves()]
        if not cands:
            cands = dom.moves()                       # ничего не предложил -> рассмотрим все
        # СТРУКТУРА: отсеять незаконные, симулировать прогресс каждого на копии, выбрать лучший
        scored = []
        rejected = []
        for m in cands:
            ok, why = dom.valid(m)
            if not ok:
                rejected.append(f"{m}✗({why})"); continue
            sim = copy.deepcopy(dom); sim.apply(m); sim.passive()
            scored.append((m, sim.progress()))
        if scored:
            best = max(scored, key=lambda s: s[1])[0]
        else:
            best = "wait" if "wait" in dom.moves() else cands[0]
        p0 = dom.progress()
        dom.apply(best); dom.passive()
        gain = dom.progress() - p0
        line = f"[{step}] {state}  ->  {best}  (progress {gain:+.2f})"
        if rejected: line += "  | gated: " + ", ".join(rejected)
        print(line)
        history.append(f"{best}")
    print(f"\nRESULT: {dom.result()}  | {'✓ SOLVED/ALIVE' if dom.succeeded() else '✗ failed'}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "equation"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else (8 if which == "equation" else 16)
    brain(DOMAINS[which](), budget)
