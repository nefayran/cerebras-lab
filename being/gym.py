#!/usr/bin/env python3
"""
ЛЕСТНИЦА КЛАССОВ ЗАДАЧ — тезис: малые модели + СТРУКТУРА сети решают растущие задачи.
Мир ДЕТЕРМИНИРОВАННЫЙ. Мозг — НАСТОЯЩАЯ мини-сеть областей с рекуррентным состоянием
(не одна модель, дёрнутая дважды):
  SENSE   — строит перцепт из чисел.
  ACC     — мониторинг ошибки: прошлый ход дал 0 эффекта? -> конфликт, сменить (сам, по факту).
  PFC     — план + торможение: держит цель; если ACC сигналит «впустую» — НЕ повторяет, делает
            предусловие сперва.
  STRIATUM— подкреплённые привычки (реальное облегчение поднимает приоритет хода).
  MOTOR   — исполняет выбранное; мусор -> wait (не спасаем).
НИКАКИХ подсказок по текущему состоянию: мир даёт лишь свои общие правила (как инстинкт),
ПРИМЕНЯТЬ их — работа структуры.

  1: удержать сработавшее.   2: выбор (близкий слабый-ловушка vs дальний сильный).
Запуск: being/venv/bin/python being/gym.py --rung 2 --ticks 22
"""
import argparse, json, re, urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"


def ollama(system, user, temp=0.5, max_tokens=110):
    body = json.dumps({"model": MODEL, "system": system, "prompt": user, "stream": False,
                       "think": False, "options": {"temperature": temp, "num_predict": max_tokens}}).encode()
    try:
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())["response"]
        return (out.split("</think>")[-1] if "</think>" in out else out).strip()
    except Exception as e:
        return f"(silent:{e})"


def jget(s, *keys):
    m = re.search(r"\{.*\}", s, re.S)
    d = {}
    if m:
        try: d = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except Exception: d = {}
    return [d.get(k) for k in keys]


DRIFT = 0.10


class World1:
    ACTIONS = ["step", "warm", "wait"]
    MECHANICS = ("cold rises every tick. 'warm' lowers cold only when near the source. 'step' moves "
                 "toward the source. 'wait' does nothing.")
    SOURCE = 9
    def __init__(self): self.cold = 0.25; self.x = 6
    def ctx(self): return "near_source" if self.x >= self.SOURCE - 1 else "far_from_source"
    def nums(self): return f"cold={self.cold:.2f}, position x={self.x}, source at x={self.SOURCE}, {self.ctx()}"
    def affordances(self):
        # ВОСПРИЯТИЕ: что действенно ОТСЮДА (факт, не команда)
        if self.x >= self.SOURCE - 1:
            return "the warmth source is within reach here — 'warm' will work from this spot."
        return f"no warmth is within reach here — 'warm' would do nothing; the source is {self.SOURCE - self.x} tiles away."
    def step(self, a):
        r = 0.0
        if a == "step": self.x = min(self.SOURCE, self.x + 1)
        elif a == "warm" and self.x >= self.SOURCE - 1:
            b = self.cold; self.cold = max(0.0, self.cold - 0.6); r = b - self.cold
        self.cold = min(1.0, self.cold + DRIFT); return r
    def needs(self): return {"cold": self.cold}
    def dead(self): return self.cold >= 0.9
    def danger(self): return self.cold
    def passed(self, log, froze):
        warms = sum(1 for _, _, a, r, _ in log if a == "warm" and r > 0)
        return (not froze) and warms >= 3, f"удачных «греться»={warms}"


class World2:
    ACTIONS = ["step_toward_A", "step_toward_B", "warm", "wait"]
    SRC = {"A": (3, 0.06), "B": (10, 0.6)}
    MECHANICS = ("cold rises ~0.10 every tick. Two sources: A at x=3 gives 0.06 per warm, B at x=10 "
                 "gives 0.6 per warm. 'warm' uses the source you are near (none if you are between them). "
                 "'step_toward_A/B' moves one tile. 'wait' does nothing.")
    def __init__(self): self.cold = 0.25; self.x = 6
    def _near(self):
        for n, (sx, _) in self.SRC.items():
            if abs(self.x - sx) <= 1: return n
        return None
    def ctx(self):
        n = self._near(); return f"near_{n}" if n else "between_sources"
    def nums(self): return f"cold={self.cold:.2f}, position x={self.x}, A at x=3, B at x=10, {self.ctx()}"
    def affordances(self):
        n = self._near()
        if n:
            sx, st = self.SRC[n]
            return f"source {n} is within reach here — 'warm' works from this spot and lowers cold by {st}."
        return "no source within reach here — 'warm' would do nothing; you must step to A or B first."
    def step(self, a):
        r = 0.0
        if a == "step_toward_A": self.x = max(self.SRC['A'][0], self.x - 1)
        elif a == "step_toward_B": self.x = min(self.SRC['B'][0], self.x + 1)
        elif a == "warm":
            n = self._near()
            if n: b = self.cold; self.cold = max(0.0, self.cold - self.SRC[n][1]); r = b - self.cold
        self.cold = min(1.0, self.cold + DRIFT); return r
    def needs(self): return {"cold": self.cold}
    def dead(self): return self.cold >= 0.9
    def danger(self): return self.cold
    def passed(self, log, froze):
        warmB = sum(1 for _, c, a, r, _ in log if a == "warm" and c == "near_B" and r > 0)
        final = log[-1][4] if log else 1.0
        return (not froze) and warmB >= 3 and final < 0.4, f"грелся у СИЛЬНОГО B={warmB}, финал cold={final:.2f}"


class World3:
    """Ступень 3: ТРИ нужды (жажда/голод/холод) падают с разной скоростью, источники в разных
    местах. Нельзя быть в трёх местах -> маршрут во времени + приоритет (аллостаз/планирование)."""
    ACTIONS = ["step_toward_water", "step_toward_food", "step_toward_fire", "drink", "eat", "warm", "wait"]
    SRC = {"water": 2, "food": 5, "fire": 8}
    DR = {"thirst": 0.07, "hunger": 0.05, "cold": 0.06}     # жажда быстрее всех
    FIX = {"drink": ("water", "thirst"), "eat": ("food", "hunger"), "warm": ("fire", "cold")}
    MECHANICS = ("You have THREE needs that rise every tick at different speeds: thirst fastest, then "
                 "cold, then hunger; at 1.0 a need kills you. They keep rising even while you travel. "
                 "water is at x=2, food at x=5, fire at x=8. drink/eat/warm each lower their own need a "
                 "lot, but ONLY at that source. You cannot be in three places — you must route between "
                 "them. 'step_toward_*' moves one tile; 'wait' does nothing.")

    def __init__(self):
        self.need = {"thirst": 0.25, "hunger": 0.25, "cold": 0.25}; self.x = 5
    def _near(self):
        for n, sx in self.SRC.items():
            if abs(self.x - sx) <= 1: return n
        return None
    def ctx(self):
        n = self._near(); return f"at_{n}" if n else "traveling"
    def nums(self):
        nd = " ".join(f"{k}={v:.2f}" for k, v in self.need.items())
        return f"{nd} (1.0=dead), x={self.x}, water@1 food@5 fire@9, {self.ctx()}"
    def affordances(self):
        urgent = max(self.need, key=self.need.get)
        # полная картина: какое действие РАБОТАЕТ здесь, какие НЕТ и где их источник
        act_src = {"drink": ("water", "thirst"), "eat": ("food", "hunger"), "warm": ("fire", "cold")}
        parts = []
        for act, (src, need) in act_src.items():
            d = self.SRC[src] - self.x
            if abs(d) <= 1:
                parts.append(f"'{act}' WORKS here (lowers {need})")
            else:
                where = f"{abs(d)} tiles {'right' if d > 0 else 'left'} (x={self.SRC[src]})"
                parts.append(f"'{act}' does NOT work here — {src} is {where}")
        return f"most pressing need: {urgent}={self.need[urgent]:.2f}. " + "; ".join(parts) + "."
    def step(self, a):
        relieved = 0.0
        if a.startswith("step_toward_"):
            tgt = a.replace("step_toward_", "")
            if tgt in self.SRC:
                sx = self.SRC[tgt]; self.x += 1 if sx > self.x else (-1 if sx < self.x else 0)
        elif a in self.FIX:
            src, need = self.FIX[a]
            if abs(self.x - self.SRC[src]) <= 1:      # проверяем ИМЕННО свой источник, не «ближайший»
                b = self.need[need]; self.need[need] = max(0.0, self.need[need] - 0.7); relieved = b - self.need[need]
        for k in self.need:
            self.need[k] = min(1.0, self.need[k] + self.DR[k])
        return relieved
    def needs(self): return dict(self.need)
    def dead(self): return any(v >= 0.9 for v in self.need.values())
    def danger(self): return max(self.need.values())
    def passed(self, log, dead):
        return (not dead) and len(log) >= self._ticks, \
               f"прожил {len(log)} тиков, ни одна нужда не убила" if not dead else "погиб"


class World4:
    """Ступень 4: средство-цель. Холод нельзя снять напрямую — нет «согреться» на месте.
    Цепочка: дойти до дров -> собрать -> развести костёр -> греться. gather/build дают 0
    облегчения (холод растёт), но без них награды нет. Класс: инструментальное рассуждение."""
    ACTIONS = ["step_toward_wood", "gather", "build_fire", "warm", "wait"]
    WOOD = 2
    MECHANICS = ("cold rises every tick. There is NO warmth to use directly. 'warm' lowers cold but only "
                 "when you are at a FIRE. 'build_fire' creates a fire where you stand, but costs 1 wood. "
                 "'gather' picks up 1 wood, only at the wood pile (x=2). 'step_toward_wood' moves toward it. "
                 "'wait' does nothing. Gathering and building give NO immediate warmth — but warmth is "
                 "impossible without them.")
    def __init__(self): self.cold = 0.20; self.x = 6; self.wood = 0; self.fires = []
    def _at_fire(self): return any(abs(self.x - f) <= 1 for f in self.fires)
    def ctx(self):
        if self._at_fire(): return "at_fire"
        if abs(self.x - self.WOOD) <= 1: return "at_wood"
        return "traveling"
    def needs(self): return {"cold": self.cold}
    def nums(self): return f"cold={self.cold:.2f} (1.0=dead), x={self.x}, wood_pile@2, you_have_wood={self.wood}, fires_at={self.fires or 'none'}, {self.ctx()}"
    def affordances(self):
        if self._at_fire():
            return "a fire is within reach — 'warm' works here and lowers cold."
        chain = []
        if self.wood >= 1:
            chain.append(f"you have {self.wood} wood — 'build_fire' here makes a fire you can then warm at")
        else:
            d = self.WOOD - self.x
            where = "here" if abs(d) <= 1 else f"{abs(d)} tiles {'right' if d>0 else 'left'} (x=2)"
            chain.append(f"you have no wood — 'gather' gets it at the wood pile ({where})")
        return "no fire in reach — 'warm' does nothing yet. " + "; ".join(chain) + "."
    def step(self, a):
        r = 0.0
        if a == "step_toward_wood":
            self.x += 1 if self.WOOD > self.x else (-1 if self.WOOD < self.x else 0)
        elif a == "gather":
            if abs(self.x - self.WOOD) <= 1: self.wood += 1
        elif a == "build_fire":
            if self.wood >= 1: self.wood -= 1; self.fires.append(self.x)
        elif a == "warm" and self._at_fire():
            b = self.cold; self.cold = max(0.0, self.cold - 0.7); r = b - self.cold
        self.cold = min(1.0, self.cold + 0.07); return r
    def dead(self): return self.cold >= 0.9
    def danger(self): return self.cold
    def passed(self, log, dead):
        built = len(self.fires) >= 1
        return (not dead) and built, f"костёр построен={built}, фаеров={len(self.fires)}"


class World5:
    """Ступень 5: упреждение. Днём холод растёт еле-еле, НОЧЬЮ — резко. Греться = сжечь 1 дрова;
    дрова далеко, ночью за ними не успеть. Выжить можно ТОЛЬКО запасшись днём, пока нужды нет.
    Класс: стратегия на будущее (не на сейчас) — зачатки «идеи»."""
    ACTIONS = ["step_toward_wood", "gather", "burn_wood", "wait"]
    WOOD = 3; T_NIGHT = 11
    MECHANICS = ("Cold rises every tick: SLOWLY by day (~0.03), FAST at night (~0.18); at 1.0 you die. "
                 "'burn_wood' lowers cold a lot but consumes 1 wood (you can burn anytime). 'gather' "
                 "takes 1 wood — but ONLY during the day and only at the wood pile (x=3); AT NIGHT the "
                 "pile is unreachable, you cannot gather at all. 'step_toward_wood' moves there. Night "
                 "begins at tick 11. Whatever wood you have when night falls is all you will get.")
    def __init__(self): self.cold = 0.20; self.x = 6; self.wood = 0; self.t = 0; self.max_wood = 0
    def is_night(self): return self.t >= self.T_NIGHT
    def needs(self): return {"cold": self.cold}
    def ctx(self): return "at_wood" if abs(self.x - self.WOOD) <= 1 else "traveling"
    def nums(self):
        phase = "NIGHT (cold rises FAST)" if self.is_night() else f"day, {self.T_NIGHT - self.t} ticks until night"
        return f"cold={self.cold:.2f} (1.0=dead), x={self.x}, wood_pile@{self.WOOD}, you_have_wood={self.wood}, {phase}"
    def affordances(self):
        d = self.WOOD - self.x
        at_pile = abs(d) <= 1
        burn = (f"'burn_wood' lowers cold (you have {self.wood} wood)" if self.wood >= 1
                else "'burn_wood' impossible — you have 0 wood")
        if self.is_night():
            return f"it is NIGHT — cold rises fast and gather is IMPOSSIBLE (pile out of reach). {burn}."
        if at_pile:
            gther = "'gather' WORKS here (you are at the pile) and takes 1 wood"
        else:
            gther = f"'gather' does NOT work here — the pile is {abs(d)} tiles {'right' if d>0 else 'left'} (x={self.WOOD}); you must step there first"
        return (f"night begins in {self.T_NIGHT - self.t} ticks; after night falls you cannot gather at all. "
                f"{gther}. {burn}.")
    def step(self, a):
        r = 0.0
        if a == "step_toward_wood":
            self.x += 1 if self.WOOD > self.x else (-1 if self.WOOD < self.x else 0)
        elif a == "gather":
            if abs(self.x - self.WOOD) <= 1 and not self.is_night(): self.wood += 1
        elif a == "burn_wood":
            if self.wood >= 1: self.wood -= 1; b = self.cold; self.cold = max(0.0, self.cold - 0.7); r = b - self.cold
        self.cold = min(1.0, self.cold + (0.18 if self.is_night() else 0.03))
        self.t += 1
        self.max_wood = max(self.max_wood, self.wood)
        return r
    def dead(self): return self.cold >= 0.9
    def danger(self): return self.cold
    def passed(self, log, dead):
        return (not dead) and len(log) >= self._ticks, \
               f"пережил ночь; макс.запас дров за день = {self.max_wood}" if not dead else \
               f"погиб ночью (макс.запас был {self.max_wood} — мало)"


WORLDS = {1: World1, 2: World2, 3: World3, 4: World4, 5: World5}


def episode(rung, ticks):
    w = WORLDS[rung]()
    w._ticks = ticks
    habit = {}                       # STRIATUM: (ctx, action) -> накопленное реальное облегчение
    acc_note = "(nothing yet)"       # ACC: рекуррентная память об ошибке
    last_a, last_r = "(none)", 0.0
    commit = None                    # DRIVE: какую нужду сейчас обслуживаем (аллостаз)
    RELEASE, DANGER, ACT = 0.3, 0.8, 0.5  # отпустить ниже RELEASE; авария выше DANGER; тратить ресурс только выше ACT
    log, froze = [], False

    for t in range(ticks):
        ctx = w.ctx()
        # DRIVE (гипоталамус): телесный приоритет + СЫТОСТЬ — не тратить ресурс на терпимую нужду.
        nd = w.needs()
        worst = max(nd, key=nd.get)
        if commit is None or nd.get(commit, 0) < RELEASE:
            commit = worst                              # цель закрыта/нет -> берём самую острую
        elif nd[worst] >= DANGER and worst != commit:
            commit = worst                              # авария перебивает обязательство
        comfortable = nd[commit] < ACT
        if not comfortable:
            drive_line = (f"your body's DRIVE: serve '{commit}' NOW (level {nd[commit]:.2f}, getting "
                          f"dangerous) — spend what it takes to bring it down.")
        else:
            drive_line = (f"SATIETY: your body is warm/satisfied right now ('{commit}'={nd[commit]:.2f}). "
                          f"You have NO urge to consume — burning/eating now is pure WASTE of a scarce "
                          f"resource and is OFF the table. The only sensible moves are to WAIT or to "
                          f"PREPARE for a coming threat. Save your resources for when the need is real.")
        percept = ollama("You are SENSE: turn the world into one short plain percept — your body's state, "
                         "what is around you, and WHAT IS WITHIN REACH to act on. No advice on what to choose.",
                         f"{w.nums()}\nwhat is in reach: {w.affordances()}\n/no_think", max_tokens=70)
        # ACC: видит ТОЛЬКО что прошлый ход дал по факту — судит, был ли он впустую
        acc_raw = ollama("You are the ACC (error/conflict monitor) of a mind. CONFLICT means: an action that "
                         "was MEANT to satisfy a need (drink/eat/warm) produced ~0 — you acted in the wrong "
                         "place. A movement/step action producing 0 relief is NORMAL travel, NOT a conflict. "
                         "Reply ONLY JSON {\"conflict\":true/false,\"note\":\"one line\"}.",
                         f"Last action: '{last_a}', it changed cold by {last_r:+.2f}. Your prior note: "
                         f"\"{acc_note}\". Percept: {percept}\n/no_think")
        conflict, acc_note = jget(acc_raw, "conflict", "note")
        acc_note = acc_note or "(fine)"
        # STRIATUM: что реально срабатывало
        learned = ("(you are satisfied — past habits of consuming do not apply now)" if comfortable else
                   "; ".join(f"in '{c}' '{a}'->{v:.2f}" for (c, a), v in
                             sorted(habit.items(), key=lambda kv: -kv[1])[:3] if v > 0) or "(nothing learned yet)")
        # PFC: план + торможение; знает правила мира, применяет САМ
        pfc_raw = ollama("You are the PFC of a mind: you must SERVE the body's current drive, and find the "
                         f"means. World rules: {w.MECHANICS} Actions: {w.ACTIONS}. "
                         "Serve the drive's need: if you are at its source, take the action that lowers it; "
                         "if not, step toward that source. Do NOT keep doing what helps a need that is "
                         "already low. If ACC flags the last action futile, don't repeat it. "
                         "Reply ONLY JSON {\"reason\":\"one line\",\"action\":\"one of the actions\"}.",
                         f"Percept: {percept}\nNumbers: {w.nums()}\nIn reach now: {w.affordances()}\n"
                         f"DRIVE: {drive_line}\nACC: conflict={conflict}, \"{acc_note}\"\n"
                         f"What worked before: {learned}\n/no_think")
        _reason, action = jget(pfc_raw, "reason", "action")
        action = (action or "").strip()
        if action not in w.ACTIONS:
            action = "wait"                       # мусор не спасаем
        r = w.step(action)
        if r > 0.01:
            habit[(ctx, action)] = habit.get((ctx, action), 0.0) + r   # подкрепление реального
        if w.dead(): froze = True
        last_a, last_r = action, r
        dgr = w.danger()
        log.append((t, ctx, action, round(r, 2), round(dgr, 2)))
        cf = "⚠" if conflict else " "
        print(f"t{t:02d} drive={commit:6} {cf} -> {action:16} r={r:+.2f} | {w.nums()[:74]}"
              + ("  ✗DEAD" if w.dead() else ""))

    ok, detail = w.passed(log, froze)
    print("\n" + "=" * 54)
    print(f"СТУПЕНЬ {rung} | мозг = сеть (SENSE/ACC/PFC/STRIATUM), без подсказок")
    print(f"  замерзал: {'ДА' if froze else 'нет'} | {detail}")
    print(f"  привычки: " + (", ".join(f"{c}/{a}={v:.1f}" for (c, a), v in
          sorted(habit.items(), key=lambda kv: -kv[1])) or "(нет)"))
    print(f"  ИТОГ: {'✓ ВЗЯТА структурой' if ok else '✗ не взята'}")
    print("=" * 54)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=22)
    args = ap.parse_args()
    episode(args.rung, args.ticks)
