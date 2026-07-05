#!/usr/bin/env python3
"""АВТОНОМНЫЙ РОСТ. Мозг сам себе ставит задачи (генераторы с эталоном), решает, проверяет,
и поднимает СЛОЖНОСТЬ класса, когда освоил. Память растёт по ходу (RAG); «сон» (LoRA) — отдельно.
Куррикулум = лестница: освоил рунг (точность ≥ порог) -> следующий, труднее.
Стойкий провал -> флаг «нужен орган» (границу — написание органа — делает человек/кодген).
Запуск: being/venv/bin/python being/autoloop.py [rounds] [K]
"""
import sys, io, re, random, contextlib, itertools, json, os, urllib.request
import connectome, sleep as sleepmod, grow_organ, self_model

random.seed(11)
NAMES = ["Ann", "Bob", "Cara", "Dan", "Eve", "Finn"]
ITEMS = ["cat", "dog", "fish", "bird", "rabbit", "hamster"]
PASS = 0.75          # порог освоения рунга
SLEEP_EVERY = 2      # каждые N раундов — консолидация в веса личности («сон»)
LOG = os.path.join(os.path.dirname(__file__), "growth.jsonl")


# ---- генераторы задач С ЭТАЛОНОМ (растущая сложность d) ----
def gen_arith(d):
    val = random.randint(2, 9) * random.randint(2, 9)
    parts = [f"Start with {val}."]
    for _ in range(d):
        op = random.choice(["add", "sub", "mul"])
        n = random.randint(2, 12)
        if op == "add": val += n; parts.append(f"Add {n}.")
        elif op == "sub": val -= n; parts.append(f"Subtract {n}.")
        else: val *= n; parts.append(f"Multiply by {n}.")
    return " ".join(parts) + " What is the result?", val


def _sols(agents, items, cons):
    out = []
    for perm in itertools.permutations(items):
        m = dict(zip(agents, perm))
        if all((m[a] != i) if rel == "not" else (m[a] == i) for a, rel, i in cons):
            out.append(m)
    return out


def gen_logic(n):
    agents, items = NAMES[:n], ITEMS[:n]
    for _ in range(200):
        truth = dict(zip(agents, random.sample(items, n)))
        a0 = random.choice(agents)
        cons = [(a0, "is", truth[a0])]                       # один прямой ключ
        for _ in range(n + 1):                               # несколько «не»
            a, i = random.choice(agents), random.choice(items)
            if truth[a] != i and (a, "not", i) not in cons: cons.append((a, "not", i))
        if len(_sols(agents, items, cons)) == 1: break       # требуем ЕДИНСТВЕННОСТЬ
    qi = random.choice([i for i in items if i != truth[a0]])  # не спрашивать про прямо-раскрытый предмет
    ans = next(a for a in agents if truth[a] == qi)
    clues = "; ".join(f"{a} owns the {i}" if rel == "is" else f"{a} does not own the {i}" for a, rel, i in cons)
    q = f"{', '.join(agents)} each own a different pet: {', '.join(items)}. {clues}. Who owns the {qi}?"
    return q, ans


def gen_compare(n):
    people = random.sample(NAMES, n)
    order = people[:]; random.shuffle(order)             # order[0] = самый высокий
    clues = [f"{order[i]} is taller than {order[i+1]}" for i in range(n - 1)]
    random.shuffle(clues)
    ask = random.choice(["tallest", "shortest"])
    ans = order[0] if ask == "tallest" else order[-1]
    return ". ".join(clues) + f". Who is the {ask}?", ans


def gen_clock(d):
    h, m = random.randint(0, 23), random.randint(0, 59)
    total = h * 60 + m
    parts = [f"It is {h:02d}:{m:02d}."]
    for _ in range(d):
        a = random.randint(10, 180); total = (total + a) % (24 * 60); parts.append(f"Add {a} minutes.")
    hh, mm = divmod(total, 60)
    return " ".join(parts) + " What time is it now (HH:MM, 24h)?", f"{hh:02d}:{mm:02d}"


def to_roman(n):
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
            (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    r = ""
    for v, s in vals:
        while n >= v: r += s; n -= v
    return r


def gen_roman(d):
    cap = [0, 0, 49, 499, 1999, 3999][min(d, 5)] or 49
    n = random.randint(1, cap)
    return f"Convert the number {n} to a Roman numeral.", to_roman(n)


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
def gen_dow(d):
    start = random.randint(0, 6); off = random.randint(20, 60 * d)
    return f"Today is {DAYS[start]}. What day of the week will it be in {off} days?", DAYS[(start + off) % 7]


CLASSES = {"arith": (gen_arith, 2), "logic": (gen_logic, 3), "compare": (gen_compare, 3),
           "clock": (gen_clock, 2), "roman": (gen_roman, 2), "dow": (gen_dow, 2)}   # (генератор, старт. сложность)


def check(kind, truth, out):
    s = str(out).lower()
    if kind == "arith":
        return any(abs(float(x) - truth) < 1e-6 for x in re.findall(r"-?\d+\.?\d*", s.replace(",", "")))
    if kind == "clock":
        return str(truth) in str(out)                    # HH:MM
    return str(truth).lower() in s                       # logic, compare, roman


def consolidate():
    """«Сон»: дообучить веса личности на прожитом, затем перечитать адаптеры в тёплом сервере."""
    print("  --- SLEEP: consolidating experience into personality weights ---")
    try:
        sleepmod.main()
        urllib.request.urlopen("http://127.0.0.1:11500/reload", timeout=120).read()
        print("  --- weights updated, server reloaded ---")
    except Exception as e:
        print(f"  --- sleep skipped: {e} ---")


def brain(task):
    with contextlib.redirect_stdout(io.StringIO()):
        return connectome.run(task)


def organ_path(c): return os.path.join(os.path.dirname(__file__), "organs", f"{c}.py")


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    classes = [c for c in CLASSES if c == sys.argv[3]] if len(sys.argv) > 3 else list(CLASSES)
    level = {c: CLASSES[c][1] for c in classes}
    stuck = {c: 0 for c in classes}
    open(LOG, "w").close()
    print(f"AUTONOMOUS GROWTH — {rounds} rounds x {K} tasks/class — classes: {classes}\n")
    for rnd in range(rounds):
        for c in classes:
            gen = CLASSES[c][0]; d = level[c]
            ok = 0
            for _ in range(K):
                q, truth = gen(d)
                r = brain(q)
                hit = check(c, truth, r)
                ok += hit
                print(f"  r{rnd} {c} d{d}  {'OK ' if hit else 'xMISS'}  truth={truth}  got={str(r)[:34]!r}")
            acc = ok / K
            grew = acc >= PASS
            if grew: level[c] += 1; stuck[c] = 0
            else: stuck[c] += 1
            self_model.note(c, grew)
            with open(LOG, "a") as f:
                f.write(json.dumps({"round": rnd, "class": c, "difficulty": d, "acc": acc, "level_now": level[c]}) + "\n")
            print(f"  == r{rnd} {c}: acc {acc:.2f} at d{d}{'  -> LEVEL UP' if grew else ''}\n")

            # САМО-РАСШИРЕНИЕ: застрял и органа ещё нет -> мозг сам отращивает себе руку
            if not grew and stuck[c] >= 2 and not os.path.exists(organ_path(c)):
                print(f"  -> {c}: stuck. GROWING a new organ for myself...")
                gacc, gmsg, _ = grow_organ.grow(c)
                print(f"     organ[{c}]: acc={gacc} -> {gmsg}")
                if gmsg and "ACCEPTED" in gmsg:
                    connectome.reload_organs(); stuck[c] = 0
                    print(f"     -> grew a new hand, reloaded it. (autonomous self-extension)\n")
        if (rnd + 1) % SLEEP_EVERY == 0:                 # периодическая консолидация: веса + само-убеждения
            consolidate(); self_model.consolidate()
    print("REACHED LEVELS:", {c: level[c] for c in classes})


if __name__ == "__main__":
    main()
