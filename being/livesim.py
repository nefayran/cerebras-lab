#!/usr/bin/env python3
"""
Единая система: НАСТОЯЩИЕ существа в ЗАЗЕМЛЁННОМ мире.
- существо = база qwen3-4b (MLX) + СВОЙ адаптер -> его веса меняются от прожитого;
- мир: нужды (жажда/голод/холод) падают, в ноль -> здоровье тает; ресурсы по углам -> обмен;
- непрерывность: состояние мира/существ/отношения на диске, переживает перезапуск;
- обучение каждый «день»: дообучение на тех ходах, что ПОМОГЛИ выжить (сигнал от мира, не от себя);
- окно (pygame) сверху.

  верстак:  being/venv/bin/python being/livesim.py --nogui   # проверить суть в логе
  смотреть: being/venv/bin/python being/livesim.py           # окно
"""
import argparse, json, os, re, random, subprocess, threading, time, gc
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


def free_mem():
    gc.collect()
    try: mx.clear_cache()
    except Exception:
        try: mx.metal.clear_cache()
        except Exception: pass


def value(b):
    """Ценность состояния: здоровье + нужды + запасы. Рост = ход реально помог."""
    return b["health"] + b["thirst"] + b["hunger"] + b["warmth"] + 8 * (b["water"] + b["food"] + b["wood"])

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(HERE, "venv", "bin", "python")
BASE = "mlx-community/Qwen3-4B-4bit"
SOULS = os.path.join(HERE, "souls")
STATE_F = os.path.join(HERE, "world_state.json")
W, H = 20, 14
DAY_LEN = 18                       # тиков в дне (по 6 на существо) -> затем обучение

PERSONAS = {
    "Vela": "You drift and feel; you share and seek connection.",
    "Koa":  "You are restless and self-first; you drive hard bargains.",
    "Sol":  "You are a builder; you plan ahead and stockpile.",
}
COLORS = {"Vela": (110, 184, 255), "Koa": (255, 115, 115), "Sol": (123, 216, 143)}

WATER = [(x, y) for y in range(1, 4) for x in range(1, 4)]
FOOD = [(x, y) for y in range(1, 4) for x in range(W - 4, W - 1)]
WOOD = [(x, y) for y in range(H - 4, H - 1) for x in range(8, 12)]

ACTIONS_HELP = ('["move",x,y] step toward; ["gather"] take resource on your tile; ["drink"]; ["eat"]; '
                '["build_fire"]; ["say","text"]; ["offer","Name","item",qty,"item",qty]; ["accept","Name"]')


def fresh_being(name, x, y):
    return {"name": name, "x": x, "y": y, "thirst": 70, "hunger": 70, "warmth": 70,
            "health": 100, "water": 0, "food": 0, "wood": 0, "alive": True, "last": "…",
            "aware": "", "goal": "", "plan": [], "born_tick": 0}


def load_state():
    if os.path.exists(STATE_F):
        return json.load(open(STATE_F))
    return {"beings": {"Vela": fresh_being("Vela", 2, 2), "Koa": fresh_being("Koa", 17, 2),
                       "Sol": fresh_being("Sol", 10, 11)},
            "fires": [], "messages": [], "offers": [], "relations": {n: "" for n in PERSONAS},
            "log": [], "tick": 0, "day": 0}


def save_state(st):
    safe = {k: v for k, v in st.items()}
    json.dump(safe, open(STATE_F, "w"), ensure_ascii=False)


def adapter_dir(name):
    return os.path.join(SOULS, name)


def load_being_model(name):
    ad = adapter_dir(name)
    if os.path.exists(os.path.join(ad, "adapters.safetensors")):
        return load(BASE, adapter_path=ad)
    return load(BASE)


def gen_mind(model, tok, name, state_json, goal, plan):
    """Ум: осознать нужду -> держать/придумать цель -> план -> действия."""
    sys = (f"You are {name}. {PERSONAS[name]} You live in a {W}x{H} world and must survive: thirst, hunger, warmth "
           f"fall over time; if any reaches 0 your health drops and you can die. water/food/wood lie in different "
           f"corners, so trading pays. You are AWARE of your needs and you THINK before acting. "
           f"Your current goal: \"{goal or '(none yet)'}\". Your current plan: {plan or '(none)'}. "
           f"Keep the goal and plan if they still serve you; change them if you are blocked or a need grew urgent; "
           f"you may invent your own goals (survive, stockpile, build a home with fire, become the one who trades X). "
           f"Output ONLY JSON: {{\"aware\":\"what you notice now\",\"goal\":\"your goal\",\"plan\":[\"step\",\"step\"],"
           f"\"actions\":[...]}}. Actions: {ACTIONS_HELP}. items water/food/wood. The actions must be the next step of "
           f"your plan. Example: {{\"aware\":\"thirst low, water is far\",\"goal\":\"keep water stocked\","
           f"\"plan\":[\"go to water\",\"gather 2\",\"drink\"],\"actions\":[[\"move\",2,2],[\"gather\"]]}} /no_think")
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": state_json}]
    try:
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    out = generate(model, tok, prompt=p, max_tokens=260, sampler=make_sampler(temp=0.6), verbose=False)
    if "</think>" in out:
        out = out.split("</think>")[-1]
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return None, "no json"
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0))), "ok"
    except Exception:
        return None, "bad json"


def nearest(p, tiles):
    return min(tiles, key=lambda t: abs(t[0] - p[0]) + abs(t[1] - p[1]))


def heuristic(b):
    """Страховка: если ум выдал кривой JSON — действуй по самой острой нужде."""
    p = (b["x"], b["y"]); worst = min(("thirst", "hunger", "warmth"), key=lambda k: b[k])
    item, tiles = {"thirst": ("water", WATER), "hunger": ("food", FOOD), "warmth": ("wood", WOOD)}[worst]
    if worst == "thirst" and b["water"] > 0: return [["drink"]]
    if worst == "hunger" and b["food"] > 0: return [["eat"]]
    if worst == "warmth" and b["wood"] > 0: return [["build_fire"]]
    if list(p) in [list(t) for t in tiles]: return [["gather"]]
    t = nearest(p, tiles); return [["move", t[0], t[1]]]


def near(a, b, c, d): return abs(a - c) <= 1 and abs(b - d) <= 1


def being_view(st, name):
    b = st["beings"][name]
    others = [{"name": o["name"], "x": o["x"], "y": o["y"]}
              for n, o in st["beings"].items() if n != name and o["alive"]]
    offers = [{"from": o["from"], "give": o["give"], "gq": o["gq"], "want": o["want"], "wq": o["wq"]}
              for o in st["offers"] if o["to"] == name]
    msgs = st["messages"][-6:]
    return json.dumps({"me": {"name": name, "x": b["x"], "y": b["y"], "thirst": b["thirst"],
                              "hunger": b["hunger"], "warmth": b["warmth"], "health": b["health"],
                              "inv": {"water": b["water"], "food": b["food"], "wood": b["wood"]}},
                       "map": {"w": W, "h": H, "water": WATER, "food": FOOD, "wood": WOOD, "fires": st["fires"]},
                       "others": others, "messages": msgs, "offers_to_me": offers}, ensure_ascii=False)


def apply_actions(st, name, acts):
    b = st["beings"][name]
    inv = {"water": "water", "food": "food", "wood": "wood"}
    for act in (acts.get("actions") or [])[:8]:
        if not isinstance(act, list) or not act:
            continue
        v = act[0]
        if v == "move" and len(act) >= 3:
            tx, ty = int(act[1]), int(act[2])
            if tx > b["x"]: b["x"] += 1
            elif tx < b["x"]: b["x"] -= 1
            elif ty > b["y"]: b["y"] += 1
            elif ty < b["y"]: b["y"] -= 1
            b["x"] = max(0, min(W - 1, b["x"])); b["y"] = max(0, min(H - 1, b["y"]))
        elif v == "gather":
            p = (b["x"], b["y"])
            if list(p) in [list(t) for t in WATER] or p in WATER: b["water"] += 1
            elif p in [tuple(t) for t in FOOD] or list(p) in [list(t) for t in FOOD]: b["food"] += 1
            elif list(p) in [list(t) for t in WOOD]: b["wood"] += 1
        elif v == "drink" and b["water"] > 0: b["water"] -= 1; b["thirst"] = min(100, b["thirst"] + 45)
        elif v == "eat" and b["food"] > 0: b["food"] -= 1; b["hunger"] = min(100, b["hunger"] + 45)
        elif v == "build_fire" and b["wood"] > 0:
            b["wood"] -= 1
            if [b["x"], b["y"]] not in st["fires"]: st["fires"].append([b["x"], b["y"]])
        elif v == "say" and len(act) >= 2:
            st["messages"].append({"from": name, "text": str(act[1])[:80]})
        elif v == "offer" and len(act) >= 6 and act[1] in st["beings"]:
            st["offers"].append({"from": name, "to": act[1], "give": act[2], "gq": int(act[3]),
                                 "want": act[4], "wq": int(act[5])})
            st["log"].append(f"{name} offers {act[1]}: {act[3]}{act[2]}->{act[5]}{act[4]}")
        elif v == "accept" and len(act) >= 2:
            for o in [x for x in st["offers"] if x["to"] == name and x["from"] == act[1]]:
                f, t = st["beings"][o["from"]], st["beings"][name]
                if f.get(o["give"], 0) >= o["gq"] and t.get(o["want"], 0) >= o["wq"]:
                    f[o["give"]] -= o["gq"]; t[o["give"]] = t.get(o["give"], 0) + o["gq"]
                    t[o["want"]] -= o["wq"]; f[o["want"]] = f.get(o["want"], 0) + o["wq"]
                    st["log"].append(f"TRADE {name}<->{o['from']}")
                st["offers"].remove(o); break


def tick_being(st, name):
    b = st["beings"][name]
    warm = any(near(b["x"], b["y"], fx, fy) for fx, fy in st["fires"])
    before_h = b["health"]
    b["thirst"] = max(0, b["thirst"] - 6); b["hunger"] = max(0, b["hunger"] - 4)
    b["warmth"] = min(100, b["warmth"] + 25) if warm else max(0, b["warmth"] - 5)
    starving = (b["thirst"] <= 0) + (b["hunger"] <= 0) + (b["warmth"] <= 0)
    b["health"] = max(0, b["health"] - 6 * starving) if starving else min(100, b["health"] + 3)
    if b["health"] <= 0: b["alive"] = False
    return b["health"] >= before_h           # «не стало хуже» = ход не навредил


def append_pending(name, view, mind_json):
    """Хороший ход -> на диск. Обучение делает ОТДЕЛЬНЫЙ процесс (не в этом, иначе клин Metal)."""
    ad = adapter_dir(name); os.makedirs(ad, exist_ok=True)
    row = {"messages": [{"role": "user", "content": view}, {"role": "assistant", "content": mind_json}]}
    with open(os.path.join(ad, "pending.jsonl"), "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sim_loop(shared, lock, stop, log_stdout):
    models = {n: load_being_model(n) for n in PERSONAS}   # грузим ОДИН раз, больше не перезагружаем
    while not stop.is_set():
        for name in list(PERSONAS):
            with lock:
                alive = shared["beings"][name]["alive"]
            if not alive:
                continue
            with lock:
                view = being_view(shared, name); shared["active"] = name
                goal = shared["beings"][name].get("goal", ""); plan = shared["beings"][name].get("plan", [])
            model, tok = models[name]
            mind, msg = gen_mind(model, tok, name, view, goal, plan)
            with lock:
                b = shared["beings"][name]
                v0 = value(b)
                mind_acts = None
                if mind:
                    if mind.get("goal"): b["goal"] = str(mind["goal"])[:80]
                    if isinstance(mind.get("plan"), list): b["plan"] = [str(s)[:50] for s in mind["plan"]][:5]
                    if mind.get("aware"): b["aware"] = str(mind["aware"])[:90]
                    if isinstance(mind.get("actions"), list) and mind["actions"]:
                        mind_acts = {"actions": mind["actions"]}
                used_mind = mind_acts is not None
                # РЕФЛЕКС: если нужда критична (<18) — действуем по ней, что бы ум ни сказал
                worst = min(("thirst", "hunger", "warmth"), key=lambda k: b[k])
                if b[worst] < 18:
                    acts = {"actions": heuristic(b)}; msg = msg + "|reflex"
                elif used_mind:
                    acts = mind_acts
                else:
                    acts = {"actions": heuristic(b)}; msg = msg + "->heur"
                apply_actions(shared, name, acts)
                tick_being(shared, name)
                v1 = value(b)
                shared["tick"] += 1; tk = shared["tick"]
                b["last"] = msg
                # награда за РЕАЛЬНЫЙ рост: раздумья ума, после которых стало ЛУЧШЕ — на диск
                if used_mind and mind and v1 > v0:
                    append_pending(name, view, json.dumps(mind, ensure_ascii=False))
                if len(shared["messages"]) > 30: shared["messages"] = shared["messages"][-30:]
                if len(shared["log"]) > 50: shared["log"] = shared["log"][-50:]
                day_now = shared["tick"] // DAY_LEN
                if log_stdout:
                    print(f"t{tk} {name} {msg} | hp{b['health']} thi{b['thirst']} hun{b['hunger']} wrm{b['warmth']} "
                          f"| GOAL: {b.get('goal','')[:40]} | AWARE: {b.get('aware','')[:40]}")
            # обучение здесь НЕ делаем (клин Metal). Просто сохраняемся на границе дня.
            if shared["tick"] % DAY_LEN == 0:
                with lock:
                    shared["day"] = shared["tick"] // DAY_LEN
                    save_state(shared)
            time.sleep(0.05)
    with lock:
        save_state(shared)


def run(nogui, max_ticks):
    shared = load_state(); shared.setdefault("active", "Vela")
    lock = threading.Lock(); stop = threading.Event()
    th = threading.Thread(target=sim_loop, args=(shared, lock, stop, nogui), daemon=True)
    th.start()
    if nogui:
        while shared["tick"] < max_ticks:
            time.sleep(1)
        stop.set(); th.join(timeout=120); return
    render_window(shared, lock, stop)


def render_window(shared, lock, stop):
    import pygame
    pygame.init()
    cell = 34; gw = W * cell; sw, sh = gw + 360, max(H * cell, 760)
    screen = pygame.display.set_mode((sw, sh)); pygame.display.set_caption("Living world")
    font = pygame.font.SysFont("menlo", 14); big = pygame.font.SysFont("menlo", 20)
    clock = pygame.time.Clock()
    def res_col(p):
        if list(p) in [list(t) for t in WATER]: return (38, 76, 140)
        if list(p) in [list(t) for t in FOOD]: return (50, 115, 56)
        if list(p) in [list(t) for t in WOOD]: return (102, 71, 38)
        return (26, 26, 33)
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
        screen.fill((13, 13, 18))
        with lock:
            fires = [tuple(f) for f in shared["fires"]]
            beings = {n: dict(b) for n, b in shared["beings"].items()}
            tick = shared["tick"]; day = shared.get("day", 0); active = shared.get("active", "")
            msgs = shared["messages"][-8:]; logs = shared["log"][-6:]
        for y in range(H):
            for x in range(W):
                c = (255, 140, 38) if (x, y) in fires else res_col((x, y))
                pygame.draw.rect(screen, c, (x * cell + 1, y * cell + 1, cell - 2, cell - 2))
        for n, b in beings.items():
            cx, cy = b["x"] * cell + cell // 2, b["y"] * cell + cell // 2
            col = COLORS[n] if b["alive"] else (90, 90, 90)
            pygame.draw.circle(screen, col, (cx, cy), cell // 3)
            screen.blit(font.render(n[0], True, (0, 0, 0)), (cx - 4, cy - 7))
        px = gw + 12; screen.blit(big.render(f"day {day}  tick {tick}", True, (255, 255, 255)), (px, 8))
        yy = 40
        for n, b in beings.items():
            tag = ">" if n == active else " "
            screen.blit(font.render(f"{tag}{n} {'DEAD' if not b['alive'] else ''}", True, COLORS[n]), (px, yy)); yy += 16
            for lab, key, c in (("thi", "thirst", (76, 153, 255)), ("hun", "hunger", (102, 216, 102)),
                                ("wrm", "warmth", (255, 153, 51)), ("hp", "health", (230, 76, 128))):
                pygame.draw.rect(screen, (40, 40, 48), (px, yy, 150, 7))
                pygame.draw.rect(screen, c, (px, yy, int(150 * b[key] / 100), 7)); yy += 9
            screen.blit(font.render(f"W{b['water']} F{b['food']} D{b['wood']} {b['last'][:16]}", True, (190, 190, 190)), (px, yy)); yy += 14
            screen.blit(font.render(f"goal: {b.get('goal','')[:34]}", True, (210, 200, 150)), (px, yy)); yy += 13
            step = (b.get('plan') or [''])[0]
            screen.blit(font.render(f"now:  {str(step)[:34]}", True, (170, 190, 160)), (px, yy)); yy += 18
        yy += 6; screen.blit(font.render("talk & trades:", True, (255, 255, 255)), (px, yy)); yy += 18
        for m in reversed(msgs):
            screen.blit(font.render(f"{m['from']}: {m['text']}"[:44], True, (200, 200, 200)), (px, yy)); yy += 15
        for l in reversed(logs):
            screen.blit(font.render(l[:44], True, (160, 170, 190)), (px, yy)); yy += 15
        pygame.display.flip(); clock.tick(12)
    stop.set(); time.sleep(0.3); pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nogui", action="store_true")
    ap.add_argument("--max-ticks", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(SOULS, exist_ok=True)
    run(args.nogui, args.max_ticks)
