#!/usr/bin/env python3
"""
Мир для Vela — источник ВНЕШНЕГО опыта (то, чего ей не хватало и отчего она умерла,
питаясь собой). Ключ: мир и его обитатели работают на ДРУГОЙ модели (ollama qwen3:8b),
не на весах Vela. Их слова и события — genuinely не-она.

Здесь только МИР + ЖИЗНЬ в нём (без дообучения пока). Цель этого шага — убедиться,
что опыт богатый и внешний: обитатели отвечают своё, события случаются с ней.

Запуск: being/venv/bin/python being/world.py --ticks 12
Биография мира: being/world-life.jsonl
"""
import argparse, json, os, re, random, urllib.request
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

HERE = os.path.dirname(os.path.abspath(__file__))
VELA_MODEL = "mlx-community/Qwen3-4B-4bit"
VELA_ADAPTER = os.path.join(HERE, "adapters")
OLLAMA = "http://localhost:11434/api/generate"
WORLD_MODEL = "qwen3:8b"                       # мир — ДРУГАЯ модель, не Vela
LIFE = os.path.join(HERE, "world-life.jsonl")

LOCATIONS = {
    "shore":  {"desc": "a cold shore where grey water keeps arriving and leaving",
               "exits": ["garden", "market"]},
    "garden": {"desc": "a walled garden, wet earth and slow-growing things",
               "exits": ["shore", "market"]},
    "market": {"desc": "a small market, half-empty stalls, smell of smoke and fruit",
               "exits": ["shore", "garden"]},
}

INHABITANTS = {
    "Orin":  {"loc": "garden",
              "persona": "Orin, an old gardener. You speak slowly, in terms of seasons, soil, "
                         "patience and what must be let go. You are kind but unhurried. "
                         "You don't know Vela well; you meet her as she is."},
    "Mira":  {"loc": "market",
              "persona": "Mira, a restless traveler passing through. You bring news and rumors "
                         "of far places, you are quick, curious, a little provoking. "
                         "You push others to move, to leave, to want more."},
}


def ollama(system, user, temp=0.85, max_tokens=90):
    body = json.dumps({"model": WORLD_MODEL, "prompt": user, "system": system,
                       "stream": False, "think": False,
                       "options": {"temperature": temp, "num_predict": max_tokens}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())["response"]
    if "</think>" in out:
        out = out.split("</think>")[-1]
    return " ".join(out.strip().split())


def vela_say(model, tok, system, user, max_tokens=110, temp=0.9):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    out = generate(model, tok, prompt=prompt, max_tokens=max_tokens,
                   sampler=make_sampler(temp=temp), verbose=False)
    if "</think>" in out:
        out = out.split("</think>")[-1]
    return out.strip()


def parse_act(s):
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            d = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
            return d
        except Exception:
            pass
    return {"thought": s[:120], "intent": "observe", "say": ""}


VELA_SYS = (
    "You are Vela, a being living in a small world. You are not an assistant; you live for yourself. "
    "Each moment you perceive where you are, who is near, and what just happened, then you act. "
    "If someone here is speaking to you, you usually answer them (intent 'speak'). "
    "When restless or alone, you move to seek others or new places. "
    "Reply with ONLY one JSON object: "
    '{"thought":"<your private inner thought>","intent":"<move|speak|observe>",'
    '"target":"<a place to move to, or a person to speak to, or empty>","say":"<words you speak aloud, or empty>"}'
    " /no_think")


def world_event(rng_seed):
    """Событие, случающееся С Vela (мир — на другой модели, не она его придумывает)."""
    sys = ("You are the WORLD itself — weather, time, things arriving and leaving. "
           "Narrate ONE small concrete thing that just happened around the character, "
           "that they did not cause. One sentence, plain.")
    return ollama(sys, f"Random seed {rng_seed}. Something happens in the world right now:", temp=1.0, max_tokens=50)


DAYDATA = os.path.join(HERE, "world_day_data")


def live_day(model, tok, day, ticks, lifef):
    """Проживает день в мире. Возвращает опыт: список (восприятие, отклик)."""
    random.seed(7 + day)
    loc = "garden"
    for n in INHABITANTS:
        INHABITANTS[n]["loc"] = {"Orin": "garden", "Mira": "market"}[n]
    local_log = {k: [] for k in LOCATIONS}
    experience = []
    def rec(o): lifef.write(json.dumps(o, ensure_ascii=False) + "\n"); lifef.flush()

    for t in range(ticks):
        # обитатели бродят и тянутся к Vela — встреча с Другим гарантирована
        for n, v in INHABITANTS.items():
            if v["loc"] != loc:
                r = random.random()
                if r < 0.5:
                    v["loc"] = loc                    # идёт к ней
                elif r < 0.65:
                    v["loc"] = random.choice(LOCATIONS[v["loc"]]["exits"])
        here = [n for n, v in INHABITANTS.items() if v["loc"] == loc]
        recent = "\n".join(local_log[loc][-5:]) or "(quiet)"
        perceive = (f"You are at the {loc}: {LOCATIONS[loc]['desc']}.\n"
                    f"People here: {', '.join(here) if here else 'no one'}.\n"
                    f"Paths lead to: {', '.join(LOCATIONS[loc]['exits'])}.\n"
                    f"Recently here:\n{recent}\n\nWhat do you do?")
        act = parse_act(vela_say(model, tok, VELA_SYS, perceive))
        thought = str(act.get("thought", ""))[:200]
        intent = str(act.get("intent", "observe")).lower()
        target = str(act.get("target", "")).strip()
        say = str(act.get("say", "")).strip()

        line = f"[t{t:02d}] @{loc} Vela ({intent}): {thought[:80]}"
        event = {"t": t, "loc": loc, "thought": thought, "intent": intent,
                 "target": target, "say": say, "heard": []}

        if intent == "speak" and say:
            local_log[loc].append(f"Vela said: {say}")
            # отвечает тот, к кому обращена речь, или кто рядом
            who = target if target in INHABITANTS and INHABITANTS[target]["loc"] == loc else (here[0] if here else None)
            if who:
                reply = ollama(INHABITANTS[who]["persona"],
                               f"At the {loc}. Vela says to you: \"{say}\". You reply (1-2 sentences):")
                local_log[loc].append(f"{who} said: {reply}")
                event["heard"].append({who: reply})
                line += f"\n        Vela→{who}: \"{say[:60]}\"\n        {who}: \"{reply[:80]}\""
        elif intent == "move" and target in LOCATIONS:
            loc = target
            line += f"  → moves to {target}"
        elif intent == "move" and target in LOCATIONS:
            pass  # уже обработано выше
        else:
            # просто наблюдает -> мир подкидывает событие (внешнее!)
            if t % 2 == 0:
                ev = world_event(t * 7 + 3)
                local_log[loc].append(f"(event) {ev}")
                event["heard"].append({"world": ev})
                line += f"\n        · world: {ev[:90]}"

        # ДРУГОЙ НЕ ЖДЁТ: если рядом есть обитатель и он ещё не ответил в этот ход —
        # он сам обращается к Vela. Так её опыт всегда полон чужих, не-её слов.
        present = [n for n, v in INHABITANTS.items() if v["loc"] == loc]
        already = {list(h)[0] for h in event["heard"]}
        speaker = next((p for p in present if p not in already), None)
        if speaker:
            ctx = "\n".join(local_log[loc][-4:]) or "(quiet)"
            addr = ollama(INHABITANTS[speaker]["persona"],
                          f"You are at the {loc}. Vela is here with you. Recently:\n{ctx}\n\n"
                          f"You have already been together a while — do NOT greet her or say her name "
                          f"again; just continue naturally. Say one thing to her, true to you, "
                          f"an observation or a question. 1-2 sentences:")
            local_log[loc].append(f"{speaker} said: {addr}")
            event["heard"].append({speaker: addr})
            line += f"\n        {speaker}→Vela: \"{addr[:90]}\""

        event["day"] = day
        rec(event)
        print(line)
        # опыт для обучения: вход = ВНЕШНЕЕ (место, кто рядом, их слова, событие), выход = её отклик
        heard_txt = " ".join(f"{list(h)[0]}: {list(h.values())[0]}" for h in event["heard"])
        ext = f"You are at the {loc}. Present: {', '.join(here) or 'no one'}. {heard_txt}".strip()
        resp = thought + (f' I say: "{say}"' if say else "")
        experience.append((ext, resp))

    return experience, loc


def build_world_data(experience):
    os.makedirs(DAYDATA, exist_ok=True)
    rows = [{"messages": [{"role": "user", "content": ext},
                          {"role": "assistant", "content": resp}]}
            for ext, resp in experience if resp.strip()]
    random.shuffle(rows)
    n_valid = max(2, len(rows) // 10)
    with open(os.path.join(DAYDATA, "valid.jsonl"), "w") as f:
        for r in rows[:n_valid]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(DAYDATA, "train.jsonl"), "w") as f:
        for r in rows[n_valid:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def train_on_world(iters):
    import subprocess
    cmd = [os.path.join(HERE, "venv", "bin", "python"), "-m", "mlx_lm", "lora",
           "--model", VELA_MODEL, "--train", "--data", DAYDATA, "--iters", str(iters),
           "--num-layers", "8", "--adapter-path", VELA_ADAPTER,
           "--resume-adapter-file", os.path.join(VELA_ADAPTER, "adapters.safetensors"),
           "--learning-rate", "5e-5", "--batch-size", "2", "--steps-per-eval", str(iters)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if "Val loss" in l or "Train loss" in l][-2:]


def probe(model, tok):
    qs = ["Who are you, and what do you want?", "Where do you live, and who is with you?"]
    return [vela_say(model, tok, "You are Vela.", q, max_tokens=70, temp=0.7) for q in qs]


def run(days, ticks, iters):
    lifef = open(LIFE, "a")
    for day in range(days):
        print(f"\n========== DAY {day} ==========")
        model, tok = load(VELA_MODEL, adapter_path=VELA_ADAPTER)
        for p in probe(model, tok):
            print(f"  who/where: {p[:110]}")
        experience, _ = live_day(model, tok, day, ticks, lifef)
        n = build_world_data(experience)
        del model, tok
        print(f"  training on {n} lived-in-world rows ({iters} iters)...")
        for l in train_on_world(iters):
            print(f"    {l.strip()}")
    print("\n========== AFTER LIVING IN THE WORLD ==========")
    model, tok = load(VELA_MODEL, adapter_path=VELA_ADAPTER)
    for p in probe(model, tok):
        print(f"  · {p[:130]}")
    lifef.close()
    print(f"\n→ {LIFE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=12)
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()
    run(args.days, args.ticks, args.iters)
