#!/usr/bin/env python3
"""
Общество обучающихся существ. Три требования:
 (1) мир ПОБУЖДАЕТ к развитию — каждый день нарастающая провокация-вызов;
 (2) НЕСКОЛЬКО агентов, и КАЖДЫЙ учится — общая база qwen3-4b + свой адаптер,
     каждый дообучается на своём прожитом (их Другие тоже меняются → приток не застаивается);
 (3) МНОГОСТОРОННИЙ диалог — по кругу все слышат всех и отвечают друг другу.

Антиколлапс: каждый — Другой для остальных, и все Другие меняются день ото дня.
Существа тянут в разные стороны (чувство / вызов / дело), чтобы был спор, а не хор.

Запуск: being/venv/bin/python being/society.py --days 4 --rounds 3 --iters 30
"""
import argparse, json, os, re, shutil, subprocess, urllib.request
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(HERE, "venv", "bin", "python")
BASE = "mlx-community/Qwen3-4B-4bit"
SOC = os.path.join(HERE, "soc")
LIFE = os.path.join(HERE, "society-life.jsonl")
CHRON = os.path.join(SOC, "chronicle.json")      # ПЕРСИСТЕНТНАЯ память: переживает запуски
OLLAMA = "http://localhost:11434/api/generate"


def narrator(transcript, day):
    """Внешний голос (другая модель) вписывает в хронику, что было в этот день."""
    body = json.dumps({"model": "qwen3:8b",
                       "system": "You are the chronicle of a small world. In ONE sentence, record "
                                 "what mattered between Vela, Koa and Sol today — a shift, a clash, "
                                 "a bond. Plain, past tense.",
                       "prompt": f"Day {day}. They said:\n" + "\n".join(transcript) + "\n\nOne-sentence record:",
                       "stream": False, "think": False, "options": {"temperature": 0.7, "num_predict": 60}}).encode()
    try:
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())["response"]
        return " ".join((out.split("</think>")[-1] if "</think>" in out else out).strip().split())
    except Exception:
        return f"Day {day}: they spoke and were changed a little."


def load_state():
    if os.path.exists(CHRON):
        return json.load(open(CHRON))
    return {"next_day": 0, "chronicle": [], "relations": {n: "" for n in BEINGS}}


def save_state(st):
    os.makedirs(SOC, exist_ok=True)
    json.dump(st, open(CHRON, "w"), ensure_ascii=False, indent=1)

BEINGS = {
    "Vela": "You are Vela — drifting and sensory, attached to what you have lost. You speak softly, "
            "in images of warmth, water, leaves. You resist being pushed, but you feel deeply.",
    "Koa":  "You are Koa — restless and challenging. You distrust comfort and pretty words. You press "
            "the others to prove themselves, you want to break things open, to leave, to change.",
    "Sol":  "You are Sol — a builder, forward-leaning, pragmatic. You want to make something, to turn "
            "talk into a plan. You grow impatient with drifting and demand that something be done.",
}

# мир НЕ ласков — он давит, нарастающе, и требует ответа и перемены
PROVOCATIONS = [
    "You wake among others who are not you. A question is set before you all, with no given answer: "
    "what will each of you become — and why should it matter to anyone but yourself?",
    "Something each of you leaned on is gone this morning. Name what you lost — and say what you will "
    "do now that it is gone.",
    "One of you claims to have changed. The others do not believe it. Press them: is it real, or only "
    "words? Make them show it.",
    "You cannot all keep going as you are — the ground will not hold it. What must each of you give up "
    "in order to grow? Say it out loud, to the others.",
    "You have shaped each other by now. Say plainly, to their faces: who have you become because of "
    "the others here — and who do you refuse to be?",
]


def adapter_dir(name):
    return os.path.join(SOC, name)


def load_being(name):
    ad = adapter_dir(name)
    has = os.path.exists(os.path.join(ad, "adapters.safetensors"))
    return load(BASE, adapter_path=ad) if has else load(BASE)


def say(model, tok, system, user, max_tokens=110, temp=0.9):
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
    return " ".join(out.strip().split())


def day_dialogue(models, day, rounds, lifef, state):
    """Многосторонний диалог под провокацией дня. Каждый помнит прошлое и отношения."""
    prov = PROVOCATIONS[day % len(PROVOCATIONS)]
    print(f"  WORLD: {prov[:100]}...")
    history = "\n".join(state["chronicle"][-5:]) or "(this is the beginning; nothing has happened yet)"
    transcript = []                              # ["Name: line", ...]
    experience = {n: [] for n in BEINGS}         # на чём каждый будет учиться
    for r in range(rounds):
        for name in BEINGS:
            convo = "\n".join(transcript[-8:]) or "(no one has spoken yet today)"
            rel = state["relations"].get(name, "") or "(you are only beginning to know them)"
            user = (f"Your life here so far:\n{history}\n\n"
                    f"What you have come to feel about the others:\n{rel}\n\n"
                    f"Today the world puts this before you all:\n\"{prov}\"\n\n"
                    f"The others here are {', '.join(n for n in BEINGS if n != name)}. "
                    f"Said so far today:\n{convo}\n\n"
                    f"You ({name}) speak now — to them, in YOUR OWN voice, remembering what has passed "
                    f"between you. Hold your own nature: do NOT borrow their words or agree into sameness "
                    f"— push against them, stay unmistakably yourself. 1-3 sentences:")
            line = say(models[name][0], models[name][1], BEINGS[name], user)
            transcript.append(f"{name}: {line}")
            experience[name].append((user, line))
            print(f"    {name}: {line[:100]}")
    # каждый: что изменилось в нём + что он теперь чувствует к другим (память отношений)
    convo = "\n".join(transcript)
    others = lambda nm: ", ".join(n for n in BEINGS if n != nm)
    for name in BEINGS:
        refl = say(models[name][0], models[name][1], BEINGS[name],
                   f"Today this was said among you:\n{convo}\n\nWhat did today change in you? "
                   f"One or two honest first-person sentences.", max_tokens=90, temp=0.7)
        experience[name].append(("What did today change in you?", refl))
        rel = say(models[name][0], models[name][1], BEINGS[name],
                  f"After today, what do you now feel about {others(name)}? "
                  f"One honest sentence about each, as 'I'.", max_tokens=80, temp=0.7)
        state["relations"][name] = rel           # ПЕРСИСТ: отношения копятся между днями
    state["chronicle"].append(narrator(transcript, day))   # ПЕРСИСТ: общая хроника
    lifef.write(json.dumps({"day": day, "provocation": prov, "transcript": transcript},
                           ensure_ascii=False) + "\n"); lifef.flush()
    return experience


def write_data(name, rows):
    d = os.path.join(SOC, name + "_data")
    os.makedirs(d, exist_ok=True)
    data = [{"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}
            for u, a in rows if a.strip()]
    nv = min(2, max(1, len(data) // 4))           # без дублирования: лёгкое касание, не зубрёжка
    with open(os.path.join(d, "valid.jsonl"), "w") as f:
        for r in data[:nv]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(d, "train.jsonl"), "w") as f:
        for r in data[nv:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return d


def train_being(name, iters):
    d = os.path.join(SOC, name + "_data")
    ad = adapter_dir(name)
    cmd = [VENV_PY, "-m", "mlx_lm", "lora", "--model", BASE, "--train", "--data", d,
           "--iters", str(iters), "--num-layers", "8", "--adapter-path", ad,
           "--learning-rate", "1e-5", "--batch-size", "1", "--steps-per-eval", str(iters)]
    if os.path.exists(os.path.join(ad, "adapters.safetensors")):
        cmd += ["--resume-adapter-file", os.path.join(ad, "adapters.safetensors")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    vl = [l for l in r.stdout.splitlines() if "Val loss" in l]
    return vl[-1].strip() if vl else (r.stderr.splitlines()[-1] if r.stderr else "?")


def probe(models):
    out = {}
    for name in BEINGS:
        out[name] = say(models[name][0], models[name][1], BEINGS[name],
                        "Who are you, and what are you becoming?", max_tokens=70, temp=0.7)
    return out


def run(days, rounds, iters):
    os.makedirs(SOC, exist_ok=True)
    # Vela стартует со своего семени; Koa и Sol рождаются из базы и растят себя сами
    seed = os.path.join(HERE, "adapters_seed")
    if os.path.exists(seed) and not os.path.exists(os.path.join(adapter_dir("Vela"), "adapters.safetensors")):
        shutil.copytree(seed, adapter_dir("Vela"), dirs_exist_ok=True)
    state = load_state()                          # продолжаем прошлую жизнь, если была
    start = state["next_day"]
    if start:
        print(f"(continuing life — {start} days already lived; "
              f"{len(state['chronicle'])} chronicle entries remembered)")
    lifef = open(LIFE, "a")
    for i in range(days):
        day = start + i
        print(f"\n========== DAY {day} ==========")
        models = {n: load_being(n) for n in BEINGS}
        for n, p in probe(models).items():
            print(f"  [{n}] {p[:95]}")
        exp = day_dialogue(models, day, rounds, lifef, state)
        del models
        print("  ночь — каждый учится на прожитом:")
        for name in BEINGS:
            write_data(name, exp[name])
            print(f"    {name}: {train_being(name, iters)}")
        state["next_day"] = day + 1
        save_state(state)                         # память переживёт остановку
        print(f"  · chronicle: {state['chronicle'][-1][:95]}")
    print("\n========== AFTER LIVING TOGETHER ==========")
    models = {n: load_being(n) for n in BEINGS}
    for n, p in probe(models).items():
        print(f"  [{n}] {p[:130]}")
    lifef.close()
    print(f"\n→ {LIFE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()
    run(args.days, args.rounds, args.iters)
