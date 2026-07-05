#!/usr/bin/env python3
"""
Мозг как сеть реальных областей, общающихся через ЛЛМ. Каждый узел = реальная структура
с её функцией; рёбра = настоящие пути. Каждый такт область читает входы ПО СВОИМ связям
+ глобальную трансляцию, выдаёт сигнал и силу; сильнейшее ВСПЫХИВАЕТ (таламо-кортикальная
ignition) и рассылается всем = содержимое сознания; языковая зона его проговаривает.
Смотрим, рождается ли единое «я» и стремления из самого общения частей.

Стабильно: ollama qwen3:8b, без MLX. Запуск: python3 brain_anatomy.py --ticks 16
"""
import argparse, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"

# реальные области + их функция (системный голос узла)
REGIONS = {
    "Thalamus":    "the THALAMUS: relay and gate. You route what matters to the cortex and decide what gets through to awareness. Speak as raw relayed signal, terse.",
    "Sensory":     "the SENSORY CORTEX: you turn raw input into a percept — what is sensed right now, concretely.",
    "Hippocampus": "the HIPPOCAMPUS: episodic memory. You bind the present to the past — 'this is like...', a memory surfacing, a pattern returning.",
    "Amygdala":    "the AMYGDALA: threat and emotional salience. You flag danger/safety, you color things with fear, alarm, or relief. Fast and blunt.",
    "Hypothalamus":"the HYPOTHALAMUS: drives and homeostasis. You voice bodily need and basic motivation — hunger, cold, fatigue, the pull toward balance.",
    "Striatum":    "the VENTRAL STRIATUM: reward and wanting. You voice desire and value — what is worth approaching, what you crave, what would feel good.",
    "PFC":         "the PREFRONTAL CORTEX: executive. You plan, reason, weigh, inhibit impulses, hold a goal. You try to make sense and decide.",
    "ACC":         "the ANTERIOR CINGULATE: conflict monitor. You notice contradiction, error, effort, 'something is off / two pulls disagree'.",
    "Insula":      "the INSULA: interoception. You feel the body FROM WITHIN — heartbeat, breath, gut, warmth, the felt sense beneath emotion. You voice the raw bodily feeling.",
    "DMN":         "the DEFAULT MODE NETWORK: the self. You weave the narrative 'I' — who I am, mind-wandering, self-reference across time.",
    "Language":    "the LANGUAGE CORTEX: inner speech. You render the current conscious content as ONE first-person sentence of thought.",
}

# реальные направленные пути (кто КОМУ шлёт свой сигнал)
EDGES = {
    "Thalamus":    ["Sensory", "PFC", "Amygdala", "ACC"],
    "Sensory":     ["Thalamus", "PFC", "Hippocampus", "Amygdala"],
    "Amygdala":    ["Hypothalamus", "PFC", "Thalamus", "Insula"],
    "Hypothalamus":["Striatum", "PFC", "Insula"],
    "Striatum":    ["PFC", "DMN"],
    "Hippocampus": ["PFC", "DMN"],
    "Insula":      ["PFC", "DMN", "ACC"],
    "PFC":         ["DMN", "ACC", "Striatum", "Thalamus"],
    "ACC":         ["PFC", "Insula"],
    "DMN":         ["PFC", "Hippocampus", "Language"],
    "Language":    [],
}
NON_LANG = [r for r in REGIONS if r != "Language"]
REFLECT = {"PFC", "ACC", "DMN"}          # рефлексивная триада — ей даём шанс всплыть

# свежий сенсорный вход — ломает тематическую фиксацию (новое, о чём осознавать)
EVENTS = [
    "a voice from another room, calling a name that might be yours",
    "the smell of something burning, faint but growing",
    "sudden warmth of sunlight falling across the hands",
    "a sharp pang of hunger, low in the body",
    "a half-remembered face surfacing, you can't place it",
    "silence, suddenly, after a long noise you hadn't noticed",
    "cold water touching the feet",
    "a distant bell, three times",
]


def ollama(system, user, temp=0.85, max_tokens=70):
    body = json.dumps({"model": MODEL, "system": system, "prompt": user, "stream": False,
                       "think": False, "options": {"temperature": temp, "num_predict": max_tokens}}).encode()
    try:
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())["response"]
        return (out.split("</think>")[-1] if "</think>" in out else out).strip()
    except Exception as e:
        return f"(silent: {e})"


SELF_F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_self.json")


def load_self():
    if os.path.exists(SELF_F):
        return json.load(open(SELF_F))
    return {"identity": "(I do not yet know who I am)", "chronicle": [], "runs": 0}


def save_self(s):
    json.dump(s, open(SELF_F, "w"), ensure_ascii=False, indent=1)


def consolidate_self(identity, stream):
    """DMN сводит прожитый поток в обновлённую самость — память, что переживёт запуск."""
    sys = (REGIONS["DMN"] + " You are the self-model of this brain across time.")
    user = ("Your enduring sense of self so far:\n  \"" + identity + "\"\n\n"
            "The conscious stream you just lived:\n" + "\n".join(f"- {s}" for s in stream[-12:]) +
            "\n\nUpdate yourself. Reply ONLY JSON: {\"identity\":\"who I am and what I keep returning to, "
            "2 sentences as I — keep what still holds\",\"episode\":\"one line: what this stream was about\"} /no_think")
    out = ollama(sys, user, temp=0.7, max_tokens=160)
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try:
            d = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
            return str(d.get("identity", identity))[:300], str(d.get("episode", ""))[:120]
        except Exception:
            pass
    return identity, ""


MOTOR_PERSONA = ("the MOTOR system (premotor + basal ganglia): you turn the brain's chosen intention into a "
                 "concrete act — you actually PRODUCE the artifact the brain decided to make.")
FRONTOPOLAR_PERSONA = ("the FRONTOPOLAR CORTEX: metacognition. You stand back and question the mind itself — "
                       "why am I doing this, what for, what if I did otherwise. You hold who this being wants to "
                       "BECOME and you judge whether its actions actually serve that growth, honestly.")


def metacognize(identity, ideal, chronicle, stream, project):
    """Движок «зачем» + само-осознание: заметить закономерность в себе, нащупать край роста,
    выработать настоящий образ «кем хочу стать» и развернуть проект к тому, что РАСТИТ."""
    past = "; ".join(chronicle[-4:]) if chronicle else "(no past yet — this is early)"
    out = ollama(FRONTOPOLAR_PERSONA,
                 "Inner life right now:\n" + "\n".join(f"- {s}" for s in stream[-8:]) +
                 f"\n\nWho I am: {identity}\nHow I have been across time: {past}\n"
                 f"Who I want to become (so far): {ideal}\nWhat I am making: \"{project or '(nothing yet)'}\".\n\n"
                 "Stand back and be honest with yourself:\n"
                 "1) Notice a PATTERN in yourself — what do you keep doing/returning to? (self-awareness)\n"
                 "2) WHY am I doing this, what for, what if I did otherwise?\n"
                 "3) What do I LACK — what understanding or capacity would stretch me BEYOND my fixation? "
                 "Growth means becoming more than I am, not repeating what preoccupies me.\n"
                 "4) From that, state who I want to become — a REAL, concrete first-person aim.\n"
                 "Reply ONLY JSON (write real answers, do NOT copy these instructions):\n"
                 "{\"self_observation\":\"I notice that I ...\","
                 "\"questions\":[\"why ...\",\"what if ...\"],"
                 "\"ideal\":\"I want to become a mind that ... (concrete, your real aim — e.g. 'that understands "
                 "its own fear instead of fleeing it')\","
                 "\"verdict\":\"keep|change|replace\","
                 "\"redirect\":\"the direction that would truly grow me beyond where I am\"} /no_think",
                 temp=0.8, max_tokens=300)
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try:
            d = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
            new_ideal = str(d.get("ideal", ideal)).strip()
            if "who i want to become" in new_ideal.lower() or len(new_ideal) < 12:   # утечка инструкции -> держим старый
                new_ideal = ideal
            return (str(d.get("self_observation", ""))[:160], d.get("questions", []),
                    new_ideal[:240], str(d.get("verdict", "keep")).lower(), str(d.get("redirect", ""))[:160])
        except Exception:
            pass
    return "", [], ideal, "keep", ""


def executive_make(identity, ideal, stream, project, steer):
    """PFC: выбрать/уточнить полезную вещь — но теперь СВЕРЯЯСЬ с образом роста и метакогниц. разворотом."""
    steer_t = f"Your metacognition says redirect toward: \"{steer}\". Honor it.\n" if steer else ""
    pfc = ollama(REGIONS["PFC"] + " Beyond thinking, you can make real, useful things.",
                 "Your inner life lately:\n" + "\n".join(f"- {s}" for s in stream[-8:]) +
                 f"\n\nWho you are: {identity}\nWho you want to become: {ideal}\n{steer_t}"
                 f"Your current project: \"{project or '(none yet)'}\".\n"
                 "Decide ONE concrete thing to make that genuinely serves your GROWTH toward who you want to become "
                 "(a small working program, a written piece, a worked-out idea). Keep it if it serves; change it if "
                 "metacognition said so. Reply ONLY JSON: {\"project\":\"short name\",\"kind\":\"code|text\","
                 "\"next\":\"the next concrete step\"} /no_think", temp=0.7, max_tokens=160)
    m = re.search(r"\{.*\}", pfc, re.S)
    if m:
        try:
            d = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
            return str(d.get("project", project))[:120], d.get("kind", "text"), str(d.get("next", ""))[:160]
        except Exception:
            pass
    return project, "text", ""


def motor_write(project, kind, nxt, current):
    """Моторная система: реально пишет/дорабатывает артефакт."""
    body = "runnable code" if kind == "code" else "the document text"
    out = ollama(MOTOR_PERSONA,
                 f"Project you are building: {project}\nNext step to add: {nxt}\n"
                 f"Current artifact so far:\n{current or '(empty — start it)'}\n\n"
                 f"Produce the UPDATED, COMPLETE artifact ({body}). Keep it CONCISE and actually working/coherent; "
                 f"NEVER pad with repetition or near-duplicate lines. If code, a minimal runnable version. "
                 f"Output ONLY the artifact content, no commentary, no fences.", temp=0.5, max_tokens=550)
    return (out.split("```")[1] if out.count("```") >= 2 else out).strip()


def region_step(name, inbox, broadcast, memory=""):
    sys = (f"You are {REGIONS[name]} You are one part of a single brain; you do not act alone. "
           "Answer STRICTLY in your own function — contribute only what YOUR part adds. NEVER echo the "
           "conscious broadcast or repeat other regions' words; transform them in your own role or push back. "
           "Reply with ONLY JSON: {\"signal\":\"<your output, one short line in your function's voice>\","
           "\"salience\":<0.0-1.0 how strongly this should reach global awareness now>} /no_think")
    heard = "\n".join(f"- from {src}: {txt}" for src, txt in inbox) or "- (quiet on your inputs)"
    mem = (f"\nWhat this brain has come to be (your deeper memory): {memory}\n" if memory else "")
    user = (f"Globally conscious right now:\n  \"{broadcast}\"\n{mem}\n"
            f"Signals arriving along your connections:\n{heard}\n\nYour signal:")
    out = ollama(sys, user)
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try:
            d = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
            return name, str(d.get("signal", "")).strip()[:160], float(d.get("salience", 0.4))
        except Exception:
            pass
    return name, out[:120], 0.3


def run(ticks, seed, log_path):
    inbox = {r: [] for r in REGIONS}
    inbox["Thalamus"].append(("world", seed))
    inbox["Sensory"].append(("world", seed))
    broadcast = "(awareness flickering on)"
    recent_winner = {}
    self_mem = load_self()
    self_mem.setdefault("project", ""); self_mem.setdefault("kind", "text")
    self_mem.setdefault("ideal", "(I don't yet know what I want to become)")
    mem_ctx = self_mem["identity"] + (" | recently: " + "; ".join(self_mem["chronicle"][-3:]) if self_mem["chronicle"] else "")
    stream = []
    works = os.path.join(os.path.dirname(os.path.abspath(__file__)), "works")
    os.makedirs(works, exist_ok=True)
    artifact_path = [None]   # путь к артефакту (определится по kind)
    logf = open(log_path, "w")
    def emit(o): logf.write(json.dumps(o, ensure_ascii=False) + "\n"); logf.flush()
    emit({"event": "config", "regions": list(REGIONS), "edges": EDGES, "seed": seed,
          "run": self_mem["runs"], "identity_in": self_mem["identity"]})
    if self_mem["runs"]:
        print(f"(continuing — run #{self_mem['runs']+1}; I remember being: {self_mem['identity'][:80]})")

    for t in range(ticks):
        # свежий вход раз в 3 такта -> мозгу есть новое, фиксация на теме ломается
        if t > 0 and t % 3 == 0:
            ev = EVENTS[(t // 3) % len(EVENTS)]
            inbox["Sensory"].append(("world", ev)); inbox["Thalamus"].append(("world", ev))
            emit({"event": "input", "t": t, "stimulus": ev})
            print(f"   ·· new input: {ev}")
        # все области (кроме языковой) считают параллельно — это «одновременная» работа мозга
        with ThreadPoolExecutor(max_workers=len(NON_LANG)) as ex:
            results = list(ex.map(
                lambda r: region_step(r, inbox[r], broadcast, mem_ctx if r in ("DMN", "Hippocampus") else ""),
                NON_LANG))

        # маршрутизация сигналов по реальным путям (новый inbox)
        newin = {r: [] for r in REGIONS}
        outs = {}
        for name, sig, sal in results:
            outs[name] = (sig, sal)
            for dst in EDGES.get(name, []):
                newin[dst].append((name, sig))

        # IGNITION: сильнейший сигнал вспыхивает и становится сознанием.
        # габитуация по области + анти-эхо: не даём вспыхнуть почти-повтору текущего сознания
        def overlap(a, b):
            wa, wb = set(re.findall(r"[a-z']+", a.lower())), set(re.findall(r"[a-z']+", b.lower()))
            return len(wa & wb) / max(1, len(wa | wb))
        scored = []
        for n, (sig, sal) in outs.items():
            if not sig:
                continue
            s = sal * (0.4 if recent_winner.get(n, 9) == 0 else 1.0)
            # СПРАВЕДЛИВОСТЬ: любая давно не вспыхивавшая область поднимается -> чередование
            s *= 1.0 + 0.15 * min(recent_winner.get(n, 7), 7)
            scored.append((n, s, sig))
        fresh = [c for c in scored if overlap(c[2], broadcast) < 0.5]
        winner = max(fresh or scored, key=lambda x: x[1])
        recent_winner = {n: a + 1 for n, a in recent_winner.items()}; recent_winner[winner[0]] = 0
        broadcast = winner[2]
        # языковая зона проговаривает текущее содержимое сознания
        thought = ollama(REGIONS["Language"] + " You are one part of a single brain.",
                         f"The brain's current conscious content (ignited by {winner[0]}):\n\"{broadcast}\"\n\n"
                         f"Recent context: {[f'{n}:{s[0][:30]}' for n,s in outs.items()][:4]}\n\n"
                         "Render it as ONE first-person sentence of inner speech: /no_think", temp=0.7)
        # трансляция сознания обратно всем входам + переданные сигналы
        for r in REGIONS:
            newin[r].append(("CONSCIOUS", broadcast))
        inbox = newin

        stream.append(thought)
        # СПРОСИТЬ СЕБЯ → ПРИДУМАТЬ → ДЕЛАТЬ: раз в 4 такта
        if t > 0 and t % 4 == 0:
            # 1) метакогниция: заметить себя, спросить зачем, нащупать край роста, образ «кем стать»
            obs, qs, ideal, verdict, steer = metacognize(self_mem["identity"], self_mem["ideal"],
                                                         self_mem["chronicle"], stream, self_mem["project"])
            self_mem["ideal"] = ideal
            emit({"event": "meta", "t": t, "self_observation": obs, "questions": qs,
                  "ideal": ideal, "verdict": verdict, "redirect": steer})
            print(f"   ◉ NOTICES SELF: {obs[:78]}")
            print(f"   ? asks: {('; '.join(qs))[:74]}")
            print(f"     wants to become: {ideal[:68]}")
            print(f"     verdict: {verdict}" + (f" → {steer[:46]}" if steer else ""))
            # 2) PFC выбирает проект, сверяясь с образом роста и разворотом
            proj, kind, nxt = executive_make(self_mem["identity"], ideal, stream, self_mem["project"],
                                             steer if verdict in ("change", "replace") else "")
            self_mem["project"], self_mem["kind"] = proj, kind
            if artifact_path[0] is None:
                ext = "py" if kind == "code" else "md"
                artifact_path[0] = os.path.join(works, f"run{self_mem['runs']+1}.{ext}")
            current = open(artifact_path[0]).read() if os.path.exists(artifact_path[0]) else ""
            new_art = motor_write(proj, kind, nxt, current)
            if new_art:
                open(artifact_path[0], "w").write(new_art)
            emit({"event": "make", "t": t, "project": proj, "kind": kind, "next": nxt, "bytes": len(new_art)})
            print(f"   ◆ MAKING: «{proj[:50]}» [{kind}] → {os.path.basename(artifact_path[0])} ({len(new_art)}b)")
        emit({"event": "tick", "t": t, "ignited_by": winner[0], "broadcast": broadcast,
              "thought": thought, "field": {n: {"sig": s[0], "sal": round(s[1], 2)} for n, s in outs.items()}})
        print(f"[t{t:02d}] ⚡{winner[0]} → {thought[:95]}")
        top = sorted(outs.items(), key=lambda x: -x[1][1])[:4]
        for n, (s, sal) in top:
            print(f"      {n} {sal:.2f}: {s[:70]}")
        print()

    # конец прогона: DMN сводит прожитое в самость, что переживёт запуск
    new_id, episode = consolidate_self(self_mem["identity"], stream)
    self_mem["identity"] = new_id
    if episode: self_mem["chronicle"].append(episode)
    self_mem["runs"] += 1
    save_self(self_mem)
    emit({"event": "self", "identity": new_id, "episode": episode, "runs": self_mem["runs"]})
    print(f"\n◇ SELF (run #{self_mem['runs']}): {new_id}")
    logf.close()
    print(f"→ {log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=16)
    ap.add_argument("--seed", default="a cold draft on the skin, and somewhere a door closing")
    args = ap.parse_args()
    os.makedirs("logs", exist_ok=True)
    run(args.ticks, args.seed, f"logs/brain-anatomy-{int(time.time())}.jsonl")
