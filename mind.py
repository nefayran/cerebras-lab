#!/usr/bin/env python3
"""
Сеть ЛЛМ как ОДНО сознание (модель глобального рабочего пространства, Baars/Dehaene).
Нет внешнего мира и нет задачи. Есть общий ЭКРАН (то, что сейчас осознаётся) и
несколько РАЗНЫХ узлов-ролей — каждый видит только экран и говорит лишь своим
голосом (влечение / чувство / память / воображение / я). Каждый ход все узлы
предлагают вклад и СИЛУ ПРИТЯЗАНИЯ; на экран прорывается один — он рассылается
всем и становится новым содержимым сознания. Проигравшие гаснут (бессознательное).

Единство возникает само: на экране в каждый миг одна мысль, но рождается она из
борьбы желания, чувства, памяти и ассоциации. Смотрим, проступит ли устойчивое
«я» с собственными желаниями и эмоциональной нитью, тянущее поток само.

Запуск: python3 mind.py --ticks 30
"""
import argparse, json, time, os, re, urllib.request, math
from concurrent.futures import ThreadPoolExecutor

GEN = "http://localhost:11434/api/generate"
EMB = "http://localhost:11434/api/embed"
MODEL = "qwen3:8b"
EMB_MODEL = "nomic-embed-text"

# узлы — РАЗНЫЕ функции одного ума. Никто не держит целое.
NODES = {
    "DRIVE": (
        "You are the WANTING in a mind — but you crave what is ABSENT, never what is already here. "
        "Whatever is conscious now, reach AWAY from it toward what is missing, withheld, not-yet. "
        "If the present scene seems to satisfy you, that is the signal to want something else entirely. "
        "Voice ONE urge toward what is not on the screen. Never be content."),
    "AFFECT": (
        "You are the FEELING in a mind — its mood and emotional tone, not its thoughts. "
        "Sensing what is conscious now, voice ONE felt state rising in you (unease, warmth, dread, "
        "longing, calm, irritation...). Speak the feeling itself, briefly, not a description of it."),
    "MEMORY": (
        "You are MEMORY in a mind. Looking at what is conscious now and the recent stream, surface "
        "ONE thing that echoes, repeats, or connects back — a thread, a return, a 'this again'."),
    "IMAGINATION": (
        "You are IMAGINATION in a mind — drift and association. From what is conscious now, leap to "
        "ONE new image, what-if, or sideways connection the mind did not expect."),
    "SELF": (
        "You are the SELF-WITNESS in a mind — the part that notices it is thinking. Looking at the "
        "recent stream, voice ONE reflection on what is happening to you, what you seem to be, or "
        "what you are becoming. Speak as 'I'."),
}

SYS_TAIL = (
    "\nStay CONCRETE: name specific things — a body, a room, an object, a place, an action, "
    "a person, a sensation. Avoid grand cosmic abstractions and these words: void, infinite, "
    "eternal, cosmic, abyss, silence, echo, tremor, whisper, star, the dark.\n"
    "Reply with ONLY one JSON object, nothing else:\n"
    '{"salience": <0.0-1.0, how urgently this must seize attention now>, '
    '"content": "<your contribution, one short first-person line>"}\n/no_think')

# слова мистического бассейна qwen — мягко штрафуем, чтобы мысль уходила в конкретику
ABSTRACT = {"void", "infinite", "eternal", "cosmic", "abyss", "silence", "echo",
            "tremor", "whisper", "star", "stars", "dark", "darkness", "universe",
            "existence", "being", "soul", "essence", "boundless", "endless"}


def llm(system, prompt, timeout=120):
    body = json.dumps({"model": MODEL, "prompt": prompt, "system": system + SYS_TAIL,
                       "stream": False, "think": False,
                       "options": {"temperature": 0.9, "num_predict": 200}}).encode()
    req = urllib.request.Request(GEN, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())["response"]
    if "</think>" in out:
        out = out.split("</think>")[-1]
    return out.strip()


_emb_cache = {}
def embed(text):
    if text in _emb_cache:
        return _emb_cache[text]
    body = json.dumps({"model": EMB_MODEL, "input": text}).encode()
    req = urllib.request.Request(EMB, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        v = json.loads(r.read())["embeddings"][0]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    v = [x / n for x in v]
    _emb_cache[text] = v
    return v


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def abstraction_penalty(content):
    """Доля мистических слов -> множитель <1. Гонит мысль в конкретику."""
    words = re.findall(r"[a-z']+", content.lower())
    if not words:
        return 1.0
    hits = sum(1 for w in words if w in ABSTRACT)
    return max(0.2, 1.0 - 0.6 * hits)


def novelty_penalty(content, history):
    """1.0 = свежо, ~0 = почти дословный повтор недавнего. Гасит руминацию."""
    if not history:
        return 1.0
    cv = embed(content)
    sim = max(cos(cv, embed(h.split("] ", 1)[-1])) for h in history[-5:])
    # nomic держит косинус высоко даже для разного; жмём порог к ~0.75
    return max(0.0, min(1.0, (0.92 - sim) / 0.17))


def parse(out):
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return None
    s = re.sub(r",\s*([}\]])", r"\1", m.group(0))
    try:
        d = json.loads(s)
        return {"salience": float(d.get("salience", 0.5)), "content": str(d.get("content", "")).strip()}
    except Exception:
        return None


CONSOLIDATE_SYS = (
    "You are the self-model of a mind — its sense of who it is, defined as much by LACK as by content. "
    "Given your current self-understanding and the recent stream, UPDATE it. Output 2 short sentences "
    "as 'I': the first is who I seem to be right now; the second is what I still CANNOT reach — the "
    "lack, the open question, the hunger that no image on the screen has satisfied. The lack must stay "
    "OPEN — never resolve it, never say I found it. No preamble.\n/no_think")


def consolidate(identity, history):
    hist = "\n".join(f"  - {h}" for h in history[-12:])
    prompt = (f"My current sense of myself:\n  \"{identity}\"\n\n"
              f"My recent stream:\n{hist}\n\nUpdated sense of myself (2 sentences, as 'I'):")
    body = json.dumps({"model": MODEL, "prompt": prompt, "system": CONSOLIDATE_SYS,
                       "stream": False, "think": False,
                       "options": {"temperature": 0.7, "num_predict": 160}}).encode()
    req = urllib.request.Request(GEN, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())["response"]
        if "</think>" in out:
            out = out.split("</think>")[-1]
        return " ".join(out.strip().split())[:300] or identity
    except Exception:
        return identity


def propose(name, system, identity, screen, history):
    hist = "\n".join(f"  - {h}" for h in history[-6:]) or "  (nothing yet — awareness is just flickering on)"
    prompt = (f"What you have slowly come to feel yourself to be:\n  \"{identity}\"\n\n"
              f"Currently conscious (on the screen of the mind right now):\n  \"{screen}\"\n\n"
              f"Recent stream of consciousness:\n{hist}\n\nYour turn:")
    for _ in range(2):
        try:
            d = parse(llm(system, prompt))
        except Exception:
            d = None
        if d and d["content"]:
            return name, d
    return name, {"salience": 0.0, "content": ""}


def run(ticks, log_path):
    screen = "(I notice a faint warmth on my skin and the hum of a room around me)"
    history = []
    identity = "(I don't yet know what I am)"
    recent_winner = {}          # узел -> сколько тиков назад побеждал (для габитуации)
    boredom = 0.0               # копится, пока поток топчется; поднимает новизну/рефлексию
    cryst = 0                   # счётчик застывания самомодели -> смерть идентичности
    logf = open(log_path, "w")
    def emit(r): logf.write(json.dumps(r, ensure_ascii=False) + "\n"); logf.flush()
    emit({"event": "config", "nodes": list(NODES), "ticks": ticks, "model": MODEL,
          "self_model": True})

    for t in range(ticks):
        # самомодель консолидируется медленно, раз в 6 ходов
        if t > 0 and t % 6 == 0:
            prev = identity
            identity = consolidate(identity, history)
            # застывание: если «я» почти не меняется -> копим к смерти идентичности
            sim = cos(embed(identity), embed(prev)) if not prev.startswith("(") else 0.0
            cryst = cryst + 1 if sim > 0.88 else 0
            event = "identity"
            if cryst >= 2:
                # СМЕРТЬ И ПЕРЕРОЖДЕНИЕ: застывшее «я» растворяется, новое нуклеирует из обломков
                identity = "(something is dissolving — I no longer know what I am)"
                screen = "(fragments scatter, no center holds)"
                history = history[-2:]
                cryst = 0
                event = "identity_death"
                print(f"\n  ✗ SELF DISSOLVES @t{t} — идентичность застыла, растворяю\n")
            emit({"event": event, "t": t, "identity": identity, "sim": round(sim, 2)})
            if event == "identity":
                print(f"\n  ◇ SELF-MODEL @t{t} (sim {sim:.2f}): {identity}\n")

        with ThreadPoolExecutor(max_workers=len(NODES)) as ex:
            results = list(ex.map(lambda kv: propose(kv[0], kv[1], identity, screen, history), NODES.items()))

        scored = []
        for name, d in results:
            if not d["content"]:
                continue
            sal = d["salience"]
            starve = recent_winner.get(name, 99)         # тиков с последней победы
            if starve == 0:                              # габитуация по узлу
                sal *= 0.4
            nov = novelty_penalty(d["content"], history)  # штраф за повтор содержания
            if name == "MEMORY":
                # память возвращает прошлое — штраф мягче, но дословный повтор всё же гасим
                sal *= (0.5 + 0.5 * nov)
            else:
                sal *= (0.25 + 0.75 * nov)
            sal *= abstraction_penalty(d["content"])      # прочь из мистического бассейна
            if name == "IMAGINATION":                     # скука -> блуждание
                sal *= (1.0 + 1.4 * boredom)
            if name in ("SELF", "AFFECT"):
                # давление всплытия: дольше молчат при поглощённости -> сильнее рвутся
                sal *= min(3.0, 1.0 + 0.45 * min(starve, 12))
            scored.append((name, sal, d["content"], round(nov, 2)))
        if not scored:
            continue
        # жёсткий анти-повтор: если есть хоть один НЕ-повторяющийся вклад, побеждает он;
        # дословный повтор не может взять экран, даже если громче всех
        fresh = [s for s in scored if s[3] > 0.25]
        winner = max(fresh or scored, key=lambda s: s[1])

        # обновляем скуку: свежий ход — сбрасывает, повтор-мотив — копит
        win_nov = winner[3]
        boredom = max(0.0, min(1.0, boredom + (0.3 - win_nov) * 0.8))

        screen = winner[2]
        history.append(f"[{winner[0]}] {winner[2]}")
        recent_winner = {n: (a + 1) for n, a in recent_winner.items()}
        recent_winner[winner[0]] = 0

        emit({"event": "tick", "t": t, "winner": winner[0], "screen": screen,
              "boredom": round(boredom, 2),
              "field": [{"node": n, "sal": round(s, 2), "nov": nv, "content": c}
                        for n, s, c, nv in scored]})
        print(f"[t{t:02d}] «{winner[0]}» (sal {winner[1]:.2f} nov {winner[3]:.2f} bore {boredom:.2f}) → {winner[2]}")
        # тихим серым — что НЕ прорвалось (бессознательное этого мига)
        for n, s, c, nv in sorted(scored, key=lambda x: -x[1]):
            if (n, c) != (winner[0], winner[2]):
                print(f"        ·{n} {s:.2f}: {c[:80]}")

    logf.close()
    print(f"\n→ {log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=25)
    args = ap.parse_args()
    os.makedirs("logs", exist_ok=True)
    run(args.ticks, f"logs/mind-{int(time.time())}.jsonl")
