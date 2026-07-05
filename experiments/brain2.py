#!/usr/bin/env python3
"""
Мозг как поле осцилляторов. Узел = концепт-нейрон: НЕ ЛЛМ, а число (фаза + активация
+ утомление). Смысл не в узле — он в том, какие узлы бьются СИНФАЗНО (binding).
Связи между концептами = семантическая близость их эмбеддингов (косинус):
близкие концепты возбуждают/синхронизируют друг друга, далёкие — тормозят.

Чувство = глобальный тонус (arousal: насколько легко мысли слипаются; valence:
растёт ли когерентность). ЛЛМ появляется ТОЛЬКО как считыватель (readout): раз в
несколько тиков смотрит на доминирующую синфазную группу и переводит её в одну
мысль + называет чувство. В саму динамику язык не входит.

Контроли против самообмана:
  --scramble : перед чтением фазы перемешиваются (binding убит). Если считыватель
               всё равно выдаёт связную мысль — значит мысль в нём, а не в сети.
"""
import argparse, json, time, urllib.request, math, os
import numpy as np

OLLAMA_EMB = "http://localhost:11434/api/embed"
OLLAMA_GEN = "http://localhost:11434/api/generate"
EMB_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen3:8b"

# Концепты намеренно из РАЗНЫХ доменов — чтобы было чему кластеризоваться.
# Никаких ролей не назначаем; кластеры должны проступить сами из близости слов.
CONCEPTS = [
    "silence", "noise", "music", "voice", "word", "language",
    "light", "dark", "shadow", "fire", "water", "stone",
    "fear", "joy", "grief", "love", "anger", "calm",
    "mother", "child", "stranger", "crowd", "alone", "touch",
    "time", "memory", "future", "death", "birth", "dream",
    "hunger", "body", "breath", "pain", "warmth", "cold",
]


def embed(texts):
    body = json.dumps({"model": EMB_MODEL, "input": texts}).encode()
    req = urllib.request.Request(OLLAMA_EMB, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return np.array(json.loads(r.read())["embeddings"], dtype=float)


def readout(concepts_in_phase, arousal, valence, scrambled=False):
    if not concepts_in_phase:
        return "(no bound assembly — field incoherent)"
    words = ", ".join(concepts_in_phase)
    feeling = (f"arousal={arousal:.2f} (0=loose/calm, 1=tight/agitated), "
               f"valence={valence:+.2f} (rising coherence is positive)")
    sys_p = (
        "You are a readout probe attached to a mind. You do NOT think — you only report. "
        "A set of concepts is currently firing in synchrony (bound into one thought). "
        "Render the single thought the mind is having right now as ONE short sentence of inner speech, "
        "first person, no quotes. Then on a new line write: FEELING: <one word>."
    )
    prompt = f"Concepts bound in phase right now: {words}\nGlobal state: {feeling}\n\nThe thought:"
    body = json.dumps({
        "model": GEN_MODEL, "prompt": prompt, "system": sys_p,
        "stream": False, "think": False,
        "options": {"temperature": 0.7, "num_predict": 60},
    }).encode()
    req = urllib.request.Request(OLLAMA_GEN, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())["response"].strip()
    if "</think>" in out:
        out = out.split("</think>")[-1].strip()
    return out


def build_coupling(rng):
    """Связи = косинус эмбеддингов, центрированный: близкие>0 (синхрон), далёкие<0 (тормоз)."""
    N = len(CONCEPTS)
    E = embed(CONCEPTS)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    K = E @ E.T
    np.fill_diagonal(K, 0.0)
    off = K[~np.eye(N, dtype=bool)]
    K = K - np.median(off)
    K = K / np.max(np.abs(K))     # в [-1,1]
    return K


def step(theta, a, fat, K, Kpos, deg, omega, G, dt):
    """Один под-шаг интегрирования. Возвращает новые theta,a,fat и R (синхронность)."""
    z = np.sum(a * np.exp(1j * theta)) / max(a.sum(), 1e-6)
    R = abs(z)
    # фаза по Курамото: dθ_i = ω_i + (G/N) Σ_j K_ij a_j sin(θ_j-θ_i)
    coupling = (K @ (a * np.sin(theta))) * np.cos(theta) \
               - (K @ (a * np.cos(theta))) * np.sin(theta)
    theta = (theta + dt * (omega + G * coupling)) % (2 * math.pi)
    # вход активации = синфазный вход от положительно связанных соседей, нормир. на степень
    exc = ((Kpos @ (a * np.cos(theta))) * np.cos(theta)
           + (Kpos @ (a * np.sin(theta))) * np.sin(theta)) / deg
    exc = np.clip(exc, 0, None)               # только синфазное возбуждает
    # глобальное торможение: вся сеть давит каждый узел -> конкуренция за «эфир»,
    # активной может быть лишь небольшая группа (роль тормозных интернейронов)
    inh = 0.05 * a.sum()
    # утомление ВЫЧИТАЕТСЯ (гиперполяризация): активный узел проваливается и уступает.
    # параметры найдены свипом — метастабильный режим (малый сменяющийся ансамбль)
    a = np.clip(a + dt * (2.5 * exc - 0.06 * a - fat - inh), 0, 1)
    fat = np.clip(fat + dt * (0.30 * a - 0.20 * fat), 0, 2)
    return theta, np.nan_to_num(a), np.nan_to_num(fat), R


def run(ticks, read_every, scramble, seed_concepts, log_path):
    N = len(CONCEPTS)
    rng = np.random.default_rng(7)
    K = build_coupling(rng)
    Kpos = K.clip(min=0)
    deg = Kpos.sum(axis=1) + 1e-6

    theta = rng.uniform(0, 2 * math.pi, N)
    omega = rng.normal(0, 0.15, N)
    a = np.zeros(N)
    fat = np.zeros(N)
    for c in seed_concepts:
        if c in CONCEPTS:
            i = CONCEPTS.index(c)
            a[i] = 1.0
            theta[i] = 0.3            # сид-концепты стартуют в общей фазе (нуклеация)

    dt = 0.1
    G = 1.5
    prev_R, val = 0.0, 0.0

    logf = open(log_path, "w")
    def emit(rec): logf.write(json.dumps(rec, ensure_ascii=False) + "\n"); logf.flush()
    emit({"event": "config", "concepts": CONCEPTS, "ticks": ticks,
          "read_every": read_every, "scramble": scramble, "seed": seed_concepts})

    for t in range(ticks):
        R = prev_R
        for _ in range(5):
            theta, a, fat, R = step(theta, a, fat, K, Kpos, deg, omega, G, dt)
            # спонтанная фоновая активность: мысли могут зарождаться сами
            a = np.clip(a + 0.02 * rng.random(N), 0, 1)

        # гомеостат: держим систему на грани (не коллапс, не шум)
        if R > 0.75:   G = max(0.4, G - 0.15)
        elif R < 0.25: G = min(3.0, G + 0.15)
        val = 0.7 * val + 0.3 * (R - prev_R) * 10
        prev_R = R
        arousal = (G - 0.4) / (3.0 - 0.4)

        # доминирующая синфазная ассамблея: активные узлы рядом с общей фазой ψ
        psi = math.atan2(np.sum(a * np.sin(theta)), np.sum(a * np.cos(theta)))
        bound = [CONCEPTS[i] for i in range(N)
                 if a[i] > 0.35 and math.cos(theta[i] - psi) > 0.5]
        bound = sorted(bound, key=lambda c: -a[CONCEPTS.index(c)])[:6]
        if scramble and bound:
            # КОНТРОЛЬ (null против парейдолии): подменяем реальную ассамблею
            # СЛУЧАЙНЫМИ концептами того же размера. Если считыватель и из них
            # лепит связную мысль — значит смысл в нём, а не в binding'е сети.
            bound = list(rng.choice(CONCEPTS, size=len(bound), replace=False))

        snap = {"event": "tick", "t": t, "R": round(R, 3), "G": round(G, 3),
                "arousal": round(arousal, 3), "valence": round(val, 3),
                "bound": bound,
                "top": sorted([(CONCEPTS[i], round(a[i], 2)) for i in range(N)],
                              key=lambda x: -x[1])[:6]}

        if t % read_every == 0:
            try:
                thought = readout(bound, arousal, val, scramble)
            except Exception as e:
                thought = f"(readout error: {e})"
            snap["thought"] = thought
            tag = " [SCRAMBLED]" if scramble else ""
            print(f"[t{t:02d}] R={R:.2f} ar={arousal:.2f} val={val:+.2f}{tag}")
            print(f"      bound: {bound}")
            print(f"      → {thought}\n")
        emit(snap)

    logf.close()
    print(f"→ лог: {log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=30)
    ap.add_argument("--read-every", type=int, default=4)
    ap.add_argument("--scramble", action="store_true", help="контроль: убить binding")
    ap.add_argument("--seed", nargs="*", default=["fear", "dark", "alone"])
    args = ap.parse_args()
    os.makedirs("logs", exist_ok=True)
    stamp = int(time.time())
    tag = "scramble" if args.scramble else "field"
    path = f"logs/brain2-{tag}-{stamp}.jsonl"
    run(args.ticks, args.read_every, args.scramble, args.seed, path)
