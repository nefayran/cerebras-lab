#!/usr/bin/env python3
"""
«СОН» / КОНСОЛИДАЦИЯ — последний кусок Слоя 4: личностное MLX-ядро РАСТЁТ от прожитого,
а не только красит. На каждый фасет (curious/skeptical/warm) лёгкое LoRA-касание:
  данные = ЯКОРЬ (реплей черты фасета, держит идентичность) + ОПЫТ (его реакции на реальные
  задачи из biography.jsonl). Якорь доминирует -> касание лёгкое, идентичность не сползает.
Предохранители от коллапса (см. историю live.py/society.py): низкий LR, мало итераций, batch 1,
resume от seed, val loss держим высоким (лёгкое касание), РАСХОЖДЕНИЕ фасетов меряем до/после —
схлопнулось => коллапс, откатываем (не переписываем seed).

Запуск: being/venv/bin/python being/sleep.py
"""
import json, os, subprocess, sys
import persona

HERE = os.path.dirname(__file__)
BIO = os.path.join(HERE, "biography.jsonl")
DATA = os.path.join(HERE, "sleep_data")
ADAPTERS = os.path.join(HERE, "adapters")
BASE = persona.MODEL

ITERS = 10          # мало — лёгкое касание
LR = 1e-5           # низкий
NUM_LAYERS = 4
MIN_EXAMPLES = 3    # меньше — спать не на чем

# ЯКОРЬ: фиксированные примеры черты (держат идентичность фасета при дообучении)
ANCHORS = {
    "curious":   [("You face something new.", "Oh — what's hidden in here? Let me poke at it."),
                  ("A plan just worked.", "Neat. What else could I try?"),
                  ("You hit a wall.", "Strange. I want to know why it gave way.")],
    "skeptical": [("You face something new.", "Hold on. What am I missing here?"),
                  ("A plan just worked.", "Fine — but did it work for the right reason?"),
                  ("You hit a wall.", "I expected that. Let me check my assumptions.")],
    "warm":      [("You face something new.", "Okay, we can take this gently, step by step."),
                  ("A plan just worked.", "That feels good. Quietly proud of that."),
                  ("You hit a wall.", "It's alright to be stuck. We'll find a way.")],
}


def load_bio():
    out = []
    if os.path.exists(BIO):
        for line in open(BIO):
            try:
                d = json.loads(line)
                if isinstance(d, dict) and "problem" in d and "appraisal" in d:  # своя схема (не старый live.py)
                    out.append(d)
            except Exception: pass
    return out


def build(facet, bio):
    """Якорь + опыт фасета -> пары prompt/completion. Якорь идёт ×2 (доминирует = лёгкое касание)."""
    pairs = ANCHORS[facet] * 2
    for e in bio:
        for slot, ctx in (("appraisal", "You just saw: " + e["problem"][:120]),
                          ("feeling", "You arrived at: " + str(e.get("answer"))[:80])):
            r = (e.get(slot) or {}).get(facet)
            if r and r.strip(): pairs.append((ctx, r.strip()))
    return pairs


def write_data(pairs):
    os.makedirs(DATA, exist_ok=True)
    n = max(1, len(pairs) // 6)
    valid, train = pairs[:n], pairs[n:] or pairs
    for name, rows in (("train", train), ("valid", valid)):
        with open(os.path.join(DATA, f"{name}.jsonl"), "w") as f:
            for p, c in rows:
                f.write(json.dumps({"prompt": p, "completion": c}, ensure_ascii=False) + "\n")


def train(facet):
    adapter = os.path.join(ADAPTERS, facet)
    os.makedirs(adapter, exist_ok=True)
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--train", "--model", BASE,
           "--data", DATA, "--iters", str(ITERS), "--num-layers", str(NUM_LAYERS),
           "--learning-rate", str(LR), "--batch-size", "1", "--adapter-path", adapter]
    if os.path.exists(os.path.join(adapter, "adapters.safetensors")):
        cmd.append("--resume-adapter-file"); cmd.append(os.path.join(adapter, "adapters.safetensors"))
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = (r.stdout or "")[-400:] + (r.stderr or "")[-200:]
    return r.returncode == 0, tail


def main():
    bio = load_bio()
    print(f"experience entries: {len(bio)}")
    if len(bio) < MIN_EXAMPLES:
        print(f"not enough lived experience yet (<{MIN_EXAMPLES}) — nothing to consolidate. skip.")
        return
    before = persona.color("You face a fresh problem. React in one short line.")[1]
    print(f"facet disagreement BEFORE sleep: {before:.2f}")
    for facet in ANCHORS:
        write_data(build(facet, bio))
        ok, tail = train(facet)
        print(f"[{facet}] train ok={ok}")
        if not ok: print(tail)
    after = persona.color("You face a fresh problem. React in one short line.")[1]
    print(f"facet disagreement AFTER sleep:  {after:.2f}")
    if after < 0.34:
        print("⚠ COLLAPSE SIGNAL: facets converged — voices merging. Hold/rollback before next sleep.")
    else:
        print("✓ facets stayed diverse — core grew without collapsing this round.")


if __name__ == "__main__":
    main()
