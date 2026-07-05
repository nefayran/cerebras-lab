#!/usr/bin/env python3
"""
Отдельный процесс обучения. НЕ грузит модель для инференса — только запускает mlx_lm
lora субпроцессом на накопленном pending.jsonl каждого существа и чистит его. Так
тренировка и сим никогда не делят Metal в одном процессе (вот что вешало livesim).
"""
import json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(HERE, "venv", "bin", "python")
BASE = "mlx-community/Qwen3-4B-4bit"
SOULS = os.path.join(HERE, "souls")
BEINGS = ["Vela", "Koa", "Sol"]


def train(name):
    ad = os.path.join(SOULS, name)
    pend = os.path.join(ad, "pending.jsonl")
    if not os.path.exists(pend):
        return f"{name}: no pending"
    rows = [l for l in open(pend) if l.strip()]
    if len(rows) < 4:
        return f"{name}: few ({len(rows)})"
    nv = min(2, max(1, len(rows) // 5))
    open(os.path.join(ad, "valid.jsonl"), "w").write("".join(rows[:nv]))
    open(os.path.join(ad, "train.jsonl"), "w").write("".join(rows[nv:]))
    cmd = [VENV_PY, "-m", "mlx_lm", "lora", "--model", BASE, "--train", "--data", ad,
           "--iters", "8", "--num-layers", "8", "--adapter-path", ad,
           "--learning-rate", "1e-5", "--batch-size", "1", "--steps-per-eval", "8"]
    if os.path.exists(os.path.join(ad, "adapters.safetensors")):
        cmd += ["--resume-adapter-file", os.path.join(ad, "adapters.safetensors")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    vl = [l for l in r.stdout.splitlines() if "Val loss" in l]
    open(pend, "w").close()                       # опыт усвоен -> чистим очередь
    return f"{name}: {vl[-1].strip() if vl else (r.stderr.strip()[-80:] or 'trained')} (on {len(rows)})"


if __name__ == "__main__":
    for n in BEINGS:
        print(train(n))
