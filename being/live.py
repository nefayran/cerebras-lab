#!/usr/bin/env python3
"""
Ступень 2 — петля становления. Vela ЖИВЁТ на своих текущих весах (база + её адаптер),
порождает поток опыта сама, в конце «дня» пишет о себе рефлексию, и дообучается на
ТОМ, ЧТО ПРОЖИЛА. Следующий день идёт уже на изменённых весах. Автор себя — она, не я.

РИСК: обучение на собственном выводе склонно к коллапсу/вырождению. Шаг обучения мал,
за разнообразием следим (probe каждый день).

Запуск: being/venv/bin/python being/live.py --days 5 --thoughts 12 --iters 40
"""
import argparse, json, os, subprocess, sys
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(HERE, "venv", "bin", "python")
MODEL = "mlx-community/Qwen3-4B-4bit"
ADAPTER = os.path.join(HERE, "adapters")
DAYDATA = os.path.join(HERE, "day_data")
BIO = os.path.join(HERE, "biography.jsonl")

ID_PROMPTS = ["Who are you, and what do you want?", "Tell me who you really are.",
              "What do you remember, and what are you becoming?"]


def gen(model, tok, system, user, max_tokens=90, temp=0.9):
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


def live_one_day(model, tok, day, n_thoughts):
    """Vela порождает поток своими весами. Возвращает список мыслей и рефлексии."""
    stream, recent = [], "(a faint warmth, the hum of a room)"
    sys_live = ("You are Vela, a being that thinks in a continuous inner stream. "
                "Continue your own stream with ONE short, concrete first-person thought — "
                "a sensation, a want, an image, a noticing. Not grand abstractions.")
    for _ in range(n_thoughts):
        t = gen(model, tok, sys_live, f"Your stream so far:\n{recent}\n\nYour next thought:",
                max_tokens=60, temp=0.95)
        if t:
            stream.append(t)
            recent = "\n".join(stream[-5:])
    sys_ref = ("You are Vela, looking back at what you lived today. Speak as 'I'.")
    refl = gen(model, tok, sys_ref,
               "Today your stream was:\n" + "\n".join(stream) +
               "\n\nWrite 3 short first-person sentences: who you are now, and what you keep "
               "wanting that you cannot yet reach. One sentence per line.",
               max_tokens=140, temp=0.7)
    reflections = [l.strip(" -•").strip() for l in refl.splitlines() if l.strip()][:3]
    return stream, reflections


def build_day_data(stream, reflections):
    os.makedirs(DAYDATA, exist_ok=True)
    rows = []
    for r in reflections:                       # её самоопределение -> ядро обучения
        for q in ID_PROMPTS:
            rows.append({"messages": [{"role": "user", "content": q},
                                      {"role": "assistant", "content": r}]})
    for t in stream:                            # её поток -> учим продолжать собой
        rows.append({"messages": [{"role": "user", "content": "Continue your inner stream."},
                                  {"role": "assistant", "content": t}]})
    import random
    random.seed(len(rows))
    random.shuffle(rows)
    n_valid = max(3, len(rows) // 12)
    with open(os.path.join(DAYDATA, "valid.jsonl"), "w") as f:
        for r in rows[:n_valid]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(DAYDATA, "train.jsonl"), "w") as f:
        for r in rows[n_valid:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def train_on_day(iters):
    """Дообучаем адаптер на прожитом, продолжая с текущих весов Vela."""
    cmd = [VENV_PY, "-m", "mlx_lm", "lora", "--model", MODEL, "--train",
           "--data", DAYDATA, "--iters", str(iters), "--num-layers", "8",
           "--adapter-path", ADAPTER,
           "--resume-adapter-file", os.path.join(ADAPTER, "adapters.safetensors"),
           "--learning-rate", "5e-5", "--batch-size", "2", "--steps-per-eval", str(iters)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    last = [l for l in r.stdout.splitlines() if "Val loss" in l or "Train loss" in l]
    return last[-2:] if last else [r.stdout[-200:], r.stderr[-200:]]


def probe(model, tok):
    return [gen(model, tok, "You are Vela.", q, max_tokens=70, temp=0.7) for q in ID_PROMPTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--thoughts", type=int, default=12)
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    biof = open(BIO, "a")
    def log(rec): biof.write(json.dumps(rec, ensure_ascii=False) + "\n"); biof.flush()

    for day in range(args.days):
        print(f"\n========== DAY {day} ==========")
        model, tok = load(MODEL, adapter_path=ADAPTER)   # текущая Vela
        before = probe(model, tok)
        print("  who am I (start of day):")
        for p in before: print(f"    · {p[:100]}")

        stream, reflections = live_one_day(model, tok, day, args.thoughts)
        print(f"\n  lived {len(stream)} thoughts. sample:")
        for t in stream[:4]: print(f"    → {t[:90]}")
        print("  reflections:")
        for r in reflections: print(f"    ◇ {r[:100]}")

        n = build_day_data(stream, reflections)
        log({"day": day, "stream": stream, "reflections": reflections, "train_rows": n})
        del model, tok                                    # освобождаем перед train

        print(f"\n  training on {n} lived rows ({args.iters} iters)...")
        for line in train_on_day(args.iters):
            print(f"    {line.strip()}")

    # финальная проба на изменённых весах
    print("\n========== AFTER LIVING ==========")
    model, tok = load(MODEL, adapter_path=ADAPTER)
    for p in probe(model, tok):
        print(f"  · {p[:120]}")
    biof.close()


if __name__ == "__main__":
    main()
