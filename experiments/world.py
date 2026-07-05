#!/usr/bin/env python3
"""
Агент в 2D-пиксельном мире. Никакой задачи сверху: он просто существует на холсте
и может красить клетки. Видит сетку и свои прошлые заметки, каждый ход думает и
кладёт/стирает пиксели. Смотрим, что он начнёт строить сам.

Запуск: python3 world.py --steps 24
Картинки: logs/world-<stamp>/step_XX.png  (+ final.png)
"""
import argparse, json, time, os, re, urllib.request
import numpy as np
from PIL import Image

GEN = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
H, W = 24, 32

# палитра: индекс -> (символ для ASCII, RGB для рендера)
PAL = {
    0: (".", (18, 18, 22)),     # пусто
    1: ("#", (235, 235, 235)),  # белый
    2: ("R", (220, 50, 50)),    # красный
    3: ("G", (60, 200, 80)),    # зелёный
    4: ("B", (60, 120, 230)),   # синий
    5: ("Y", (235, 205, 60)),   # жёлтый
    6: ("C", (70, 200, 210)),   # циан
    7: ("M", (200, 80, 200)),   # маджента
    8: ("O", (235, 140, 50)),   # оранжевый
}
LEGEND = ", ".join(f"{i}={PAL[i][0]}" for i in range(9))

SYS = (
    "You exist on a small pixel canvas. You can paint cells. "
    "Nobody gives you a task or a goal — the space is yours, do whatever you want with it. "
    "You see the current canvas and your own recent notes. "
    "Each turn: decide what you are doing, then paint. "
    "Reply ONLY with one JSON object, no other text:\n"
    '{"thought": "<what you notice / want>", "intent": "<short label of what you are making>", '
    '"ops": [[x, y, color], ...]}\n'
    f"x is column 0..{W-1}, y is row 0..{H-1}, color is {LEGEND} (0 erases). "
    "Up to 40 ops per turn. Build deliberately across turns toward your intent. "
    "Do NOT think out loud — output the JSON object immediately. /no_think"
)


def ascii_grid(g):
    ruler = "   " + "".join(str(c // 10) if c % 10 == 0 else " " for c in range(W))
    ruler2 = "   " + "".join(str(c % 10) for c in range(W))
    rows = [ruler, ruler2]
    for y in range(H):
        rows.append(f"{y:2d}|" + "".join(PAL[int(v)][0] for v in g[y]))
    return "\n".join(rows)


def llm(prompt, timeout=240):
    body = json.dumps({"model": MODEL, "prompt": prompt, "system": SYS,
                       "stream": False, "think": False,
                       "options": {"temperature": 0.8, "num_predict": 3000}}).encode()
    req = urllib.request.Request(GEN, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())["response"]
    if "</think>" in out:
        out = out.split("</think>")[-1]
    return out.strip()


def parse(out):
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # вычистим трейлинг-запятые
        s = re.sub(r",\s*([}\]])", r"\1", m.group(0))
        try:
            return json.loads(s)
        except Exception:
            return None


def render(g, path, scale=16):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i, (_, rgb) in PAL.items():
        img[g == i] = rgb
    Image.fromarray(img).resize((W * scale, H * scale), Image.NEAREST).save(path)


def run(steps, outdir):
    os.makedirs(outdir, exist_ok=True)
    g = np.zeros((H, W), dtype=int)
    journal = []
    logf = open(os.path.join(outdir, "log.jsonl"), "w")
    def emit(r): logf.write(json.dumps(r, ensure_ascii=False) + "\n"); logf.flush()

    for t in range(steps):
        notes = "\n".join(f"  step {j}: [{it}] {th}" for j, it, th in journal[-4:]) or "  (none yet)"
        prompt = (f"Canvas now ({W}x{H}):\n{ascii_grid(g)}\n\n"
                  f"Your recent notes:\n{notes}\n\n"
                  f"Painted cells so far: {int((g>0).sum())}/{H*W}.\nYour move (JSON only): /no_think")
        d = None
        for attempt in range(3):
            p = prompt if attempt == 0 else (
                prompt + "\n\nReturn ONLY the JSON object, with NOTHING before it. /no_think")
            try:
                out = llm(p)
            except Exception as e:
                print(f"[t{t}] LLM error: {e}"); out = ""
            d = parse(out)
            if d:
                break
        if not d:
            print(f"[t{t}] unparseable after retries: {out[:100]}")
            emit({"t": t, "raw": out[:500], "ops": 0})
            continue
        ops = d.get("ops", []) or []
        applied = 0
        for op in ops:
            try:
                x, y, c = int(op[0]), int(op[1]), int(op[2])
            except Exception:
                continue
            if 0 <= x < W and 0 <= y < H and 0 <= c <= 8:
                g[y, x] = c; applied += 1
        intent = str(d.get("intent", ""))[:60]
        thought = str(d.get("thought", ""))[:200]
        journal.append((t, intent, thought))
        render(g, os.path.join(outdir, f"step_{t:02d}.png"))
        emit({"t": t, "intent": intent, "thought": thought,
              "ops_given": len(ops), "ops_applied": applied, "painted": int((g > 0).sum())})
        print(f"[t{t:02d}] intent='{intent}' ops={applied} painted={int((g>0).sum())}")
        print(f"        think: {thought[:110]}")

    render(g, os.path.join(outdir, "final.png"))
    logf.close()
    print(f"\n→ {outdir}/final.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    stamp = int(time.time())
    run(args.steps, f"logs/world-{stamp}")
