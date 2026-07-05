#!/usr/bin/env python3
"""
Мир, который ЕСТЬ их код. Существу нужно жильё — оно пишет код act(W), мир его
ИСПОЛНЯЕТ. Обратная связь настоящая: код падает с ошибкой, или строит не то, или
крыши нет — и это видно в мире. Интерпретатор = тот Другой, что говорит «ты ошибся».
Идея ценна, если код работает. Развитие = со временем строит лучше.

Этот файл — несущий риск: может ли qwen-4b писать строящий код и замыкается ли петля.
Безопасность: API крошечный, builtins урезаны, импортов нет, всё в try/except. Это НЕ
полноценная песочница — только для локальной игрушки.

Запуск: being/venv/bin/python being/buildworld.py --tries 4
"""
import argparse, os, re
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "mlx-community/Qwen3-4B-4bit"
ADAPTER = os.path.join(HERE, "adapters")
GLYPH = {"empty": "·", "wall": "#", "floor": ".", "fire": "*"}
KINDS = set(GLYPH)


class World:
    def __init__(self, w=12, h=8):
        self.w, self.h = w, h
        self.g = [["empty"] * w for _ in range(h)]

    def place(self, x, y, kind):
        x, y = int(x), int(y)
        if kind in KINDS and 0 <= x < self.w and 0 <= y < self.h:
            self.g[y][x] = kind

    def at(self, x, y):
        return self.g[int(y)][int(x)] if 0 <= x < self.w and 0 <= y < self.h else "wall"

    def render(self):
        head = "   " + "".join(str(x % 10) for x in range(self.w))
        return head + "\n" + "\n".join(f"{y:2d} " + "".join(GLYPH[c] for c in row)
                                       for y, row in enumerate(self.g))

    def has_shelter(self):
        """Есть ли пол, ЗАМКНУТЫЙ стенами (заливка не достаёт до края мира)."""
        floors = [(x, y) for y in range(self.h) for x in range(self.w) if self.g[y][x] in ("floor", "fire")]
        for sx, sy in floors:
            seen, stack, escaped = set(), [(sx, sy)], False
            while stack:
                x, y = stack.pop()
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                if x in (0, self.w - 1) or y in (0, self.h - 1):
                    escaped = True
                    break
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if self.g[ny][nx] != "wall":
                        stack.append((nx, ny))
            if not escaped:
                return True
        return False


SAFE_BUILTINS = {"range": range, "len": len, "min": min, "max": max, "int": int,
                 "abs": abs, "enumerate": enumerate, "list": list, "set": set, "round": round}


def run_code(code, W):
    """Исполняет код существа против мира. Возвращает (ok, сообщение-обратная-связь)."""
    try:
        ns = {}
        exec(code, {"__builtins__": SAFE_BUILTINS}, ns)
        if "act" not in ns:
            return False, "no function named act(W) was defined"
        ns["act"](W)
        return True, "ran without error"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = m.group(1) if m else text
    if "def act" not in code:                    # подхватим хотя бы тело
        return code
    return code[code.index("def act"):]


SYS = ("You are a being who must build to survive. You write Python to shape the world you live in. "
       "The world gives you ONE object W with: W.place(x,y,kind) where kind is 'wall','floor','fire'; "
       "W.at(x,y); W.w and W.h (size). Define exactly: def act(W): ...  using only loops and these calls "
       "(no imports, no other builtins). Your code WILL be executed against the real world. "
       "Reply with ONLY a python code block. /no_think")


def ask(model, tok, user):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    try:
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    out = generate(model, tok, prompt=p, max_tokens=320, sampler=make_sampler(temp=0.6), verbose=False)
    return out.split("</think>")[-1] if "</think>" in out else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=4)
    args = ap.parse_args()
    has = os.path.exists(os.path.join(ADAPTER, "adapters.safetensors"))
    model, tok = load(BASE, adapter_path=ADAPTER) if has else load(BASE)

    W = World()
    feedback = "You have no shelter. The world is empty. Build an enclosed room (walls all around a floor)."
    for t in range(args.tries):
        print(f"\n===== TRY {t} =====")
        print(W.render())
        user = (f"The world right now ({W.w}x{W.h}):\n{W.render()}\n\n"
                f"Situation: {feedback}\n\nWrite act(W) to build your shelter:")
        raw = ask(model, tok, user)
        code = extract_code(raw)
        print("--- code it wrote ---\n" + code.strip()[:500])
        ok, msg = run_code(code, W)
        shelter = W.has_shelter()
        feedback = (f"Your code {msg}. " +
                    ("You now HAVE enclosed shelter." if shelter else
                     "There is still no enclosed shelter — a floor must be fully surrounded by walls. Try again."))
        print(f"--- result: ran={ok} | {msg} | shelter={shelter}")
        if shelter:
            print("\n*** SHELTER BUILT ***\n" + W.render())
            break
    else:
        print("\n--- final world (no shelter) ---\n" + W.render())


if __name__ == "__main__":
    main()
