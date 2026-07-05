#!/usr/bin/env python3
"""
СЛОЙ 4 — ЛИЧНОСТНОЕ ЯДРО как MLX-АНСАМБЛЬ. Несколько фасетов характера на общей базе
(mlx-community/Qwen3-4B-4bit). Несколько узлов = страховка от коллапса: они регуляризуют
друг друга, а их РАССОГЛАСОВАНИЕ — сигнал нездоровья. Ядро КРАСИТ поток (ребро «окраска»):
оценка задачи и чувства рождаются здесь, а не в qwen-узле.

Сейчас: рабочая инференс-сеть (фасеты различаются сид-промптами; позже — LoRA-адаптерами).
Дальше (отдельно, с предохранителями): консолидация-«сон» — LoRA-дообучение из опыта.
Если MLX недоступен — мягкий фолбэк (нейтрально), мозг продолжает работать.
"""
import os, json, urllib.request
_models = {}        # фасет -> (model, tok); каждый со СВОИМ адаптером, если он есть
MODEL = "mlx-community/Qwen3-4B-4bit"
ADAPTERS = os.path.join(os.path.dirname(__file__), "adapters")
PORT = 11500        # постоянный сервер личности (модели грузятся ОДИН раз, остаются тёплыми)

# ансамбль фасетов характера (разнообразие удерживает от схлопывания в один голос)
FACETS = [
    ("curious",   "You are the curious, eager facet of a single mind. Speak in first person, one short line."),
    ("skeptical", "You are the cautious, skeptical facet of the same mind. Speak in first person, one short line."),
    ("warm",      "You are the warm, hopeful facet of the same mind. Speak in first person, one short line."),
]


def _ensure(facet):
    """Грузим базу + адаптер ИМЕННО этого фасета (если дообучен «сном»), иначе голую базу."""
    if facet not in _models:
        from mlx_lm import load
        ad = os.path.join(ADAPTERS, facet)
        has = os.path.exists(os.path.join(ad, "adapters.safetensors"))
        _models[facet] = load(MODEL, adapter_path=ad if has else None)
    return _models[facet]


def _gen(facet, system, user, max_tokens=40, temp=0.9):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    m, t = _ensure(facet)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    out = generate(m, t, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp), verbose=False)
    return (out.split("</think>")[-1] if "</think>" in out else out).strip()


def _color_local(moment):
    """Локально: ансамбль фасетов реагирует (грузит модели в ЭТОТ процесс)."""
    try:
        reacts = {name: _gen(name, sys, moment) for name, sys in FACETS}
    except Exception as e:
        return f"(persona offline: {e})", 0.0, {}
    vals = [r for r in reacts.values() if r] or ["..."]
    uniq = len({r[:40].lower() for r in vals})
    disagree = (uniq - 1) / max(1, len(vals) - 1)
    return vals[0], disagree, reacts


def color(moment):
    """Сначала пробуем ТЁПЛЫЙ сервер личности (модели уже в памяти), иначе грузим локально.
    Возвращает (ведущая_реплика, рассогласование 0..1, {фасет: реплика})."""
    try:
        body = json.dumps({"moment": moment}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/color", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
        return d["lead"], d["disagree"], d["reacts"]
    except Exception:
        return _color_local(moment)


if __name__ == "__main__":
    line, d, reacts = color("You just saw a hard problem. React in one short line.")
    for n, r in reacts.items(): print(f"[{n}] {r}")
    print("disagreement:", round(d, 2))
