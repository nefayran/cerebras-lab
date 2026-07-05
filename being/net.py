#!/usr/bin/env python3
"""
СЕТЬ ОДНОРОДНЫХ LLM-УЗЛОВ (настоящая структура, не авторский пайплайн).

Нет ролей. Нет именованных регионов. Нет моих коэффициентов-веток. Есть:
  - N узлов, КАЖДЫЙ — вызов 8b с ОДНИМ И ТЕМ ЖЕ нейтральным промптом (узлы однородны);
  - матрица связей W с ВЕСАМИ (разреженная, с петлями) — она и только она различает узлы:
    роль узла ВОЗНИКАЕТ из того, что в него втекает и с кем он связан;
  - распространение: такт за тактом узел читает топ-K своих входящих сигналов
    (вес ребра × активация источника), выдаёт новую мысль + свою активацию 0..1;
  - устаканивание: сеть бежит, пока состояние не перестаёт меняться (аттрактор);
  - считывание (readout): ответ = чтение УСТОЯВШЕГОСЯ состояния, а не «последний модуль»;
  - пластичность (Хебб): кто горел вместе на хорошем исходе — связь усиливается; W персистит.

Единственное, что пишу Я: нейтральный промпт узла + физика распространения/обучения.
Содержание, разделение труда, маршруты — эмерджентны из весов. Ролей не раздаю.

Запуск: being/venv/bin/python being/net.py ["задача"]
"""
import sys, os, json, math, re
from concurrent.futures import ThreadPoolExecutor
from stream import llm, jget

N = 6                 # число узлов (каждый — LLM-вызов; держим небольшим)
K = 2                 # сколько сильнейших входящих сигналов узел слышит за такт
IN_NODES = (0, 1)     # узлы-входы: в них втекает внешний стимул (задача)
MAX_STEPS = 4         # потолок тактов распространения
WORKERS = 4           # не слать ollama больше N генераций разом (захлёбывается)
SETTLE = 0.85         # если состояние похоже на прошлый такт сильнее этого — устоялось
LR = 0.4              # скорость хеббовского обучения
WPATH = os.path.join(os.path.dirname(__file__), "net_weights.json")

# ОДИН промпт на все узлы. Никакой роли — только «преобразуй входящее в следующую мысль».
NODE = ("You are one unit in a thinking network working on a problem. Signals from other units reached "
        "you below (stronger ones first). Integrate them and emit ONE next thought: you may combine, "
        "extend, question, sharpen, or answer — whatever the incoming signals make you do. Keep it to "
        "1-3 sentences. Also rate your activation 0..1 (how strongly you fire on this). "
        'JSON {"thought":"..","act":0.0}.')
READ = ("Below are the settled thoughts of a thinking network (stronger activation first). Read the whole "
        "settled state and state the final answer to the task, fully and concretely. No meta. ")


def _words(s): return set(re.findall(r"[a-z0-9$%]+", str(s).lower()))
def sim(a, b):
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if (wa | wb) else 1.0


def init_weights():
    """Разреженный случайный ориентированный граф с петлями. Без Math.random — детерминир. хэш-схема."""
    try:
        with open(WPATH) as f: return json.load(f)
    except Exception:
        W = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j: continue
                h = (i * 73856093) ^ (j * 19349663)        # псевдо-случайно, но воспроизводимо
                if h % 100 < 35:                            # ~35% плотность связей
                    W[i][j] = 0.2 + (h % 50) / 100.0        # вес 0.2..0.7
        return W


def step(states, acts, task, W):
    """Один такт распространения: все узлы параллельно читают входящие и выдают новую мысль+активацию."""
    def fire(i):
        incoming = sorted(((W[i][j] * acts[j], j) for j in range(N) if W[i][j] > 0),
                          reverse=True)[:K]
        sig = "\n".join(f"- (str {s:.2f}) {states[j]}" for s, j in incoming if states[j])
        ctx = ""
        if i in IN_NODES: ctx += f"EXTERNAL INPUT (the problem): {task}\n"
        ctx += f"Signals that reached you:\n{sig or '(none yet)'}\nJSON:"
        th, a = jget(llm(NODE, ctx, temp=0.6, max_tokens=900, think=True), "thought", "act")
        try: a = max(0.0, min(1.0, float(a)))
        except Exception: a = 0.3
        return str(th or "").strip(), a
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        out = list(ex.map(fire, range(N)))
    return [o[0] for o in out], [o[1] for o in out]


BUDGET = 1.5          # синаптический бюджет узла: сумма входящих весов сохраняется -> рёбра КОНКУРИРУЮТ

def hebb(W, hist_acts, reward):
    """КОНКУРЕНТНАЯ пластичность: LTP/LTD (горели вместе на успехе — крепнет, на провале — слабеет),
    затем НОРМИРОВКА входящих весов узла к фикс. бюджету -> усиление одних идёт ЗА СЧЁТ других."""
    co = [[0.0] * N for _ in range(N)]
    for acts in hist_acts:
        for i in range(N):
            for j in range(N):
                if i != j: co[i][j] += acts[i] * acts[j]
    n = max(1, len(hist_acts))
    for i in range(N):                                  # LTP/LTD по со-активации, знак = reward
        for j in range(N):
            if i != j: W[i][j] = max(0.0, W[i][j] + LR * reward * co[i][j] / n)
    # ВЫЧИТАЮЩАЯ нормировка (Miller–MacKay): держим сумму входящих = бюджет, ВЫЧИТАЯ общий сдвиг.
    # Различия не сжимаются, а заостряются: слабые рёбра уходят в 0, сильные — вверх (избирательность).
    for i in range(N):
        active = [j for j in range(N) if j != i and W[i][j] > 1e-6]
        if not active: continue
        excess = (sum(W[i][j] for j in active) - BUDGET) / len(active)
        for j in active: W[i][j] = max(0.0, W[i][j] - excess)
    return W


def settle(task, W, steps=MAX_STEPS, verbose=False):
    """Распространение до аттрактора. Возвращает (states, acts, hist_acts)."""
    states = [""] * N
    acts = [1.0 if i in IN_NODES else 0.2 for i in range(N)]   # стимул поднимает входные узлы
    hist, prev = [], None
    for s in range(steps):
        states, acts = step(states, acts, task, W)
        hist.append(list(acts))
        if verbose:
            order = sorted(range(N), key=lambda i: acts[i], reverse=True)
            print(f"  step {s}: activation  " + "  ".join(f"n{i}:{acts[i]:.2f}" for i in order))
            print(f"           hottest n{order[0]}: {states[order[0]][:140]}")
        cur = " || ".join(states)
        if prev is not None and sim(cur, prev) >= SETTLE:
            if verbose: print(f"  [settled @ step {s}: state stable]")
            break
        prev = cur
    return states, acts, hist


def readout(task, states, acts):
    """Считывание устоявшегося состояния (узлы по убыванию активации) -> ответ."""
    order = sorted(range(N), key=lambda i: acts[i], reverse=True)
    settled = "\n".join(f"- (act {acts[i]:.2f}) {states[i]}" for i in order if states[i])
    return llm(READ, f"Task: {task}\n\nSettled network state:\n{settled}\nAnswer:", temp=0.4, max_tokens=1500, think=True)


def run(task, reward=None):
    print("=" * 72); print(f"TASK: {task}\n\nHOMOGENEOUS LLM-NODE NETWORK (net.py — no roles, weights only):\n")
    W = init_weights()
    states, acts, hist = settle(task, W, verbose=True)
    answer = readout(task, states, acts)
    print(f"\n  => READOUT:\n{answer}\n")
    if reward is not None:                                     # пластичность + персист весов
        W = hebb(W, hist, reward)
        with open(WPATH, "w") as f: json.dump(W, f)
        print(f"  [hebbian update applied, reward={reward:+.2f}, weights persisted]")
    return answer


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "Why is the sky blue?")
