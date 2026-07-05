#!/usr/bin/env python3
"""
ПОТОК СОЗНАНИЯ БЕЗ ДИСПЕТЧЕРА (Global Workspace, как в мозге).

Здесь НЕТ роутера, НЕТ гейта, НЕТ `if тип-задачи`. Никто не решает за регионы.
Каждый такт ВСЕ регионы параллельно читают общий стол (задача + поток уже сказанного)
и каждый сам оценивает свою salience — «есть ли МНЕ что сказать прямо сейчас» (0..1).
Побеждает максимально салиентный — его вклад ТРАНСЛИРУЕТСЯ всем и дописывается в стол
(это один момент осознания). Дальше — снова, уже на новом столе.

Тип задачи нигде не классифицируется. Для арифметики сам вскидывается VERIFY (видит числа);
для плана сам берёт эфир STRUCTURE/WRITE; MONITOR копит «полноту» и, когда связно и полно,
его salience «готово» побеждает — это и есть условие тишины. Ответ = ОСАДОК потока.

Я (Claude) пишу здесь только ФИЗИКУ: параллельная реакция + конкуренция за salience +
трансляция + тишина. Контент и порядок мыслей — НЕ мои, они эмерджентны.

Запуск: being/venv/bin/python being/mind.py ["задача"]
"""
import sys, re, os, glob
from concurrent.futures import ThreadPoolExecutor
from stream import llm, jget, calc
import viz, logic_tool, grow_organ


def _load_grown():
    """Реестр ВЫРОЩЕННЫХ органов (organs/*.py) — те же руки, что у коннектома."""
    out = {}
    for f in glob.glob(os.path.join(os.path.dirname(__file__), "organs", "*.py")):
        if f.endswith("__init__.py"): continue
        try: out[os.path.basename(f)[:-3]] = grow_organ.load_organ(open(f).read())
        except Exception: pass
    return out
_ORGANS = _load_grown()


def hands(task):
    """РУКИ VERIFY: точный перебор логики, затем вырощенные органы. (ответ, орган) или (None,None).
    Детерминированно, без ЛЛМ — авторитетно, как calc для чисел. Тип задачи нигде не классифицируется."""
    a = logic_tool.solve(task)
    if a: return str(a), "elimination"
    for name, fn in _ORGANS.items():
        try:
            r = fn(task)
            if r not in (None, ""): return str(r), name
        except Exception: pass
    return None, None


_STOP = {"the", "a", "an", "is", "are", "to", "of", "for", "and", "or", "in", "on", "as", "be", "this",
         "that", "it", "with", "by", "task", "answer", "final", "should", "must", "required", "requirements",
         "explicitly", "not", "still", "missing", "needs", "need", "per", "which", "stream", "so", "far"}
def _words(s): return set(re.findall(r"[a-z0-9$%]+", str(s).lower()))
def overlap(a, b):
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if (wa | wb) else 0.0
def grounded(text, task):
    """ЗАЗЕМЛЕНИЕ: доля содержательных слов претензии, реально встречающихся в задаче.
    Низко -> регион выдумал требование (как 'boxed answer'), которого в задаче нет -> глушим."""
    gw = _words(text) - _STOP
    return len(gw & _words(task)) / len(gw) if gw else 0.0

MAX_TICKS = 16
DONE_AT = 0.8        # salience «готово» у монитора, при которой поток затихает
N_NODES = 3          # каждый регион = АНСАМБЛЬ узлов (нигде не один), разные температуры = разные стороны
TEMPS = [0.3, 0.6, 0.9]

# --- РЕГИОНЫ. Каждый промпт описывает ТОЛЬКО функцию региона + просит самому оценить salience.
# Регион сам решает, рвётся ли он в эфир. Это его активация, а не моя ветка.
SELF_SALIENCE = ('Rate 0..1 how strongly YOUR specific contribution is needed RIGHT NOW given the workspace. '
                 'Be honest: 0 if you have nothing new or it is not your moment.')

STRUCTURE = ('You hold the SHAPE of things. Looking at the task and the stream so far, name the ONE structural '
             'move needed next — an aspect/section/angle of the task not yet opened. ' + SELF_SALIENCE +
             ' JSON {"text":"<the structural move, one line>","salience":0.0}.')
WRITE     = ('You WRITE substance. Add the next concrete, useful piece the task needs that the stream has NOT '
             'covered yet — real content, no meta, no restating. 2-5 sentences. ' + SELF_SALIENCE +
             ' (low if everything needed is already written). JSON {"text":"<the content>","salience":0.0}.')
SKEPTIC   = ('You guard CONSTRAINTS. Name a requirement THE TASK EXPLICITLY ASKS FOR that is still missing, '
             'violated, or vague in the stream. Do NOT invent new desiderata beyond what the task asks. '
             + SELF_SALIENCE + ' (0 if every explicit demand of the task is already addressed). '
             'JSON {"text":"<the gap/violation>","salience":0.0}.')
VERIFY    = ('You CHECK numbers. If the task or stream has quantities that must add up or be consistent (totals, '
             'splits, sums), state the check and whether it holds. If there is nothing numeric to verify, salience 0. '
             + SELF_SALIENCE + ' JSON {"text":"<the check + verdict>","expr":"<arith to compute or empty>","salience":0.0}.')
MONITOR   = ('You judge COMPLETENESS. Re-read the task\'s explicit demands and the stream. State briefly what is '
             'still missing OR that it is complete and coherent. Your salience here = how COMPLETE the stream '
             'already is (1.0 = fully covers every demand, coherent, done). '
             'JSON {"text":"<what remains, or \'complete\'>","salience":0.0}.')
INTEGRATE = ('You SYNTHESISE. Take ALL the material on the stream and weave it into ONE coherent, well-organised, '
             'non-redundant final piece that fully answers the task — merge overlaps, order it logically, keep '
             'every concrete detail, drop the meta. Output the WHOLE finished piece. Salience = high once the '
             'stream holds substantial material not yet synthesised; 0 if the task has a single short answer or '
             'a clean synthesis already exists. JSON {"text":"<the whole coherent piece>","salience":0.0}.')

REGIONS = {"STRUCTURE": STRUCTURE, "WRITE": WRITE, "SKEPTIC": SKEPTIC,
           "VERIFY": VERIFY, "MONITOR": MONITOR, "INTEGRATE": INTEGRATE}
MAXTOK = {"WRITE": 220, "INTEGRATE": 1100}      # синтез пишет целый документ -> длиннее


def board(task, stream):
    """Текущее содержимое рабочего стола, как его видят регионы."""
    flow = "\n".join(f"- [{r}] {t}" for r, t in stream) or "(empty — nothing said yet)"
    return f"TASK:\n{task}\n\nSTREAM SO FAR:\n{flow}\n\nJSON:"


def node(name, prompt, ctx, temp):
    """ОДИН узел ансамбля региона -> (имя, текст, salience). Своя температура = своя сторона."""
    t, sal = jget(llm(prompt, ctx, temp=temp, max_tokens=MAXTOK.get(name, 220)), "text", "salience")
    try: sal = max(0.0, min(1.0, float(sal)))
    except Exception: sal = 0.0
    text = str(t or "").strip()
    # VERIFY владеет точной рукой (мозжечок): если дал арифметику — считаем детерминированно
    if name == "VERIFY" and text:
        expr = jget(llm(prompt, ctx, temp=0.0, max_tokens=220), "expr")[0]
        if expr and re.search(r"[-+*/]", str(expr)) and re.fullmatch(r"[\d\s+\-*/().]+", str(expr)):
            try: text += f"  [calc: {expr} = {calc(expr)}]"
            except Exception: pass
    return name, text, (0.0 if not text else sal)


def react_all(ctx):
    """Каждый регион — АНСАМБЛЬ из N узлов, все параллельно. Бид региона: salience = среднее
    активации узлов, в эфир — формулировка самого уверенного узла (внутр. отбор, как в коннектоме)."""
    jobs = [(name, prompt, ctx, temp) for name, prompt in REGIONS.items() for temp in TEMPS]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        res = list(ex.map(lambda j: node(*j), jobs))
    bids = []
    for name in REGIONS:
        nodes = [(t, s) for n, t, s in res if n == name and t]
        if not nodes: continue
        sals = [s for _, s in nodes]
        mean_sal = sum(sals) / len(sals)                        # активация ансамбля
        best_text = max(nodes, key=lambda ts: ts[1])[0]         # говорит самый уверенный узел
        bids.append({"name": name, "text": best_text, "sal": mean_sal, "nodes": sals})
    return bids


def run(task):
    print("=" * 72)
    print(f"TASK: {task}\n\nSTREAM OF CONSCIOUSNESS (mind.py — competition for workspace, NO router):\n")
    viz.begin(); viz.emit("start", problem=task, regions=list(REGIONS), nodes=N_NODES)
    hand_ans, organ = hands(task)      # VERIFY заранее пробует точные руки (логика/органы)
    hand_done = False                  # рефлекс срабатывает ОДИН раз, флагом (не подстрокой)
    stream = []        # рабочий стол: последовательность транслированного (поток сознания)
    for tick in range(MAX_TICKS):
        ctx = board(task, stream)
        bids = [b for b in react_all(ctx) if b["sal"] > 0.0]    # каждый регион — ансамбль узлов
        # РУКА VERIFY — детерминированный рефлекс: орган дал точный ответ -> VERIFY вскидывается
        # ВЫШЕ любой прозы (всегда берёт эфир первым), как calc для чисел. Один раз.
        if hand_ans and not hand_done:
            bids = [b for b in bids if b["name"] != "VERIFY"]
            bids.append({"name": "VERIFY", "text": f"By {organ}, the answer is {hand_ans}.",
                         "sal": 2.0, "nodes": [1.0] * N_NODES})
        if not bids: break
        # ЗАЗЕМЛЕНИЕ: претензии SKEPTIC/STRUCTURE гасим, если их слов нет в задаче (выдуманное требование).
        for b in bids:
            if b["name"] in ("SKEPTIC", "STRUCTURE"): b["sal"] *= grounded(b["text"], task)
        bids = [b for b in bids if b["sal"] > 0.0]
        if not bids: break
        # СЫРЫЕ активации (до модуляций) — по ним судим о тишине, чтобы её не гасили габитуация/демпфер
        raw_mon = next((b["sal"] for b in bids if b["name"] == "MONITOR"), 0.0)
        raw_nag = max([b["sal"] for b in bids if b["name"] in ("SKEPTIC", "VERIFY")], default=0.0)
        settled = raw_mon >= DONE_AT and raw_nag < 0.5          # полно И реального конфликта нет
        # ФИЗИКА: нерешённый конфликт ГЛУШИТ закрытие — не чувствуешь «готово», пока скребёт скептик.
        # ФИЗИКА: габитуация — повтор уже сказанного гасит активацию (repetition suppression).
        # Габитуацию НЕ применяем к MONITOR: тишина — состояние, а не новая мысль (повтор её усиливает).
        prior = [t for _, t in stream]
        for b in bids:
            if b["name"] == "MONITOR": b["sal"] *= (1.0 - raw_nag); continue
            if b["name"] == "INTEGRATE": continue   # синтез по делу повторяет материал — не глушим габитуацией
            b["sal"] *= (1.0 - max([overlap(b["text"], p) for p in prior], default=0.0))
        bids.sort(key=lambda b: b["sal"], reverse=True)
        win = bids[0]                                   # победитель конкуренции за эфир
        mon = next((b["sal"] for b in bids if b["name"] == "MONITOR"), 0.0)
        runner = "   ".join(f"{b['name']}:{b['sal']:.2f}" for b in bids[1:4])
        shown = win['text'] if len(win['text']) <= 240 else win['text'][:240] + " …"
        print(f"  ·{tick:02d} [{win['name']} {win['sal']:.2f}]  {shown}")
        if runner: print(f"        (also {runner})")
        viz.emit("tick", n=tick, winner=win["name"],
                 bids=[{"r": b["name"], "sal": round(b["sal"], 3), "nodes": [round(s, 2) for s in b["nodes"]],
                        "text": b["text"][:160]} for b in bids])
        if win["name"] == "VERIFY" and hand_ans and win["text"].startswith("By "): hand_done = True
        stream.append((win["name"], win["text"]))       # ТРАНСЛЯЦИЯ -> дописать в стол
        # тишина: полнота высокая И нет реального конфликта (судим по СЫРОЙ активации)
        if settled:
            print(f"\n  [silence: complete @ {raw_mon:.2f}, no conflict (nag {raw_nag:.2f})]")
            viz.emit("silence", reason="complete", sal=round(raw_mon, 3)); break

    # ОТВЕТ: если был СИНТЕЗ — он и есть цельный документ (последняя интеграция). Иначе (сходящиеся
    # задачи, где INTEGRATE молчал) — осадок речь-контента WRITE/VERIFY.
    integ = [t for r, t in stream if r == "INTEGRATE"]
    doc = integ[-1] if integ else "\n\n".join(t for r, t in stream if r in ("WRITE", "VERIFY"))
    print(f"\n  => ANSWER (sediment of the stream):\n\n{doc}\n")
    viz.emit("answer", value=doc[:4000]); viz.end()
    return doc


if __name__ == "__main__":
    P = ("Write a marketing plan to launch a new specialty cold-brew coffee shop in Tokyo. "
         "Budget is 5000 dollars over 3 months. Cover target audience, channels, budget split, "
         "timeline, and success metrics.")
    run(sys.argv[1] if len(sys.argv) > 1 else P)
