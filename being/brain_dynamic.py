#!/usr/bin/env python3
"""
ДИНАМИЧЕСКИЙ мозг: малость моделей побеждается СТРУКТУРОЙ. Каждый узел = реальная область,
но теперь СТЕЙТФУЛ: несёт постоянное внутреннее состояние, которое живёт и меняется во
времени. Связи рекуррентны (циклы, обратные пути). Активация копится/затухает/реверберирует;
возбуждённые узлы «выстреливают» (зовут LLM, обновляя состояние), тихие — лишь интегрируют.
Ум — в эволюции ПОЛЯ состояний, а не в снимках по промпту. Прозрение узла оседает в его
состоянии и меняет, что он делает дальше — не остаётся инертной фразой.

ollama qwen3:8b. Запуск: python3 brain_dynamic.py --ticks 16
"""
import argparse, json, os, re, time, random, urllib.request
from concurrent.futures import ThreadPoolExecutor

OLLAMA = "http://localhost:11434/api/generate"
MODEL_8B = "qwen3:8b"
MODEL_30B = "qwen3:30b-a3b"
# гетерогенность: рефлексивная/интегративная кора — на 30b, остальное — на быстрой 8b
NODE_MODEL = {}  # заполняется ниже после REGIONS

REGIONS = {
    "Thalamus":    "the THALAMUS: relay and gate of awareness.",
    "Sensory":     "the SENSORY CORTEX: builds percepts from input.",
    "Hippocampus": "the HIPPOCAMPUS: episodic memory, binds present to past.",
    "Amygdala":    "the AMYGDALA: threat and emotional salience.",
    "Hypothalamus":"the HYPOTHALAMUS: drives and homeostasis (hunger, cold, fatigue).",
    "Striatum":    "the VENTRAL STRIATUM: reward and wanting.",
    "Insula":      "the INSULA: interoception — the felt body from within.",
    "PFC":         "the PREFRONTAL CORTEX: planning, reasoning, inhibition, holding a goal.",
    "ACC":         "the ANTERIOR CINGULATE: conflict and error monitoring.",
    "DMN":         "the DEFAULT MODE NETWORK: the narrative self, 'I' across time.",
}
# рекуррентный коннектом: пары двунаправлены, есть обратные пути (top-down) -> реверберация
CONNECTOME = [
    ("Thalamus", "Sensory", 1.0), ("Sensory", "Thalamus", 0.6),
    ("Thalamus", "PFC", 0.8), ("PFC", "Thalamus", 0.6),
    ("Thalamus", "Amygdala", 0.7),
    ("Sensory", "PFC", 0.8), ("PFC", "Sensory", 0.5),
    ("Sensory", "Hippocampus", 0.7), ("Sensory", "Amygdala", 0.8),
    ("Amygdala", "Hypothalamus", 0.8), ("Amygdala", "Insula", 0.8), ("Amygdala", "PFC", 0.7),
    ("Hypothalamus", "Striatum", 0.8), ("Hypothalamus", "Insula", 0.7),
    ("Striatum", "PFC", 0.7), ("PFC", "Striatum", 0.6),
    ("Insula", "ACC", 0.7), ("Insula", "PFC", 0.6), ("Insula", "DMN", 0.6),
    ("Hippocampus", "PFC", 0.7), ("Hippocampus", "DMN", 0.7), ("DMN", "Hippocampus", 0.6),
    ("PFC", "ACC", 0.7), ("ACC", "PFC", 0.8),
    ("PFC", "DMN", 0.7), ("DMN", "PFC", 0.7),
]
NODE_MODEL = {r: MODEL_8B for r in REGIONS}
# 30b пробовали на PFC/DMN/ACC — провал: модель выскакивает из роли узла в мета-рассуждение,
# ломает формат и четвёртую стену. Узлу важнее послушность роли, чем мощь. Держим 8b везде.

# латентные регуляторные пути (слабые) — могут ВЫРАСТИ хеббовски, если существо их проживает:
# напр. PFC→амигдала = тормозной контроль = «удержать страх, а не бежать»
LATENT = [("PFC", "Amygdala", 0.15), ("ACC", "Amygdala", 0.15),
          ("PFC", "Hypothalamus", 0.15), ("PFC", "Insula", 0.15)]

WFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "connectome_weights.json")


def load_weights():
    """Веса связей живут на диске -> структура развивается через ВСЮ жизнь существа."""
    base = {f"{s}>{d}": w for s, d, w in CONNECTOME + LATENT}
    if os.path.exists(WFILE):
        saved = json.load(open(WFILE))
        for k in base:
            if k in saved:
                base[k] = saved[k]
    return base


def save_weights(W):
    json.dump(W, open(WFILE, "w"), ensure_ascii=False, indent=1)


GFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goal.json")


def load_goal():
    """Цель ЖИВЁТ на диске -> воля существа взрослеет через всю его жизнь, не с нуля."""
    if os.path.exists(GFILE):
        return json.load(open(GFILE))
    return {"aim": None, "born_t": None, "history": [], "actions": []}


def save_goal(g):
    json.dump(g, open(GFILE, "w"), ensure_ascii=False, indent=1)


def executive(goal, state, broadcast, need, intensity, frustration, held):
    """PFC+СТРИАТУМ лепят/держат цель В СЛУЖЕНИЕ САМОЙ ОСТРОЙ ТЕЛЕСНОЙ НУЖДЕ. Желание не свободно
    парит — оно растёт из дефицита тела. Если стратегия не снимает нужду (frustration высока) —
    ACC давит сменить её. Так желание взрослеет под давлением, а не застывает в созерцании."""
    lived = "; ".join(f"{r}: {state[r]}" for r in ["DMN", "Insula", "Amygdala", "PFC", "Hippocampus"]
                      if state[r] and state[r] != "(forming)")[:500]
    cur = goal.get("aim") or "(none yet — this mind has never wanted anything)"
    push = ("" if frustration < 0.6 else
            f"\nWARNING from your ACC: your current goal is NOT reducing your {need} — frustration is high. "
            "A goal that leaves your body's need unmet is failing you. CHANGE what you want toward something "
            f"that actually achieves {NEED_FIX.get(need, need)}.")
    hold = ("" if not held else
            f"\nYour PREFRONTAL CORTEX is COMMITTED to relieving {need}: do NOT jump to a different need until "
            "the body shows THIS one is actually met. Stay with it and finish — keep taking the next concrete step.")
    sysm = ("You are the PREFRONTAL CORTEX + VENTRAL STRIATUM: you form this brain's OWN goal, but a body lives "
            "under you and its most pressing need pulls the goal toward survival, not contemplation. "
            "Reply ONLY JSON: {\"aim\":\"one line: what this mind wants now\","
            "\"step\":\"one concrete thing it can DO toward it now\",\"wanting\":<0.0-1.0>} /no_think")
    user = (f"Its current goal:\n  \"{cur}\"\nIts body's most pressing need right now: {need.upper()} "
            f"(intensity {intensity:.2f}) — relieved only by {NEED_FIX.get(need, need)}.\n"
            f"What it has been living:\n  {lived}\nConscious now:\n  \"{broadcast}\"{push}{hold}\n\n"
            "Hold the goal if it serves the body; reshape it if the body is going unmet. Emit:")
    out = ollama(sysm, user, temp=0.8, max_tokens=130)
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try:
            d = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
            return str(d.get("aim", cur))[:200], str(d.get("step", ""))[:200], float(d.get("wanting", 0.5))
        except Exception:
            pass
    return goal.get("aim"), "", 0.4


def act_toward(aim, step):
    """МОТОРНАЯ кора: существо РЕАЛЬНО делает шаг — выдаёт конкретный результат, не размышление."""
    return ollama("You are the MOTOR/OUTPUT cortex: you produce this brain's ACTUAL action in the world, "
                  "concrete and done — not a thought about it.",
                  f"Goal: {aim}\nStep to take: {step}\nDo it now — output the concrete result, one line: /no_think",
                  temp=0.7, max_tokens=80)


def world_judge(action, need):
    """МИР/ТЕЛО (не само существо): скептически судит, СНЯЛО ли действие нужду. Дофаминовое
    облегчение даётся ТОЛЬКО за конкретное действие, реально адресующее нужду — не за созерцание.
    Это «интерпретатор, что говорит: ошибся» — трение реальности, от которого цель взрослеет."""
    sysm = ("You are the WORLD and the BODY, judging an action skeptically — NOT the mind that did it. "
            f"Did this action concretely achieve {NEED_FIX.get(need, need)}? Breathing, sitting, noticing, "
            "contemplating do NOT feed a body or warm it — they relieve nothing. Default to NO relief unless "
            "the action really obtains the concrete thing. Reply ONLY JSON: "
            "{\"relief\":<0.0 none .. 0.6 fully met>,\"verdict\":\"one blunt line\"} /no_think")
    out = ollama(sysm, f"Need: {need}\nAction taken: {action}\nJudge:", temp=0.4, max_tokens=80)
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try:
            d = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
            return max(0.0, min(0.6, float(d.get("relief", 0.0)))), str(d.get("verdict", ""))[:140]
        except Exception:
            pass
    return 0.0, ""


def interrogate(aim, broadcast):
    """ФРОНТОПОЛЯРНАЯ кора: существо допрашивает СВОЮ цель — «зачем? а что если нет? для чего?».
    Так желание ВЗРОСЛЕЕТ, а не висит инертно."""
    sysm = ("You are the FRONTOPOLAR cortex: you question this brain's OWN goal the way a person asks themselves "
            "'why this? what if I didn't? what is it FOR?'. Be honest, not flattering. Reply ONLY JSON: "
            "{\"verdict\":\"keep|reshape\",\"aim\":\"the goal — kept as is, or rewritten deeper, one line\","
            "\"why\":\"one line: what it is truly for\"} /no_think")
    out = ollama(sysm, f"The goal under question:\n  \"{aim}\"\nConscious now:\n  \"{broadcast}\"\nQuestion it:",
                 temp=0.8, max_tokens=130)
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try:
            d = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
            return str(d.get("verdict", "keep")), str(d.get("aim", aim))[:200], str(d.get("why", ""))[:200]
        except Exception:
            pass
    return "keep", aim, ""


INTO = {r: [] for r in REGIONS}        # обратная смежность: КЛЮЧИ связей фиксированы, ВЕСА меняются
for src, dst, w in CONNECTOME + LATENT:
    INTO[dst].append(src)
# целевая сумма входящих весов на узел (баланс из базового коннектома) — для гомеостаза
TARGET_IN = {r: sum(w for s, d, w in CONNECTOME + LATENT if d == r) or 1.0 for r in REGIONS}

EVENTS = ["a voice from another room calling a name", "the smell of something burning",
          "warm sunlight across the hands", "a sharp pang of hunger", "a half-remembered face",
          "sudden silence after long noise", "cold water at the feet", "a distant bell, three times"]
# события — не только восприятие, но и ВОЗМУЩЕНИЯ ТЕЛА (интероцепция): меняют уставки гомеостаза
BODY_FX = {"a sharp pang of hunger": {"hunger": +0.35}, "warm sunlight across the hands": {"cold": -0.25},
           "cold water at the feet": {"cold": +0.3}}
# каждый тик тело ДРЕЙФУЕТ к дефициту -> нужда копится сама -> рождает давление (как настоящий голод)
DRIFT = {"hunger": 0.05, "cold": 0.025, "fatigue": 0.035}
# чем нужда снимается: дофамин даёт облегчение ТОЛЬКО за действие, реально адресующее нужду
NEED_FIX = {"hunger": "obtaining and eating food", "cold": "finding warmth or shelter",
            "fatigue": "resting or stopping effort"}


def ollama(system, user, model=MODEL_8B, temp=0.85, max_tokens=120):
    body = json.dumps({"model": model, "system": system, "prompt": user, "stream": False,
                       "think": False, "options": {"temperature": temp, "num_predict": max_tokens}}).encode()
    try:
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=240) as r:
            out = json.loads(r.read())["response"]
        return (out.split("</think>")[-1] if "</think>" in out else out).strip()
    except Exception as e:
        return f"(silent:{e})"


def fire(name, state, incoming, broadcast):
    """Возбуждённый узел: обновляет своё ПОСТОЯННОЕ состояние и выдаёт сигнал."""
    sys = (f"You are {REGIONS[name]} You are a STATEFUL node in a dynamic brain: you carry an inner state "
           "that PERSISTS and evolves over time — it is yours, it accumulates, it changes you. "
           "Speak only in your function. Reply ONLY JSON: "
           "{\"state\":\"your updated inner state, 1-2 lines — carry what still holds, change what shifted\","
           "\"signal\":\"what you send out now, one short line\",\"salience\":<0.0-1.0>} /no_think")
    user = (f"Your inner state right now (you carry it):\n  \"{state}\"\n\n"
            f"Globally conscious:\n  \"{broadcast}\"\n\nSignals reaching you:\n{incoming or '- (quiet)'}\n\n"
            "Update your inner state and emit:")
    out = ollama(sys, user, model=NODE_MODEL[name])
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try:
            d = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
            return (str(d.get("state", state))[:200], str(d.get("signal", ""))[:160],
                    float(d.get("salience", 0.4)))
        except Exception:
            pass
    # JSON не распарсился (бывает у 30b — выдаёт прозу): состояние ВСЁ РАВНО эволюционирует из текста
    if out.startswith("(silent"):
        return state, "", 0.2
    clean = " ".join(out.split())[:200]
    return clean, clean[:160], 0.4


def run(ticks, seed, log_path):
    rng = random.Random(7)
    W = load_weights()                       # живые веса связей (хеббовская пластичность)
    goal = load_goal()                       # ВОЛЯ: цель живёт между запусками и взрослеет
    body = goal.get("body") or {"hunger": 0.3, "cold": 0.3, "fatigue": 0.2}  # ТЕЛО: нужды живут тоже
    expect = goal.get("expect", 0.2)         # предсказание облегчения (для дофаминовой ошибки)
    frustration = goal.get("frustration", 0.0)  # копится, когда цель не снимает нужду -> ACC давит
    commit = goal.get("commit") or {"need": None, "left": 0}  # аллостаз: PFC держит нужду до закрытия
    SATISFIED = 0.3                          # нужда ниже этого = закрыта, можно отпустить
    HORIZON = 4                              # на сколько тиков вперёд PFC предвидит дрейф тела
    DANGER = 0.85                            # нужда выше этого перебивает любое обязательство (аварийный приоритет)
    state = {r: "(forming)" for r in REGIONS}
    act = {r: 0.0 for r in REGIONS}
    fat = {r: 0.0 for r in REGIONS}          # утомление: активный узел спадает, уступает
    since_fired = {r: 0 for r in REGIONS}    # давно не выстреливал -> поднимаем в очередь (справедливость)
    signal = {r: "" for r in REGIONS}        # последний выстрел узла (для рекуррентной передачи)
    act["Thalamus"] = act["Sensory"] = 0.9
    signal["Sensory"] = seed; signal["Thalamus"] = seed
    broadcast = "(awareness flickering on)"
    recent_winner = {}
    history = []                             # недавние осознанные мысли -> душим мантру
    logf = open(log_path, "w")
    def emit(o): logf.write(json.dumps(o, ensure_ascii=False) + "\n"); logf.flush()
    emit({"event": "config", "regions": list(REGIONS), "connectome": CONNECTOME, "seed": seed})

    for t in range(ticks):
        fresh_ev = None
        if t > 0 and t % 3 == 0:             # свежий вход ВТОРГАЕТСЯ — перезаписывает сенсорику, не дописывается
            ev = EVENTS[(t // 3) % len(EVENTS)]
            fresh_ev = ev
            signal["Sensory"] = ev; signal["Thalamus"] = ev
            state["Sensory"] = f"a new thing just arrived: {ev}"; state["Thalamus"] = f"relaying: {ev}"
            act["Sensory"] = 1.0; act["Thalamus"] = 0.9
            fat["Sensory"] = fat["Thalamus"] = 0.0   # вход прорывается сквозь усталость
            for nd, dv in BODY_FX.get(ev, {}).items():           # вход возмущает ТЕЛО, не только восприятие
                body[nd] = max(0.0, min(1.0, body[nd] + dv))
            emit({"event": "input", "t": t, "stimulus": ev}); print(f"   ·· input: {ev}")

        # ТЕЛО дрейфует к дефициту -> нужда копится сама; самая острая = доминирующее давление
        for nd in body:
            body[nd] = max(0.0, min(1.0, body[nd] + DRIFT[nd]))
        need = max(body, key=body.get); intensity = body[need]   # ЧУВСТВУЕМАЯ нужда (интероцепция)
        total_def = sum(body.values())
        # АЛЛОСТАЗ: цель PFC целится в ПРЕДСКАЗАННО критичную нужду и ДЕРЖИТСЯ её до закрытия,
        # а не бросается на самый громкий сиюминутный позыв.
        pred = {nd: body[nd] + DRIFT[nd] * HORIZON for nd in body}
        danger_nd = max(body, key=body.get)
        interrupt = body[danger_nd] >= DANGER and danger_nd != commit["need"]   # авария перебивает обязательство
        if interrupt:
            goal_need = danger_nd                                # угроза жизни забирает руль
            commit = {"need": goal_need, "left": 2}
        elif commit["need"] and body[commit["need"]] > SATISFIED and commit["left"] > 0:
            goal_need = commit["need"]                           # держим обязательство
        else:
            goal_need = max(pred, key=pred.get)                  # иначе берём предсказанно-критичную
            commit = {"need": goal_need, "left": 2}              # и обязуемся на 2 исполнительных цикла
        goal_intensity = body[goal_need]

        # ДИНАМИКА: drive = персистенция(затухание) + рекуррентный вход от соседей + шум
        def jaccard(a, b):
            wa, wb = set(re.findall(r"[a-z']+", a.lower())), set(re.findall(r"[a-z']+", b.lower()))
            return len(wa & wb) / max(1, len(wa | wb))
        total = sum(act.values())            # глобальное торможение -> поле не насыщается в 1.0
        drive = {}
        for r in REGIONS:
            inp = sum(W[f"{s}>{r}"] * act[s] for s in INTO[r])
            deg = sum(W[f"{s}>{r}"] for s in INTO[r]) or 1.0
            echo = jaccard(signal[r], broadcast) if signal[r] else 0.0   # эхо поля -> новизна гасит drive
            drive[r] = 0.65 * act[r] + 0.6 * (inp / deg) - 0.07 * total - 0.5 * fat[r] - 0.4 * echo + 0.05 * rng.random()
            drive[r] = max(0.0, drive[r])
        drive["Hypothalamus"] += 0.7 * intensity      # острая нужда поднимает влечение гипоталамуса
        drive["Insula"] += 0.5 * total_def            # тело копит -> инсула чувствует дискомфорт громче
        # отбор выстреливающих: drive + справедливость (давно молчавшие поднимаются)
        score = {r: drive[r] + 0.12 * min(since_fired[r], 6) for r in REGIONS}
        order = sorted(REGIONS, key=lambda r: -score[r])
        firing = [r for r in order if score[r] > 0.45][:5]
        if len(firing) < 3:
            firing = order[:3]
        # ГАРАНТИЯ ХОДА: 2 самых давно молчавших выстреливают обязательно (рефлексия не голодает)
        for r in sorted(REGIONS, key=lambda r: -since_fired[r])[:2]:
            if r not in firing:
                firing.append(r)

        # параллельный выстрел возбуждённых узлов: каждый обновляет своё СОСТОЯНИЕ
        feel = f"- (body): {need} rising to {intensity:.2f}; whole-body deficit {total_def:.2f}"
        def do(r):
            incoming = "\n".join(f"- {s}: {signal[s]}" for s in INTO[r] if signal[s])[:600]
            if goal.get("aim") and r in ("PFC", "Striatum", "ACC", "DMN"):
                incoming = f"- (standing intention you hold): {goal['aim']}\n" + incoming   # цель тянет поток
            if r in ("Hypothalamus", "Insula"):
                incoming = feel + "\n" + incoming                  # тело говорит телесным узлам
            return r, fire(r, state[r], incoming, broadcast)
        with ThreadPoolExecutor(max_workers=len(firing)) as ex:
            fired = dict(ex.map(do, firing))

        # обновляем поле: состояние/сигнал у выстреливших; активация у всех (с реверберацией)
        for r in REGIONS:
            if r in fired:
                st, sg, sal = fired[r]
                state[r] = st; signal[r] = sg
                act[r] = min(1.0, 0.6 * drive[r] + 0.5 * sal)
                since_fired[r] = 0
            else:
                act[r] = max(0.0, min(1.0, drive[r]))     # тихий узел лишь интегрирует/затухает
                since_fired[r] += 1
            fat[r] = min(2.0, 0.75 * fat[r] + 0.45 * act[r])   # утомление копится с активностью, спадает

        # ХЕББ: связь крепнет, когда ОБА конца выстрелили в ЭТОТ тик (со-фаер = событие, не фон);
        # слабеет медленно. Рост перевешивает распад на совпавших путях -> отбор, а не размытие.
        for k in W:
            s, d = k.split(">")
            cofire = 1.0 if (s in fired and d in fired) else 0.0
            grow = 0.18 * cofire * act[s] * act[d]
            W[k] = max(0.0, min(1.6, W[k] + grow - 0.004 * W[k]))
        # ГОМЕОСТАЗ (synaptic scaling): нормируем сумму входов узла к базовому балансу.
        # Хебб решает КАКИЕ связи сильнее ОТНОСИТЕЛЬНО соседей; один путь не может сожрать узел.
        for r in REGIONS:
            cur = sum(W[f"{s}>{r}"] for s in INTO[r])
            if cur > 0:
                scale = (0.7 * TARGET_IN[r] + 0.3 * cur) / cur   # мягко тянем к таргету, не жёстко
                for s in INTO[r]:
                    W[f"{s}>{r}"] = min(1.6, W[f"{s}>{r}"] * scale)

        # ВОЛЯ: раз в 5 тиков существо лепит/держит свою цель из прожитого и делает шаг к ней;
        # раз в 10 — допрашивает цель («зачем? для чего?») и может переписать её глубже.
        if t > 0 and t % 5 == 0:
            held = commit["left"] > 0 and commit["need"] == goal_need
            new_aim, step, wanting = executive(goal, state, broadcast, goal_need, goal_intensity, frustration, held)
            event = "form" if goal.get("aim") is None else ("reshape" if jaccard(new_aim, goal.get("aim") or "") < 0.5 else "hold")
            if goal.get("aim") is None:
                goal["born_t"] = t
            goal["aim"] = new_aim
            did = act_toward(new_aim, step) if step else ""
            # МИР судит: сняло ли действие НУЖДУ, на которую существо обязалось. Дофамин = ошибка предсказания.
            relief, verdict_w = world_judge(did, goal_need) if did else (0.0, "")
            body[goal_need] = max(0.0, body[goal_need] - relief)  # реальное облегчение тела
            commit["left"] -= 1
            if body[goal_need] <= SATISFIED:                      # нужда закрыта -> обязательство снято
                commit = {"need": None, "left": 0}
            dopamine = relief - expect                            # RPE: лучше/хуже ожидания
            expect = max(0.0, min(0.6, expect + 0.4 * dopamine))  # ожидание подтягивается к опыту
            frustration = max(0.0, min(1.5, frustration + (0.4 if relief < 0.1 else -0.6)))  # не помогло -> копится
            act["Striatum"] = min(1.0, act["Striatum"] + 0.5 * wanting + 0.5 * max(0.0, dopamine))
            goal["history"].append({"t": t, "aim": new_aim, "event": event, "wanting": round(wanting, 2),
                                    "need": goal_need, "relief": round(relief, 2), "frustration": round(frustration, 2)})
            if did:
                goal["actions"].append({"t": t, "step": step, "did": did, "need": goal_need,
                                        "relief": round(relief, 2), "world": verdict_w})
            emit({"event": "goal", "t": t, "kind": event, "aim": new_aim, "step": step, "did": did,
                  "need": goal_need, "intensity": round(goal_intensity, 2), "relief": round(relief, 2),
                  "dopamine": round(dopamine, 2), "frustration": round(frustration, 2), "held": held})
            if interrupt:
                print(f"   ⚠ ПРЕРЫВАНИЕ: {goal_need}={goal_intensity:.2f} — угроза перебила обязательство")
            print(f"   ◆ GOAL [{event}{' ·held' if held else ''}] {new_aim}")
            print(f"   ▸ did: {did[:80]}")
            print(f"   ⊙ body:{goal_need}={goal_intensity:.2f} relief={relief:.2f} dope={dopamine:+.2f} frust={frustration:.2f} «{verdict_w[:50]}»")
            if t % 10 == 0:
                verdict, deeper, why = interrogate(new_aim, broadcast)
                if verdict == "reshape" and jaccard(deeper, new_aim) < 0.6:
                    goal["aim"] = deeper
                    goal["history"].append({"t": t, "aim": deeper, "event": "interrogated", "why": why})
                    emit({"event": "goal", "t": t, "kind": "interrogated", "aim": deeper, "why": why})
                    print(f"   ? WHY -> reshaped: {deeper}  (for: {why[:60]})")
                else:
                    print(f"   ? WHY -> kept  (for: {why[:60]})")

        # IGNITION: сильнейший свежий сигнал вспыхивает -> сознание
        def overlap(a, b):
            wa, wb = set(re.findall(r"[a-z']+", a.lower())), set(re.findall(r"[a-z']+", b.lower()))
            return len(wa & wb) / max(1, len(wa | wb))
        cand = [(r, act[r] * (0.4 if recent_winner.get(r, 9) == 0 else 1.0), signal[r])
                for r in fired if signal[r]]
        # анти-мантра: штрафуем сигнал, перепевающий недавние мысли (не только текущий broadcast)
        def stale(sig): return max([overlap(sig, h) for h in history] or [0.0])
        cand = [(r, sc * (1.0 - 0.7 * stale(sig)), sig) for r, sc, sig in cand]
        fresh = [c for c in cand if overlap(c[2], broadcast) < 0.5 and stale(c[2]) < 0.6]
        if not fresh and fresh_ev:           # всё несвежее -> внешняя реальность пробивает мантру
            winner = ("Sensory", 1.0, signal["Sensory"])
        else:
            winner = max(fresh or cand, key=lambda x: x[1])
        history.append(winner[2]); history[:] = history[-6:]
        recent_winner = {n: a + 1 for n, a in recent_winner.items()}; recent_winner[winner[0]] = 0
        broadcast = winner[2]
        thought = ollama("the LANGUAGE CORTEX: inner speech of this brain.",
                         f"Conscious content now (ignited by {winner[0]}): \"{broadcast}\"\n"
                         "Say it as ONE first-person sentence of inner speech: /no_think", temp=0.7, max_tokens=60)

        emit({"event": "tick", "t": t, "ignited_by": winner[0], "thought": thought,
              "act": {r: round(act[r], 2) for r in REGIONS},
              "weights": {k: round(v, 3) for k, v in W.items()},
              "states": {r: state[r] for r in REGIONS}})
        hot = sorted(REGIONS, key=lambda r: -act[r])[:5]
        print(f"[t{t:02d}] ⚡{winner[0]:11} {thought[:80]}")
        print(f"      hot: " + " ".join(f"{r}{act[r]:.1f}" for r in hot))
        print()

    goal["body"] = body; goal["expect"] = expect; goal["frustration"] = frustration; goal["commit"] = commit
    save_weights(W)                          # структура переживёт запуск -> развитие через жизнь
    save_goal(goal)                          # воля+тело переживут запуск -> желание взрослеет через жизнь
    logf.close()
    # РАССКАЗ СЛОВАМИ: как жило и менялось желание существа за эту жизнь
    print("\n" + "=" * 60 + "\nЖИЗНЬ ЖЕЛАНИЯ:")
    if not goal["history"]:
        print("  (за этот прогон цель не родилась)")
    else:
        for h in goal["history"]:
            tag = {"form": "родилась", "reshape": "переросла в", "hold": "держит",
                   "interrogated": "после «зачем?» стала"}.get(h["event"], h["event"])
            line = f"  t{h['t']:>2}  {tag}: {h['aim']}"
            if h.get("why"):
                line += f"\n         (ради чего: {h['why']})"
            print(line)
    if goal["actions"]:
        print("\nЧТО СДЕЛАЛО РУКАМИ (и ответил ли мир):")
        for a in goal["actions"]:
            r = a.get("relief", 0)
            mark = "✓ помогло" if r >= 0.2 else "✗ телу не легче"
            print(f"  t{a['t']:>2}  [{a.get('need','?')}] {a['did'][:70]}")
            print(f"        мир: «{a.get('world','')[:60]}» -> {mark} ({r:.2f})")
    print(f"\nТЕЛО НА КОНЕЦ ЖИЗНИ: " + ", ".join(f"{k}={v:.2f}" for k, v in body.items())
          + f" | фрустрация={frustration:.2f}")
    print("=" * 60)
    print(f"→ {log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=16)
    ap.add_argument("--seed", default="a cold draft on the skin, and somewhere a door closing")
    args = ap.parse_args()
    os.makedirs("logs", exist_ok=True)
    run(args.ticks, args.seed, f"logs/brain-dynamic-{int(time.time())}.jsonl")
