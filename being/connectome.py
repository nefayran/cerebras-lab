#!/usr/bin/env python3
"""
КОННЕКТОМ АНСАМБЛЕЙ ВОКРУГ ОБЩЕГО РАБОЧЕГО СТОЛА (Global Workspace).
Не конвейер слоёв, а области-ансамбли, связанные во все стороны вокруг общего стола:

                 ┌──────────── GLOBAL WORKSPACE ────────────┐
                 │ focus (стек), facts, hypotheses, notes    │
                 └──▲────────▲────────▲────────▲─────────▲───┘
       bottom-up │ top-down │ recall │ ACC петля │ окраска │
        ВОСПРИЯТИЕ  ПАМЯТЬ    РАССУЖД.  МОНИТОР     ЛИЧНОСТЬ
        (ансамбль) (RAG-сеть) (ансамбль) (ACC,анс.) (стиль)

ЦИКЛ СОЗНАНИЯ: на каждом такте все области читают стол и ПИШУТ в него; победившая
гипотеза транслируется (global broadcast) и становится очередной мыслью ПОТОКА.
Связи: top-down (память/фокус правят восприятие), рекуррентные (ACC -> память/восприятие),
латеральные (рассуждение <-> монитор). Выбор НЕСОВЕРШЕН (несколько вариантов, иногда мимо).

Каждая область = АНСАМБЛЬ узлов (несколько вызовов 8b), нигде не один.
Запуск: being/venv/bin/python being/connectome.py ["задача"]
"""
import sys, re, random, os
from concurrent.futures import ThreadPoolExecutor
from stream import llm, calc, jget, say, voice, factstr, perceive as perceive_net
import memory, persona, viz, logic_tool, glob, grow_organ, self_model, control

WRITER = ('You are an expert. Answer the task directly, fully and concretely. No meta-commentary, no restating the task.')
LAST = {}   # последнее (features, action) для подкрепления гейта извне


def passthrough(problem):
    """ПОЛ: один проход базовой модели. Это НЕ пайплайн — это сама база, без машинерии мозга."""
    return llm(WRITER, f"Task: {problem}", temp=0.5, max_tokens=900)


def _load_grown():
    """Реестр ВЫРОЩЕННЫХ органов (organs/*.py), загружаются в песочнице как при отращивании."""
    out = {}
    d = os.path.join(os.path.dirname(__file__), "organs")
    for f in glob.glob(os.path.join(d, "*.py")):
        if f.endswith("__init__.py"): continue
        try: out[os.path.basename(f)[:-3]] = grow_organ.load_organ(open(f).read())
        except Exception: pass
    return out
_ORGANS = _load_grown()


def reload_organs():
    """Перечитать вырощенные органы (после того как мозг отрастил новую руку на ходу)."""
    global _ORGANS; _ORGANS = _load_grown(); return list(_ORGANS)


def deterministic(text):
    """Детерминированные руки по порядку: логика, затем вырощенные органы. (answer, organ) или (None,None)."""
    a = logic_tool.solve(text)
    if a: return a, "logic"
    for name, fn in _ORGANS.items():
        try:
            r = fn(text)
            if r not in (None, ""): return str(r), name
        except Exception: pass
    return None, None

random.seed(7)
MAX_CYCLES = 14
N_PROPOSE = 3      # размер ансамбля рассуждения (варианты)
N_CRITIC = 2       # размер ансамбля монитора (ACC)
ACCEPT = 0.6       # порог уверенности для принятия ответа

# --- области (каждая — ансамбль; промпты крошечные, только функция) ---
MQUERY  = ('List 2-3 different things worth recalling to answer this. JSON {"queries":["..",".."]}.')
RECALL1 = ('What do you know about this — concrete facts or rough estimates with numbers? JSON {"facts":["..",".."]}.')
GATE    = ('GIVEN facts are ground truth. From the numbered RECALLED claims, list the indices that CONTRADICT a '
           'given fact or are unrelated to the question. JSON {"drop":[0,2]}.')
PROPOSE = ('Look at the facts and the focus. {angle} If arithmetic, give the expression. '
           'JSON {"kind":"subq" or "answer","text":"..","expr":".. or \\"\\""}.')
# линзы рассуждения — разные «стороны», как разные системы мозга (а не одна N раз)
LENSES = [
    ("direct",  "Give a concrete direct answer — a value, number or statement. Do NOT restate the question."),
    ("split",   "Take ONE certain fact and ask what it FORCES next (work step by step by elimination). kind=subq."),
    ("skeptic", "Challenge the easy answer: state a doubt or a counter-example."),
    ("analogy", "Answer through an analogy or association from another domain."),
]
CRITIC  = ('Does this thought fit the facts and help answer the focus? Score 0..1 and say why briefly. '
           'JSON {"score":0.0,"why":".."}.')
COMMIT  = ('Give your best DIRECT answer to the focus now, even if unsure — do not ask another question. '
           'If arithmetic, give the expression. JSON {"text":"..","expr":".. or \\"\\""}.')
PERSONA = ('In one short first-person feeling, react to how this is going. JSON {"thought":".."}.')
VERIFY  = ('The GROUND-TRUTH facts are certain. Check the answer against EACH fact: does it hold without '
           'contradicting any of them? If it contradicts even one, ok=false. JSON {"ok":true/false,"why":".."}.')
# System 2 — медленное forward-рассуждение в рабочей памяти
DERIVE  = ('From the established facts, state ONE NEW fact that NECESSARILY follows (is forced) — a step forward. '
           'If nothing new is forced, done=true. JSON {"derived":"..","done":false}.')
ANSWER2 = ('Using ONLY the established facts, answer the goal directly. If arithmetic, give the expression. '
           'JSON {"text":"..","expr":".. or \\"\\""}.')


class Workspace:
    """Общий рабочий стол: все области читают и пишут сюда."""
    def __init__(self):
        self.facts, self.notes, self.stack = [], [], []
        self.given = []        # исходные воспринятые факты = ИСТИНА (для верификации, не отравить памятью)
        self.ruled_out = []    # ответы, отвергнутые верификацией (сигнал ошибки ACC -> избегать)
        self.deliberated = set()   # фокусы, где уже включали System 2 (не зацикливать)
        self.answer = None; self.recalled_for = set(); self.visits = {}
    def avoid(self):
        return ("Do NOT answer any of these — they contradict the facts: " + "; ".join(self.ruled_out) + "\n") if self.ruled_out else ""
    def remember(self, new):
        have = {str(x).strip().lower() for x in self.facts}
        for f in new or []:
            k = str(f).strip().lower()
            if k and k not in have: have.add(k); self.facts.append(str(f).strip())
    def factstr(self): return "; ".join(str(f) for f in self.facts)
    def notestr(self): return "\n".join(f"- {q} = {v}" for q, v in self.notes) or "(none)"


# --- ПАМЯТЬ: многоузловая RAG-сеть (query-expansion -> 2 источника -> слияние) ---
def recall_net(focus, given=""):
    queries = list(dict.fromkeys((jget(llm(MQUERY, f"Question: {focus}\nJSON:", temp=0.6), "queries")[0] or []) + [focus]))[:3]
    def one(q):
        f = jget(llm(RECALL1, f"About: {q}\nJSON:", temp=0.4), "facts")[0] or []   # СЕМАНТИЧЕСКАЯ (параметрика)
        return f + memory.search(q, k=3)                                            # ЭПИЗОДИЧЕСКАЯ (вектор-стор)
    facts = []
    with ThreadPoolExecutor(max_workers=max(1, len(queries))) as ex:
        for fl in ex.map(one, queries): facts += fl
    # ГЕЙТ СОГЛАСОВАННОСТИ: примирить память с реальностью — выкинуть противоречащее данным / нерелевантное
    if facts and given:
        listing = "\n".join(f"{i}: {f}" for i, f in enumerate(facts))
        drop = jget(llm(GATE, f"GIVEN facts:\n{given}\n\nRECALLED:\n{listing}\nQuestion: {focus}\nJSON:"), "drop")[0] or []
        drop = {int(i) for i in drop if str(i).strip().lstrip("-").isdigit()}
        facts = [f for i, f in enumerate(facts) if i not in drop]
    return facts


# --- РАССУЖДЕНИЕ: ансамбль РАЗНОУГЛОВЫХ линз ПАРАЛЛЕЛЬНО (разные «стороны», как системы мозга) ---
def propose_ensemble(ws, focus):
    ctx = f"Facts: {ws.factstr()}\nNotes:\n{ws.notestr()}\n{ws.avoid()}Focus: {focus}\nJSON:"
    def one(item):
        name, angle = item
        kind, text, expr = jget(llm(PROPOSE.replace("{angle}", angle), ctx, temp=0.6), "kind", "text", "expr")
        return {"kind": (kind or "answer"), "text": str(text), "expr": expr, "lens": name} if text else None
    with ThreadPoolExecutor(max_workers=len(LENSES)) as ex:
        return [h for h in ex.map(one, LENSES) if h]


# --- МОНИТОР (ACC): ансамбль критиков ПАРАЛЛЕЛЬНО ---
def score_hyp(ws, focus, h):
    def crit(i):
        s = jget(llm(CRITIC, f"Facts: {ws.factstr()}\nFocus: {focus}\nThought: {h['text']}\nJSON:",
                     temp=0.3 + 0.2 * i), "score")[0]
        try: return float(s)
        except Exception: return 0.5
    with ThreadPoolExecutor(max_workers=N_CRITIC) as ex:
        scores = list(ex.map(crit, range(N_CRITIC)))
    return sum(scores) / len(scores) if scores else 0.5


def select(scored):
    """Глобальная трансляция: вес ~ оценка, но выбор стохастичен -> иногда берём не лучшее (человечно)."""
    if not scored: return None, 0.0
    weights = [max(0.01, sc) ** 2 for _, sc in scored]
    h, sc = random.choices(scored, weights=weights, k=1)[0]
    return h, sc


def is_arith(e):
    """Чисто арифметическое выражение: только цифры/операторы/скобки и есть оператор (не '8:00 + 35 minutes')."""
    e = str(e or "")
    return bool(re.fullmatch(r"[\d\s+\-*/().]+", e)) and bool(re.search(r"[-+*/]", e))


def value_of(h):
    if is_arith(h.get("expr")):
        try: return calc(h["expr"])
        except Exception: pass
    return h["text"]


def verify(answer, given):
    """Опровержение против ИСТИНЫ (воспринятых фактов): противоречит хоть одному -> False."""
    if not given: return True
    ok = jget(llm(VERIFY, f"Ground-truth facts:\n{given}\nAnswer: {answer}\nJSON:", temp=0.2), "ok")[0]
    return ok is not False     # None/parse-fail -> не блокируем


def deliberate(ws, focus, indent, problem):
    """System 2. Сперва тянемся за ДЕТЕРМИНИРОВАННОЙ РУКОЙ (орган логики, как calc для чисел);
    если задача не на назначение — идём ВПЕРЁД от достоверного свободным выводом."""
    tool = logic_tool.solve(problem)                  # орган логики: точный перебор по структуре
    if tool:
        say(f"work it out by elimination: {tool}", indent)
        viz.emit("region", name="CALC"); viz.emit("speak", text=f"by elimination → {tool}")
        return {"kind": "answer", "text": str(tool), "expr": "", "trusted": True}
    base = "\n".join(ws.given) or ws.factstr()
    derived = []
    for _ in range(5):
        ctx = base + ("\nDerived so far:\n" + "\n".join(derived) if derived else "")
        # ВАЖНО: цель НЕ показываем — иначе шаг прыгает к догадке вместо распространения
        d, done = jget(llm(DERIVE, f"Established facts:\n{ctx}\nWhat ONE new fact is forced next?\nJSON:", temp=0.2), "derived", "done")
        d = str(d or "").strip()
        if done or not d or d in derived: break
        derived.append(d); say(d, indent)
        viz.emit("region", name="REASONING"); viz.emit("speak", text=d)
    t, e = jget(llm(ANSWER2, f"Established facts:\n{base}\n" + "\n".join(derived) + f"\nGoal: {focus}\nJSON:", temp=0.2),
                "text", "expr")
    return {"kind": "answer", "text": str(t or ""), "expr": e}


def run(problem, force_action=None):
    print("=" * 72)
    print(f"TASK: {problem}\n\nSTREAM OF CONSCIOUSNESS (connectome):\n")
    # ВЫУЧЕННЫЙ ГЕЙТ: включать машинерию или не мешать и дать базе ответить (пол: не хуже базы)
    action, feats = (force_action, control.features(problem)) if force_action else control.choose(problem)
    LAST["features"], LAST["action"] = feats, action
    if action == "passthrough":
        say("[gate: pass through — my machinery won't beat my base here]", "  ")
        doc = passthrough(problem)
        print(f"\n  => ANSWER:\n{doc}\n")
        return doc
    viz.begin(); viz.emit("start", problem=problem)
    ws = Workspace()
    lead, dis, appraisal = persona.color(f"You just saw this problem: {problem}\nReact in one short first-person line.")
    say(lead, "  ")                         # ЛИЧНОСТЬ (MLX-ансамбль) красит вход
    viz.emit("persona", phase="appraisal", reacts=appraisal, disagree=dis, lead=lead)
    if dis >= 1.0: say("  (facets pull different ways)", "  ")
    pf = perceive_net(problem)              # СЛОЙ восприятия (bottom-up), ансамбль перцепторов
    ws.remember(pf); ws.given = list(pf)    # воспринятое = ИСТИНА для верификации
    viz.emit("region", name="PERCEPTION"); viz.emit("facts", items=ws.facts[:12])
    tool_ans, organ_name = deterministic(problem)   # детерминированная рука (логика / вырощенный орган) авторитетна
    me = self_model.reflect(problem, organ_name)     # МЕТАКОГНИЦИЯ: сверяюсь с собой перед задачей
    if me: say(me, "  "); viz.emit("region", name="PERSONALITY"); viz.emit("speak", text=me)
    ws.stack.append(problem)

    def norm(s): return str(s).strip().lower()[:60]

    for c in range(MAX_CYCLES):
        if not ws.stack: break
        focus = ws.stack[-1]
        indent = "  " + "    " * (len(ws.stack) - 1)
        viz.emit("focus", stack=list(ws.stack))
        # ЕДИНО-ДОМЕННАЯ задача: распознал тип -> сразу дёрнул орган (как calc для чистой арифметики).
        # Композит сюда не попадёт: deterministic(целое) вернёт None, пойдёт дробление+под-шаговый роутинг.
        if focus == problem and tool_ans and focus not in ws.deliberated:
            ws.deliberated.add(focus)
            how = "by elimination" if organ_name == "logic" else f"with my {organ_name} sense"
            say(f"I recognise this kind — work it out {how}: {tool_ans}", indent)
            viz.emit("region", name="CALC"); viz.emit("speak", text=f"{how} → {tool_ans}")
            ws.notes.append((focus[:50], tool_ans)); ws.answer = tool_ans
            ws.stack.pop(); viz.emit("focus_pop", note=str(tool_ans)[:40]); continue
        # ОРКЕСТРАЦИЯ: под-вопрос-назначение роутим в орган логики С ПОЛНЫМ условием задачи
        if len(ws.stack) > 1 and re.search(r"\bwho (owns|has|is)\b|\bwhat does\b", focus.lower()) and focus not in ws.deliberated:
            sub = logic_tool.solve(problem, query=focus)
            if sub:
                ws.deliberated.add(focus)
                say(f"work it out by elimination: {sub}", indent)
                viz.emit("flow", src="REASONING", dst="CALC"); viz.emit("region", name="CALC"); viz.emit("speak", text=f"by elimination → {sub}")
                ws.notes.append((focus[:60], sub)); ws.answer = sub
                ws.stack.pop(); viz.emit("focus_pop", note=str(sub)[:40]); continue
        ws.visits[focus] = ws.visits.get(focus, 0) + 1
        stalled = ws.visits[focus] >= 3                          # застряли на фокусе -> пора коммитить
        # top-down: память тянется под текущий фокус (один раз на новый фокус)
        if focus not in ws.recalled_for:
            ws.recalled_for.add(focus)
            got = recall_net(focus, ws.factstr())     # сверяем вспомненное с уже данными фактами
            ws.remember(got)
            viz.emit("flow", src="PERCEPTION", dst="MEMORY")     # фокус из восприятия будит память
            viz.emit("region", name="MEMORY"); viz.emit("recall", items=got[:8])
            if got: voice(f"You bring to mind: {factstr(got)[:140]}", indent)

        if stalled:                                             # КОММИТ: породить прямой ответ, не зацикливаться
            text, expr = jget(llm(COMMIT, f"Facts: {ws.factstr()}\nNotes:\n{ws.notestr()}\n{ws.avoid()}Focus: {focus}\nJSON:",
                                  temp=0.7), "text", "expr")
            winner = {"kind": "answer", "text": str(text or ""), "expr": expr}
            say(winner["text"], indent)
            viz.emit("region", name="REASONING"); viz.emit("speak", text=winner["text"], commit=True)
            val = value_of(winner)
            if is_arith(winner.get("expr")):
                say(f"  ({winner['expr']} = {val})", indent); viz.emit("region", name="CALC"); viz.emit("calc", expr=str(winner["expr"]), val=str(val))
            ws.notes.append((focus[:50], val)); ws.answer = val; ws.stack.pop()
            continue

        # латераль: рассуждение предлагает варианты -> монитор их судит
        hyps = propose_ensemble(ws, focus)
        if not hyps: ws.stack.pop(); continue
        viz.emit("flow", src="MEMORY", dst="REASONING")          # факты памяти кормят гипотезы
        viz.emit("region", name="REASONING")
        viz.emit("propose", hyps=[h["text"][:80] for h in hyps], lenses=[h.get("lens", "") for h in hyps])
        scored = [(h, score_hyp(ws, focus, h)) for h in hyps]
        viz.emit("flow", src="REASONING", dst="MONITOR")         # гипотезы идут НАПРЯМУЮ критикам
        viz.emit("region", name="MONITOR"); viz.emit("critic", scores=[round(s, 2) for _, s in scored])
        winner, sc = select(scored)
        if ws.ruled_out:                       # уже отвергали прямой ответ -> ДЕКОМПОЗИРУЙ, не пере-гадывай
            subq = next((h for h, _ in scored if h["kind"] == "subq" or str(h["text"]).strip().endswith("?")), None)
            if subq is not None: winner, sc = subq, 0.5
        say(winner["text"], indent)
        viz.emit("speak", text=winner["text"], score=round(sc, 2), kind=winner["kind"])

        text = str(winner["text"]).strip()
        is_q = winner["kind"] == "subq" or text.endswith("?")           # вопрос НЕ может стать ответом
        repeat = norm(text) == norm(focus) or norm(text) in [norm(s) for s in ws.stack]
        if is_q and not repeat and len(ws.stack) < 3:
            ws.stack.append(text); viz.emit("focus_push", q=text[:80])  # ДЕРЕВО: преследуем под-вопрос
        else:
            if is_q:                                                    # углубляться некуда -> коммитим прямой ответ
                ct, cexpr = jget(llm(COMMIT, f"Facts: {ws.factstr()}\nNotes:\n{ws.notestr()}\n{ws.avoid()}Focus: {focus}\nJSON:",
                                     temp=0.7), "text", "expr")
                winner = {"kind": "answer", "text": str(ct or ""), "expr": cexpr}; sc = ACCEPT
                say(winner["text"], indent); viz.emit("speak", text=winner["text"], commit=True)
            val = value_of(winner)
            if is_arith(winner.get("expr")):
                viz.emit("flow", src="REASONING", dst="CALC")
                say(f"  ({winner['expr']} = {val})", indent); viz.emit("region", name="CALC"); viz.emit("calc", expr=str(winner["expr"]), val=str(val))
            # ВЕРИФИКАЦИЯ опровержением против ИСТИНЫ (не против отравленного памятью стола)
            vok = verify(winner["text"], "\n".join(ws.given)) if not is_q else True
            if sc >= ACCEPT and vok:
                if tool_ans and focus == problem and focus not in ws.deliberated:
                    # детерминированный орган АВТОРИТЕТЕН поверх догадки (как calc для чисел)
                    ws.deliberated.add(focus)
                    how = "by elimination" if organ_name == "logic" else f"with my {organ_name} sense"
                    voice("Let me be sure — work it out exactly.", indent)
                    say(f"work it out {how}: {tool_ans}", indent)
                    viz.emit("region", name="CALC"); viz.emit("speak", text=f"{how} → {tool_ans}")
                    val = tool_ans
                ws.notes.append((focus[:50], val)); ws.answer = val
                ws.stack.pop(); viz.emit("focus_pop", note=f"{focus[:40]} = {val}")
            elif focus not in ws.deliberated:
                # System 1 догадка не прошла -> РЕКРУТИРУЕМ System 2: идём вперёд от достоверного
                ws.deliberated.add(focus)
                if not vok and winner["text"].strip() not in ws.ruled_out: ws.ruled_out.append(winner["text"].strip())
                voice("Let me work forward from what's certain, step by step.", indent)
                viz.emit("region", name="MONITOR"); viz.emit("doubt")
                cand = deliberate(ws, focus, indent, problem)
                cval = value_of(cand)
                # туле (детерминир. орган) верим как калькулятору — без перепроверки шатким ЛЛМ
                if cand["text"].strip() and (cand.get("trusted") or verify(cand["text"], "\n".join(ws.given))):
                    ws.notes.append((focus[:50], cval)); ws.answer = cval
                    ws.stack.pop(); viz.emit("focus_pop", note=str(cval)[:40])
                else:
                    ws.recalled_for.discard(focus)
            else:
                why = "it contradicts a given fact" if not vok else "not convincing"
                if not vok and winner["text"].strip() not in ws.ruled_out:
                    ws.ruled_out.append(winner["text"].strip())
                voice(f"Wait — {why}; I look again.", indent)
                viz.emit("flow", src="MONITOR", dst="PERCEPTION"); viz.emit("flow", src="MONITOR", dst="MEMORY")
                viz.emit("region", name="MONITOR"); viz.emit("doubt")
                ws.recalled_for.discard(focus)

    if ws.answer in (None, "", "None"):     # финал пуст -> взять самый содержательный узел-заметку
        ws.answer = max((str(v) for _, v in ws.notes if v not in (None, "", "None")),
                        key=len, default=ws.answer)
    lead, dis2, feeling = persona.color(f"You arrived at the answer: {ws.answer}\nReact in one short first-person feeling.")
    say(lead, "  ")                         # ЛИЧНОСТЬ (MLX-ансамбль) красит финал
    viz.emit("persona", phase="feeling", reacts=feeling, disagree=dis2, lead=lead)
    viz.emit("answer", value=str(ws.answer)); viz.end()
    print(f"\n  => ANSWER: {ws.answer}\n")
    memory.add(f"{problem.strip()} -> {ws.answer}")     # консолидация в вектор-стор
    for q, v in ws.notes:                               # и ключевые надуманные факты
        if v not in (None, "", "None"): memory.add(f"{q}: {v}")
    bio = os.path.join(os.path.dirname(__file__), "biography.jsonl")   # опыт для «сна»
    with open(bio, "a") as f:
        import json as _j
        f.write(_j.dumps({"problem": problem, "answer": str(ws.answer),
                          "appraisal": appraisal, "feeling": feeling}, ensure_ascii=False) + "\n")
    return ws.answer


if __name__ == "__main__":
    P = ("A shop had 3 crates of 24 apples each. They sold 17 apples on Monday and twice as many on Tuesday. "
         "On Wednesday they received 2 more crates of 24. How many apples does the shop have now?")
    run(sys.argv[1] if len(sys.argv) > 1 else P)
