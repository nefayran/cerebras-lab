#!/usr/bin/env python3
"""ОРГАН ЛОГИКИ — детерминированная «рука» для дедукции/назначений, как calc для арифметики.
ЛЛМ парсит условие в структуру (агенты × предметы × ограничения), а ПЕРЕБОР детерминированно
находит единственное согласованное назначение. Общий для класса «кто-владеет-чем», не под задачу.
Возвращает ответ (оригинальный регистр) либо None, если это не пазл назначения / неоднозначно."""
import itertools, re
from stream import llm, jget

PARSE = ('Parse this ONLY if it is an assignment puzzle (each agent gets exactly one distinct item — a bijection). '
         'agents = the entities; items = the distinct things assigned; '
         'constraints = list of [agent, "is" or "not", item]; '
         'query = {"by":"item" or "agent", "key":"the item or agent the question asks about"}. '
         'If it is NOT such a puzzle, set agents to []. '
         'JSON {"agents":[..],"items":[..],"constraints":[[a,rel,i]],"query":{"by":"..","key":".."}}.')


def _n(s): return str(s).strip().lower()


def solve(problem, query=None):
    """problem — несёт ОГРАНИЧЕНИЯ; query (если задан) — под-вопрос, на который отвечаем
    (для композитов: ограничения из целого, а спрашиваем про под-шаг)."""
    raw = llm(PARSE, f"Problem: {problem}\nJSON:", temp=0.1, max_tokens=320)
    agents = [str(a) for a in (jget(raw, "agents")[0] or [])]
    items = [str(i) for i in (jget(raw, "items")[0] or [])]
    cons = jget(raw, "constraints")[0] or []
    if len(agents) < 2 or len(agents) != len(items):
        return None                                   # не пазл назначения -> пусть решает ЛЛМ

    # QUERY берём ДЕТЕРМИНИРОВАННО из текста (ЛЛМ тут ненадёжен); для под-шага — из явного query
    if query:
        qsent = _n(query)
    else:
        qs = re.findall(r"[^.?!]*\?", problem)
        qsent = _n(qs[-1] if qs else problem)
    by = key = None
    for i in items:                                   # предмет в вопросе -> «кто владеет предметом»
        if _n(i) in qsent: by, key = "item", _n(i); break
    if not by:
        for a in agents:                              # иначе агент в вопросе -> «что у агента»
            if _n(a) in qsent: by, key = "agent", _n(a); break
    if not by:
        return None
    items_n = [_n(i) for i in items]
    C = [( _n(c[0]), _n(c[1]), _n(c[2]) ) for c in cons if isinstance(c, (list, tuple)) and len(c) == 3]

    sols = []
    for perm in itertools.permutations(items_n):
        amap = {_n(a): p for a, p in zip(agents, perm)}
        ok = True
        for a, rel, i in C:
            if a not in amap or i not in items_n:      # шумная ссылка -> пропустить
                continue
            if "not" in rel:
                if amap[a] == i: ok = False; break
            else:
                if amap[a] != i: ok = False; break
        if ok: sols.append(amap)
    if not sols:
        return None

    def keymatch(x, k): return x == k or k in x or x in k
    ans = set()
    for s in sols:
        if by == "item":
            for a, i in s.items():
                if keymatch(i, key): ans.add(a)
        elif by == "agent":
            for a, i in s.items():
                if keymatch(a, key): ans.add(i)
    if len(ans) != 1:
        return None                                   # неоднозначно -> не врём
    a = ans.pop()
    for orig in agents + items:                       # вернуть оригинальный регистр
        if _n(orig) == a: return orig
    return a


if __name__ == "__main__":
    print(solve("Ann, Bob and Cara each own a different pet: a cat, a dog, or a fish. "
                "Ann does not own the cat. Bob owns the dog. Who owns the fish?"))
