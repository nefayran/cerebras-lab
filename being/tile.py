#!/usr/bin/env python3
"""
РЕКУРСИВНОЕ ЗАМОЩЕНИЕ С РАЗДЕЛЁННЫМИ РОЛЯМИ (tile.py).

Прошлая версия ломалась: узел сам судил атомарность (переоптимистично) и на листе
КОПИРОВАЛ родителя. Теперь три РАЗНЫЕ операции, ни одной роли-диктатора:

  1. РАЗБИЕНИЕ (decompose): несколько узлов независимо бьют регион на под-части,
     их списки ОБЪЕДИНЯЮТСЯ и дедупятся -> коллективная разбивка региона (много умов).
  2. ГОЛОС О СОСТАВНОСТИ (vote): панель узлов голосует «эта часть ещё составная или
     уже атомарна?» -> большинство. Не самосуд одного узла.
  3. ЛИСТ (leaf): атомарную часть один узел пишет ПОЛНОСТЬЮ и конкретно.

Рекурсия идёт по НОВЫМ под-частям из разбиения -> ребёнок не равен родителю.
Кто делит — сами узлы (разбиением + голосованием). Размер куска — голос о составности.
Запуск: being/venv/bin/python being/tile.py ["задача"]
"""
import sys, re, math
from concurrent.futures import ThreadPoolExecutor
from stream import llm, jget
import viz, memory

N_SPLIT = 4           # узлов-разбивателей на регион (каждый предлагает свою разбивку)
N_VOTE = 3            # узлов-голосователей о составности
MAX_DEPTH = 2         # 0 корень -> регионы -> под-части ...
MAX_CHILDREN = 6      # потолок под-частей на регион после дедупа
SIM_DUP = 0.88        # семантический дедуп меток под-частей
THINK = False

DECOMPOSE = ("Break the REGION below into its main DISTINCT sub-parts — the pieces a team would split it "
             "into to cover it fully, no overlaps. Give 3-6 short labels. "
             'JSON {"parts":["..","..",".."]}.')
VOTE = ("Is the item below a SINGLE focused piece that one person can write directly and well, or is it "
        "still several distinct sub-parts that should be split further? "
        'JSON {"verdict":"atomic" or "composite"}.')
LEAF = ("Write THIS specific part of the task fully and concretely as a finished, self-contained piece — "
        "real content, no plan, no meta, no restating the label.")
STITCH_SEC = ("Below are the pieces of ONE section of a deliverable. Weave them into a coherent, non-redundant "
              "section: keep every concrete detail, remove overlap, order logically. Output the section only.")
STITCH_DOC = ("Below are the finished SECTIONS of one deliverable. Weave them into ONE coherent, well-ordered "
              "deliverable with smooth transitions: keep the detail, remove cross-section overlap. Output the "
              "whole, nothing else.")


def proj2d(emb):
    if not emb: return 0.0, 0.0
    x = sum(e * math.sin(0.7 * i + 1.3) for i, e in enumerate(emb))
    y = sum(e * math.cos(0.5 * i + 2.1) for i, e in enumerate(emb))
    return round(x, 4), round(y, 4)


def dedup_labels(labels):
    """Объединить близкие по смыслу метки в одну (коллективная разбивка без повторов)."""
    out = []
    for lab in labels:
        lab = str(lab).strip()
        if not lab: continue
        emb = memory.embed(lab)
        if emb and any(memory._cos(emb, e) > SIM_DUP for _, e in out): continue
        out.append((lab, emb))
    return out


def decompose(task, region):
    """N_SPLIT узлов независимо бьют регион -> ОБЪЕДИНЕНИЕ их частей -> дедуп. Много умов, не один."""
    ctx = f"TASK:\n{task}\n\nREGION TO BREAK DOWN: {region}\n\nJSON:"
    def one(_):
        return jget(llm(DECOMPOSE, ctx, temp=0.8, max_tokens=(700 if THINK else 200), think=THINK), "parts")[0] or []
    with ThreadPoolExecutor(max_workers=N_SPLIT) as ex:
        allp = [p for lst in ex.map(one, range(N_SPLIT)) for p in (lst or [])]
    return dedup_labels(allp)[:MAX_CHILDREN]


def is_composite(task, label):
    """Панель голосует: часть ещё составная? Большинство решает (не самосуд одного)."""
    ctx = f"TASK:\n{task}\n\nITEM: {label}\n\nJSON:"
    def vote(i):
        v = jget(llm(VOTE, ctx, temp=0.3 + 0.2 * i, max_tokens=(500 if THINK else 60), think=THINK), "verdict")[0]
        return str(v or "").strip().lower().startswith("comp")
    with ThreadPoolExecutor(max_workers=N_VOTE) as ex:
        votes = list(ex.map(vote, range(N_VOTE)))
    return sum(votes) > len(votes) / 2


def write_leaf(task, label):
    # ЛИСТ ДУМАЕТ: глубина каждого куска (устраняем конфаунд think=False против думающего pure)
    return llm(LEAF, f"TASK:\n{task}\n\nPART TO WRITE: {label}\n\nThe piece:", temp=0.6,
               max_tokens=1100, think=True)


_CTR = [0]
def _id(): _CTR[0] += 1; return _CTR[0] - 1


def tile_region(task, region, depth, parent, leaves, top):
    children = decompose(task, region)
    for label, emb in children:
        pid = _id(); px, py = proj2d(emb)
        this_top = label if depth == 0 else top          # верхний регион-предок (для секций сшивки)
        composite = depth < MAX_DEPTH and is_composite(task, label)
        viz.emit("patch", id=pid, parent=parent, depth=depth, part=label[:40],
                 atomic=not composite, x=px, y=py, snip="")
        indent = "  " + "    " * depth
        print(f"{indent}+ [{label[:46]}]  ({'split' if composite else 'leaf'})")
        if composite:
            tile_region(task, label, depth + 1, pid, leaves, this_top)   # рекурсия по НОВОЙ под-части
        else:
            content = write_leaf(task, label)
            leaves.append({"part": label, "content": content, "top": this_top})
            viz.emit("leaf", id=pid, part=label[:40], snip=content[:120])


def run(task):
    print("=" * 72); print(f"TASK: {task}\n\nRECURSIVE TILING (decompose + compositeness vote + leaf):\n")
    viz.begin(); viz.emit("start", problem=task, nodes=N_SPLIT)
    _CTR[0] = 0; leaves = []
    tile_region(task, "(the whole task)", 0, -1, leaves, None)
    # ИЕРАРХИЧЕСКАЯ СШИВКА: листья региона -> секция (вход ограничен), потом секции -> целое.
    # Так вход каждой сшивки мал и think=True не пустеет на больших деревьях.
    groups = {}
    for l in leaves:
        groups.setdefault(l["top"] or l["part"], []).append(l)
    sections = []
    for top, items in groups.items():
        if len(items) == 1:
            sec = items[0]["content"]
        else:
            body = "\n\n".join(f"### {it['part']}\n{it['content']}" for it in items)
            sec = llm(STITCH_SEC, f"Section: {top}\n\nPieces:\n{body}\n\nSection:", temp=0.4,
                      max_tokens=1600, think=True).strip() or body
        sections.append((top, sec))
    # ФИНАЛ: секции уже связно сшиты по-отдельности; собираем прямым склеиванием под заголовками.
    # Без гигантской финальной think=True-сшивки — она вырождается на большом входе (огрызок/пусто).
    doc = "\n\n".join(f"## {t}\n{s}" for t, s in sections)
    print(f"\n  => DELIVERABLE ({len(leaves)} leaves -> {len(sections)} sections):\n\n{doc}\n")
    viz.emit("answer", value=doc[:4000], patches=len(leaves)); viz.end()
    return doc


if __name__ == "__main__":
    P = ("Design a complete onboarding experience for a new mobile banking app: everything a thoughtful "
         "team would specify before building it.")
    run(sys.argv[1] if len(sys.argv) > 1 else P)
