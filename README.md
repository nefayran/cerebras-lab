# llm-brain

A research sandbox for one question: if you wire many copies of the same language
model into a network with no assigned roles, does anything mind-like emerge from the
dynamics — or do you only ever read your own reflection back out?

Everything runs locally against [Ollama](https://ollama.com) (`qwen3:8b` for
generation, `nomic-embed-text` for embeddings). No API keys, no cloud.

## The experiments, in order

### 1. `brain.py` — a thought stream from LLM nodes

N nodes, all identical and neutral, arranged on a ring. Each hears scraps from its
neighbours and continues one short "inner-speech" thought. No node is a person, none
is an assistant, none has a role — so any differentiation or structure would have to
be emergent.

What actually happened: **mode collapse.** Left to echo each other, the nodes
converge — the stream narrows to a few repeating attractor phrases instead of
diversifying. Interesting as a negative result: identical LLM nodes on a plain
message bus don't self-organise into structure, they synchronise into a rut.

### 2. `brain2.py` — a field of oscillators, LLM only at the edge

Rethink the substrate. A node is no longer an LLM — it's a **concept-neuron**: a
number (phase + activation + fatigue). Meaning doesn't live in a node; it lives in
which nodes fire **in phase** (binding). Edges between concepts are the cosine
similarity of their embeddings — near concepts excite and synchronise, far ones
inhibit. A global tone reads out as feeling (arousal: how easily thoughts clump;
valence: whether coherence is rising).

The LLM enters **only as a readout**: every few ticks it looks at the dominant
in-phase group and turns it into one thought plus a named feeling. Language never
touches the dynamics themselves.

### The control that mattered — `--scramble`

The obvious trap: a fluent LLM reader will narrate *anything* as a coherent thought,
so a nice-looking transcript proves nothing. So before the readout, `--scramble`
permutes the phases and destroys the binding. If the reader still emits a coherent
thought from noise, the thought was in the **reader**, not the network.

It did. That's the honest finding: much of the apparent "inner life" was **pareidolia
in the readout**. What is real and worth keeping is the *dynamics* — the
synchronisation, the clumping, the tone — not the words the LLM drapes over them.

## `being/` — the expansion

`being/` takes the substrate further into a larger architecture: a connectome,
concept "organs", memory, a self-model, persona, and long-running live simulations
with a visualiser (`mindviz`). It is the messier, more ambitious end of the sandbox.
Heavy artifacts (model weights, adapters, generated data, run logs) are gitignored;
the source and a few small sample outputs are here.

## Run it

```bash
# prerequisites
ollama pull qwen3:8b
ollama pull nomic-embed-text

# experiment 1 — LLM-node thought stream
python3 brain.py --nodes 8 --ticks 60 --topology ring

# experiment 2 — oscillator field with LLM readout
python3 brain2.py
python3 brain2.py --scramble   # the control: does coherence survive killed binding?
```

## Repo map

| Path | What it is |
|---|---|
| `brain.py` | Experiment 1 — thought stream from identical LLM nodes |
| `brain2.py` | Experiment 2 — oscillator field, LLM as readout, `--scramble` control |
| `mind.py`, `world.py` | Early scaffolding around the stream |
| `being/` | The expanded "being" — connectome, organs, memory, self-model, live sim |

## Status

A sandbox, not a product — negative results and dead ends are part of the record.
The takeaway so far: emergence lived in the **dynamics**, and the words were mostly
the reader's. MIT licensed; read it as a lab notebook.
