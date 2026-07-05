#!/usr/bin/env python3
"""Лёгкий эмиттер событий мышления для визуализации (Rust/macroquad читает viz_events.jsonl).
Поведение мозга не меняет — только пишет, какая область сработала и что попало на стол."""
import json, os

PATH = os.path.join(os.path.dirname(__file__), "viz_events.jsonl")
_f = None


def begin():
    global _f
    _f = open(PATH, "w")


def emit(k, **data):
    if _f is None: return
    data["k"] = k
    _f.write(json.dumps(data, ensure_ascii=False) + "\n"); _f.flush()


def end():
    global _f
    if _f: _f.close(); _f = None
