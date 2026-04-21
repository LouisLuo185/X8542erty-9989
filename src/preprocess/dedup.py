"""Lightweight text dedup helpers: SimHash-style 64-bit fingerprint and exact-hash clusters."""

from __future__ import annotations

import hashlib

import pandas as pd


def _gram_vector(gram: str) -> list[int]:
    digest = hashlib.md5(gram.encode("utf-8")).digest()
    n = int.from_bytes(digest[:8], "big")
    return [1 if (n >> i) & 1 else -1 for i in range(64)]


def compute_simhash_u64(text: str) -> int:
    text = (text or "").strip()
    if len(text) < 2:
        return 0
    accum = [0] * 64
    for i in range(len(text) - 1):
        vec = _gram_vector(text[i : i + 2])
        for k in range(64):
            accum[k] += vec[k]
    out = 0
    for k in range(64):
        if accum[k] >= 0:
            out |= 1 << k
    return out


def simhash_hamming(a: int, b: int) -> int:
    x = a ^ b
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


def enrich_dedup_columns(df: pd.DataFrame, text_col: str = "clean_text", near_dup: bool = True) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["simhash_u64"] = pd.Series(dtype="str")
        out["dedup_cluster_id"] = pd.Series(dtype="str")
        out["is_dedup_representative"] = pd.Series(dtype="bool")
        return out

    out = df.copy()
    hashes: list[int] = []
    for text in out[text_col].astype(str):
        hashes.append(compute_simhash_u64(text))
    out["simhash_u64"] = [format(h, "016x") for h in hashes]

    use_near_dup = near_dup and len(out) <= 3000

    if not use_near_dup:
        out["dedup_cluster_id"] = out["simhash_u64"]
        first_seen: dict[str, int] = {}
        for idx, cid in enumerate(out["dedup_cluster_id"]):
            if cid not in first_seen:
                first_seen[cid] = idx
        out["is_dedup_representative"] = [first_seen[cid] == i for i, cid in enumerate(out["dedup_cluster_id"])]
        return out

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    indices = list(range(len(out)))
    for i in indices:
        for j in range(i + 1, len(out)):
            if simhash_hamming(hashes[i], hashes[j]) <= 5:
                union(i, j)

    roots = {find(i) for i in indices}
    root_to_cluster = {root: f"dup_{root}" for root in sorted(roots)}
    cluster_ids = [root_to_cluster[find(i)] for i in indices]
    out["dedup_cluster_id"] = cluster_ids

    first_seen = {}
    for idx, cid in enumerate(out["dedup_cluster_id"]):
        if cid not in first_seen:
            first_seen[cid] = idx
    out["is_dedup_representative"] = [first_seen[cid] == i for i, cid in enumerate(out["dedup_cluster_id"])]
    return out
