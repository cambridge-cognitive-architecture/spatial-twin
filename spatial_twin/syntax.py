"""Space syntax on a zone graph.

Standard definitions (Hillier & Hanson 1984) applied to zones as nodes:

    connectivity     degree
    mean depth       mean shortest-path length to every other reachable zone
    RA               2 (MD - 1) / (n - 2)
    RRA              RA / D_n, normalised by the diamond value so systems of
                     different size compare; lower is more integrated
    integration      1 / RRA (HH); higher is more integrated. Undefined, NaN,
                     where RA = 0: a zone one step from every other has no
                     finite value, as in depthmapX. RA and RRA stay finite and
                     keep the ordering, so they are reported beside it.
    integration_local  the same within radius 3
    choice           betweenness (Brandes, unweighted)
    control          sum over neighbours of 1 / their degree
    intelligibility  r(connectivity, integration) over zones: a property of the
                     system, not of a zone

This is the zone graph, not a visibility graph of grid points.
"""
from __future__ import annotations

from collections import deque

import numpy as np

LOCAL_RADIUS = 3


def diamond(n: int) -> float:
    """D_n, the RA of a diamond-shaped system of n nodes."""
    if n < 3:
        return np.nan
    return 2.0 * (n * (np.log2((n + 2) / 3.0) - 1.0) + 1.0) / ((n - 1) * (n - 2))


def space_syntax(adj: dict[str, list[str]], local_radius: int = LOCAL_RADIUS) -> dict:
    nodes = sorted(adj)
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    if n < 3:
        out = {v: {"connectivity": float(len(adj[v])), "mean_depth": np.nan, "integration": np.nan,
                   "integration_local": np.nan, "choice": 0.0, "control": np.nan} for v in nodes}
        return {"zones": out, "intelligibility": np.nan, "n_zones": n}

    depth = np.full((n, n), np.inf)
    for s in nodes:
        si = idx[s]
        depth[si, si] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if not np.isfinite(depth[si, idx[w]]):
                    depth[si, idx[w]] = depth[si, idx[u]] + 1
                    q.append(w)

    local = {}
    for v in nodes:
        row = depth[idx[v]]
        near = row[np.isfinite(row) & (row <= local_radius)]
        k = len(near)                     # includes v, at depth 0
        if k < 3:
            local[v] = np.nan
            continue
        md = float(near.sum() / (k - 1))
        ra = 2.0 * (md - 1.0) / max(k - 2, 1)
        d = diamond(k)
        local[v] = float(1.0 / (ra / d)) if ra > 0 and d > 0 else np.nan     # undefined where RA = 0

    bet = {v: 0.0 for v in nodes}
    for s in nodes:
        stack, pred = [], {v: [] for v in nodes}
        sigma = {v: 0.0 for v in nodes}; sigma[s] = 1.0
        dist = {v: -1 for v in nodes}; dist[s] = 0
        q = deque([s])
        while q:
            v = q.popleft(); stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w]:
                    delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                bet[w] += delta[w]
    for v in bet:
        bet[v] /= 2.0                      # undirected: each pair counted twice

    dn = diamond(n)
    out = {}
    for v in nodes:
        reach = depth[idx[v]][np.isfinite(depth[idx[v]])]
        md = float(reach.sum() / max(len(reach) - 1, 1))
        ra = 2.0 * (md - 1.0) / (n - 2)
        rra = ra / dn if dn else np.nan
        out[v] = {
            "connectivity": float(len(adj[v])),
            "mean_depth": md,
            "ra": ra,
            "rra": float(rra),
            # RA == 0: one step from every other zone. 1/RRA has no finite value there, so it is
            # undefined rather than infinite; RRA = 0 still ranks the zone as the most integrated.
            "integration": float(1.0 / rra) if rra > 0 else np.nan,
            "integration_undefined": bool(rra == 0),
            "integration_local": local[v],
            "choice": float(bet[v]),
            "control": float(sum(1.0 / max(len(adj[w]), 1) for w in adj[v])),
            "n_reachable": int(len(reach) - 1),
        }
    conn = np.array([out[v]["connectivity"] for v in nodes])
    integ = np.array([out[v]["integration"] for v in nodes], float)
    ok = np.isfinite(integ)
    intel = float(np.corrcoef(conn[ok], integ[ok])[0, 1]) if ok.sum() > 2 and conn[ok].std() > 0 else np.nan
    return {"zones": out, "intelligibility": intel, "n_zones": n}
