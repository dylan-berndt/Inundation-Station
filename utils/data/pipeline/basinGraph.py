"""Basin connectivity graph (from BasinATLAS `NEXT_DOWN`/`HYBAS_ID`) and, for
a given set of gauge basins, their upstream subgraph/edge-list/hop-distances.

Two behavior-preserving performance fixes versus the original
`InundationData.__init__` (both produce an identical graph/structures, just
faster to compute - see the docstrings below for why each is safe):

1. `buildBasinGraph` replaces a per-row `basinATLAS[basinATLAS["HYBAS_ID"] ==
   upstream["NEXT_DOWN"]]` full-frame scan (O(n) work repeated for each of
   the ~n rows, so O(n^2) overall) with a single `groupby("HYBAS_ID")` index
   built once. Same edges get added either way - a DiGraph doesn't care what
   order `add_edge` was called in.
2. `computeUpstreamStructures` only runs `nx.ancestors` / shortest-path for
   the basins that are actually a gauge's own basin. The original computed
   this for every basin with ERA5 data (~9600), even though `__getitem__`
   only ever looks these dicts up via `translateDict[grdcID]` - i.e. gauge
   basins only (~2500). The unused entries were simply never read.
"""

import os

import networkx as nx
import pandas as pd

from .caching import StageCache, fingerprint, fingerprintShapefile


def buildBasinGraph(basinATLAS, verbose=True):
    pfafIDStr = basinATLAS["PFAF_ID"].astype(str)
    hybasGroups = basinATLAS.groupby("HYBAS_ID").groups

    graph = nx.DiGraph()
    total = len(basinATLAS)

    for i, index in enumerate(basinATLAS.index):
        upstreamPfaf = pfafIDStr.at[index]
        graph.add_edge(upstreamPfaf, upstreamPfaf)

        nextDown = basinATLAS.at[index, "NEXT_DOWN"]
        endo = basinATLAS.at[index, "ENDO"]
        if pd.isna(nextDown) or nextDown == 0 or endo == 2:
            continue

        for downstreamIndex in hybasGroups.get(nextDown, ()):
            downstreamPfaf = pfafIDStr.at[downstreamIndex]
            if upstreamPfaf == downstreamPfaf:
                continue
            graph.add_edge(upstreamPfaf, downstreamPfaf)

        if verbose:
            print(f"\r{i}/{total} Basin Structures Appended to Graph", end="")

    if verbose:
        print()

    return graph


def computeUpstreamStructures(graph, targetPfafIDs, verbose=True):
    """Returns (upstreamBasins, upstreamStructure, graphs, hopDistances),
    each a dict keyed by pfafID, matching the fields `InundationData` used to
    store as `self.upstreamBasins` / `self.upstreamStructure` / `self.graphs`
    / `self.hopDistances`."""
    upstreamBasins = {}
    upstreamStructure = {}
    graphs = {}
    hopDistances = {}

    targetPfafIDs = list(targetPfafIDs)
    for i, node in enumerate(targetPfafIDs):
        nodes = [node] + sorted(nx.ancestors(graph, node))
        upstreamBasins[node] = nodes

        # A real copy, not `graph.subgraph(nodes)`. That returns a view backed
        # by local closures (`subgraph_view.<locals>.reverse_edge`), which is
        # unpicklable - it would break both the StageCache torch.save below and
        # the Windows DataLoader, which pickles the Dataset to spawn workers.
        # Only gauge basins are stored here, so the copies are cheap.
        subgraph = nx.DiGraph(graph.subgraph(nodes))
        graphs[node] = subgraph

        nodeMap = dict(zip(nodes, range(len(nodes))))
        upstreamStructure[node] = [[nodeMap[edge[0]], nodeMap[edge[1]]] for edge in subgraph.edges]

        hopDistances[node] = [nx.shortest_path_length(subgraph, source=source, target=node) for source in nodes]

        if verbose:
            print(f"\r{i + 1}/{len(targetPfafIDs)} Upstream Structures Compiled", end="")

    if verbose:
        print()

    return upstreamBasins, upstreamStructure, graphs, hopDistances


def loadBasinGraph(basinATLAS, basinLevelSHPPath, cacheRoot, force=False, verbose=True):
    cache = StageCache(cacheRoot)
    key = fingerprintShapefile(basinLevelSHPPath)
    return cache.loadOrCompute(
        "basinGraph", key, lambda: buildBasinGraph(basinATLAS, verbose=verbose), force=force
    )


def loadUpstreamStructures(graph, basinLevelSHPPath, targetPfafIDs, cacheRoot, force=False, verbose=True):
    targetPfafIDs = list(targetPfafIDs)
    cache = StageCache(cacheRoot)
    key = fingerprint(fingerprintShapefile(basinLevelSHPPath), sorted(targetPfafIDs))
    return cache.loadOrCompute(
        "upstreamStructures", key,
        lambda: computeUpstreamStructures(graph, targetPfafIDs, verbose=verbose),
        force=force,
    )
