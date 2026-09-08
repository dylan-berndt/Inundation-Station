"""Generic on-disk caching primitives used by every pipeline stage.

Cache validity is decided by a *fingerprint* the caller supplies - typically a
hash of the config fields a stage depends on, plus the size/mtime of the raw
files it reads - never by the age of the cache file. That makes caches
self-invalidating: change `rolling`/`scales`/`downsampling`, or touch a raw
ERA5/GRDC file, and the next load recomputes instead of silently serving
stale data. Bump CACHE_VERSION to invalidate every cache at once (e.g. after
changing what a stage stores).
"""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import torch

CACHE_VERSION = 1


def _stableRepr(value):
    if isinstance(value, dict):
        return {str(k): _stableRepr(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, (list, tuple, set)):
        return [_stableRepr(v) for v in value]
    return value


def fingerprint(*parts):
    payload = json.dumps(_stableRepr(list(parts)), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def fingerprintFiles(paths):
    entries = []
    for path in sorted(paths):
        if os.path.exists(path):
            stat = os.stat(path)
            entries.append((os.path.basename(path), stat.st_size, int(stat.st_mtime)))
        else:
            entries.append((os.path.basename(path), None, None))
    return fingerprint(CACHE_VERSION, entries)


def fingerprintShapefile(shpPath):
    stem = shpPath[:-len(".shp")] if shpPath.endswith(".shp") else shpPath
    return fingerprintFiles([stem + ext for ext in (".shp", ".dbf", ".shx")])


def fingerprintDirectory(directory, pattern="*"):
    entries = []
    for path in sorted(glob(os.path.join(directory, pattern))):
        stat = os.stat(path)
        entries.append((os.path.basename(path), stat.st_size, int(stat.st_mtime)))
    return fingerprint(CACHE_VERSION, len(entries), entries)


class StageCache:
    """One cache file per (name, fingerprint) pair. For a handful of large,
    monolithic artifacts (a graph, an index map, a gauge-series dict)."""

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, name, key, ext="pt"):
        return os.path.join(self.root, f"{name}_{key}.{ext}")

    def get(self, name, key, ext="pt"):
        path = self.path(name, key, ext)
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        return None

    def set(self, name, key, value, ext="pt"):
        path = self.path(name, key, ext)
        tmpPath = path + f".tmp{os.getpid()}"
        torch.save(value, tmpPath)
        os.replace(tmpPath, path)
        return value

    def loadOrCompute(self, name, key, compute, ext="pt", force=False):
        if not force:
            cached = self.get(name, key, ext)
            if cached is not None:
                return cached
        return self.set(name, key, compute(), ext)


class ShardedCache:
    """Directory-of-files cache for large collections of small items (one
    tensor per basin, for example). Lets a stage stream results to disk as
    they're computed instead of holding a second full copy in memory, and
    lets later loads pull items back in parallel."""

    def __init__(self, root, key):
        self.directory = os.path.join(root, key)

    def exists(self):
        return os.path.isdir(self.directory) and len(os.listdir(self.directory)) > 0

    def itemPath(self, itemID):
        return os.path.join(self.directory, f"{itemID}.pt")

    def save(self, itemID, value):
        os.makedirs(self.directory, exist_ok=True)
        path = self.itemPath(itemID)
        tmpPath = path + f".tmp{os.getpid()}"
        torch.save(value, tmpPath)
        os.replace(tmpPath, path)

    def load(self, itemID):
        path = self.itemPath(itemID)
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        return None

    def loadAll(self, itemIDs, maxWorkers=None, onProgress=None):
        paths = [self.itemPath(itemID) for itemID in itemIDs]

        def loadOne(pair):
            index, path = pair
            value = torch.load(path, weights_only=False)
            if onProgress is not None:
                onProgress(index)
            return value

        with ThreadPoolExecutor(max_workers=maxWorkers or os.cpu_count() * 4) as executor:
            values = list(executor.map(loadOne, enumerate(paths)))
        return dict(zip(itemIDs, values))
