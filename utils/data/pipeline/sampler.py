"""Batches samples by total upstream-node count (`config.nodesPerBatch`)
rather than a fixed batch size, since basin graphs vary widely in size.

Faithful port of `utils/data/dataset.py`'s `GraphSizeSampler`, with one
deliberate UX change: the original unconditionally popped up 3 matplotlib
histograms *every time a sampler was constructed* (so twice per `.split()`
call, with no way to opt out) - fine in a notebook, but it blocks/clutters
when training runs as a plain script. Plotting is now opt-in via
`display=False` (default off); pass `display=True` to get the old behavior
back. This has zero effect on batch contents or model input - `self.batches`
is built identically either way.
"""

import random

import numpy as np
from scipy.stats import mode
from torch.utils.data import Sampler


class GraphSizeSampler(Sampler):
    def __init__(self, dataset, nodesPerBatch=500, dropLast=False, force=False, shuffle=True, startPoint=0, display=False):
        self.dataset = dataset
        self.nodesPerBatch = nodesPerBatch
        self.dropLast = dropLast
        self.shuffle = shuffle

        self.batches = []
        self.start = startPoint

        if hasattr(dataset, "graphSizes"):
            indices = range(len(dataset))
            sizes = dataset.graphSizes
        else:
            under = dataset.dataset
            subsetIndices = dataset.indices
            indices = range(len(subsetIndices))
            sizes = [under.graphSizes[subsetIndices[i]] for i in indices]

        if self.shuffle:
            combined = list(zip(indices, sizes))
            random.shuffle(combined)
            indices, sizes = zip(*combined)

        batch = []
        batchSizes = []
        batchSum = 0
        for i in range(len(indices)):
            if batchSum + sizes[i] > nodesPerBatch and len(batch) != 0:
                self.batches.append(batch)
                batchSizes.append(batchSum)
                batch = []
                batchSum = 0

            batch.append(indices[i])
            batchSum += sizes[i]

        self.batches.append(batch)
        batchSizes.append(batchSum)

        # For diagnosing memory leaks
        if force:
            self.batches = [self.batches[i] for i in range(len(self.batches)) if batchSizes[i] == nodesPerBatch]
            batchSizes = [size for size in batchSizes if size == nodesPerBatch]
            batchSize = mode(np.array([len(batch) for batch in self.batches])).mode
            batchSizes = [batchSizes[i] for i in range(len(batchSizes)) if len(self.batches[i]) == batchSize]
            self.batches = [batch for batch in self.batches if len(batch) == batchSize]

        if display:
            self._plotDistributions(sizes, batchSizes)

    def _plotDistributions(self, sizes, batchSizes):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.title("Node Count Distribution per Sample")
        plt.hist(sizes)
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.title("Node Count Distribution per Batch")
        plt.hist(batchSizes)
        plt.grid()

        plt.subplot(1, 3, 3)
        plt.title("Data Samples Distribution per Batch")
        plt.hist([len(batch) for batch in self.batches])
        plt.grid()
        plt.show()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        batchList = self.batches[self.start:] + self.batches[:self.start]
        for batch in batchList:
            yield batch

    def __len__(self):
        return len(self.batches)
