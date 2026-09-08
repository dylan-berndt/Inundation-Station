import numpy as np
# from utils.config import Config
import copy
import json
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import wilcoxon


# Return periods, in years, that threshold column i corresponds to.
#   schema 2 (current):  [2, 5, 10]
#   schema 1 (legacy):   [1, 2, 5, 10], where the "1 year" column is really the
#                        1st percentile of the fitted distribution - a low-flow
#                        threshold exceeded on about half of all days. It is
#                        dropped on load so legacy and current runs line up.
RETURN_PERIODS = [2, 5, 10]
LEGACY_RETURN_PERIODS = [1, 2, 5, 10]


def readGauge(entry):
    """Normalizes one gauge record to the current schema.

    In schema 1 the two keys were written the wrong way round in test.ipynb -
    "targetDev" held the mean and "targetMean" held the deviation - so every
    KGE computed from those files was wrong. Files written after that fix carry
    "schema": 2 and need no correction."""
    legacy = int(entry.get("schema", 1)) < 2

    observedMean = entry["targetDev"] if legacy else entry["targetMean"]
    observedDev = entry["targetMean"] if legacy else entry["targetDev"]

    tp = np.array(entry["tp"]).sum(axis=0)
    fp = np.array(entry["fp"]).sum(axis=0)
    fn = np.array(entry["fn"]).sum(axis=0)

    if legacy and len(tp) == len(LEGACY_RETURN_PERIODS):
        tp, fp, fn = tp[1:], fp[1:], fn[1:]

    return tp, fp, fn, observedMean, observedDev


def calcMetrics(metricSet):
    periods = len(RETURN_PERIODS)
    calculated = {
        "recall": np.zeros([len(metricSet), periods]),
        "precision": np.zeros([len(metricSet), periods]),
        "f1": np.zeros([len(metricSet), periods]),
        "nodes": np.zeros([len(metricSet)]),
        "nrmse": np.zeros([len(metricSet)]),
        "nse": np.zeros([len(metricSet)]),
        "kge": np.zeros([len(metricSet)]),
        "names": np.empty([len(metricSet)], dtype=object),
        "totalPositives": np.zeros([len(metricSet), periods])
    }

    for i, name in enumerate(metricSet):
        calculated["nodes"][i] = metricSet[name]["nodes"]

        tp, fp, fn, observedMean, observedDev = readGauge(metricSet[name])

        totalPositives = tp + fn
        calculated["totalPositives"][i] = totalPositives

        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        calculated["nrmse"][i] = np.mean(metricSet[name]["rmse"])

        calculated["recall"][i] = recall
        calculated["precision"][i] = precision
        calculated["f1"][i] = f1

        calculated["nse"][i] = 1 - (metricSet[name]["nseNum"] / metricSet[name]["nseDenom"])

        # Gupta et al. (2009): alpha is the variability ratio, beta the bias
        # ratio. `predDev` is stored as a variance by StreamingPearson.
        alpha = math.sqrt(metricSet[name]["predDev"]) / observedDev
        beta = metricSet[name]["predMean"] / observedMean
        corr = metricSet[name]["correlation"]
        calculated["kge"][i] = 1 - math.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        calculated["names"][i] = name

    return calculated


def alignNames(source, target):
    """Return (sourceIdx, targetIdx) indexing into `source`/`target`'s ["names"]
    restricted to gauges present in both, skipping any that are missing from either."""
    targetNameToIndex = {name: idx for idx, name in enumerate(target["names"].tolist())}
    sourceNames = source["names"].tolist()

    sourceIdx = [i for i, name in enumerate(sourceNames) if name in targetNameToIndex]
    targetIdx = [targetNameToIndex[sourceNames[i]] for i in sourceIdx]

    skipped = len(sourceNames) - len(sourceIdx)
    if skipped > 0:
        print(f"Skipping {skipped} gauge(s) not present in both datasets.")

    return sourceIdx, targetIdx


def plotMetrics(metrics, names, colors):
    calculated = [calcMetrics(metricSet) for metricSet in metrics]

    basis = calculated[0]["totalPositives"]
    for i in range(1, len(calculated)):
        basisIdx, comparisonIdx = alignNames(calculated[0], calculated[i])
        currentBasis = basis[basisIdx]
        comparison = calculated[i]["totalPositives"][comparisonIdx]
        print(np.allclose(currentBasis, comparison))
        print(np.max(np.abs(currentBasis - comparison), axis=0))

        for j in range(len(RETURN_PERIODS)):
            mismatch = np.abs(currentBasis - comparison)[:, j]
            plt.hist(mismatch[mismatch > 0])
            plt.show()

    print(np.nanmean(calculated[0]["f1"], axis=0))
    print(np.nanmean(calculated[1]["f1"], axis=0))

    labels = [f"{period} Year Return Period" for period in RETURN_PERIODS]

    def plotMetric(m, name):
        plt.figure(figsize=(2.7 * len(RETURN_PERIODS), 4))
        for i in range(len(RETURN_PERIODS)):
            plt.subplot(1, len(RETURN_PERIODS), i + 1)
            plt.title(labels[i])
            for j, metricSet in enumerate(calculated):
                scores = metricSet[m][:, i].T
                scores = scores[~np.isnan(scores)]
                plot = plt.boxplot(scores, positions=[j], widths=0.5, label=names[j], patch_artist=True, showfliers=False)

                for patch in plot['boxes']:
                    patch.set_facecolor(colors[j])
            
                for line in plot['medians']:
                    line.set_color('black')

            # plt.legend()

            plt.grid()
            plt.xticks(np.arange(len(names)), names)
            plt.xlabel("Model")

            if i == 0:
                plt.ylabel(name)

            plt.tight_layout()

        plt.show()

    plotMetric("f1", "F1 Score")
    plotMetric("recall", "Recall")
    plotMetric("precision", "Precision")

    # plt.figure(figsize=(10, 6))
    # for i in range(4):
    #     plt.subplot(1, 4, i + 1)
    #     plt.title(labels[i])
    #     currentX = nodeX[:, i]
    #     currentModelY = modelY[:, i]
    #     currentFloodY = floodY[:, i]
    #     plt.scatter(currentX, currentModelY, alpha=0.5, c='tab:blue')
    #     plt.scatter(currentX, currentFloodY, alpha=0.5, c='tab:orange')
    #     plt.ylim(0, 1)

    #     modelFit = np.polyfit(currentX, currentModelY, 1)
    #     plt.plot(np.arange(np.max(currentX)), modelFit[0] * np.arange(np.max(currentX)) + modelFit[1], c='tab:blue', label="GNN Correlation")

    #     floodFit = np.polyfit(currentX, currentFloodY, 1)
    #     plt.plot(np.arange(np.max(currentX)), floodFit[0] * np.arange(np.max(currentX)) + floodFit[1], c='tab:orange', label="Flood Hub Correlation")

    #     plt.legend()

    #     plt.grid()
    #     plt.xlabel("Total Upstream Basin Nodes")
    #     plt.ylabel("F1 Score")

    # plt.show()

    plt.figure(figsize=(6, 3))
    bins = np.linspace(0, 0.02, 6)
    ax = plt.gca()
    ax.hist([calculated[i]["nrmse"] for i in range(len(calculated))], bins, label=names, color=colors)
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.legend(loc="upper right")
    plt.ylabel("Basins")
    plt.xlabel("Root Mean Squared Error")
    plt.xticks(bins)
    plt.show()

    plt.figure(figsize=(6, 3))

    for i, metricSet in enumerate(calculated):
        cdf = np.array([np.sum(metricSet["nse"] < (threshold / 1000)) / len(metricSet["nse"]) for threshold in range(-1000, 1000)])
        plt.plot(np.arange(-1000, 1000) / 1000, cdf, label=names[i], color=colors[i])

    plt.title("Cumulative Distribution of NSE")
    plt.xlabel("NSE")
    plt.ylabel("CDF")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 3))

    for i, metricSet in enumerate(calculated):
        cdf = np.array([np.sum(metricSet["kge"] < (threshold / 1000)) / len(metricSet["kge"]) for threshold in range(-1000, 1000)])
        plt.plot(np.arange(-1000, 1000) / 1000, cdf, label=names[i], color=colors[i])

    plt.title("Cumulative Distribution of KGE")
    plt.xlabel("KGE")
    plt.ylabel("CDF")
    plt.grid()
    plt.legend()
    plt.show()

    tests = ["f1", "nrmse", "nse", "kge"]
    testNames = [f"{period} Year Flood F1" for period in RETURN_PERIODS] + ["NRMSE", "NSE", "KGE"]
    floodHubIndex = names.index("Flood Hub")
    floodHubMetrics = calculated[floodHubIndex]

    for i in range(len(calculated)):
        if i == floodHubIndex:
            continue

        xIdx, yIdx = alignNames(calculated[i], floodHubMetrics)

        values = []
        samples = []
        for j, test in enumerate(tests):
            # Align X and Y to the shared set of gauges to perform a paired test with Wilcoxon
            x = calculated[i][test][xIdx]
            y = floodHubMetrics[test][yIdx]

            if test == "f1":
                for k in range(len(RETURN_PERIODS)):
                    xSample = x[:, k]
                    ySample = y[:, k]
                    mask = np.logical_and(~np.isnan(xSample), ~np.isnan(ySample))
                    pValue = wilcoxon(xSample, ySample, alternative="greater", nan_policy='omit').pvalue
                    samples.append(np.sum(mask))
                    values.append(pValue)
            else:
                mask = np.logical_and(~np.isnan(x), ~np.isnan(y))

                pValue = wilcoxon(x, y, alternative="greater" if test != "nrmse" else "less", nan_policy='omit').pvalue
                samples.append(np.sum(mask))
                values.append(pValue)
            
        lines = "\n\t".join(f"{testNames[j]} (N={samples[j]}): {values[j]}" for j in range(len(testNames)))
        print(f"{names[i]} P-Values:\n\t{lines}")


# paths = ["2026-01-24 05-33 Combo ChebBlock5", "2026-01-25 06-45 FloodHub"]
paths = ["2026-08-14 16-38 HierarchicalSAGE", "2026-01-28 23-16 Combo ChebBlock5", "2026-01-26 00-53 FloodHub"]
names = ["SAGE", "STGNN", "Flood Hub"]
colors = ["tab:green", "tab:blue", "tab:orange"]

metrics = [json.load(open(os.path.join("checkpoints", paths[i], "metrics.json"))) for i in range(len(paths))]

plotMetrics(metrics, names, colors)
