import numpy as np
# from utils.config import Config
import copy
import json
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import wilcoxon


def calcMetrics(metricSet):
    calculated = {
        "recall": np.zeros([len(metricSet), 4]),
        "precision": np.zeros([len(metricSet), 4]),
        "f1": np.zeros([len(metricSet), 4]),
        "nodes": np.zeros([len(metricSet)]),
        "nmae": np.zeros([len(metricSet)]),
        "nse": np.zeros([len(metricSet)]),
        "kge": np.zeros([len(metricSet)]),
        "names": np.empty([len(metricSet)], dtype=object)
    }

    for i, name in enumerate(metricSet):
        calculated["nodes"][i] = metricSet[name]["nodes"]

        tp = np.array(metricSet[name]["tp"]).sum(axis=0)
        fp = np.array(metricSet[name]["fp"]).sum(axis=0)
        fn = np.array(metricSet[name]["fn"]).sum(axis=0)

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

        f1 = 2 * recall * precision / (recall + precision + 1e-6)

        calculated["nmae"][i] = np.mean(metricSet[name]["mae"])

        calculated["recall"][i] = recall
        calculated["precision"][i] = precision
        calculated["f1"][i] = f1

        calculated["nse"][i] = 1 - (metricSet[name]["nseNum"] / metricSet[name]["nseDenom"])

        alpha = metricSet[name]["predMean"] / metricSet[name]["targetMean"]
        # I have mixed variance and deviation in my previous script. I am sorry.
        beta = math.sqrt(metricSet[name]["predDev"]) / metricSet[name]["targetDev"]
        corr = metricSet[name]["correlation"]
        kge = 1 - math.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        calculated["kge"][i] = kge

        calculated["names"][i] = name

    return calculated


def plotMetrics(metrics, names, colors):
    calculated = [calcMetrics(metricSet) for metricSet in metrics]

    plt.figure(figsize=(10, 6))

    labels = ["1 Year Return Period", "2 Year Return Period", "5 Year Return Period", "10 Year Return Period"]

    def plotMetric(m, name):
        for i in range(4):
            plt.subplot(1, 4, i + 1)
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
            plt.ylabel(name)

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

    tests = ["f1", "nmae", "nse", "kge"]
    testNames = ["1 Year Flood F1", "2 Year Flood F1", "5 Year Flood F1", "10 Year Flood F1", "NMAE", "NSE", "KGE"]
    floodHubIndex = names.index("Flood Hub")
    floodHubMetrics = calculated[floodHubIndex]

    for i in range(len(calculated)):
        if i == floodHubIndex:
            continue

        values = []
        samples = []
        for j, test in enumerate(tests):
            x = calculated[i][test]
            y = floodHubMetrics[test]

            # Sort Y to perform paired test with Wilcoxon
            y = np.array([y[floodHubMetrics["names"].tolist().index(name)] for name in calculated[i]["names"]])

            if test == "f1":
                for k in range(4):
                    xSample = x[:, k]
                    ySample = y[:, k]
                    mask = np.logical_and(~np.isnan(xSample), ~np.isnan(ySample))
                    pValue = wilcoxon(xSample[mask], ySample[mask], alternative="greater").pvalue
                    samples.append(np.sum(mask))
                    values.append(pValue)
            else:
                mask = np.logical_and(~np.isnan(x), ~np.isnan(y))

                pValue = wilcoxon(x[mask], y[mask], alternative="greater" if test != "nmae" else "less").pvalue
                samples.append(np.sum(mask))
                values.append(pValue)
            
        print(f"{names[i]} P-Values:\n\t{"\n\t".join([f'{testNames[j]} (N={samples[j]}): {values[j]:.3f}' for j in range(len(testNames))])}")


paths = ["2026-01-06 20-50 ChebBlock5", "2026-01-11 15-24 GCNBlock", "2026-01-04 19-02 floodHub"]
names = ["Combo GCNBlock5", "GCNBlock5", "Flood Hub"]
colors = ["tab:blue", "tab:cyan", "tab:orange"]

metrics = [json.load(open(os.path.join("checkpoints", paths[i], "metrics.json"))) for i in range(len(paths))]

plotMetrics(metrics, names, colors)
