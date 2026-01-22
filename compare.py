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
        "x": np.zeros([len(metricSet), 4]),
        "nmae": np.zeros([len(metricSet), 4]),
        "nse": np.zeros([len(metricSet)]),
        "kge": np.zeros([len(metricSet)]),
        "names": np.empty([len(metricSet)], dtype=object)
    }

    for i, name in enumerate(metricSet):
        calculated["x"][i, :] = metricSet[name]["nodes"]

        tp = np.array(metricSet[name]["tp"]).sum(axis=0)
        fp = np.array(metricSet[name]["fp"]).sum(axis=0)
        fn = np.array(metricSet[name]["fn"]).sum(axis=0)

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

        f1 = 2 * recall * precision / (recall + precision)

        calculated["nmae"][i, :] = np.nan_to_num(f1, nan=0)

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
                plot = plt.boxplot(scores, positions=j, widths=0.5, label=names[j], patch_artist=True, showfliers=False)

                for patch in plot['boxes']:
                    patch.set_facecolor(colors[j])
            
                for line in plot['medians']:
                    line.set_color('black')

            plt.legend()

            plt.grid()
            plt.xticks(np.arange(len(names)), names)
            plt.xlabel("Forecast Horizon")
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
        plt.plot(np.arange(-1000, 1000) / 1000, cdf, label=names[i])

    plt.title("Cumulative Distribution of NSE")
    plt.xlabel("NSE")
    plt.ylabel("CDF")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 3))

    for i, metricSet in enumerate(calculated):
        cdf = np.array([np.sum(metricSet["kge"] < (threshold / 1000)) / len(metricSet["kge"]) for threshold in range(-1000, 1000)])
        plt.plot(np.arange(-1000, 1000) / 1000, cdf, label=names[i])

    plt.title("Cumulative Distribution of KGE")
    plt.xlabel("KGE")
    plt.ylabel("CDF")
    plt.grid()
    plt.legend()
    plt.show()

    tests = ["f1", "recall", "precision", "nmae", "nse", "kge"]
    testNames = ["F1", "Recall", "Precision", "NMAE", "NSE", "KGE"]
    floodHubIndex = names.index("Flood Hub")
    floodHubMetrics = calculated[floodHubIndex]

    for i in range(len(calculated)):
        if i == floodHubIndex:
            continue

        values = []
        for j, test in enumerate(tests):
            x = calculated[i][test]
            y = floodHubMetrics[test]

            # Sort Y to perform paired test with Wilcoxon
            y = [y[floodHubMetrics["names"].index(name)] for name in calculated[i]["names"]]

            pValue = wilcoxon(x, y, alternative="greater").pvalue
            values.append(pValue)

        print(f"{names[i]} P-Values:", f"{" | ".join([f'{testNames[j]}: {values[j]}' for j in range(testNames)])}")


paths = ["", ""]
names = ["", "Flood Hub"]
colors = ["tab:blue", "tab:orange"]

metrics = {names[i]: json.load(open(os.path.join("checkpoints", paths[i], "metrics.json"))) for i in range(len(paths))}

plotMetrics(metrics, names, colors)
