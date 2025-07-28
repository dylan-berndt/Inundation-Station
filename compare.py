import numpy as np
# from utils.config import Config
import copy
import json
import matplotlib.pyplot as plt
import os


def plotMetrics(floodHubMetrics, modelMetrics, config):
    model = {
        "recall": np.zeros([len(modelMetrics), config.future, 4]),
        "precision": np.zeros([len(modelMetrics), config.future, 4]),
        "f1": np.zeros([len(modelMetrics), config.future, 4]),
        "y": np.zeros([len(modelMetrics), 4])
    }
    flood = copy.deepcopy(model)

    nodeX = np.zeros([len(modelMetrics), 4])
    modelY = np.zeros([len(modelMetrics), 4])
    floodY = np.zeros([len(modelMetrics), 4])

    for i, name in enumerate(modelMetrics):
        nodeX[i, :] = modelMetrics[name]["nodes"]

        tp = np.array(modelMetrics[name]["tp"])
        fp = np.array(modelMetrics[name]["fp"])
        fn = np.array(modelMetrics[name]["fn"])

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

        f1 = 2 * recall * precision / (recall + precision)

        f1Mask = ~np.isnan(f1)
        f1Val = np.where(f1Mask, f1, 0)
        modelY[i, :] = np.nan_to_num(np.sum(f1Val, axis=0) / np.sum(f1Mask, axis=0), nan=0)

        model["recall"][i] = recall
        model["precision"][i] = precision
        model["f1"][i] = f1

        tp = np.array(floodHubMetrics[name]["tp"])
        fp = np.array(floodHubMetrics[name]["fp"])
        fn = np.array(floodHubMetrics[name]["fn"])

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

        f1 = 2 * recall * precision / (recall + precision)

        f1Mask = ~np.isnan(f1)
        f1Val = np.where(f1Mask, f1, 0)
        floodY[i, :] = np.nan_to_num(np.sum(f1Val, axis=0) / np.sum(f1Mask, axis=0), nan=0)

        flood["recall"][i] = recall
        flood["precision"][i] = precision
        flood["f1"][i] = f1

    plt.figure(figsize=(10, 6))

    labels = ["1 Year Return Period", "2 Year Return Period", "5 Year Return Period", "10 Year Return Period"]

    def plotMetric(m, name):
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.title(labels[i])
            modelF1 = model[m][:, :, i].T
            modelF1 = [box[~np.isnan(box)] for box in modelF1]
            modelPlot = plt.boxplot(modelF1, positions=np.arange(7) + 0.25, widths=0.5, label="GNN Model", patch_artist=True)
            floodF1 = flood[m][:, :, i].T
            floodF1 = [box[~np.isnan(box)] for box in floodF1]
            floodPlot = plt.boxplot(floodF1, positions=np.arange(7) - 0.25, widths=0.5, label="Flood Hub", patch_artist=True)
            plt.ylim(0, 1)

            mod = model[m][:, :, i]
            flh = flood[m][:, :, i]
            modMask = ~np.isnan(mod)
            modVal = np.where(modMask, mod, 0)
            flhMask = ~np.isnan(flh)
            flhVal = np.where(flhMask, flh, 0)
            plt.plot(np.arange(7), np.sum(modVal, axis=0) / np.sum(modMask, axis=0), c='tab:blue')
            plt.plot(np.arange(7), np.sum(flhVal, axis=0) / np.sum(flhMask, axis=0), c='tab:orange')

            for patch in modelPlot['boxes']:
                patch.set_facecolor('tab:blue')

            for line in modelPlot['medians']:
                line.set_color('black')

            for patch in floodPlot['boxes']:
                patch.set_facecolor('tab:orange')

            for line in floodPlot['medians']:
                line.set_color('black')

            plt.legend()

            plt.grid()
            plt.xticks(np.arange(7), np.arange(7) + 1)
            plt.xlabel("Forecast Horizon")
            plt.ylabel(name)

        plt.show()

    plotMetric("f1", "F1 Score")
    plotMetric("recall", "Recall")
    plotMetric("precision", "Precision")

    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(labels[i])
        currentX = nodeX[:, i]
        currentModelY = modelY[:, i]
        currentFloodY = floodY[:, i]
        plt.scatter(currentX, currentModelY, alpha=0.5, c='tab:blue')
        plt.scatter(currentX, currentFloodY, alpha=0.5, c='tab:orange')
        plt.ylim(0, 1)

        modelFit = np.polyfit(currentX, currentModelY, 1)
        plt.plot(np.arange(np.max(currentX)), modelFit[0] * np.arange(np.max(currentX)) + modelFit[1], c='tab:blue', label="GNN Correlation")

        floodFit = np.polyfit(currentX, currentFloodY, 1)
        plt.plot(np.arange(np.max(currentX)), floodFit[0] * np.arange(np.max(currentX)) + floodFit[1], c='tab:orange', label="Flood Hub Correlation")

        plt.legend()

        plt.grid()
        plt.xlabel("Total Upstream Basin Nodes")
        plt.ylabel("F1 Score")

    plt.show()

    modelNSE = 1 - np.array([modelMetrics[name]["nseNum"] / modelMetrics[name]["nseDenom"] for name in modelMetrics])
    modelCDF = np.array([np.sum(modelNSE < (threshold / 1000)) / len(modelNSE) for threshold in range(-1000, 1000)])

    floodHubNSE = 1 - np.array([floodHubMetrics[name]["nseNum"] / floodHubMetrics[name]["nseDenom"] for name in floodHubMetrics])
    floodHubCDF = np.array([np.sum(floodHubNSE < (threshold / 1000)) / len(floodHubNSE) for threshold in range(-1000, 1000)])

    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(-1000, 1000) / 1000, modelCDF, label="GNN Model")
    plt.plot(np.arange(-1000, 1000) / 1000, floodHubCDF, label="Flood Hub")
    plt.title("Cumulative Distribution of NSE")
    plt.xlabel("NSE")
    plt.ylabel("CDF")
    plt.grid()
    plt.legend()
    plt.show()


floodHubPath = os.path.join("checkpoints", "2025-07-09")
modelPath = os.path.join("checkpoints", "2025-07-16")
# config = Config().load(os.path.join(modelPath, "config.json"))

with open(os.path.join(floodHubPath, "metrics.json")) as file:
    floodHubMetrics = json.load(file)

with open(os.path.join(modelPath, "metrics.json")) as file:
    modelMetrics = json.load(file)


class Ugh:
    pass


config = Ugh()
config.future = 7


plotMetrics(floodHubMetrics, modelMetrics, config)
