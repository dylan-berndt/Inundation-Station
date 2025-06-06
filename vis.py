import socket
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math
from scipy.interpolate import UnivariateSpline

MAX_DATA_POINTS = 5000
DIVISOR = 2

MAX_DISPLAY_POINTS = 100

DISPLAY_REFRESH = 0.1


class Client:
    def __init__(self, host, port=12001):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))

    def send(self, name, data):
        collated = f"  {name}||{data:.3f}"
        self.socket.send(bytes(collated, encoding='utf-8'))


class Server:
    def __init__(self, port=12945):
        self.data = {}
        self.time = {}

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('', port))

        self.socket.setblocking(False)

        self.socket.listen(1)
        self.client = None

        self.accept()

    def accept(self):
        found = False
        while not found:
            try:
                self.client, _ = self.socket.accept()
                found = True
            except BlockingIOError:
                pass

    def receive(self):
        stringData = self.client.recv(512).decode(encoding='utf-8')

        dataPoints = stringData.split("  ")
        for data in dataPoints:
            if not data:
                continue

            name, data = data.split("||")
            data = float(data)

            if name in self.data:
                self.data[name].append(data)
                self.time[name].append(datetime.now().timestamp())
            else:
                self.data[name] = [data]
                self.time[name] = [datetime.now().timestamp()]

            if len(self.data[name]) >= MAX_DATA_POINTS:
                datum = np.array(self.data[name]).reshape([-1, 2])
                datum = np.mean(datum, axis=-1).squeeze()
                point = np.array(self.time[name]).reshape([-1, 2])
                point = np.mean(point, axis=-1).squeeze()
                self.data[name] = list(datum)
                self.time[name] = list(point)


def plotData(data):
    plt.clf()

    plots = len(data.keys())
    if plots == 0:
        return

    width = math.ceil(plots / 3 * 2)
    height = int(plots / width) * 4

    for n, name in enumerate(data.keys()):
        point = np.array(server.time[name])
        datum = np.array(server.data[name])

        if len(datum) > MAX_DISPLAY_POINTS:
            plt.subplot(height, width, n + plots + 1)
            points = point[len(datum) - (MAX_DISPLAY_POINTS + 1):]
            display = datum[len(datum) - (MAX_DISPLAY_POINTS + 1):]
            plt.plot(points, display, label=name)

            plt.axhline(np.mean(display), linestyle='--', color='orange')

            plt.legend()

            # divisor = len(datum) // MAX_DISPLAY_POINTS
            # datum = datum[::divisor]

        plt.subplot(height, width, n+1)
        plt.plot(point, datum, label=name)

        plt.legend()

    plt.pause(0.01)


if __name__ == "__main__":
    server = Server()

    last = datetime.now()

    plt.ion()
    fig = plt.figure()
    # fig.canvas.manager.window.attributes('-topmost', 0)

    while True:
        try:
            server.receive()
        except ValueError:
            pass
        except BlockingIOError:
            pass

        if (datetime.now() - last).total_seconds() > DISPLAY_REFRESH:
            last = datetime.now()
            plotData(server.data)

