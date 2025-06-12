import socket
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from datetime import datetime
import math
import threading
import time

MAX_DATA_POINTS = 5000
DIVISOR = 2

MAX_DISPLAY_POINTS = 100

DISPLAY_REFRESH = 0.1


class Client:
    def __init__(self, host, port=12001):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))

    def send(self, name, data):
        collated = f" | {name}||{data:.3f}"
        self.socket.sendall(bytes(collated, encoding='utf-8'))


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
        stringData = self.client.recv(4096).decode(encoding='utf-8')

        dataPoints = stringData.split(" | ")
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


# def plotData(data):
#     plt.clf()

#     plots = len(data.keys())
#     if plots == 0:
#         return

#     width = math.ceil(plots / 3 * 2)
#     height = int(plots / width) * 4

#     for n, name in enumerate(data.keys()):
#         point = np.array(server.time[name])
#         datum = np.array(server.data[name])

#         if len(datum) > MAX_DISPLAY_POINTS:
#             plt.subplot(height, width, n + plots + 1)
#             points = point[len(datum) - (MAX_DISPLAY_POINTS + 1):]
#             display = datum[len(datum) - (MAX_DISPLAY_POINTS + 1):]
#             plt.plot(points, display, label=name)

#             plt.axhline(np.mean(display), linestyle='--', color='orange')

#             plt.legend()

#             # divisor = len(datum) // MAX_DISPLAY_POINTS
#             # datum = datum[::divisor]

#         plt.subplot(height, width, n+1)
#         plt.plot(point, datum, label=name)

#         plt.legend()

#     plt.pause(0.01)


class DataGUI:
    def __init__(self, server):
        self.server = server
        self.root = tk.Tk()
        self.root.title("Model Tracker")
    
        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.frame)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollFrame = ttk.Frame(self.canvas)

        self.scrollFrame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.create_window((0, 0), window=self.scrollFrame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<MouseWheel>", self._on_mousewheel)

        self.plots = {}

        self.running = True
        self.data_thread = threading.Thread(target=self.dataCollection, daemon=True)
        self.data_thread.start()

        self.updatePlots()

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def dataCollection(self):
        while self.running:
            try:
                self.server.receive()
            except (ValueError, BlockingIOError):
                pass
            time.sleep(0.01)

    def createPlot(self, name, row, col):
        plotFrame = ttk.Frame(self.scrollFrame)
        plotFrame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

        self.scrollFrame.grid_columnconfigure(col, weight=1)
        self.scrollFrame.grid_rowconfigure(row, weight=1)

        fig = Figure(figsize=(10, 4), dpi=100)
        ax1 = fig.add_subplot(121)
        ax1.set_title(name + " Overall")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(122)
        ax2.set_title(name + " Recent")
        ax2.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, plotFrame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        return {
            "frame": plotFrame,
            "figure": fig,
            "ax1": ax1,
            "ax2": ax2,
            "canvas": canvas
        }

    def updatePlots(self):
        if not self.server.data:
            self.root.after(int(DISPLAY_REFRESH * 1000), self.updatePlots)
            return
        
        currentDataSeries = set(self.server.data.keys())
        existingPlots = set(self.plots.keys())

        for key in currentDataSeries - existingPlots:
            row = len(self.plots)
            col = 0
            self.plots[key] = self.createPlot(key, row, col)
            self.canvas.update_idletasks()

        for name, plot in self.plots.items():
            point = np.array(self.server.time[name])
            datum = np.array(self.server.data[name])

            ax1 = plot["ax1"]
            ax1.clear()
            ax1.set_title(name + " Overall")
            ax1.grid(True, alpha=0.3)

            ax1.plot(point, datum, label=name, alpha=0.7)

            ax2 = plot["ax2"]
            ax2.clear()
            ax2.set_title(name + " Recent")
            ax2.grid(True, alpha=0.3)

            if len(datum) > MAX_DISPLAY_POINTS:
                points = point[len(datum) - (MAX_DISPLAY_POINTS + 1):]
                display = datum[len(datum) - (MAX_DISPLAY_POINTS + 1):]
                ax2.plot(points, display, label=name, alpha=0.7)
                ax2.axhline(np.mean(display), linestyle='--', color='orange')
            
            ax1.legend()
            ax2.legend()

            plot["canvas"].draw()

        self.root.after(int(DISPLAY_REFRESH * 1000), self.updatePlots)

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.running = False


if __name__ == "__main__":
    server = Server()
    gui = DataGUI(server)
    gui.run()

