from tkinter import *

import matplotlib.gridspec as gridspec
from matplotlib import ticker
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

from lib.cpa import COUNT_HYP, Statistics


def raw(ax, traces, limit=16, chunk=None):
    chunk = (chunk or 0) + 1
    ax.set(xlabel="Time Samples", ylabel="Quantification")
    return [ax.plot(trace, label=f"iteration {d * chunk}") for d, trace in enumerate(traces[:limit])]


def avg(ax, trace):
    ax.set(xlabel="Time Samples", ylabel="Quantification")
    return ax.plot(trace, color="grey")


def fft(ax, freq, spectrum, f):
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, *_: f"{int(v / 1e6)}"))
    ax.set(xlabel="Frequency (MHz)", ylabel="Hamming Weight")
    *_, line = ax.magnitude_spectrum(spectrum, Fs=200e6, color="red")
    return line


def iterations(ax, stats, byte):
    maxs = Statistics.graph(stats.bests)
    ax.set(xlabel="Traces acquired", ylabel="Pearson Correlation")
    plot_key = None
    plot_guess = None
    plots = []
    for h in range(COUNT_HYP):
        if h == stats.key[byte] and h == stats.guesses[-1][byte]:
            plot_key, = plot_guess, = ax.plot(stats.iterations, maxs[byte, h], color="r", zorder=10)
        elif h == stats.key[byte]:
            plot_key, = ax.plot(stats.iterations, maxs[byte, h], color="b", zorder=10)
        elif h == stats.guesses[-1][byte]:
            plot_guess, = ax.plot(stats.iterations, maxs[byte, h], color="c", zorder=10)
        else:
            plots.append(ax.plot(stats.iterations, maxs[byte, h], color="grey"))
    return plot_key, plot_guess, plots


def temporal(ax, stats, byte):
    corr_guess = stats.corr[byte, stats.guesses[-1][byte]]
    corr_key = stats.corr[byte, stats.key[byte]]
    ax.set(xlabel="Time Samples", ylabel="Pearson Correlation")
    ax.fill_between(range(stats.corr.shape[2]), stats.corr_max[byte], stats.corr_min[byte], color="grey")
    if stats.exacts[-1][byte]:
        plot_key, = plot_guess, = ax.plot(corr_guess, color="r")
    else:
        plot_guess, = ax.plot(corr_guess, color="c")
        plot_key, = ax.plot(corr_key, color="b")
    return plot_key, plot_guess


class PlotFrame(LabelFrame):
    def __init__(self, master):
        super().__init__(master, text="Plots")
        self.gs = gridspec.GridSpec(2, 1, hspace=0.5)
        self.fig = Figure(figsize=(16, 9))
        self.ax1 = self.fig.add_subplot(self.gs[0])
        self.ax2 = self.fig.add_subplot(self.gs[1])
        self.plot1 = None
        self.plot2 = None
        self.annotation = None
        self.gs.tight_layout(self.fig)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=TOP)

    def clear(self):
        self.ax1.clear()
        self.ax2.clear()
        self.fig.clear()
        self.gs.tight_layout(self.fig)
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.canvas.figure = self.fig
        self.canvas.draw()

    def draw_stats(self, data):
        mean, spectrum, freq = data
        for legend in self.fig.legends:
            legend.remove()
        self.ax1.clear()
        self.ax2.clear()
        self.plot1, = avg(self.ax1, mean)
        self.plot2 = fft(self.ax2, freq, spectrum, freq)
        self.fig.suptitle("Leakage statistics")
        self.fig.legend((self.plot1, self.plot2),
                        ("Temporal average", "Spectrum average"))
        self.fig.canvas.draw()

    def draw_corr(self, data, byte):
        byte = byte or 0
        stats = data
        for legend in self.fig.legends:
            legend.remove()
        self.ax1.clear()
        self.ax2.clear()
        plot_key, plot_guess, _ = iterations(self.ax1, stats, byte)
        temporal(self.ax2, stats, byte)
        self.fig.suptitle(f"Correlation byte {byte}")
        self.fig.legend((plot_key, plot_guess),
                        (f"key 0x{stats.key[byte]:02x}", f"guess 0x{stats.guesses[-1][byte]:02x}"))
        self.fig.canvas.draw()
