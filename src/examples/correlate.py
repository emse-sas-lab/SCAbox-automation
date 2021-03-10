"""Import acquisition data and perform CPA.

The data are imported from CSV files produced by ``acquire.py``
in order to avoid parsing which is computationally expensive.

The temporal correlations are plot and the key guess is displayed
to validate the attack.

"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from lib import traces as tr, data
from lib.cpa import Handler, Statistics
from lib.data import Request

print(f"{'started':<16}{datetime.now():%Y-%m-%d %H:%M:%S}")

# Creating leakage signal filters according to the sensors sampling frequency
f_sampling = 200e6
f_nyquist = f_sampling / 2
f_cut = 13e6
w_cut = f_cut / f_nyquist
order = 4
b, a, *_ = signal.butter(order, w_cut, btype="highpass", output="ba")

# Creating request object to generate the command to send to SoC and latter used filenames
request = Request({
    "iterations": 1024,
    "target": "/dev/ttyUSB1",
    "path": "./testdata",
    "model": 1
})

# Select a byte to plot the correlation
byte = 0

# Load acquisition data from CSV files
channel = data.Channel(os.path.join(request.path, request.filename("channel", ".csv")))
leak = data.Leak(os.path.join(request.path, request.filename("leak", ".csv")))
meta = data.Meta(os.path.join(request.path, request.filename("meta", ".csv")))

# Filter leakage signal in order to decrease noise
traces = np.array(tr.crop(leak.traces))
for trace in traces:
    trace[:] = signal.filtfilt(b, a, trace)

# Perform correlation on the current data and attack statistics
handler = Handler(model=request.model, traces=traces, channel=channel)
stats = Statistics(handler)

# Plot the temporal correlation of the selected byte and saves the figure with the other acquisition data
plt.fill_between(range(stats.corr.shape[2]), stats.corr_max[byte], stats.corr_min[byte], color="grey")
if stats.exacts[-1][byte]:
    plt.plot(stats.corr[byte, stats.key[byte]], color="r", label=f"key 0x{stats.key[byte]:02x}")
else:
    plt.plot(stats.corr[byte, stats.guesses[-1][byte]], color="c", label=f"guess 0x{stats.guesses[-1][byte]:02x}")
    plt.plot(stats.corr[byte, stats.key[byte]], color="b", label=f"key 0x{stats.key[byte]:02x}")

plt.savefig(os.path.join(request.path, request.filename(f"corr_b{byte}")))
plt.close()

print(stats)
print(f"{'ended':<16}{datetime.now():%Y-%m-%d %H:%M:%S}")
