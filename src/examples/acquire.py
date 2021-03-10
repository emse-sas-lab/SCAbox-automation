"""Export and plot leakage from SoC.

The data acquired from SoC consists on the side-channel leakage
and encryption channel data used later to perform and validate correlation.

The side-channel leakage is plot to evaluate the quality of the acquisition.
The data is exported in separated files.

"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from serial import Serial, PARITY_NONE

from lib import data
from lib import traces as tr
from lib.data import Request, Keywords

print(f"{'started':<16}{datetime.now():%Y-%m-%d %H:%M:%S}")

# Creating request object to generate the command to send to SoC and latter used filenames
request = Request({
    "iterations": 1024,
    "target": "/dev/ttyUSB1",
    "path": "./testdata"
})

os.makedirs(request.path)

# Send acquisition command and read acquisition data from serial port
with Serial(request.target, 921_600, parity=PARITY_NONE, xonxoff=False) as ser:
    command = f"{request.command('sca')}"
    print(f"{'sending':<16}{command}")
    ser.flush()
    ser.write(f"{command}\n".encode())
    s = bytearray(ser.read(16))
    while s[-16:].find(Keywords.END_ACQ_TAG) == -1:
        while ser.in_waiting == 0:
            continue
        while ser.in_waiting != 0:
            s += ser.read_all()

# Parse received serial data to perform deserialization
parser = data.Parser(s, direction=request.direction, verbose=request.verbose)
parsed = len(parser.channel)

# Save parsed and raw data to file system
with open(os.path.join(request.path, request.filename(suffix=".bin")), "wb+") as file:
    file.write(s)
parser.channel.write_csv(os.path.join(request.path, request.filename("channel", ".csv")))
parser.leak.write_csv(os.path.join(request.path, request.filename("leak", ".csv")))
parser.meta.write_csv(os.path.join(request.path, request.filename("meta", ".csv")))
parser.noise.write_csv(os.path.join(request.path, request.filename("noise", ".csv")))

print(f"{'size':<16}{len(s)} B")
print(f"{'parsed':<16}{parsed}/{request.iterations}")
print(f"{'total':<16}{parser.meta.iterations}/{(request.chunks or 1) * request.iterations}")

# Compute mean the trace and the spectrum of the mean trace
traces = np.array(tr.crop(parser.leak.traces))
mean = np.sum(traces.copy(), axis=0) / parser.meta.iterations
spectrum = np.absolute(fft.fft(mean - np.mean(mean)))

# Plot the raw traces, the mean and the spectrum
for trace in traces[:16]:
    plt.plot(trace)
plt.savefig(os.path.join(request.path, request.filename("traces")))
plt.close()
plt.plot(mean)
plt.savefig(os.path.join(request.path, request.filename("mean")))
plt.close()
plt.plot(spectrum)
plt.savefig(os.path.join(request.path, request.filename("spectrum")))
plt.close()

print(f"{'ended':<16}{datetime.now():%Y-%m-%d %H:%M:%S}")
