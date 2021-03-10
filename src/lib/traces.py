"""Perform signal processing on power consumption traces.

This module is designed to provide fast signal processing
function for power consumption signals.

"""

import numpy as np
from scipy import stats, fft, signal


def _pearsonr_from_ref(r, st, sh):
    return list(map(lambda s: stats.pearsonr(r, st[s])[0], sh))


class Statistics:
    def __init__(self, leak=None, filters=None, noise=None):
        self.iterations = 0
        self.cropped = None
        self.sum = None
        self.mean = None
        self.spectrum = None
        self.freqs = None

        if leak is not None:
            self.update(leak, filters, noise)

    def update(self, leak, filters=None, noise=None):
        self.iterations += len(leak)
        self.cropped = signal.detrend(adjust(leak.traces, None if self.sum is None else self.sum.shape[0]), axis=1)
        self.cropped -= np.mean(self.cropped, axis=1).reshape((self.cropped.shape[0], 1))
        if filters is not None:
            for f in filters:
                if f is None:
                    continue
                b, a, *_ = f
                self.cropped = signal.filtfilt(b, a, self.cropped, axis=1)
        elif noise is not None and noise.iterations > 0:
            filtered = fft.fft(self.cropped, axis=1) - fft.fft(noise.cropped, axis=1)
            self.cropped = np.real(fft.ifft(filtered, axis=1))

        if self.sum is None:
            self.sum = np.sum(self.cropped, axis=0)
        else:
            self.sum += np.sum(self.cropped, axis=0)
        self.mean = np.divide(self.sum, self.iterations)
        self.spectrum = np.absolute(fft.fft(self.mean))
        size = len(self.spectrum)
        self.freqs = np.argsort(np.fft.fftfreq(size, 1.0 / 200e6)[:size // 2] / 1e6)

    def clear(self):
        self.iterations = 0
        self.cropped = None
        self.sum = None
        self.mean = None
        self.spectrum = None


def crop(traces, end=None):
    """Crops all the traces signals to have the same duration.

    If ``end`` parameter is not provided the traces are cropped to have
    the same duration as the shortest given trace.

    Parameters
    ----------
    traces : list[list[int]]
        2D list of numbers representing the trace signal.
    end : int, optional
        Index after which the traces are truncated.
        Must be inferior to the length of the shortest trace.

    Returns
    -------
    list[list[int]]
        Cropped traces.

    """
    m = min(map(len, traces))
    m = min(end or m, m)
    return [trace[:m] for trace in traces]


def pad(traces, fill=0, end=None):
    """Pads all the traces signals have the same duration.

    If ``end`` parameter is not provided the traces are padded to have
    the same duration as the longest given trace.

    Parameters
    ----------
    traces : list[list[int]]
        2D list of numbers representing the trace signal.
    fill : int, optional
        Padding value to insert after the end of traces.
    end : int, optional
        New count of samples of the traces.
        Must be greater than the length of the longest trace.
    Returns
    -------
    list[list[int]]
        Padded traces.

    """
    samples = list(map(len, traces))
    m = max(samples)
    m = max(end or m, m)
    return [trace + [fill] * (m - read) for trace, read in zip(traces, samples)]


def adjust(traces, n=None, fill=0):
    cropped = crop(traces)
    m = len(cropped[0])
    if not n or m == n:
        return cropped
    if m > n:
        return crop(cropped, end=n)
    elif n > m:
        return pad(cropped, end=n, fill=fill)


def sync(traces, step=1, stop=None):
    """Synchronize trace signals by correlating them

    WARNING: this method may cause segfault
    when the memory adjacent to traces cannot be used

    This function implements an algorithm based on Pearson's
    correlation to synchronize signals peaks.

    More precisely, it compares the traces to a reference trace
    by rolling theses forward or backward. The algorithm search
    for the roll value that maximizes pearson correlation.
    
    Parameters
    ----------
    traces : np.ndarray
        2D numbers array representing cropped or padded traces data.
    step : int, optional
        Rolling step, if equals n, the trace will be rolled
        n times in both directions at each rolling iteration.
    stop : int, optional
        Rolling stop, maximum roll to perform.

    Returns
    -------
        np.ndarray
            2D array representing synchronized traces.

    """
    ref = traces[0]
    n, m = traces.shape
    strides_pos = ref.strides * 2
    strides_neg = (-strides_pos[0], strides_pos[1])
    shape = (m, m)
    stop = min(stop or m, m)
    shifts = list(range(0, stop, step))

    for trace in traces:
        strided = np.lib.stride_tricks.as_strided(trace, shape, strides_pos)
        try:
            buffer = _pearsonr_from_ref(ref, strided, shifts)
        except ValueError:
            continue

        argmax_pos = np.int(np.argmax(buffer))
        max_pos = buffer[argmax_pos]
        strided = np.lib.stride_tricks.as_strided(trace, shape, strides_neg)
        try:
            buffer = _pearsonr_from_ref(ref, strided, shifts)
        except ValueError:
            continue
        argmax_neg = np.int(np.argmax(buffer))
        max_neg = buffer[argmax_neg]
        if max_neg < max_pos:
            trace[:] = np.roll(trace, -shifts[argmax_pos])
        else:
            trace[:] = np.roll(trace, shifts[argmax_neg])

    trace = traces[n - 1]
    shifts = list(range(-stop, stop, step))
    buffer = list(map(lambda shift: stats.pearsonr(ref, np.roll(trace, shift))[0], shifts))
    trace[:] = np.roll(trace, np.argmax(buffer) - stop)

    return traces
