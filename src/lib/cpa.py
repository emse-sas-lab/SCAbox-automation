"""Numpy based module for CPA side-channel attacks.

This module is provides abstractions to handle
side channel attacks via power consumption acquisition.

It features a trace accumulator to avoid storing all the
traces in memory. It also implements a fast Pearson correlation algorithm
to retrieve attack results in a reasonable amount of time.

"""
import concurrent
from enum import Enum
from itertools import product

import numpy as np

from lib import aes

COUNT_HYP = 256  # Count of key hypothesis for one byte
COUNT_CLS = 256  # Traces with the same byte value in a given position
BLOCK_SIZE = aes.BLOCK_SIZE


class Statistics:
    def __init__(self, handler=None):
        self.corr = None
        self.corr_min = None
        self.corr_max = None

        self.key = None
        self.guesses = []
        self.exacts = []
        self.ranks = []
        self.bests = []
        self.iterations = []
        self.divergences = []

        if handler and handler.iterations > 0:
            self.update(handler)

    def update(self, handler):
        self.corr = handler.correlations()
        self.key = handler.key
        guess, best, exact, rank = Statistics.guess_stats(self.corr, handler.key)
        self.corr_max, self.corr_min = Statistics.guess_envelope(self.corr, guess)
        self.guesses.append(guess)
        self.exacts.append(exact)
        self.ranks.append(rank)
        self.bests.append(best)
        self.iterations.append(handler.iterations)
        self.divergences.append(self.div_idxs())

    def clear(self):
        self.corr = None
        self.corr_min = None
        self.corr_max = None
        self.guesses.clear()
        self.exacts.clear()
        self.ranks.clear()
        self.bests.clear()
        self.iterations.clear()

    def __repr__(self):
        return f"Statistics({self.corr})"

    def __str__(self):
        ret = f"{'Byte':<8}{'Exact':<8}{'Key':<8}{'(%)':<8}{'Guess':<8}{'(%)':<8}{'Rank':<8}{'Divergence':<8}\n"
        for b in range(BLOCK_SIZE):
            ret += f"{b:<8}" \
                   f"{bool(self.exacts[-1][b]):<8}" \
                   f"{self.key[b]:<8x}{100 * self.bests[-1][b, self.key[b]]:<5.2f}{'%':<3}" \
                   f"{self.guesses[-1][b]:<8x}{100 * self.bests[-1][b, self.guesses[-1][b]]:<5.2f}{'%':<3}" \
                   f"{self.ranks[-1][b]:<8}" \
                   f"{self.divergences[-1][b]:<8}\n"

        return ret

    def div_idxs(self, n=0.2):
        div = np.full((BLOCK_SIZE,), fill_value=-1)
        for b in range(BLOCK_SIZE):
            if self.key[b] != self.guesses[-1][b]:
                continue
            for chunk, mx in enumerate(self.bests):
                mx_second = mx[b, np.argsort(mx[b])[-2]]
                mx_key = mx[b, self.key[b]]
                if (mx_key - mx_second) / mx_key > n:
                    div[b] = self.iterations[chunk]
                    break
        return div

    @classmethod
    def graph(cls, data):
        data = np.array(data)
        n = len(data.shape)
        r = tuple(range(n))
        return np.moveaxis(data, r, tuple([r[-1]] + list(r[:-1])))

    @classmethod
    def guess_stats(cls, cor, key):
        """Computes the best guess key from correlation data.

        Parameters
        ----------
        cor : np.ndarray
            Temporal correlation per block byte and hypothesis.
        key : np.ndarray

        Returns
        -------
        guess : np.ndarray
            Guessed key block.
        maxs : np.ndarray
            Maximums of temporal correlation per hypothesis.
        exact : np.ndarray
            ``True`` if the guess is exact for each byte position.
        rank : np.ndarray
            Rank of the true key in terms of correlation.

        See Also
        --------
        correlations : Compute temporal correlation.

        """
        best = np.amax(np.absolute(cor), axis=2)
        guess = np.argmax(best, axis=1)
        rank = COUNT_HYP - np.argsort(np.argsort(best, axis=1), axis=1)
        rank = np.array([rank[b, key[b]] for b in range(BLOCK_SIZE)])
        exact = guess == key
        return guess, best, exact, rank

    @classmethod
    def guess_envelope(cls, cor, guess):
        """Computes the envelope of correlation.

        The envelope consists on two curves representing
        respectively the max and min of temporal correlation
        at each instant.

        This feature is mainly useful to plot
        temporal correlations curve.

        Parameters
        ----------
        cor : np.ndarray
            Temporal correlation per block position and hypothesis.
        guess : np.ndarray
            Guessed block matrix.
        Returns
        -------
        cor_max : np.ndarray
            Maximum correlation at each instant.
        cor_min : np.ndarray
            Minimum correlation at each instant.

        See Also
        --------
        correlations : Compute temporal correlation.

        """
        env = np.moveaxis(cor.copy(), (0, 1, 2), (0, 2, 1))
        for b in range(BLOCK_SIZE):
            env[b, :, guess[b]] -= env[b, :, guess[b]]

        return np.max(env, axis=2), np.min(env, axis=2)


class Handler:
    """CPA correlation handler interface.

    Attributes
    ----------
    blocks: np.ndarray
        Encrypted data blocks for each trace.
    key: np.ndarray
        Key data block for all the traces.
    hypothesis: np.ndarray
        Value of power consumption for each hypothesis and class.
    lens: np.ndarray
        Count of traces per class.
    sums: np.ndarray
        Average trace per class.
    sums2: np.ndarray
        Standard deviation trace per class.
    sum: np.ndarray
        Average trace for all classes.
    sum2: np.ndarray
        Standard deviation trace for all classes.

    """

    class Models(Enum):
        """CPA power consumption models.

        """

        SBOX_R0 = 0
        INV_SBOX_R10 = 1

    def __init__(self, model=None, channel=None, traces=None, samples=None):
        """Allocates memory, accumulates traces and initialize model.

        Parameters
        ----------
        channel : data.Channel
            Channel data blocks for each trace.
        traces : np.ndarray
            Leak traces matrix.
        model : Handler.Models
            Hypothesis model.
        samples : int
            Count of time samples in the signals.
        """
        self.model = model or Handler.Models(value=0)
        self.blocks = None
        self.key = None
        self.iterations = 0
        self.samples = None
        self.hypothesis = None
        self.lens = None

        self.sums = None
        self.sums2 = None
        self.sum = None
        self.sum2 = None

        if traces is not None and channel is not None:
            samples = samples or traces.shape[1]
            self.clear(samples).set_model(model).set_key(channel).set_blocks(channel).accumulate(traces)
        else:
            self.clear(samples or 0).set_model(self.model)

    def clear(self, samples=0):
        self.iterations = 0
        self.samples = samples
        self.blocks = None
        self.key = np.zeros((BLOCK_SIZE,), dtype=np.int)
        self.hypothesis = np.zeros((COUNT_HYP, COUNT_CLS), dtype=np.uint8)
        self.lens = np.zeros((BLOCK_SIZE, COUNT_CLS), dtype=np.int)
        self.sums = np.zeros((BLOCK_SIZE, COUNT_CLS, samples), dtype=np.float)
        self.sums2 = np.zeros((BLOCK_SIZE, COUNT_CLS, samples), dtype=np.float)
        self.sum = np.zeros(samples, dtype=np.float)
        self.sum2 = np.zeros(samples, dtype=np.float)
        return self

    @classmethod
    def _accumulate_lens(cls, lens, blocks, traces, iterations):
        iterations += traces.shape[0]
        for b, (block, trace) in product(range(BLOCK_SIZE), zip(blocks, traces)):
            lens[b, block[b]] += 1
        return lens, iterations

    @classmethod
    def _accumulate_sums(cls, sums, blocks, traces, sum):
        sum += np.sum(traces, axis=0)
        for b, (block, trace) in product(range(BLOCK_SIZE), zip(blocks, traces)):
            sums[b, block[b]] += trace
        return sums, sum

    @classmethod
    def _accumulate_sums2(cls, sums2, blocks, traces, sum2):
        sum2 += np.sum(traces * traces, axis=0)
        for b, (block, trace) in product(range(BLOCK_SIZE), zip(blocks, traces)):
            sums2[b, block[b]] += np.square(trace)
        return sums2, sum2

    def accumulate(self, traces):
        """Sorts traces by class and compute means and deviation.

        Parameters
        ----------
        traces : np.ndarray
            Traces matrix.

        Returns
        -------
        sca-automation.lib.cpa.Handler
            Reference to self.

        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            processes = (executor.submit(Handler._accumulate_lens, self.lens, self.blocks, traces, self.iterations),
                         executor.submit(Handler._accumulate_sums, self.sums, self.blocks, traces, self.sum),
                         executor.submit(Handler._accumulate_sums2, self.sums2, self.blocks, traces, self.sum2))
            self.lens, self.iterations = processes[0].result()
            self.sums, self.sum = processes[1].result()
            self.sums2, self.sum2 = processes[2].result()
        return self

    def set_key(self, channel):
        shape = (BLOCK_SIZE,)
        if self.model == Handler.Models.SBOX_R0:
            self.key = aes.words_to_block(channel.keys[0]).reshape(shape)
        elif self.model == Handler.Models.INV_SBOX_R10:
            self.key = aes.key_expansion(aes.words_to_block(channel.keys[0]))[10].T.reshape(shape)
        else:
            raise ValueError(f"unknown model: {self.model}")
        return self

    def set_blocks(self, channel):
        shape = (BLOCK_SIZE,)
        if self.model == Handler.Models.SBOX_R0:
            self.blocks = np.array([aes.words_to_block(block).reshape(shape) for block in channel.plains])
        elif self.model == Handler.Models.INV_SBOX_R10:
            self.blocks = ([aes.words_to_block(block).reshape(shape) for block in channel.ciphers])
        else:
            raise ValueError(f"unknown model: {self.model}")
        return self

    def set_model(self, model):
        """Initializes power consumption model.

        Parameters
        ----------
        model : Handler.Models
            Model value.

        Returns
        -------
        sca-automation.lib.cpa.Handler
            Reference to self.

        """

        if model == Handler.Models.SBOX_R0:
            for h, k in product(range(COUNT_HYP), range(COUNT_CLS)):
                self.hypothesis[h, k] = bin(aes.S_BOX[k ^ h]).count("1")
        elif model == Handler.Models.INV_SBOX_R10:
            for h, k in product(range(COUNT_HYP), range(COUNT_CLS)):
                self.hypothesis[h, k] = bin(aes.INV_S_BOX[k ^ h] ^ k).count("1")
        else:
            raise ValueError(f"unknown model: {self.model}")
        self.model = model
        return self

    @classmethod
    def _byte_correlations(cls, data):
        b, n, sums, lens, hypothesis, mean, dev, samples = data
        ret = np.empty((COUNT_HYP, samples))
        mean_ij = np.nan_to_num(np.divide(sums[b], lens[b].reshape((COUNT_CLS, 1))))
        for h in range(COUNT_HYP):
            y = np.array(hypothesis[h] * lens[b], dtype=np.float)
            y_mean = np.divide(np.sum(y), n)
            y_dev = np.sqrt(np.divide(np.sum(hypothesis[h] * y), n) - y_mean * y_mean)
            xy = np.divide(np.sum(y.reshape((COUNT_HYP, 1)) * mean_ij, axis=0), n)
            ret[h] = np.nan_to_num(np.divide(np.divide(xy - mean * y_mean, dev), y_dev))
        return ret

    def correlations(self):
        """Computes Pearson's correlation coefficients on current data.

        Returns
        -------
        np.ndarray
            Temporal correlation per byte and hypothesis.

        """
        n = self.iterations
        mean = self.sum / n
        dev = self.sum2 / n
        dev -= np.square(mean)
        dev = np.sqrt(dev)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            args = ((b, n, self.sums, self.lens, self.hypothesis, mean, dev, self.samples) for b in range(BLOCK_SIZE))
            results = executor.map(Handler._byte_correlations, args)
        return np.array(list(results))
