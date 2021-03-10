import os
import unittest
from math import sqrt

from numpy.linalg import norm

from lib import data, aes

LOG_PATH = os.path.join("..", *["..", "data", "acquisition"])


class DataTest(unittest.TestCase):
    def setUp(self):
        self.data_path_hw_256 = os.path.join(LOG_PATH, *["hw", "data_hw_256.csv"])
        self.data_path_hw_65536 = os.path.join(LOG_PATH, *["hw", "data_hw_65536.csv"])

    def test_read(self):
        data = data.Channel.from_csv(self.data_path_hw_256)
        n = len(data.plains)
        self.assertEqual(n, len(data.ciphers), "ciphers len mismatch")
        self.assertEqual(n, len(data.keys), "keys len mismatch")
        self.assertEqual(n, 256, "plains len mismatch")

        handler = aes.Handler(aes.words_to_block(data.keys[0]))
        for plain, cipher in zip(data.plains, data.ciphers):
            plain = aes.words_to_block(plain)
            cipher = aes.words_to_block(cipher)
            self.assertEqual(norm(handler.encrypt(plain) - cipher), 0, "incorrect encryption")

        data = data.Channel.from_csv(self.data_path_hw_65536)
        n = len(data.plains)
        self.assertEqual(n, len(data.ciphers), "ciphers len mismatch")
        self.assertEqual(n, len(data.keys), "keys len mismatch")
        self.assertGreaterEqual(n, 65536 * 0.8, "< 80% of plains retrieved")

    def test_write(self):
        old_data = data.Channel.from_csv(self.data_path_hw_256)
        old_data.write_csv("_buffer.csv", )
        new_data = data.Channel.from_csv("_buffer.csv")

        diff_p = (new != old for new, old in zip(new_data.plains, old_data.plains))
        diff_c = (new != old for new, old in zip(new_data.ciphers, old_data.ciphers))
        diff_k = (new != old for new, old in zip(new_data.keys, old_data.keys))

        self.assertEqual(sum(diff_p), False, "incorrect plains")
        self.assertEqual(sum(diff_c), False, "incorrect ciphers")
        self.assertEqual(sum(diff_k), False, "incorrect keys")

        os.remove("_buffer.csv")


class LeakTest(unittest.TestCase):
    def setUp(self):
        self.leak_path_hw_256 = os.path.join(LOG_PATH, *["hw", "leak_hw_256.csv"])
        self.leak_path_hw_65536 = os.path.join(LOG_PATH, *["hw", "leak_hw_65536.csv"])

    def test_read(self):
        leak = data.Leak.from_csv(self.leak_path_hw_256)
        n = len(leak.traces)
        self.assertEqual(n, 256)

        leak = data.Leak.from_csv(self.leak_path_hw_65536)
        n = len(leak.traces)
        self.assertGreaterEqual(n, 65536 * 0.8, "< 80% of traces retrieved")
        m = [len(trace) for trace in leak.traces]
        m.sort()
        m2 = [i * i for i in m]
        m_max = m[-1]
        m_min = m[0]
        m_med = m[n // 2]
        m_avg = sum(m) / n
        m_dev = sqrt(sum(m2) / n - m_avg * m_avg)
        self.assertEqual(sum((i - j for i, j in zip(m, leak.samples))), 0)
        self.assertAlmostEqual(m_med, m_avg, delta=m_dev)
        self.assertLess(abs(m_max - m_avg), 3 * m_dev)
        self.assertLess(abs(m_min - m_avg), 3 * m_dev)

    def test_write(self):
        old_data = data.Leak.from_csv(self.leak_path_hw_256)
        old_data.write_csv("_buffer.csv", )
        new_data = data.Leak.from_csv("_buffer.csv")
        for new_trace, old_trace in zip(new_data.traces, old_data.traces):
            diff = sum((abs(new - old) for new, old in zip(new_trace, old_trace)))
            self.assertEqual(diff, 0)

        os.remove("_buffer.csv")


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.cmd_path_hw_256 = os.path.join(LOG_PATH, *["hw", "cmd_hw_256.log"])
        self.cmd_path_hw_65536 = os.path.join(LOG_PATH, *["hw", "cmd_hw_65536.log"])

    def test_parse(self):
        parser = data.Parser.from_bytes(read.file(self.cmd_path_hw_256))
        n = len(parser.leak.traces)
        self.assertEqual(n, len(parser.channel.plains), "plains len mismatch")
        self.assertEqual(n, parser.meta.iterations, "iterations mismatch")
        self.assertEqual(n, 256, "traces len mismatch")

        parser = data.Parser.from_bytes(read.file(self.cmd_path_hw_65536))
        n = len(parser.leak.traces)
        self.assertEqual(n, len(parser.channel.plains), "plains len mismatch")
        self.assertEqual(n, parser.meta.iterations, "iterations mismatch")
        self.assertGreaterEqual(n, 65536 * 0.8, "< 80% of traces retrieved")


if __name__ == "__main__":
    unittest.main()
