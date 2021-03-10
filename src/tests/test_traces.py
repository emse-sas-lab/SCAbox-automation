import unittest

import numpy as np

from lib import traces as tr


class TracesTest(unittest.TestCase):
    def setUp(self):
        self.x0 = np.sin(np.linspace(0, 2 * np.pi, 256)).tolist()
        self.x1 = np.cos(np.linspace(0, 2 * np.pi, 256)).tolist()
        self.x2 = np.sin(np.linspace(np.pi / 4, 2 * np.pi + np.pi / 2, 320)).tolist()

    def test_crop(self):
        traces = np.array(tr.crop([self.x0, self.x2]))
        self.assertEqual(traces.shape, (2, 256))
        self.assertEqual(traces.dtype, np.float64)

    def test_pad(self):
        traces = np.array(tr.pad([self.x0, self.x2]))
        self.assertEqual(traces.shape, (2, 320))
        self.assertEqual(traces.dtype, np.float64)

    def test_sync(self):
        traces = tr.sync(np.array([self.x0, self.x1]))
        self.assertAlmostEqual(np.linalg.norm(traces[0] - traces[1]), 0, delta=0.15)


if __name__ == "__main__":
    unittest.main()
