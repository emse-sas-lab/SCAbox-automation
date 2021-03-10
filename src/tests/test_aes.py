import unittest

import numpy as np

from lib import aes


class HandlerTest(unittest.TestCase):

    def setUp(self):
        self.plain_str = "00112233445566778899aabbccddeeff"

        self.plain = np.array([
            [0x00, 0x44, 0x88, 0xcc],
            [0x11, 0x55, 0x99, 0xdd],
            [0x22, 0x66, 0xaa, 0xee],
            [0x33, 0x77, 0xbb, 0xff]
        ], dtype=np.uint8)

        self.key = np.array([
            [0x00, 0x04, 0x08, 0x0c],
            [0x01, 0x05, 0x09, 0x0d],
            [0x02, 0x06, 0x0a, 0x0e],
            [0x03, 0x07, 0x0b, 0x0f]
        ], dtype=np.uint8)

        self.cipher = np.array([
            [0x69, 0x6a, 0xd8, 0x70],
            [0xc4, 0x7b, 0xcd, 0xb4],
            [0xe0, 0x04, 0xb7, 0xc5],
            [0xd8, 0x30, 0x80, 0x5a]
        ], dtype=np.uint8)

        np.set_printoptions(formatter={"int": hex})

    def test_encrypt(self):
        handler = aes.Handler(self.key)
        block = handler.encrypt(self.plain)
        self.assertEqual(np.linalg.norm(block - self.cipher), 0)

    def test_decrypt(self):
        handler = aes.Handler(self.key)
        block = handler.decrypt(self.cipher)
        self.assertEqual(np.linalg.norm(block - self.plain), 0)

    def test_block_to_words(self):
        word = aes.block_to_words(self.plain)
        self.assertEqual(word, self.plain_str)

    def test_words_to_block(self):
        block = aes.words_to_block(self.plain_str)
        self.assertEqual(np.linalg.norm(block - self.plain), 0)


if __name__ == "__main__":
    unittest.main()
