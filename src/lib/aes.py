"""Numpy based AES module for side-channel attacks

This module is designed to allow easily build power consumption models.

It separately provides all the round operations used in the AES algorithm.
It also features a handler class used to encapsulate key-expansion and
ease encryption and decryption.

Examples
--------
>>> from lib import aes
>>> import numpy as np
>>> key = np.ones((4,4), dtype=np.uint8)
>>> plain = np.eye(4, dtype=np.uint8)
>>> ark = aes.add_round_key(plain, key)
>>> handler = aes.Handler(key)
>>> cipher = handler.encrypt(plain)

Unlike the usual AES implementation, this one keeps a full trace
of the intermediate operations during encryption.

More precisely, this module stores the value of the message block
at each round and for each round operation.

Examples
--------
>>> from lib import aes
>>> import numpy as np
>>> key = np.ones((4,4), dtype=np.uint8)
>>> plain = np.eye(4, dtype=np.uint8)
>>> handler = aes.Handler(key)
>>> cipher = handler.encrypt(plain)
>>> ark = handler.blocks[1, aes.Stages.ADD_ROUND_KEY]

"""

import numpy as np

S_BOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)

INV_S_BOX = np.array([
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
])

R_CON = np.array([
    [0x01, 0x00, 0x00, 0x00],
    [0x02, 0x00, 0x00, 0x00],
    [0x04, 0x00, 0x00, 0x00],
    [0x08, 0x00, 0x00, 0x00],
    [0x10, 0x00, 0x00, 0x00],
    [0x20, 0x00, 0x00, 0x00],
    [0x40, 0x00, 0x00, 0x00],
    [0x80, 0x00, 0x00, 0x00],
    [0x1b, 0x00, 0x00, 0x00],
    [0x36, 0x00, 0x00, 0x00],
    [0x6c, 0x00, 0x00, 0x00],
    [0xd8, 0x00, 0x00, 0x00],
    [0xab, 0x00, 0x00, 0x00]
], dtype=np.uint8)

N_ROUNDS = 10  # Count of rounds excluding the first one
BLOCK_LEN = 4  # Size of the AES block matrix
BLOCK_SIZE = 16  # Size of the AES block in bytes


def __xtime(x):
    return (x << 1) ^ (((x >> 7) & 1) * 0x1b)


def __multiply(x, y):
    return (((y & 1) * x) ^
            ((y >> 1 & 1) * __xtime(x)) ^
            ((y >> 2 & 1) * __xtime(__xtime(x))) ^
            ((y >> 3 & 1) * __xtime(__xtime(__xtime(x)))) ^
            ((y >> 4 & 1) * __xtime(__xtime(__xtime(__xtime(x))))))


def word_to_col(w):
    """Splits a hexadecimal string to a bytes column.

    Parameters
    ----------
    w : str
        Hexadecimal 32-bit word.

    Returns
    -------
    list
        4 bytes column containing integers representing the input string.

    """
    x = int(w, 16)
    return [x >> 24, (x >> 16) & 0xff, (x >> 8) & 0xff, x & 0xff]


def col_to_word(c):
    """Formats a bytes column as a hexadecimal string.


    Parameters
    ----------
    c : np.ndarray
        4 bytes column to format
    Returns
    -------
    str:
        Hexadecimal 32 bits words representing the input column
    Raises
    ------
    ValueError
        If the column has a length different from 4

    """

    if len(c) != BLOCK_LEN:
        raise ValueError("input column must be of length 4")
    return "%02x%02x%02x%02x" % tuple(c)


def words_to_block(words):
    """Transforms hexadecimal message string into an AES block matrix.

    Parameters
    ----------
    words : str
        4 32-bits words as strings representing block's columns

    Returns
    -------
    np.ndarray
        4x4 matrix column major message block's matrix.

    Raises
    ------
    ValueError
        If the count of words provided differs from 4.

    See Also
    --------
        word_to_col : Column conversion from word.

    """

    ret = np.empty((BLOCK_LEN, BLOCK_LEN))
    ret[0] = word_to_col(words[0:8])
    ret[1] = word_to_col(words[8:16])
    ret[2] = word_to_col(words[16:24])
    ret[3] = word_to_col(words[24:32])
    return np.array(ret, dtype=np.uint8).T


def block_to_words(block):
    """Formats a state block into an hexadecimal string.

    Parameters
    ----------
    block : np.ndarray
        4x4 row major block matrix.

    Returns
    -------
    str
        4 32-bit words string representing the column major block.


    Raises
    ------
    ValueError
        If count of rows in block provided differs from 4.

    See Also
    --------
        col_to_word : column conversion to hexadecimal string.
    """
    if len(block) != BLOCK_LEN:
        raise ValueError("input block must be of length 4")
    return "%s%s%s%s" % tuple(col_to_word(c) for c in block.T)


def add_round_key(block, key):
    """Performs a bitwise XOR between a state block and the key.

    Parameters
    ----------
    block : np.ndarray
        4x4 column major block matrix.
    key : np.ndarray
        4x4  column major block matrix.

    Returns
    -------
    np.ndarray
        4x4 result block matrix

    """
    return block ^ key.T


def sub_bytes(block):
    """Applies sbox to a state block.

    Parameters
    ----------
    block : np.ndarray
        4x4 column major block matrix.

    Returns
    -------
    np.ndarray
        4x4 result block matrix

    """
    ret = block.copy()
    ret[0] = np.fromiter(map(lambda b: S_BOX[b], ret[0]), dtype=np.uint8)
    ret[1] = np.fromiter(map(lambda b: S_BOX[b], ret[1]), dtype=np.uint8)
    ret[2] = np.fromiter(map(lambda b: S_BOX[b], ret[2]), dtype=np.uint8)
    ret[3] = np.fromiter(map(lambda b: S_BOX[b], ret[3]), dtype=np.uint8)
    return ret


def inv_sub_bytes(block):
    """Applies inverse sbox to a state block.

    Parameters
    ----------
    block : np.ndarray
        4x4 column major block matrix.

    Returns
    -------
    np.ndarray
        4x4 result block matrix.

    """
    ret = block.copy()
    ret[0] = np.fromiter(map(lambda b: INV_S_BOX[b], ret[0]), dtype=np.uint8)
    ret[1] = np.fromiter(map(lambda b: INV_S_BOX[b], ret[1]), dtype=np.uint8)
    ret[2] = np.fromiter(map(lambda b: INV_S_BOX[b], ret[2]), dtype=np.uint8)
    ret[3] = np.fromiter(map(lambda b: INV_S_BOX[b], ret[3]), dtype=np.uint8)
    return ret


def shift_rows(block):
    """Shifts rows of a state block.

    Parameters
    ----------
    block : np.ndarray
        4x4 column major block matrix.

    Returns
    -------
    np.ndarray
        4x4 result block matrix.

    """
    ret = block.copy()
    ret[1] = np.roll(ret[1], -1)
    ret[2] = np.roll(ret[2], -2)
    ret[3] = np.roll(ret[3], -3)
    return ret


def inv_shift_rows(block):
    """Reverses shift rows of a state block.

    Parameters
    ----------
    block : np.ndarray
        4x4 column major block matrix.

    Returns
    -------
    np.ndarray
        4x4 result block matrix.

    """
    ret = block.copy()
    ret[1] = np.roll(ret[1], 1)
    ret[2] = np.roll(ret[2], 2)
    ret[3] = np.roll(ret[3], 3)
    return ret


def mix_columns(block):
    """Mix columns of a state block.

    Parameters
    ----------
    block : np.ndarray
        4x4 column major block matrix.

    Returns
    -------
    np.ndarray
        4x4 result block matrix.

    """
    ret = block.copy()
    t = block[0]
    tmp = block[0] ^ block[1] ^ block[2] ^ block[3]
    ret[0] ^= __xtime(block[0] ^ block[1]) ^ tmp
    ret[1] ^= __xtime(block[1] ^ block[2]) ^ tmp
    ret[2] ^= __xtime(block[2] ^ block[3]) ^ tmp
    ret[3] ^= __xtime(block[3] ^ t) ^ tmp
    return ret


def inv_mix_columns(block):
    """Reverses mix columns of a state block.

    Parameters
    ----------
    block : np.ndarray
        4x4 column major block matrix.

    Returns
    -------
    np.ndarray
        4x4 result block matrix.

    """
    ret = block.copy()
    ret[0] = (__multiply(block[0], 0x0e)
              ^ __multiply(block[1], 0x0b)
              ^ __multiply(block[2], 0x0d)
              ^ __multiply(block[3], 0x09))
    ret[1] = (__multiply(block[0], 0x09)
              ^ __multiply(block[1], 0x0e)
              ^ __multiply(block[2], 0x0b)
              ^ __multiply(block[3], 0x0d))
    ret[2] = (__multiply(block[0], 0x0d)
              ^ __multiply(block[1], 0x09)
              ^ __multiply(block[2], 0x0e)
              ^ __multiply(block[3], 0x0b))
    ret[3] = (__multiply(block[0], 0x0b)
              ^ __multiply(block[1], 0x0d)
              ^ __multiply(block[2], 0x09)
              ^ __multiply(block[3], 0x0e))
    return ret


def sub_word(col):
    """Applies SBOX to a key column.

    Parameters
    ----------
    col : np.ndarray
        4 bytes array.

    Returns
    -------
    np.ndarray
        Result column.

    """
    return np.fromiter(map(lambda b: S_BOX[b], col), dtype=np.uint8)


def rot_word(col):
    """Rotates a 4 key column.

    Parameters
    ----------
    col : np.ndarray
        4 bytes array.

    Returns
    -------
    np.ndarray
        Result column.

    """
    return np.roll(col, -1)


def key_expansion(key):
    """Performs key expansion.

    Parameters
    ----------
    key : np.ndarray
        4x4 input block key matrix.

    Returns
    -------
    np.ndarray
        4x4 input block key matrices for each round.

    """
    keys = np.zeros((N_ROUNDS + 1, BLOCK_LEN, BLOCK_LEN), dtype=np.uint8)
    keys[0] = key.T
    for cur, prev, con in zip(keys[1:], keys, R_CON):
        cur[0] = con ^ sub_word(rot_word(prev[3])) ^ prev[0]
        cur[1] = cur[0] ^ prev[1]
        cur[2] = cur[1] ^ prev[2]
        cur[3] = cur[2] ^ prev[3]

    return keys


class Stages:
    """AES round stages enumeration.

    The index of the stage correspond to its order into the round.
    """
    START = 0

    SUB_BYTES = 1
    SHIFT_ROWS = 2
    MIX_COLUMNS = 3
    ADD_ROUND_KEY = 4

    INV_SHIFT_ROWS = 1
    INV_SUB_BYTES = 2
    INV_ADD_ROUND_KEY = 3
    INV_MIX_COLUMNS = 4

    def __init__(self):
        raise NotImplementedError("Cannot instantiate an object from a static class")


class Handler:
    """AES computations handler interface.

    Attributes
    ----------
    blocks : np.ndarray
        4x4 block state matrix for each round and each operation.
    keys : np.ndarray
        4x4 block key matrix for each round.

    """

    def __init__(self, key):
        """Allocates memory and performs key expansion.

        Parameters
        ----------
        key : np.ndarray
            4x4 input block key matrix

        """
        self.blocks = np.zeros((N_ROUNDS + 1, 5, BLOCK_LEN, BLOCK_LEN), dtype=np.uint8)
        self.keys = np.zeros((N_ROUNDS + 1, BLOCK_LEN, BLOCK_LEN), dtype=np.uint8)
        self.keys = key_expansion(key)

    def encrypt(self, block):
        """Performs AES cipher algorithm.

        Parameters
        ----------
        block : np.ndarray
            Plain 4x4 column major block matrix.

        Returns
        -------
        np.ndarray
            Cipher 4x4 column major block matrix.
        """
        cur, key = self.blocks[0], self.keys[0]
        cur[Stages.START] = np.copy(block)
        cur[Stages.SUB_BYTES] = np.copy(block)
        cur[Stages.SHIFT_ROWS] = np.copy(block)
        cur[Stages.MIX_COLUMNS] = np.copy(block)
        cur[Stages.ADD_ROUND_KEY] = add_round_key(block, key)

        for (cur, prev, key) in zip(self.blocks[1:-1], self.blocks[:-2], self.keys[1:-1]):
            cur[Stages.START] = prev[Stages.ADD_ROUND_KEY].copy()
            cur[Stages.SUB_BYTES] = sub_bytes(cur[Stages.START])
            cur[Stages.SHIFT_ROWS] = shift_rows(cur[Stages.SUB_BYTES])
            cur[Stages.MIX_COLUMNS] = mix_columns(cur[Stages.SHIFT_ROWS])
            cur[Stages.ADD_ROUND_KEY] = add_round_key(cur[Stages.MIX_COLUMNS], key)

        cur, prev, key = self.blocks[N_ROUNDS], self.blocks[N_ROUNDS - 1], self.keys[N_ROUNDS]
        cur[Stages.START] = prev[Stages.ADD_ROUND_KEY].copy()
        cur[Stages.SUB_BYTES] = sub_bytes(cur[Stages.START])
        cur[Stages.SHIFT_ROWS] = shift_rows(cur[Stages.SUB_BYTES])
        cur[Stages.MIX_COLUMNS] = cur[Stages.SHIFT_ROWS].copy()
        cur[Stages.ADD_ROUND_KEY] = add_round_key(cur[Stages.MIX_COLUMNS], key)

        return cur[-1].copy()

    def decrypt(self, block):
        """Performs AES inverse cipher algorithm.

        Parameters
        ----------
        block : np.ndarray
            Cipher 4x4 column major block matrix.

        Returns
        -------
        np.ndarray
            Plain 4x4 column major block matrix.

        """
        cur, key = self.blocks[N_ROUNDS], self.keys[N_ROUNDS]
        cur[Stages.START] = np.copy(block)
        cur[Stages.INV_SHIFT_ROWS] = np.copy(block)
        cur[Stages.INV_SUB_BYTES] = np.copy(block)
        cur[Stages.INV_ADD_ROUND_KEY] = add_round_key(block, key)
        cur[Stages.INV_MIX_COLUMNS] = np.copy(cur[Stages.INV_ADD_ROUND_KEY])

        for (cur, prev, key) in zip(self.blocks[-2::-1], self.blocks[:0:-1], self.keys[-2::-1]):
            cur[Stages.START] = prev[Stages.INV_MIX_COLUMNS].copy()
            cur[Stages.INV_SHIFT_ROWS] = inv_shift_rows(cur[Stages.START])
            cur[Stages.INV_SUB_BYTES] = inv_sub_bytes(cur[Stages.INV_SHIFT_ROWS])
            cur[Stages.INV_ADD_ROUND_KEY] = add_round_key(cur[Stages.INV_SUB_BYTES], key)
            cur[Stages.INV_MIX_COLUMNS] = inv_mix_columns(cur[Stages.INV_ADD_ROUND_KEY])

        cur, prev, key = self.blocks[0], self.blocks[1], self.keys[0]
        cur[Stages.START] = prev[Stages.INV_MIX_COLUMNS].copy()
        cur[Stages.INV_SHIFT_ROWS] = inv_shift_rows(cur[Stages.START])
        cur[Stages.INV_SUB_BYTES] = inv_sub_bytes(cur[Stages.INV_SHIFT_ROWS])
        cur[Stages.INV_ADD_ROUND_KEY] = add_round_key(cur[Stages.INV_SUB_BYTES], key)
        cur[Stages.INV_MIX_COLUMNS] = cur[Stages.INV_ADD_ROUND_KEY].copy()

        return cur[-1].copy()
