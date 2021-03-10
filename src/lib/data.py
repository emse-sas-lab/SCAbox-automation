"""Python-agnostic SCA data parsing, logging and reporting module.

This module is designed as a class library providing *entity classes*
to represent the side channel data acquired from SoC.

It provides binary data parsing in order to read
acquisition data from a serial sources or a binary file and
retrieve it in the entity classes.

All the entity classes of the module provide CSV support
to allow processing acquisition data without parsing.
It also provides formatting data in a more human-readable format.

"""

import csv
import math
import os
from collections.abc import *
from enum import Enum
from warnings import warn

from lib.cpa import Handler


class Serializable:
    def write_csv(self, path: str, append: bool) -> None:
        pass


class Deserializable:
    def read_csv(self, path: str, count: int, start: int) -> None:
        pass


class Channel(MutableSequence, Reversible, Sized, Serializable, Deserializable):
    """Encryption channel data.

    This class is designed to represent AES 128 encryption data
    for each trace acquired.

    data are represented as 32-bit hexadecimal strings
    in order to accelerate IO operations on the encrypted block.

    Attributes
    ----------
    plains: list[str]
        Hexadecimal plain data of each trace.
    ciphers: list[str]
        Hexadecimal cipher data of each trace.
    keys: list[str]
        Hexadecimal key data of each trace.

    Raises
    ------
    ValueError
        If the three list attributes does not have the same length.

    """

    STR_MAX_LINES = 32

    def __init__(self, path=None, count=None, start=0):
        """Imports encryption data from CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV to read.
        count : int, optional
            Count of rows to read.
        start: int, optional
            Index of first row to read.
        Returns
        -------
        Channel
            Imported encryption data.
        """
        self.plains = []
        self.ciphers = []
        self.keys = []

        if not path:
            return

        self.read_csv(path, count, start)

    def __getitem__(self, item):
        return self.plains[item], self.ciphers[item], self.keys[item]

    def __setitem__(self, item, value):
        plain, cipher, key = value
        self.plains[item] = plain
        self.ciphers[item] = cipher
        self.keys[item] = key

    def __delitem__(self, item):
        del self.plains[item]
        del self.ciphers[item]
        del self.keys[item]

    def __len__(self):
        return len(self.plains)

    def __iter__(self):
        return zip(self.plains, self.ciphers, self.keys)

    def __reversed__(self):
        return zip(reversed(self.plains), reversed(self.ciphers), reversed(self.keys))

    def __repr__(self):
        return f"{type(self).__name__}({self.plains!r}, {self.ciphers!r}, {self.keys!r})"

    def __str__(self):
        n = len(self.plains[0]) + 4
        ret = f"{'plains':<{n}s}{'ciphers':<{n}s}{'keys':<{n}s}"
        for d, (plain, cipher, key) in enumerate(self):
            if d == Channel.STR_MAX_LINES:
                return ret + f"\n{len(self) - d} more..."
            ret += f"\n{plain:<{n}s}{cipher:<{n}s}{key:<{n}s}"
        return ret

    def __iadd__(self, other):
        self.plains += other.plains
        self.ciphers += other.ciphers
        self.keys += other.keys
        return self

    def clear(self):
        """Clears all the attributes.

        """
        self.plains.clear()
        self.ciphers.clear()
        self.keys.clear()

    def append(self, item):
        plain, cipher, key = item
        self.plains.append(plain)
        self.ciphers.append(cipher)
        self.keys.append(key)

    def pop(self, **kwargs):
        self.plains.pop()
        self.ciphers.pop()
        if len(self.keys) > 1 and len(self.keys) == len(self.plains):
            self.keys.pop()

    def insert(self, index: int, item):
        plain, cipher, key = item
        self.plains.insert(index, plain)
        self.ciphers.insert(index, cipher)
        self.keys.insert(index, key)

    def write_csv(self, path, append=False):
        """Exports encryption data to a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file to write.
        append : bool, optional
            True to append the data to an existing file.
        """
        try:
            file = open(path, "x+" if append else "w+", newline="")
            append = False
        except FileExistsError:
            file = open(path, "a+")
        writer = csv.DictWriter(file, [Keywords.PLAIN, Keywords.CIPHER, Keywords.KEY], delimiter=";")
        if not append:
            writer.writeheader()
        for plain, cipher, key in self:
            writer.writerow({Keywords.PLAIN: plain,
                             Keywords.CIPHER: cipher,
                             Keywords.KEY: key})
        file.close()

    def read_csv(self, path=None, count=None, start=0):
        count = count or math.inf
        with open(path, "r", newline="") as file:
            reader = csv.DictReader(file, delimiter=";")
            for d, row in enumerate(reader):
                if d < start:
                    continue
                if d >= count + start:
                    break
                self.plains.append(row[Keywords.PLAIN])
                self.ciphers.append(row[Keywords.CIPHER])
                self.keys.append(row[Keywords.KEY])

        if len(self.plains) != len(self.ciphers) or len(self.plains) != len(self.keys):
            raise ValueError("Inconsistent plains, cipher and keys length")


class Leak(MutableSequence, Reversible, Sized, Serializable, Deserializable):
    """Side-channel leakage data.

    This class represents the power consumption traces
    acquired during encryption in order to process these.

    Attributes
    ----------
    samples: list[int]
        Count of samples for each traces.
    traces: list[list[int]]
        Power consumption leakage signal for each acquisition.

    """

    STR_MAX_LINES = 32

    def __init__(self, path=None, count=None, start=0):
        """Imports leakage data from CSV.

        Parameters
        ----------
        path : str
            Path to the CSV to read.
        count : int, optional
            Count of rows to read.
        start: int, optional
            Index of first row to read.
        Returns
        -------
        Leak
            Imported leak data.
        """

        self.traces = []
        self.samples = []
        if not path:
            return
        self.read_csv(path, count, start)

    def __getitem__(self, item):
        return self.traces[item]

    def __setitem__(self, item, value):
        self.traces[item] = value
        self.samples[item] = len(value)

    def __delitem__(self, item):
        del self.traces[item]
        del self.samples[item]

    def __len__(self):
        return len(self.traces)

    def __iter__(self):
        return iter(self.traces)

    def __reversed__(self):
        return iter(reversed(self.traces))

    def __repr__(self):
        return f"{type(self).__name__}({self.traces!r})"

    def __str__(self):
        ret = f"{'no.':<16s}traces"
        for d, trace in enumerate(self):
            if d == Leak.STR_MAX_LINES:
                return ret + f"\n{len(self) - d} more..."
            ret += f"\n{d:<16d}"
            for t in trace:
                ret += f"{t:<4d}"
        return ret

    def __iadd__(self, other):
        self.traces += other.traces
        self.samples += other.samples
        return self

    def clear(self):
        self.traces.clear()
        self.samples.clear()

    def append(self, item):
        self.traces.append(item)
        self.samples.append(len(item))

    def pop(self, **kwargs):
        self.traces.pop()
        self.samples.pop()

    def insert(self, index, item):
        self.traces.insert(index, item)
        self.samples.insert(index, len(item))

    def write_csv(self, path, append=False):
        """Exports leakage data to CSV.

        Parameters
        ----------
        path : str
            Path to the CSV file to write.
        append : bool, optional
            True to append the data to an existing file.
        """
        with open(path, "a+" if append else "w+", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerows(self.traces)

    def read_csv(self, path, count=None, start=0):
        count = count or math.inf
        with open(path, "r", newline="") as file:
            reader = csv.reader(file, delimiter=";")
            for d, row in enumerate(reader):
                if d < start:
                    continue
                if d >= count + start:
                    break
                self.traces.append(list(map(lambda x: int(x), row)))
                self.samples.append(len(self.traces[-1]))


class Meta(Serializable, Deserializable):
    """Meta-data of acquisition.

   This class is designed to represent additional infos
   about the current side-channel acquisition run.

    Attributes
    ----------
    mode : str
        Encryption mode.
    direction : str
        Encryption direction, either encrypt or decrypt.
    target : int
        Sensors calibration target value.
    sensors : int
        Count of sensors.
    iterations : int
        Requested count of traces.
    offset : int
        If the traces are ASCII encoded, code offset
    """

    def __init__(self, path=None, count=None, start=0):
        """Imports meta-data from CSV file.

        If the file is empty returns an empty meta data object.

        Parameters
        ----------
        path : str
            Path to the CSV to read.

        Returns
        -------
        Meta
            Imported meta-data.
        """

        self.mode = None
        self.algo = None
        self.direction = None
        self.target = 0
        self.sensors = 0
        self.iterations = 0
        self.offset = 0

        if not path:
            return

        self.read_csv(path, count, start)

    def __repr__(self):
        return f"{type(self).__name__}(" \
               f"{self.mode!r}, " \
               f"{self.direction!r}, " \
               f"{self.target!r}, " \
               f"{self.sensors!r}, " \
               f"{self.iterations!r}, " \
               f"{self.offset!r})"

    def __str__(self):
        return f"{Keywords.MODE:<16}{self.mode}\n" \
               f"{Keywords.DIRECTION:<16}{self.direction}\n" \
               f"{Keywords.TARGET:<16}{self.target}\n" \
               f"{Keywords.SENSORS:<16}{self.sensors}\n" \
               f"{Keywords.ITERATIONS:<16}{self.iterations}\n" \
               f"{Keywords.OFFSET:<16}{self.offset}"

    def clear(self):
        """Resets meta-data.

        """
        self.mode = None
        self.algo = None
        self.direction = None
        self.target = 0
        self.sensors = 0
        self.iterations = 0
        self.offset = 0

    def write_csv(self, path, append=False):
        """Exports meta-data to a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file to write.
        append : bool, optional
        """
        try:
            file = open(path, "x+" if append else "w+", newline="")
        except FileExistsError:
            meta = Meta(path)
            self.iterations += meta.iterations
            file = open(path, "w+", newline="")

        fieldnames = [Keywords.MODE,
                      Keywords.DIRECTION,
                      Keywords.TARGET,
                      Keywords.SENSORS,
                      Keywords.ITERATIONS,
                      Keywords.OFFSET]
        writer = csv.DictWriter(file, fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerow({Keywords.MODE: self.mode,
                         Keywords.DIRECTION: self.direction,
                         Keywords.TARGET: self.target,
                         Keywords.SENSORS: self.sensors,
                         Keywords.ITERATIONS: self.iterations,
                         Keywords.OFFSET: self.offset})
        file.close()

    def read_csv(self, path, count=None, start=0):
        with open(path, "r", newline="") as file:
            reader = csv.DictReader(file, delimiter=";")
            try:
                row = next(reader)
            except StopIteration:
                return
            self.mode = row[Keywords.MODE]          
            self.direction = row[Keywords.DIRECTION]
            self.target = int(row[Keywords.TARGET])
            self.sensors = int(row[Keywords.SENSORS])
            self.iterations = int(row[Keywords.ITERATIONS])
            self.offset = int(row[Keywords.OFFSET])

            count = count or math.inf
            for d, row in enumerate(reader):
                if d < start:
                    continue
                if d >= count + start:
                    break
                self.iterations += int(row[Keywords.ITERATIONS])


class Request:
    """Data processing request.

    This class provides a simple abstraction to wrap
    file naming arguments during acquisition, import or export.

    The these arguments combined together form a *data request*
    specifying all the characteristics of the target data-set.

    Attributes
    ----------
    target : str
        Serial port id or file prefix according to the source mode.
    iterations : int
        Requested count of traces.
    mode : str
        Encryption mode.
    direction : str
        Encrypt direction.
    source : str
        Source mode.
    verbose : True
        True to perform verbose acquisition.
    chunks : int, optional
        Count of chunks to acquire or None if not performing chunk acquisition
    """

    ACQ_CMD_NAME = "sca"

    class Algos(Enum):
        AES = "aes"
        PRESENT = "present"
        KLEIN = "klein"
        CRYPTON = "crypton"
		
    class Modes(Enum):
        HARDWARE = "hw"
        TINY = "tiny"
        OPENSSL = "ssl"
        DHUERTAS = "dhuertas"
        PRESENT = "present"
        KLEIN = "klein"
        CRYPTON = "crypton"
		
    class Directions(Enum):
        ENCRYPT = "enc"
        DECRYPT = "dec"

    def __init__(self, args=None):
        """Initializes a request with a previously parsed command.

        Parameters
        ----------
        args
            Parsed arguments.

        """
        self.target = ""
        self.iterations = 0
        self.mode = Request.Modes.HARDWARE
        self.algo = Request.Algos.AES
        self.model = Handler.Models.SBOX_R0
        self.direction = Request.Directions.ENCRYPT
        self.verbose = False
        self.noise = False
        self.chunks = None
        self.start = None
        self.end = None
        self.path = ""

        if type(args) == dict:
            self._from_dict(args)
        else:
            self._from_namespace(args)

    def _from_namespace(self, args):
        if hasattr(args, "iterations"):
            self.iterations = args.iterations
        if hasattr(args, "target"):
            self.target = args.target
        if hasattr(args, "mode"):
            self.mode = next(filter(lambda e: args.mode == e.name, Request.Modes))
            self._set_algo(self.mode)
        if hasattr(args, "model"):
            self.model = next(filter(lambda e: args.model == e.name, Handler.Models))
        if hasattr(args, "direction"):
            self.direction = args.direction
        if hasattr(args, "verbose"):
            self.verbose = args.verbose
        if hasattr(args, "chunks"):
            self.chunks = args.chunks
        if hasattr(args, "noise"):
            self.noise = args.noise
        if hasattr(args, "start"):
            self.start = args.start
        if hasattr(args, "end"):
            self.end = args.end
        if hasattr(args, "path"):
            self.path = args.path

    def _from_dict(self, args):
        if "iterations" in args:
            self.iterations = args["iterations"]
        if "target" in args:
            self.target = args["target"]
        if "mode" in args:
            self.mode = args["mode"]
            self._set_algo(self.mode)
        if "model" in args:
            self.model = args["model"]
        if "direction" in args:
            self.direction = args["direction"]
        if "verbose" in args:
            self.verbose = args["verbose"]
        if "chunks" in args:
            self.chunks = args["chunks"]
        if "noise" in args:
            self.noise = args["noise"]
        if "start" in args:
            self.start = args["start"]
        if "end" in args:
            self.end = args["end"]
        if "path" in args:
            self.path = args["path"]

    def _set_algo(self, mode):

        if(mode == Request.Modes.HARDWARE):
            self.algo = Request.Algos.AES
        elif(mode == Request.Modes.TINY):
            self.algo = Request.Algos.AES
        elif(mode == Request.Modes.OPENSSL):
            self.algo = Request.Algos.AES
        elif(mode == Request.Modes.DHUERTAS):
            self.algo = Request.Algos.AES
        elif(mode == Request.Modes.PRESENT):
            self.algo = Request.Algos.PRESENT
        elif(mode == Request.Modes.KLEIN):
            self.algo = Request.Algos.KLEIN
        elif(mode == Request.Modes.CRYPTON):
            self.algo = Request.Algos.CRYPTON
        else:
            print("unknown mode")


    def __repr__(self):
        return f"{type(self).__name__}" \
               f"({self.target!r}, " \
               f"{self.iterations!r}, " \
               f"{self.mode!r}, " \
               f"{self.model!r}, " \
               f"{self.direction!r}, " \
               f"{self.verbose!r}, " \
               f"{self.noise!r}, " \
               f"{self.start!r}, " \
               f"{self.end!r}, " \
               f"{self.chunks!r})"

    def __str__(self):
        return f"{'target':<16}{self.target}\n" \
               f"{'iterations':<16}{self.iterations}\n" \
               f"{'mode':<16}{self.mode.name}\n" \
               f"{'model':<16}{self.model.name}\n" \
               f"{'direction':<16}{self.direction.name}\n" \
               f"{'verbose':<16}{self.verbose}\n" \
               f"{'start':<16}{self.start}\n" \
               f"{'end':<16}{self.end}\n" \
               f"{'chunks':<16}{self.chunks}\n" \
               f"{'path':<16}{os.path.abspath(self.path)}"

    def filename(self, prefix=None, suffix=""):
        """Creates a filename based on the request.

        This method allows to consistently name the files
        according to request's characteristics.

        Parameters
        ----------
        prefix : str
            Prefix of the filename.
        suffix : str, optional
            Suffix of the filename.

        Returns
        -------
        str :
            The complete filename.

        """
        iterations = self.iterations * (self.chunks or 1)
        return f"{prefix or self.target.split(os.sep)[-1]}_{self.mode}_{self.direction}_{iterations}{suffix}"

    def command(self, name):
        return f"{name}" \
               f" -t {self.iterations}" \
               f" -m {self.mode.value}" \
               f"{' -i' if self.direction == Request.Directions.DECRYPT else ''}" \
               f"{' -r' if self.noise else ''}" \
               f"{' -v' if self.verbose else ''}" \
               f"{f' -s {self.start}' if self.start else ''}" \
               f"{f' -e {self.end}' if self.end else ''}"

    @property
    def total(self):
        return self.iterations if self.chunks is None else self.iterations * self.chunks

    def requested(self, chunk=None):
        return self.iterations if chunk is None else chunk * self.iterations


class Keywords:
    """Iterates over the binary log keywords.

    This iterator allows to represent the keywords and
    the order in which they appear in the binary log consistently.

    This feature is useful when parsing log lines in order
    to avoid inserting a sequence that came in the wrong order.

    At each iteration the next expected keyword is returned.

    ``meta`` attribute allows to avoid expect again meta keywords
    when an error occurs and the iterator must be reset.

    Attributes
    ----------
    idx: int
        Current keyword index.
    meta: bool
        True if the meta-data keyword have already been browsed.
    inv: bool
        True if the keywords follow the inverse encryption sequence.
    metawords: str
        Keywords displayed once at the beginning of acquisition.
    datawords: str
        Keywords displayed recurrently during each trace acquisition.
    value: str
        Value of the current keyword.

    """

    MODE = "mode"
    DIRECTION = "direction"
    SENSORS = "sensors"
    TARGET = "target"
    KEY = "keys"
    PLAIN = "plains"
    CIPHER = "ciphers"
    SAMPLES = "samples"
    CODE = "code"
    WEIGHTS = "weights"
    OFFSET = "offset"
    ITERATIONS = "iterations"

    DELIMITER = b":"
    START_RAW_TAG = b"\xfd\xfd\xfd\xfd"
    START_TRACE_TAG = b"\xfe\xfe\xfe\xfe"
    END_ACQ_TAG = b"\xff\xff\xff\xff"
    END_LINE_TAG = b";;\n"

    def __init__(self, meta=False, inv=False, verbose=False, noise=False):
        """Initializes a new keyword iterator.

        Parameters
        ----------
        meta : bool
            If true, meta-data keywords will be ignored.
        inv: bool
            True if the keywords follow the inverse encryption sequence.
        """
        self.idx = 0
        self.meta = meta
        self.value = ""
        self.__build_metawords()
        self.__build_datawords(inv, verbose, noise)
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        current = self.value
        if self.meta:
            self.idx = (self.idx + 1) % len(self.datawords)
            self.value = self.datawords[self.idx]
            return current

        if self.idx < len(self.metawords):
            self.idx += 1

        if self.idx == len(self.metawords):
            self.reset(meta=True)
            return current

        self.value = self.metawords[self.idx]
        return current

    def reset(self, meta=None):
        self.idx = 0
        self.meta = meta or self.meta
        self.value = self.datawords[0] if self.meta else self.metawords[0]

    def __build_metawords(self):
        self.metawords = [Keywords.SENSORS, Keywords.TARGET, Keywords.MODE, Keywords.DIRECTION, Keywords.KEY]

    def __build_datawords(self, inv, verbose, noise):
        keywords = [Keywords.SAMPLES, Keywords.WEIGHTS] if verbose else [Keywords.SAMPLES, Keywords.CODE]
        datawords = [Keywords.CIPHER, Keywords.PLAIN] if inv else [Keywords.PLAIN, Keywords.CIPHER]
        if noise:
            self.datawords = keywords + datawords
        else:
            self.datawords = datawords
        self.datawords += keywords

    def all(self):
        return self.metawords + self.datawords


class Parser:
    """Binary data parser.

    This class is designed to parse binary acquisition data
    and store parsed data in the entity classes in order
    to later import and export it.

    Attributes
    ----------
    leak : Leak
        Leakage data.
    channel : Channel
        Encryption data.
    meta : Meta
        Meta-data.

    See Also
    --------
    Keywords : iterator representing the keyword sequence

    """

    def __init__(self, s=b"", direction=Request.Directions.ENCRYPT, noise=False, verbose=False):
        """Initializes an object with binary data.

        Parameters
        ----------
        s : bytes
            Binary data.
        direction : str
            Encryption direction.
        Returns
        -------
        Parser
            Parser initialized with data.

        """
        self.leak = Leak()
        self.noise = Leak()
        self.channel = Channel()
        self.meta = Meta()
        self.parse(s, direction=direction, noise=noise, verbose=verbose)

    def pop(self):
        """Pops acquired value until data lengths matches.

        This method allows to guarantee that the data contained
        in the parser will always have the same length which is
        the number of traces parsed.

        Returns
        -------
        Parser
            Reference to self.
        """
        lens = list(map(len, [
            self.channel.plains, self.channel.ciphers, self.leak.samples, self.leak.traces,
            self.noise.samples or self.leak.samples, self.noise.traces or self.leak.samples
        ]))
        n_min = min(lens)
        n_max = max(lens)

        if n_max == n_min and n_max != 0:
            if len(self.noise.samples) > 0 or len(self.noise.traces) > 0:
                self.noise.pop()
            self.leak.pop()
            self.channel.pop()
            self.meta.iterations -= 1
            return
        elif n_max == 0:
            return

        Parser.__pop_until(self.leak.samples, n_min)
        Parser.__pop_until(self.leak.traces, n_min)

        Parser.__pop_until(self.noise.samples, n_min)
        Parser.__pop_until(self.noise.traces, n_min)

        Parser.__pop_until(self.channel.keys, n_min)
        Parser.__pop_until(self.channel.plains, n_min)
        Parser.__pop_until(self.channel.ciphers, n_min)

        return self

    def clear(self):
        """Clears all the parser data.

        """
        self.noise.clear()
        self.leak.clear()
        self.meta.clear()
        self.channel.clear()

    def parse(self, s, direction=Request.Directions.ENCRYPT, noise=False, verbose=False, warns=False):
        """Parses the given bytes to retrieve acquisition data.

        If inv`` is not specified the parser will infer the
        encryption direction from ``s``.

        Parameters
        ----------
        s : bytes
            Binary data.
        noise : bool
            True if noise traces must be acquired before crypto-traces.
        direction : str
            Encryption direction.

        Returns
        -------
        Parser
            Reference to self.
        """
        keywords = Keywords(inv=direction == Request.Directions.DECRYPT, noise=noise, verbose=verbose)
        expected = next(keywords)
        valid = True
        noised = False
        lines = s.replace(b"\r\n", b"\n").split(Keywords.END_LINE_TAG)
        for idx, line in enumerate(lines):
            if line in (Keywords.END_ACQ_TAG, Keywords.START_TRACE_TAG, Keywords.START_RAW_TAG):
                if valid is False:
                    valid = line == (Keywords.START_RAW_TAG if noise else Keywords.START_TRACE_TAG)
                if noise:
                    noised = (line == Keywords.START_RAW_TAG) or not (line == Keywords.START_TRACE_TAG)
                continue
            try:
                if self.__parse_line(line, expected, keywords, noised):
                    expected = next(keywords)
            except (ValueError, UnicodeDecodeError, RuntimeError) as e:
                if warns:
                    warn(f"parsing error\nerror: {e}\niteration: {len(self.leak.traces)}\nline {idx}")
                keywords.reset(keywords.meta)
                expected = next(keywords)
                valid = False
                self.pop()

        if len(self.channel.keys) == 1:
            self.channel.keys = [self.channel.keys[0]] * len(self.channel)
        self.meta.iterations += len(self.channel)
        return self

    def __parse_line(self, line, expected, keywords, noise):
        split = line.strip().split(Keywords.DELIMITER)
        try:
            keyword = str(split[0], "ascii").strip().split(" ")[-1]
            data = split[1].strip()
        except IndexError:
            return False

        if keyword in keywords.all() and keyword != expected:
            raise RuntimeError(f"expected {expected} keyword not {keyword}")

        leak = self.noise if noise else self.leak
        if keyword in (Keywords.MODE, Keywords.DIRECTION):
            setattr(self.meta, keyword, str(data, "ascii"))
        elif keyword in (Keywords.SENSORS, Keywords.TARGET):
            setattr(self.meta, keyword, int(data))
        elif keyword in (Keywords.KEY, Keywords.PLAIN, Keywords.CIPHER):
            getattr(self.channel, keyword).append(f"{int(data.replace(b' ', b''), 16):032x}")
        elif keyword == Keywords.SAMPLES:
            leak.samples.append(int(data))
        elif keyword == Keywords.CODE:
            leak.traces.append(list(map(int, line[(len(Keywords.CODE) + 2):])))
        elif keyword == Keywords.WEIGHTS:
            leak.traces.append(list(map(int, data.strip().split(b","))))
        else:
            return False

        if keyword == Keywords.TARGET:
            self.meta.offset = self.meta.sensors * self.meta.target

        if keyword in (Keywords.CODE, Keywords.WEIGHTS):
            n = leak.samples[-1]
            m = len(leak.traces[-1])
            if m != n:
                raise RuntimeError(f"trace lengths mismatch {m} != {n}")

        return True

    @classmethod
    def __pop_until(cls, data, n):
        while len(data) > n:
            data.pop()
