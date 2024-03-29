import click
import ctypes
import enum
import itertools
import os
import queue
import signal
import sys
import threading
import time


class TimeoutTimer:
    """
    A convenient wrapper for triggering a TimeoutError after a given time.

    There should only be a single TimeoutTimer object at a given time.
    """

    def __init__(self, initial_timeout):
        self._initial_timeout = initial_timeout
        self._start_time = 0
        self._end_time = 0
        self._enabled = False

    def __enter__(self):
        """
        Start the timer.
        :return: This timer
        """

        def handler(signum, frame):
            raise TimeoutError()

        try:
            signal.signal(signal.SIGALRM, handler)

            if self._initial_timeout > 0:
                signal.setitimer(signal.ITIMER_REAL, self._initial_timeout)
                self._enabled = True
        except AttributeError:
            log(
                "Unable to use signals; timeout will be less effective",
                Verbosity.always,
            )
        self._start_time = time.time()
        self._end_time = self._start_time + self._initial_timeout
        return self

    def recap_timeout(self, new_timeout):
        """
        Set the new timeout of this Timer, measured from the start of the timer,
        if the new timeout would trigger sooner.

        :param new_timeout: The new timeout to set (0 indicates cancel)
        :return: None
        """
        if new_timeout == 0:
            self.cancel()
            return

        new_time_remaining = self._start_time + new_timeout - time.time()
        if new_time_remaining < 0:
            self.cancel()
            self._end_time = self._start_time + new_timeout
            raise TimeoutError()
        else:
            try:
                if signal.getitimer(signal.ITIMER_REAL)[0] > new_time_remaining:
                    signal.setitimer(signal.ITIMER_REAL, new_time_remaining)
                    self._enabled = True
            except AttributeError:
                pass
            self._end_time = self._start_time + new_timeout

    def reset_timeout(self, new_timeout):
        """
        Set the new timeout of this Timer, measured from the start of the timer.

        :param new_timeout: The new timeout to set (0 indicates cancel)
        :return: None
        """
        if new_timeout == 0:
            self.cancel()
            self._end_time = self._start_time + new_timeout
            return

        new_time_remaining = self._start_time + new_timeout - time.time()
        if new_time_remaining < 0:
            self.cancel()
            self._end_time = self._start_time + new_timeout
            raise TimeoutError()
        else:
            try:
                signal.setitimer(signal.ITIMER_REAL, new_time_remaining)
                self._enabled = True
            except AttributeError:
                pass
            self._end_time = self._start_time + new_timeout

    def __exit__(self, exit_type, value, traceback):
        """
        Cancel the timer.
        :return: None
        """
        self.cancel()

    def cancel(self):
        """
        Cancel the timer.
        :return: None
        """
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self._enabled = False
        except AttributeError:
            pass
        self._end_time = self._start_time

    def expired(self):
        return (time.time() > self._end_time) and self._enabled


class Stopwatch:
    """
    A stopwatch for easy measurement of elapsed time, optionally split into intervals.
    """

    def __init__(self):
        self.__start = time.time()
        self.__interval_start = self.__start
        self.__records = {}

    def record_interval(self, name):
        """
        Record the time elapsed since the end of the last interval.

        :param name: The name of the record to make
        :return: None
        """
        interval_end = time.time()
        self.__records[name] = interval_end - self.__interval_start
        self.__interval_start = interval_end

    def record_total(self, name):
        """
        Record the time elapsed since the creation of this stopwatch.

        :param name: The name of the record to make
        :return: None
        """
        self.__records[name] = time.time() - self.__start

    def elapsed_time(self):
        """
        Return the time elapsed since the creation of this stopwatch.
        :return: The time elapsed, in seconds
        """

        return time.time() - self.__start

    @property
    def records(self):
        return dict(self.__records)

    def report_times(self):
        for name, record in self.records.items():
            if name == "Total":
                output("-", Verbosity.plan_info)
            output_pair(
                name + " Time",
                record,
                Verbosity.always if name == "Total" else Verbosity.plan_info,
            )


class TypedChoice(click.Choice):
    """
    A modified version of click.Choice that allows the choice options to be arbitrary objects.
    The argument is compared against the string representation of each object; if it matches,
    then the object is returned.

    As with click.Choice, you should only pass a list or tuple of choices. Other iterables
    (like generators) may lead to surprising results.

    :param case_sensitive: Set to false to make choices case insensitive. Defaults to true.
    """

    name = "typedchoice"

    def __init__(self, choices, case_sensitive=True):
        self.object_choices = choices

        click.Choice.__init__(
            self, list(map(str, choices)), case_sensitive=case_sensitive
        )

    def convert(self, value, param, ctx):
        # Exact match
        if value in self.choices:
            return self.object_choices[self.choices.index(value)]

        # Match through normalization and case sensitivity
        # first do token_normalize_func, then lowercase
        # preserve original `value` to produce an accurate message in
        # `self.fail`
        normed_value = value
        normed_choices = self.choices

        if ctx is not None and ctx.token_normalize_func is not None:
            normed_value = ctx.token_normalize_func(value)
            normed_choices = [
                ctx.token_normalize_func(choice) for choice in self.choices
            ]

        if not self.case_sensitive:
            normed_value = normed_value.lower()
            normed_choices = [choice.lower() for choice in normed_choices]

        if normed_value in normed_choices:
            return self.object_choices[normed_choices.index(normed_value)]

        self.fail(
            "invalid choice: %s. (choose from %s)" % (value, ", ".join(self.choices)),
            param,
            ctx,
        )

    def __repr__(self):
        return "TypedChoice(%r)" % list(self.choices)


class TaggedChoice(click.Choice):
    """
    A modified version of click.Choice that allows the choice options to be provided as a
    dictionary. The argument is compared against the keys of the dictionary; if it matches,
    then the corresponding value is returned.

    :param case_sensitive: Set to false to make choices case insensitive. Defaults to true.
    """

    name = "taggedchoice"

    def __init__(self, options, case_sensitive=True):
        self.options = options

        click.Choice.__init__(self, list(options.keys()), case_sensitive=case_sensitive)

    def convert(self, value, param, ctx):
        # Exact match
        if value in self.options:
            return self.options[value]

        # Match through normalization and case sensitivity
        # first do token_normalize_func, then lowercase
        # preserve original `value` to produce an accurate message in
        # `self.fail`
        def normalize(val):
            if ctx is not None and ctx.token_normalize_func is not None:
                val = ctx.token_normalize_func(val)
            if not self.case_sensitive:
                val = val.lower()
            return val

        normalized_value = normalize(value)
        for key in self.options:
            if normalize(key) == normalized_value:
                return self.options[key]

        self.fail(
            "invalid choice: %s. (choose from %s)" % (value, ", ".join(self.options)),
            param,
            ctx,
        )

    def __repr__(self):
        return "TypedChoice(%r)" % list(self.choices)


class FileLocator:
    """
    A class to aid in the lookup of files that may or may not be in a Singularity image.

    Files in a Singularity image are relative to root.
    Other files are relative to the local directory.
    """

    def __getitem__(self, location):
        if os.path.exists(location):
            return location
        elif os.path.exists("/" + location):
            return "/" + location
        elif os.path.exists("../" + location):
            return "../" + location
        else:
            raise EnvironmentError("Unable to locate " + location)


class DimacsStream:
    """
    A class to aid in parsing of a DIMACS-style filestream.
    """

    def __init__(
        self,
        stream,
        comment_prefixes=frozenset({"c", "O"}),
        process_comment=lambda x: None,
    ):
        """
        :param stream: Input stream to parse.
        :param comment_prefixes: A set of characters of prefixes indicating a comment line.
        :param process_comment: A method to call on all comments discovered during the parse.
        """
        self.__stream = stream
        self.__comment_prefixes = comment_prefixes
        self.__process_comment = process_comment

    def parse_line(self, allowed_prefixes=None):
        """
        Locate and parse the next line of a DIMACS-style stream, ignoring comments.

        Raises a RuntimeError if this line has an unexpected prefix.

        :param allowed_prefixes: A set of characters of prefixes to allow.
        :return: A list of space-separated elements of the next line, or None if EOF.
        """
        for line in self.__stream:
            if len(line) == 0:
                continue
            elif line[0] in self.__comment_prefixes:
                self.__process_comment(line.rstrip())
                continue
            elif allowed_prefixes is None or line[0] in allowed_prefixes:
                return line.split()
            else:
                raise RuntimeError("Unexpected line prefix in: {0}".format(line))


class Verbosity(enum.IntEnum):
    always = 0
    stages = 1
    plan_info = 2
    progress = 3
    solver_output = 4
    debug = 5


output_verbosity = Verbosity.debug


def set_verbosity(verbosity):
    """
    Set the level of information to output, globally
    :param verbosity: 0 (minimal), 1, 2, 3, 4, 5 (everything)
    :return: None
    """
    global output_verbosity
    output_verbosity = verbosity


def output(arg, verbosity=Verbosity.debug):
    """
    Output the text to stdout according to the global log level

    :param arg: Text to output
    :param verbosity: 0 (always), 1, 2, 3, 4, 5 (debug only)
    :return: None
    """
    global output_verbosity
    if verbosity <= output_verbosity:
        print(arg)


def output_pair(key, value, verbosity=Verbosity.debug, flush=True):
    """
    Output the key/value pair to the global log level like "Key: Value"

    :param key: Key to output
    :param value: Value to output
    :param verbosity: 0 (always), 1, 2, 3, 4, 5 (debug only)
    :param flush: If true, flush stdout afterwards
    :return: None
    """
    global output_verbosity
    if verbosity <= output_verbosity:
        print(str(key) + ": " + str(value), flush=flush)


def log(arg, verbosity=Verbosity.debug, flush=True, **kwargs):
    """
    Output the text to stderr according to the global log level

    :param arg: Text to output
    :param verbosity: 0 (always), 1, 2, 3, 4, 5 (debug only)
    :param flush: If true, flush stderr afterwards
    :param kwargs: Other arguments, passed to stderr
    :return:
    """
    if verbosity <= output_verbosity:
        print(arg, flush=flush, file=sys.stderr, **kwargs)


def kill_on_crash(sig=None):
    """
    Ensure that the child process is killed if the parent exits (e.g. from a cython segfault).

    From https://stackoverflow.com/questions/320232/ensuring-subprocesses-are-dead-on-exiting-python-program
    """
    if sig is None:
        sig = signal.SIGKILL

    def do():
        libc = ctypes.CDLL("libc.so.6")
        return libc.prctl(1, sig)

    return do


class BufferedStream:
    """
    Buffer the output of the stream through a queue on a separate thread.

    An unbuffered process.stdout stream does not behave well with timeouts.
    """

    def __init__(self, stream, timer=None):
        self.__stream = stream
        self.__timer = timer
        self.__queue = queue.Queue()
        self.__finished = False

        def enqueue_output():
            for line in self.__stream:
                self.__queue.put(line)
            self.__stream.close()
            self.__finished = True

        self.__thread = threading.Thread(target=enqueue_output)
        self.__thread.daemon = True
        self.__thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                if self.__timer is not None and self.__timer.expired():
                    # If the timer does not successfully go off (i.e., Windows), trigger it here
                    raise TimeoutError()
                return self.__queue.get(block=True, timeout=1)
            except queue.Empty:
                if self.__finished:
                    raise StopIteration


class GroupedHelp(click.Command):
    """
    Add high-level grouping to help command output
    """

    def __init__(self, groups, **kwargs):
        click.Command.__init__(self, **kwargs)
        self.__groups = groups

    def get_help(self, ctx):
        help_text = click.Command.get_help(self, ctx)
        for indicator, group_name in self.__groups.items():
            argument = "  --" + indicator
            help_text = help_text.replace(
                argument, "\n" + group_name + ":\n" + argument
            )
        return help_text


def split_every(iterable, n):
    """
    Split the iterable into lists of size n.
    Note that the final iterable may be < n, if n does not evenly divide the number of elements.

    :param iterable: The iterable to split
    :param n: Size of groups
    :return: An iterator that yields lists of size <= n.
    """
    i = iter(iterable)
    piece = list(itertools.islice(i, n))
    while piece:
        yield piece
        piece = list(itertools.islice(i, n))


def normalize_TPU_addr(addr):
    """
    Ensure that a TPU addr always has the form grpc://.*:8470
    :param addr:
    :return:
    """
    if not addr.startswith("grpc://"):
        addr = "grpc://" + addr
    if not addr.endswith(":8470"):
        addr = addr + ":8470"
    return addr
