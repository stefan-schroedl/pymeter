import math
import copy
import statistics
from copy import deepcopy
from abc import ABC, abstractmethod
from collections import deque
from time import process_time, perf_counter
import numpy as np
import torch

_INF = float("inf")
_NAN = float("nan")


class AbstractMeter(ABC):
    """
    Incrementally aggregates a stream of numbers for lazy, constant-memory
    calculation of distributional statistics.
    Also works for sequences of numpy arrays and torch tensors. The non-scalar
    case is (almost) equivalent to feeding each element consecutively.
    """

    def __init__(self, data=None, cutoff_small=1e-5):
        """
        Args:
            data (optional): Input data to initialize the `Meter`
            cutoff_small (float): Threshold for counting 'small' values
        """
        self.cutoff_small = cutoff_small
        self.reset()

        if data is not None:
            self.update(data)

    def as_dict(self, keys=['count', 'min', 'max', 'mean', 'std']):
        return {key: getattr(self, key) for key in keys}

    def __repr__(self):
        return f'{self.__class__.__name__}{str(self.as_dict())})'

    def update(self, obj, n=1):
        """
        Aggregate an object based on type
        """

        if isinstance(obj, AbstractMeter):
            return self.update_meter(obj, 1)
        elif isinstance(obj, (int, float)):
            return self.update_scalar(obj, n)
        elif isinstance(obj, np.ndarray):
            return self.update_numpy(obj, n)
        elif torch.is_tensor(obj):
            return self.update_torch(obj, n)
        else:
            raise TypeError(f'wrong type in Meter.update(): {obj}')

    def __iadd__(self, x):
        return self.update(x)

    def __isub__(self, x):
        return self.update(-x)

    def __add__(self, other):
        if not isinstance(other, AbstractMeter):
            raise TypeError(f'expected Meter, got {other}')
        return copy.deepcopy(self).update(other)

    def __radd__(self, other):
        if not isinstance(other, AbstractMeter):
            raise TypeError(f'expected Meter, got {other}')
        return copy.deepcopy(self).update(other)

    @abstractmethod
    def reset(self):
        """
        Reset all stats
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Number of elements consumed
        """
        pass

    @abstractmethod
    def update_scalar(self, x, n=1):
        """
        Consume a native Python number.

        Args:
          x (int, float): The number
          n (int): Add the number that many times
        """
        pass

    def update_torch(self, x, n=1):
        """
        Consume a pytorch tensor. Can reduce to update_numpy
        """
        return self.update_numpy(x.cpu().numpy())

    @abstractmethod
    def update_numpy(self, x, n=1):
        """
        Consume a numpy array
        """
        pass

    @abstractmethod
    def update_meter(self, other, n=1):
        """
        Aggregate metrics from another Meter instance
        """
        pass


class Meter(AbstractMeter):

    def reset(self):
        self.count = 0        # total number of elements
        self.count_zero = 0   # total number of zero elements
        self.count_small = 0  # total number of (absolute) small elements
        self.last = _NAN      # most recent value
        self.min = _INF       # minimum number observed
        self.max = -_INF      # maximum number observed
        self.amin = _INF      # absolute minimum number observed
        self.amax = -_INF     # absolute maximum number observed
        self.sum = 0          # total sums
        self.sum2 = 0         # total sum of squares


    def __len__(self):
        return self.count


    def update_scalar(self, x, n=1):
        self.last = x
        self.count += n
        if x == 0.0:
            self.count_zero += n
        x_abs = abs(x)
        if x_abs < self.cutoff_small:
            self.count_small += n
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        self.amin = min(self.amin, x_abs)
        self.amax = max(self.amax, x_abs)
        self.sum += x * n
        self.sum2 += x * x * n

        return self


    def update_numpy(self, x, n=1):

        count = x.size
        count_nonzero = np.count_nonzero(x)
        x_abs = np.abs(x)
        count_small = (x_abs < self.cutoff_small).sum().item()
        count_zero = count - count_nonzero
        s = x.sum().item()

        self.count += count * n
        self.count_zero += count_zero * n
        self.count_small += count_small * n
        self.min = min(self.min, x.min().item())
        self.max = max(self.max, x.max().item())
        self.amin = min(self.amin, x_abs.min().item())
        self.amax = max(self.amax, x_abs.max().item())
        self.sum += s * n
        self.sum2 += (x * x).sum().item() * n

        # arguable: we don't want to store the whole tensor in self.last
        if count > 0:
            self.last = s / count

        return self


    def update_meter(self, other, n=1):

        self.last = other.last
        self.count += other.count * n
        self.count_zero += other.count_zero
        if other.cutoff_small != self.cutoff_small:
            # Inconsistent thresholds, invalidate
            self.count_small = _NAN
        else:
            self.count_small += other.count_small * n
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        self.amin = min(self.amin, other.amin)
        self.amax = max(self.amax, other.amax)
        self.sum += other.sum * n
        self.sum2 += other.sum2 * n

        return self


    @property
    def mean(self):
        """
        Average value of sequence
        """
        try:
            return self.sum / len(self)
        except ZeroDivisionError:
            return _NAN


    @property
    def zero_frac(self):
        """
        Fraction of sequence values that are zero
        """
        try:
            return self.count_zero / len(self)
        except ZeroDivisionError:
            return _NAN


    @property
    def small_frac(self):
        """
        Fraction of absolute sequence values below configured threshold
        """
        try:
            return self.count_small / len(self)
        except ZeroDivisionError:
            return _NAN


    @property
    def std(self):
        try:
            return math.sqrt(max(0.0, (self.sum2 - self.sum * self.sum / self.count)) / (self.count - 1))
        except Exception:
            return _NAN


class MovingWindowMeter(AbstractMeter):
    """
    Computes and stores the average, standard deviation, min, max, and current value of a
    moving window on a sequence of numbers, numpy arrays, or torch tensors. When the capacity
    is reached, each update removes the oldest datum.

    Uses O(n) space.
    """

    def __init__(self, capacity, cutoff_small=1e-5):
        """
        Args:
          capacity (int): Number of observations in window
          cutoff_small (float): Threshold for counting 'small' values
        """

        self.capacity = capacity
        self.cutoff_small = cutoff_small
        self.reset()


    def reset(self):
        self.q = deque(maxlen=self.capacity)


    def __len__(self):
        return len(self.q)


    def update_scalar(self, x, n=1):
        self.q.extend([x] * n)
        return self


    def update_numpy(self, x, n=1):
        raise NotImplementedError


    def update_meter(self, other, n=1):
        """
        Add stats from other MovingWindowMeter
        """

        if not isinstance(other, MovingWindowMeter):
            raise TypeError(f'wrong type in MovingWindowMeter.update(): {other}')

        self.q.extend(x for x in list(other.q) for i in range(n))

        return self


    @property
    def count(self):
        return len(self.q)


    @property
    def count_zero(self):
        return len(self.q) - np.count_nonzero(self.q)


    @property
    def count_small(self):
        return len([x for x in self.q if x < self.cutoff_small])


    @property
    def last(self):
        if len(self.q) == 0:
            return _NAN
        return self.q[-1]


    @property
    def min(self):
        if len(self.q) == 0:
            return _NAN
        return min(self.q)


    @property
    def max(self):
        if len(self.q) == 0:
            return _NAN
        return max(self.q)


    @property
    def amax(self):
        if len(self.q) == 0:
            return -_INF
        return np.abs(self.q).max()


    @property
    def amin(self):
        if len(self.q) == 0:
            return _INF
        return np.abs(self.q).min()


    @property
    def sum(self):
        if len(self.q) == 0:
            return _NAN
        return sum(self.q)


    @property
    def sum2(self):
        if len(self.q) == 0:
            return _NAN
        return sum([x * x for x in self.q])


    @property
    def mean(self):
        """
        Average value of sequence
        """
        try:
            return self.sum / len(self)
        except ZeroDivisionError:
            return _NAN


    @property
    def std(self):
        if self.count < 2:
            return _NAN
        return statistics.stdev(self.q)


    @property
    def median(self):
        if len(self.q) == 0:
            return _NAN
        return statistics.median(self.q)


    @property
    def mode(self):
        try:
            return statistics.mode(self.q)
        except Exception:
            return _NAN


    def is_complete(self):
        return len(self.q) >= self.capacity


class MeterDict(dict):

    """
    A default dictionary whose values are Meters
    """

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self.update(k, v)


    def __getitem__(self, k):
        """
        Override method to defaultdict-like behavior
        """

        if k not in self.keys():
            super().__setitem__(k, Meter())
        return super().__getitem__(k)


    def __setitem__(self, k, obj):

        if isinstance(obj, AbstractMeter):
            super().__setitem__(k, obj)
        else:
            super().__setitem__(k, Meter())
            self[k].update(obj)


    def __repr__(self):
        s = 'MeterDict(\n'
        for k, v in self.items():
            s += '  * ' + str(k) + '\n     ' + str(v) + '\n'
        s += ')'
        return s


    def update(self, key, val=None, n=1):
        """
        Aggregate a value, a meter, or another MeterDict.
        Usage:
           - md.update(md2) aggregate another MeterDict
           - md.update('key', 'value') aggregate Meter md['key'] with 'value'

        """
        if key is None:
            raise ValueError('key cannot be None')
        if val is None:
            if isinstance(key, (dict, MeterDict)):
                # add stats from other object
                for name in key.keys():
                    self[name].update(key[name], n)
            else:
                raise TypeError(f'expected MeterDict, a dictionary, or both key and value; got {key}')
        else:
            self[key].update(val, n)

        return self


    def reset(self, key=None):
        if key is None:
            for v in self.values():
                v.reset()
        else:
            self[key].reset()


    def __iadd__(self, obj):
        return self.update(obj)


    def __add__(self, other):
        if not isinstance(other, (AbstractMeter, MeterDict)):
            # NOTE: + 0 is not a no-op, contrary to numbers
            # possible gotcha when using sum() function
            raise TypeError(f'expected Meter or MeterDict, got {other}')
        return copy.deepcopy(self).update(other)


    def __radd__(self, other):
        if not isinstance(other, (AbstractMeter, MeterDict)):
            raise TypeError(f'expected Meter or MeterDict, got {other}')
        return copy.deepcopy(self).update(other)


class ProgressFormatter:
    """
    Format several Meter stats in a joint string.
    """

    def __init__(self, sep='\t', default_fmt=':.4f', property='mean'):
        self.sep = sep
        self.meters = []
        self.default_fmt = default_fmt

        self.property = property

    def add_meter(self, meter, label, fmt=None):
        if not fmt:
            fmt = self.default_fmt
        self.meters.append((meter, label, fmt))

    def __str__(self):
        return self.sep.join([('{{}}: {{{}}}'.format(f)).format(l, getattr(m, self.property)) for (m, l, f) in self.meters])


# for global accumulation.
# Useful for timing without having to pass the meter around as additional
# function arguments

global_meter = MeterDict()


def get_global_meter(key=None):
    if key is None:
        return global_meter
    return global_meter[key]


class Timing:

    """
    Context manager to record timing stats

    Note: Extensible to multiple resource functions, currently supporting wall clock and cpu time
    """

    def __init__(self, name='', output_meter=None, count=1, from_start=False):

        if output_meter is None:
            output_meter = MeterDict()
        self.meter = output_meter
        if len(name) > 0 and name[-1] != '_':
            name += '_'
        self.name = name
        # functions to call at start and stop
        self.time_fct = {(name + 'wall'): perf_counter, (name + 'cpu'): process_time}
        self.count = count
        self.start_time = {}
        if from_start:
            self.reset()


    @property
    def wall(self):
        """
        Average wall clock time accumulated between start() and stop() calls, in sec
        """
        return self.meter[self.name + 'wall'].mean


    @property
    def running_wall(self):
        """
        Currently ticking wall clock
        """
        return self.durations()[self.name + 'wall']


    @property
    def cpu(self):
        """
        Average CPU time accumulated between start() and stop() calls, in sec
        """
        return self.meter[self.name + 'cpu'].mean


    def reset(self):
        for k in self.time_fct:
            self.meter[k].reset()


    def start(self):
        self.start_time = {k: f() for k, f in self.time_fct.items()}


    def stop(self):
        dur = self.durations()
        self.meter.update(key=dur, n=self.count)


    def durations(self):
        return {k: (f() - self.start_time[k]) for k, f in self.time_fct.items()}


    def __enter__(self):
        self.start()
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def __str__(self):
        return f'Timing(wall={self.wall},cpu={self.cpu})'


def timed(key=None, count=1, from_start=False, output_meter=None):
    """
    Decorator to collect and aggregate timing on function calls
    """

    if output_meter is None:
        output_meter = global_meter

    def decorator(fct, key=None):

        if key is None:
            key = fct.__name__

        def wrapper(*args, **kwargs):
            with Timing(name=key, output_meter=output_meter, count=count, from_start=from_start):
                return fct(*args, **kwargs)

        return wrapper

    if callable(key):
        # @timed ... without arguments
        return decorator(key)
    else:
        # @timed('bla', ...)
        return lambda fct: decorator(fct, key)
