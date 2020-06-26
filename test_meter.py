import torch

import math
import time
from pytest import approx, raises
import numpy as np
import torch

from time import process_time

from meter import Meter, MovingWindowMeter, MeterDict, timed, Timing, get_global_meter, ProgressFormatter


def test_init():

    d = {'a': Meter(), 'b': 1}
    m = MeterDict(**d)
    assert len(m) == 2
    assert isinstance(m['a'], Meter)
    assert isinstance(m['b'], Meter)
    assert m['a'].count == 0
    assert m['b'].count == 1

    d = {'a': Meter(), 'b': 'hello'}
    with raises(TypeError):
        m = MeterDict(**d)


def test_uninit():

    m = Meter()
    assert m.min == float('inf')
    assert m.max == - float('inf')
    assert m.count == 0
    assert math.isnan(m.mean)


def test_basic_and_dict():

    m = MeterDict()

    with raises(TypeError):
        m['a'] = 'hello'

    m['a'] = 1
    m['a'] += 2
    m['b'] += 1

    m2 = MeterDict()
    m2['a'] += 3
    m2['a'] += 4
    m2['c'] += 1

    m += m2

    with raises(TypeError):
        m['a'] += 'hello'

    assert len(m) == 3
    assert m['a'].count == 4
    assert m['a'].last == 4
    assert m['a'].min == 1
    assert m['a'].max == 4
    assert m['a'].mean == 2.5
    assert m['a'].std == approx(1.2909944)


def test_multiplier():

    m = Meter()
    m.update(3, n=10)
    assert m.count == 10
    assert m.mean == 3
    assert m.sum == 30


def test_timing():

    count = 10000000
    tt = time.time()
    tp = process_time()

    with Timing('test') as timing:
        a = 1
        for i in range(count):
            a += 1
        time.sleep(.5)

    expected_wall = time.time() - tt
    expected_cpu = process_time() - tp

    assert expected_wall == approx(timing.wall, abs=1e-2)
    assert expected_cpu == approx(timing.cpu, abs=1e-2)


def test_timed():

    # decorator without args
    @timed
    def wait0():
        time.sleep(1)

    wait0()
    wait0()
    meter = get_global_meter()

    assert meter['wait0_wall'].mean == approx(1, abs=1e-2)
    assert meter['wait0_cpu'].mean == approx(0, abs=1e-2)

    p = ProgressFormatter()
    p.add_meter(meter['wait0_wall'], 'system time')
    p.add_meter(meter['wait0_cpu'], 'process time')
    msg = str(p)
    assert msg == f'system time: {meter["wait0_wall"].mean:.4f}\tprocess time: {meter["wait0_cpu"].mean:.4f}'

    # decorator with args
    @timed('WAIT')
    def wait1():
        time.sleep(1)

    wait1()
    wait1()
    assert meter['WAIT_wall'].mean == approx(1, abs=1e-2)
    assert meter['WAIT_cpu'].mean == approx(0, abs=1e-2)


def test_moving():
    m = MovingWindowMeter(5)
    for x in [9.8, 1.5, -1.5, 0, 7, 3.8, 4.5, 2.222, -1, 0]:
        m += x

    assert len(m) == 5
    assert m.last == 0
    assert m.min == -1
    assert m.max == 4.5
    assert m.mean == 1.9044
    assert m.std == approx(2.3713702)


def test_moving_multiplier():

    m = MovingWindowMeter(5)
    m.update(3, n=4)
    assert m.count == 4
    assert m.mean == 3
    assert m.sum == 12


def test_moving_append():
    m = MovingWindowMeter(5)
    for x in [9.8, 1.5, -1.5, 0, 7, 3.8, 4.5, 2.222, -1, 0]:
        m += x

    m2 = MovingWindowMeter(10)
    for i in range(10):
        m2 += i
    m += m2
    assert len(m) == 5
    assert m.last == 9
    assert m.min == 5
    assert m.max == 9


def test_numpy():

    m = Meter()
    tensor1 = np.arange(10.)
    m += tensor1

    assert m.count == 10
    assert m.mean == 4.5
    assert m.min == 0
    assert m.max == 9

    tensor1 = np.zeros(5)
    tensor2 = np.ones(5)

    tensor12 = np.concatenate([tensor1, tensor2])

    m1 = Meter(cutoff_small=0.3)
    m2 = Meter(cutoff_small=0.3)

    m1 += tensor1
    m2 += tensor2

    # note: default argument necessary to avoid implicit 'addition' of 0
    m12 = sum([m1, m2], Meter(cutoff_small=0.3))

    assert m12.count == 10
    assert m12.mean == tensor12.mean()
    assert m12.std == tensor12.std(ddof=1)  # np default is without ddop, torch with it
    assert m12.min == 0
    assert m12.max == 1
    assert m12.zero_frac == 0.5
    assert m12.small_frac == 0.5


def test_torch():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    m = Meter()
    tensor1 = torch.arange(10.).to(device)
    m += tensor1

    assert m.count == 10
    assert m.mean == 4.5
    assert m.min == 0
    assert m.max == 9

    tensor1 = torch.zeros(5).to(device)
    tensor2 = torch.ones(5).to(device)

    tensor12 = torch.cat([tensor1, tensor2])

    m1 = Meter(cutoff_small=0.3)
    m2 = Meter(cutoff_small=0.3)

    m1 += tensor1
    m2 += tensor2

    # note: default argument necessary to avoid implicit 'addition' of 0
    m12 = sum([m1, m2], Meter(cutoff_small=0.3))

    assert m12.count == 10
    assert m12.mean == tensor12.mean()
    assert m12.std == tensor12.std()
    assert m12.min == 0
    assert m12.max == 1
    assert m12.zero_frac == 0.5
    assert m12.small_frac == 0.5
