# PyMeter
Simple utilities for stream statistics and timing.

Examples:

```
m = MeterDict()
m['a'] = 1
m['b'] += 3
m['a'] += 2
m['b'] += 4
print(m['a'].mean, m['a'].std, m['b'].mean, m['b'].std)


for i in range(count):
   timing = Timing('test')
   with timing:
     do_something()

print(f'wall clock time: {timing.wall}; cpu time: {timing.cpu}')  
```

See `test_meter.py` for more use cases.
