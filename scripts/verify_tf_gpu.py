#!/usr/bin/env python3
"""Verify TensorFlow + Metal GPU setup. Prints version, devices, and times a matmul."""

import time
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print()

# List all physical devices
for dtype in ("CPU", "GPU"):
    devs = tf.config.list_physical_devices(dtype)
    print(f"{dtype} devices: {devs}")
print()

# Quick matmul benchmark
size = 2048
a = tf.random.normal((size, size))
b = tf.random.normal((size, size))

# Warm-up
_ = tf.matmul(a, b)

runs = 5
t0 = time.perf_counter()
for _ in range(runs):
    _ = tf.matmul(a, b)
elapsed = (time.perf_counter() - t0) / runs

print(f"Matmul {size}x{size}  avg {elapsed*1000:.1f} ms  ({runs} runs)")
