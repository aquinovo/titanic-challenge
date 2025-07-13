import time
import psutil
import os
import logging

def profile_resources(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = psutil.cpu_percent(interval=None)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        end_mem = process.memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent(interval=None)
        logging.info(f"[PROFILING] {func.__name__}: RAM used: {end_mem - start_mem:.2f} MB | "
                     f"CPU change: {end_cpu - start_cpu:.2f}% | Time: {end_time - start_time:.2f}s")
        return result
    return wrapper
