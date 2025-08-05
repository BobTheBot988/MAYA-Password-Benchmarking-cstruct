import subprocess
import time
import os
import sys
import fcntl
from time import sleep

__LOCK_PATH = "/tmp/gpu_lock/gpu_selection.lock"


def __get_lock():
    lock_fd = None
    assert os.path.isfile(__LOCK_PATH)
    try:
        lock_fd = open(__LOCK_PATH, 'a')
        fcntl.lockf(lock_fd, fcntl.LOCK_EX)
    except OSError:
        lock_fd.close()
        lock_fd = None
    return lock_fd


def __release_lock(fd):
    fd.close()


def get_device():
    while (fd := __get_lock()) is None:
        print('Error locking gpu_selection.lock')
        sleep(1)
    chosen_gpu = __get_optimal_gpu()
    __release_lock(fd)
    if chosen_gpu is not None:
        return f'cuda:{chosen_gpu}'
    else:
        print('No GPU availabe at this time')
        sys.exit(1)


def __get_visible_gpus():
    visible_gpus = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_gpus is not None:
        visible_gpus = visible_gpus.split(",")
    return visible_gpus

def __get_optimal_gpu():
    visible_gpus = __get_visible_gpus()

    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free,memory.total,', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)

    gpus_info = result.stdout.strip().split('\n')

    gpu_memory = []
    for gpu in gpus_info:
        id, gpu_utilization, memory_free, memory_total = map(int, gpu.split(','))

        if visible_gpus is not None:
            if str(id) not in visible_gpus:
                continue
            #else:
            #    id = visible_gpus.index(str(id))

        percentage_free_memory = round((memory_free / memory_total * 100), 2)
        percentage_free_gpu = 100 - gpu_utilization

        if percentage_free_memory < 25 or percentage_free_gpu < 25:
            continue

        priority_value = (percentage_free_memory * percentage_free_gpu) / 2
        gpu_memory.append((id, priority_value))

    gpu_memory = sorted(gpu_memory, key=lambda x: x[1], reverse=True)
    return gpu_memory[0][0] % len(visible_gpus) if gpu_memory else None
