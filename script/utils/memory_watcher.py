import os
import threading

import psutil
import torch
import time

class MemoryWatcher:
    def __init__(self, output_file, device, interval=5):
        self.output_file = output_file
        self.device = device
        self.interval = interval

        self.process = psutil.Process(os.getpid())

        self.count = 0
        self.ram_peak = 0
        self.ram_avg = 0
        self.vram_peak = 0
        self.vram_avg = 0
        self._stop = False

    def _watch(self):
        while not self._stop:
            self.count += 1

            ram_usage = self.get_ram_usage()
            if ram_usage > self.ram_peak:
                self.ram_peak = ram_usage
            self.ram_avg = self.ram_avg + ((ram_usage - self.ram_avg) / self.count)

            if "cuda" in str(self.device):
                vram_usage = self.get_vram_usage()
                if vram_usage > self.vram_peak:
                    self.vram_peak = vram_usage
                self.vram_avg = self.vram_avg + ((vram_usage - self.vram_avg) / self.count)
            time.sleep(self.interval)

    def start(self):
        self._stop = False
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop = True
        self.thread.join()
        self.log_to_file()

    def reset(self):
        self.count = 0
        self.ram_peak = 0
        self.vram_peak = 0
        self.ram_avg = 0
        self.vram_avg = 0

    def log_to_file(self):
        with open(self.output_file, 'a') as f:
            f.write(f"[M] - RAM: Peak = {self.ram_peak:.2f} MB | Avg = {self.ram_avg:.2f} MB\n")
            f.write(f"[M] - VRAM: Peak = {self.vram_peak:.2f} MB | Avg = {self.vram_avg:.2f} MB\n")

    def get_ram_usage(self):
        ram_usage = self.process.memory_info().rss / (1024 ** 2)  # MB
        return ram_usage

    def get_vram_usage(self):
        free, total = torch.cuda.mem_get_info(self.device)
        vram_usage = (total - free) / (1024 ** 2) # MB
        return vram_usage