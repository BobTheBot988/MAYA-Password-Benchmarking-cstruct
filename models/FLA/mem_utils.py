import psutil,os

def get_memory_usage()->float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1000 / 1000

def kill_if_low_memory(threshold_percent:float=95):
    """Checks if current system-wide memory usage exceeds threshold.
      If so, prints a warning and exits the program.


    Args:
        threshold_percent (float, optional): the percent of maximum memory occupied  Defaults to 95 must be lower than 100.

    Raises:
        ValueError: if the threshold_percent > 99.99999
    """

    if threshold_percent > 99.99999:
        raise ValueError("The threshold is too big...")

    mem = psutil.virtual_memory()
    if mem.percent >= threshold_percent:
        print(f"[ABORT] Memory usage is at {mem.percent:.1f}%, which exceeds the threshold of {threshold_percent}%.")
        sys.exit(1)  # Exit with error status