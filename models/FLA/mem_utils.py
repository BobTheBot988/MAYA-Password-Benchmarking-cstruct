import psutil,os
def get_memory_usage_byte()->float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss 


def get_memory_usage()->float:
    """Get current memory usage in MB"""
    return get_memory_usage_byte()/ 1000 / 1000

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

def analyze_heap_memory(heap: list) -> None:
    from statistics import mean
    import sys

    if not heap:
        print("Heap is empty.")
        return

    total_password_length = 0
    total_object_size = 0
    total_buffer_size = 0
    total_sizeof = 0

    for item in heap:
        pw_len = len(item.password_string)
        total_password_length += pw_len

        sizes = item.memory_size()
        total_object_size += sizes['object_size']
        total_buffer_size += sizes['password_buffer']
        total_sizeof += sys.getsizeof(item)

    n_items = len(heap)
    avg_pw_len = total_password_length / n_items
    avg_obj_size = total_object_size / n_items
    avg_buf_size = total_buffer_size / n_items
    avg_total_size = total_sizeof / n_items

    print(f"\n📊 Heap Analysis Report ({n_items} items)")
    print(f"- Average password length: {avg_pw_len:.2f} characters")
    print(f"- Avg HeapItem object size: {avg_obj_size} bytes")
    print(f"- Avg password buffer size: {avg_buf_size} bytes")
    print(f"- Avg total size (via __sizeof__): {avg_total_size:.2f} bytes")
    print(f"- Total estimated heap size: {total_sizeof / 1_000_000:.2f} MB")
