from time import time, sleep


# Time printing function
def print_time(before, returning=False, printing=True):
    time_taken = time() - before
    mins = int(time_taken // 60)
    secs = int(time_taken % 60)
    mins_str = f'0{mins}' if mins < 10 else str(mins)
    secs_str = f'0{secs}' if secs < 10 else str(secs)
    time_taken = mins_str + ':' + secs_str

    if printing:
        print(f'Time taken: {time_taken}')

    if returning:
        return time_taken


if __name__ == '__main__':
    # Example
    before = time()
    sleep(2)
    print_time(before)

