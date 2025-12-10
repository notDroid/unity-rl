import time

class Stopwatch:
    def __init__(self):
        pass

    def start(self):
        self.start_time = time.time()

    def end(self):
        return time.time() - self.start_time

def round_up(x: float | int, y: int = 1):
    return ((x + y - 1) // y) * y