import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.stop_time = None
        self.timings = list()

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()
        self.time_elapsed = self.stop_time - self.start_time
        self.timings.append(self.time_elapsed)

    def __str__(self):
        return "Runtime: {0}s".format(self.time_elapsed)

    def getAverage(self):
        return sum(self.timings)/len(self.timings)
