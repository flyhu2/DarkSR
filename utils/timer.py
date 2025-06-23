#
# timer.py
#
# This file is part of LIDN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version

import time


class Timer(object):
    def __init__(self, b_start=False):
        self.time_start = 0
        self.time_stop = 0
        self.is_running = False

        if b_start:
            self.start()
        else:
            self.stop()

    def start(self):
        if not self.is_running:
            self.time_start = time.time()
            self.time_stop = 0
        self.is_running = True

    def stop(self):
        if self.is_running:
            self.time_stop = time.time()
        self.is_running = False

    def restart(self):
        self.stop()
        self.start()

    def reset(self):
        self.is_running = False
        self.time_start = 0
        self.time_stop = 0

    def is_running(self):
        return self.is_running

    def elapsed_ticks(self):
        elapsed_time = 0
        if self.is_running:
            elapsed_time = time.time() - self.time_start
        else:
            elapsed_time = self.time_stop - self.time_start

        if elapsed_time < 0:
            print(self.time_start, self.time_stop, self.is_running)
            return 0

        return elapsed_time

    def elapsed_milliseconds(self):
        pass


def elapsed_ticks_format(elapsed_ticks):
    m, s = divmod(elapsed_ticks, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), int(s)
