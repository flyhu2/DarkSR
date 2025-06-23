#
# log.py
#
# This file is part of LIDN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version

import os
import logging


class Log(object):
    def __init__(self, filepath, formatter, b_print=True):
        self.filepath = filepath
        os.makedirs(self.filepath, exist_ok=True)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.b_print = b_print

        self.logger.handlers = []
        fh = logging.FileHandler(os.path.join(self.filepath, 'result.log'), 'a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(formatter))  # "%(message)s"
        self.logger.addHandler(fh)

        fh.close()

    def write(self, log):
        self.logger.info(log)
        if self.b_print:
            print(log)

    def info(self, log):
        self.logger.info(log)
        if self.b_print:
            print(log)

    def debug(self, log):
        self.logger.debug(log)
        if self.b_print:
            print(log)

    def warning(self, log):
        self.logger.warning(log)
        if self.b_print:
            print(log)

    def error(self, log):
        self.logger.error(log)
        if self.b_print:
            print(log)
