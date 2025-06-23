#
# filename_sort.py
#
# This file is part of LIDN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version


class FileNameSort(object):

    @staticmethod
    def _is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def _find_continuous_num(self, astr, c):
        num = ''
        # noinspection PyBroadException
        try:
            while not self._is_number(astr[c]) and c < len(astr):
                c += 1
            while self._is_number(astr[c]) and c < len(astr):
                num += astr[c]
                c += 1
        except:
            pass
        if num != '':
            return int(num)

    def _comp2filename(self, file1, file2):
        smaller_length = min(len(file1), len(file2))
        continuous_num = ''
        for c in range(0, smaller_length):
            if not self._is_number(file1[c]) and not self._is_number(file2[c]):
                # print('both not number')
                if file1[c] < file2[c]:
                    return True
                if file1[c] > file2[c]:
                    return False
                if file1[c] == file2[c]:
                    if c == smaller_length - 1:
                        # print('the last bit')
                        if len(file1) < len(file2):
                            return True
                        else:
                            return False
                    else:
                        continue
            if self._is_number(file1[c]) and not self._is_number(file2[c]):
                return True
            if not self._is_number(file1[c]) and self._is_number(file2[c]):
                return False
            if self._is_number(file1[c]) and self._is_number(file2[c]):
                if self._find_continuous_num(file1, c) < self._find_continuous_num(file2, c):
                    return True
                else:
                    return False
        # if file1 < file2:
        #     return True
        # else:
        #     return False

    @staticmethod
    def sort_insert(lst):
        for i in range(1, len(lst)):
            x = lst[i]
            j = i
            while j > 0 and lst[j - 1] > x:
                # while j > 0 and comp2filename(x, lst[j-1]):
                lst[j] = lst[j - 1]
                j -= 1
            lst[j] = x
        return lst

    def sort(self, lst):
        for i in range(1, len(lst)):
            x = lst[i]
            j = i
            # while j > 0 and lst[j-1] > x:
            while j > 0 and self._comp2filename(x, lst[j - 1]):
                lst[j] = lst[j - 1]
                j -= 1
            lst[j] = x
        return lst
