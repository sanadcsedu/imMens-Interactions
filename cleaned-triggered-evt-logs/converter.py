import glob
from datetime import datetime, timedelta
import csv


def chop_microseconds(delta):
    return delta - timedelta(microseconds=delta.microseconds)


if __name__ == "__main__":
    file_names = glob.glob("*.txt")
    for fname in file_names:
        newfname = fname.split('.')[0] + '.csv'
        # print(newfname)
        file = open(fname, 'r')
        newfile = open(newfname, 'w')
        lines = file.readlines()
        flag = True
        for line in lines:
            line = line.strip()
            line = line.split(',')
            unix_ts_ms = int(line[0]) / 1000
            unix_ts = datetime.fromtimestamp(unix_ts_ms)
            if flag:
                flag = False
                offset = unix_ts
                newline = 'time, action, visualization\n'
                newfile.write(newline)
                cur_ts = offset - offset
            else:
                cur_ts = unix_ts - offset

            cur_ts = chop_microseconds(cur_ts)
            newline = str(cur_ts) + ',' + line[1].strip(' ') + ',' + line[2].strip(' ') + '\n'
            newfile.write(newline)

        file.close()
        newfile.close()
        # break