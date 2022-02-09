import glob
from datetime import timedelta
import csv

user_list = glob.glob('p4_reformed.csv')

for user in user_list:
    fname = open(user, 'r')
    csv_reader = csv.reader(fname)
    next(csv_reader)

    vector = []
    start_time = None
    prev_viz = None
    flag = True
    for lines in csv_reader:
        # line = lines.strip()
        # line = lines.split(',')
        cur_viz = lines[4]
        if flag:
            start_time = lines[0]
            prev_viz = cur_viz
            flag = False
            continue
        if cur_viz == "undefined":
            continue

        if prev_viz != cur_viz:
            #finding the time_diff
            cur_time = lines[0].split(":")
            # print(start_time)
            start_time = start_time.split(":")
            t2 = timedelta(hours=int(cur_time[0]), minutes=int(cur_time[1]), seconds = int(cur_time[2]))
            t1 = timedelta(hours=int(start_time[0]), minutes=int(start_time[1]), seconds = int(start_time[2]))
            duration = t2 - t1
            vector.append((prev_viz, duration))
            start_time = lines[0]
            prev_viz = cur_viz

    for vizu, d in vector:
        print("{} {}\n".format(vizu, d))
    break


