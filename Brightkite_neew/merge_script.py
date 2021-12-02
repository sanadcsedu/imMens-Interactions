import pdb

import pandas as pd
import os


directory = 'p4'
parent_directory = 'D:\\imMens interaction logs\\annotated\\Brightkite-0ms\\Brightkite-state'
path = os.path.join(parent_directory, directory)

try:
    os.mkdir(path)
    # print("%s created" % directory)
except:
    print("Creation of the directory %s failed" % cur_work)

raw_interaction = open("p4-0-0-imMensEvt.txt")
df = pd.read_excel("p4-brightkite-0ms-annot.xlsx", sheet_name = "Sheet1")
# s = dataframe.to_csv(dataframe.head())
# print(df[['State', 'time']])
dfs = df.to_string()
data = []
for index, row in df.iterrows():
    # print("{} {}".format(row['State'], row['time']))
    hh = row['time'].hour
    mm = row['time'].minute
    ss = row['time'].second
    # print("{} : {} : {}".format(hh, mm, ss))
    seconds = hh * 60 + mm * 60 + ss
    data.append([row['State'], seconds, row['Subtask']])

idx = 0
fname = path + '//' + directory + '1.txt'
prev_subtask = -1
for lines in raw_interaction.readlines():
    # print(data[idx][2])
    if data[idx][2] != prev_subtask:
        if os.path.exists(fname):
            new_file.close()
        fname = path + '\\' + str(data[idx][2]) + '.txt'
        new_file = open(fname, 'w')

    lines = lines.strip()
    lines = lines.split(',')
    # user, time, x, y, z = lines
    time = lines[1].split(":")
    # print(time)
    hh = int(time[0])
    mm = int(time[1])
    ss = int(time[2])
    # print("{} : {} : {}".format(hh, mm, ss))
    seconds = hh * 60 + mm * 60 + ss
    prev_subtask = data[idx][2]
    if seconds < data[idx][1]:
        lines.append(data[idx][0])
    else:
        if idx + 1 < len(data):
            idx += 1
        lines.append(data[idx][0])
    new_file.write('%s\n' %lines)
