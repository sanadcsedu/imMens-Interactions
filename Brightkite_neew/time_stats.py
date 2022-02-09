import glob
from datetime import timedelta
import csv
import pandas as pd

def find_intervals(df, operation):
    vector = []
    start_time = None
    # prev_viz = None
    flag = True
    for index, row in df.iterrows():

        if flag:
            start_time = row['time']
            flag = False
            continue

        if row['type'] == operation:
            # finding the time_diff
            t2 = timedelta(hours=int(row['time'].hour), minutes=int(row['time'].minute),
                           seconds=int(row['time'].second))
            t1 = timedelta(hours=int(start_time.hour), minutes=int(start_time.minute), seconds=int(start_time.second))
            duration = t2 - t1
            vector.append((row['proposition'], duration))
            start_time = row['time']

    for prop, d in vector:
        print("{}".format(d))


user_list = glob.glob('*0ms-annot.xlsx')
# print(user_list)
for excel_fname in user_list:
    print(excel_fname)
    df_excel = pd.read_excel(excel_fname, sheet_name= "Sheet1", usecols="A:D, H")
    find_intervals(df_excel, "hypothesis")
    print("Gap between generalization")
    find_intervals(df_excel, "generalization")



