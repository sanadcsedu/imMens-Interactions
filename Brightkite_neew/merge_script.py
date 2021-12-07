import csv
import pdb
import glob
import pandas as pd
import os


class integrate:
    def __init__(self):
        self.raw_files = glob.glob('*_raw.csv')
        self.excel_files = glob.glob('*-annot.xlsx')
        self.path = 'D:\\imMens Learning\\Brightkite_neew\\joined\\'
        # print(self.excel_files)

    def get_files(self):
        for raw_fname in self.raw_files:
            user = raw_fname.split('_')[0]
            # print(user)
            excel_fname = [string for string in self.excel_files if user in string][0]
            # print("{} {}".format(raw_fname, excel_fname))
            self.merge(user, raw_fname, excel_fname)
            # break

    def excel_to_memory(self, df):
        data = []
        for index, row in df.iterrows():
        # print("{} {}".format(row['State'], row['time']))
            hh = row['time'].hour
            mm = row['time'].minute
            ss = row['time'].second
            # print("{} : {} : {}".format(hh, mm, ss))
            seconds = hh * 60 + mm * 60 + ss
            data.append([row['proposition'], row['State'], row['Reward'], seconds])
            # print(data[len(data) - 1])
        return data

    def merge(self, user, raw_fname, excel_fname):
        # print(user)
        raw_interaction = open(raw_fname, 'r')
        csv_reader = csv.reader(raw_interaction)
        next(csv_reader)

        # print(df_raw.head())
        # print(excel_fname)
        df_excel = pd.read_excel(excel_fname, sheet_name= "Sheet1", usecols="A:D, H")
        # print(df_excel.columns)
        data = self.excel_to_memory(df_excel)
        new_fname = self.path + user + '.csv'
        new_file = open(new_fname, 'w')

        idx = 0
        for lines in csv_reader:
            time = lines[1].split(":")
            # print(time)
            hh = int(time[0])
            mm = int(time[1])
            ss = int(time[2])
            # print("{} : {} : {}".format(hh, mm, ss))
            seconds = hh * 60 + mm * 60 + ss
            # pdb.set_trace()
            if seconds < data[idx][3]:
                new_file.write("{}, {}, {}, {}, {}, {}\n".format(lines[1], data[idx][0], data[idx][1], data[idx][2], lines[3], lines[5]))
            else:
                if idx + 1 < len(data):
                    idx += 1
                new_file.write("{}, {}, {}, {}, {}, {}\n".format(lines[1], data[idx][0], data[idx][1], data[idx][2], lines[3], lines[5]))
                    # new_file.write('%s\n' %time)
        raw_interaction.close()
        new_file.close()


if __name__ == "__main__":
    obj = integrate()
    obj.get_files()
# directory = 'p4'
# parent_directory = 'D:\\imMens interaction logs\\annotated\\Brightkite-0ms\\Brightkite-state'
# path = os.path.join(parent_directory, directory)
#
# try:
#     os.mkdir(path)
#     # print("%s created" % directory)
# except:
#     print("Creation of the directory %s failed" % cur_work)
#
# raw_interaction = open("p4-0-0-imMensEvt.txt")
# df = pd.read_excel("p4-brightkite-0ms-annot.xlsx", sheet_name = "Sheet1")
# # s = dataframe.to_csv(dataframe.head())
# # print(df[['State', 'time']])
# dfs = df.to_string()
# data = []
# for index, row in df.iterrows():
#     # print("{} {}".format(row['State'], row['time']))
#     hh = row['time'].hour
#     mm = row['time'].minute
#     ss = row['time'].second
#     # print("{} : {} : {}".format(hh, mm, ss))
#     seconds = hh * 60 + mm * 60 + ss
#     data.append([row['State'], seconds, row['Subtask']])
#
# idx = 0
# fname = path + '//' + directory + '1.txt'
# prev_subtask = -1
# for lines in raw_interaction.readlines():
#     # print(data[idx][2])
#     if data[idx][2] != prev_subtask:
#         if os.path.exists(fname):
#             new_file.close()
#         fname = path + '\\' + str(data[idx][2]) + '.txt'
#         new_file = open(fname, 'w')
#
#     lines = lines.strip()
#     lines = lines.split(',')
#     # user, time, x, y, z = lines
#     time = lines[1].split(":")
#     # print(time)
#     hh = int(time[0])
#     mm = int(time[1])
#     ss = int(time[2])
#     # print("{} : {} : {}".format(hh, mm, ss))
#     seconds = hh * 60 + mm * 60 + ss
#     prev_subtask = data[idx][2]
#     if seconds < data[idx][1]:
#         lines.append(data[idx][0])
#     else:
#         if idx + 1 < len(data):
#             idx += 1
#         lines.append(data[idx][0])
#     new_file.write('%s\n' %lines)
