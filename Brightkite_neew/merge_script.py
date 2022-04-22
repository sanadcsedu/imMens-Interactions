import csv
import pdb
import glob
import pandas as pd
import os


class integrate:
    def __init__(self):
        # self.raw_files = glob.glob('*_raw.csv') imMensEvt.txt
        self.raw_files = glob.glob('*imMensEvt.txt')
        self.excel_files = glob.glob('*-annot.xlsx')
        self.path = 'D:\\imMens Learning\\Brightkite_neew\\joined\\'

        # print(self.excel_files)

    def get_files(self):
        for raw_fname in self.raw_files:
            user = raw_fname.split('-')[0]
            # print(user)
            excel_fname = [string for string in self.excel_files if user in string][0]
            print("{} {}".format(raw_fname, excel_fname))
            self.merge2(user, raw_fname, excel_fname)
            # break

    def excel_to_memory(self, df):
        data = []
        for index, row in df.iterrows():
        # print("{} {}".format(row['State'], row['time']))
            hh = row['time'].hour
            mm = row['time'].minute
            ss = row['time'].second
            # print("{} : {} : {}".format(hh, mm, ss))
            seconds = mm * 60 + ss
            if row['State'] == "None": #When reading from excel do not consider states = None
                continue
            data.append([row['proposition'], row['State'], row['Reward'], row['Subtask'], seconds])
            # print(data[len(data) - 1])
        return data

    #Used for merging CSV files with the Excel files
    def merge1(self, user, raw_fname, excel_fname):
        # print(user)
        raw_interaction = open(raw_fname, 'r')
        csv_reader = csv.reader(raw_interaction)
        next(csv_reader)

        # print(df_raw.head())
        # print(excel_fname)
        df_excel = pd.read_excel(excel_fname, sheet_name= "Sheet1(2)", usecols="A:D, H")
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
                new_file.write("{},\"{}\",{},{},{},{}\n".format(lines[1], data[idx][0], data[idx][1], data[idx][2], lines[3], lines[5]))
            else:
                if idx + 1 < len(data):
                    idx += 1
                new_file.write("{},\"{}\",{},{},{},{}\n".format(lines[1], data[idx][0], data[idx][1], data[idx][2], lines[3], lines[5]))
                    # new_file.write('%s\n' %time)
        raw_interaction.close()
        new_file.close()

    #For merging RAW_interactions (original dataset .txt file) with user annotations (Excel file)
    def merge2(self, user, raw_fname, excel_fname):

        raw_interaction = open(raw_fname, 'r')

        df_excel = pd.read_excel(excel_fname, sheet_name="Sheet1(2)", usecols="A:D, H:I")
        data = self.excel_to_memory(df_excel)

        new_fname = self.path + user + '_new.csv'
        new_file = open(new_fname, 'w')

        new_file.write("Time, Proposition, State,action,reward,visualization,subtask\n")
        idx = 0
        for lines in raw_interaction.readlines():
            lines = lines.strip()
            lines = lines.split(',')
            time = lines[1].split(':')
            hh = int(time[0])
            mm = int(time[1])
            ss = int(time[2])
            # print("{} : {} : {}".format(hh, mm, ss))
            seconds = hh * 60 + mm * 60 + ss
            # pdb.set_trace()

            if seconds < data[idx][4]:
                new_file.write("{},{},({}+{}),{},{},{},{}\n".format(data[idx][0],lines[1], data[idx][1], lines[3], lines[2], data[idx][2], lines[3], data[idx][3]))
            else:
                if idx + 1 < len(data):
                    idx += 1
                new_file.write("{},{},({}+{}),{},{},{},{}\n".format(data[idx][0], lines[1], data[idx][1], lines[3], lines[2], data[idx][2], lines[3], data[idx][3]))

        raw_interaction.close()
        new_file.close()

class reform:
    def __init__(self):
        # self.raw_files = glob.glob('*_raw.csv') imMensEvt.txt
        self.path = 'D:\\imMens Learning\\Brightkite_neew\\joined\\'
        # self.raw_files = glob.glob(self.path + '*_new.txt', recursive=True)
        #Finding the files from a different subdirectory because the glob cannot do that
        self.raw_files = [os.path.join(dirpath, f)
                          for dirpath, dirnames, files in os.walk(self.path)
                          for f in files if f.endswith('_new.csv')]
        # self.approved_actions = ['brush', 'range select', 'pan', 'zoom', 'clear']
        self.approved_actions = ['brush', 'range select', 'pan', 'zoom'] #Without using the action clear.

    def run_reform(self):
        for files in self.raw_files:
            raw_interaction = open(files, "r")

            temp = files.split("\\")
            user = temp[len(temp) - 1].split('_')[0]
            new_fname = self.path + user + '_reformed.csv'
            new_file = open(new_fname, 'w')

            self.reform(raw_interaction, new_file)
            raw_interaction.close()
            new_file.close()


    def reform(self, old, new):
        flag = True
        prev_time = None
        prev_state = None
        for lines in old.readlines():
            curline = lines.strip()
            curline = lines.split(',')
            if flag:
                new.write(lines)
                flag = False
                continue
            # pdb.set_trace()
            if curline[2] in self.approved_actions and (curline[0] != prev_time or curline[1] != prev_state):
                new.write(lines)
            prev_time = curline[0]
            prev_state = curline[1]


if __name__ == "__main__":

    #The Integrate class helps to integrate the raw interaction files with feedback (.excel) files.
    obj = integrate()
    obj.get_files()

    #This class reform cleans the data a bit, removing concurrent similar actions in the same time step / visualization
    #We can use this because for now we are working in a high level and the differentiating factors on those interactions
    # are not being used for now.
    # obj = reform()
    # obj.run_reform()

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
