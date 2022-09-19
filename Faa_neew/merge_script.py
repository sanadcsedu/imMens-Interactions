import csv
import pdb
import glob
import pandas as pd
import os
from collections import defaultdict

class integrate:
    def __init__(self):
        # self.raw_files = glob.glob('*_raw.csv') imMensEvt.txt
        self.raw_files = glob.glob('*imMensEvt_fix.txt')
        self.excel_files = glob.glob('*-0ms.xlsx')
        # self.raw_files = glob.glob('p15-1-0-imMensEvt_fix.txt')
        # self.excel_files = glob.glob('p15-faa-0ms.xlsx')
        # print(self.excel_files)
        self.path = 'D:\\imMens Learning\\Faa_neew\\merged_new\\'
        # self.temp = defaultdict()
        # print(self.excel_files)

    def get_files(self):
        for raw_fname in self.raw_files:
            user = raw_fname.split('-')[0]
            # print(user)
            excel_fname = [string for string in self.excel_files if user in string][0]
            # print("{} {}".format(raw_fname, excel_fname))
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
            if row['State'] == 'None': #When reading from excel do not consider states = None
                # pdb.set_trace()
                continue
            # self.temp[row['Action']] = 1
            # print("{} {} {}".format(row['time'], row['State'], row['Action']))
            # data.append([row['proposition'], row['State'], row['Reward'], row['Subtask'], seconds])
            data.append([row['State'], row['Action'], row['Reward'], row['Subtask'], seconds])
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
        df_excel = pd.read_excel(excel_fname, sheet_name= "Sheet(2)", usecols="A:E, H")
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

        df_excel = pd.read_excel(excel_fname, sheet_name="Sheet(2)", usecols="A:E, I:J")
        data = self.excel_to_memory(df_excel)

        new_fname = self.path + user + '_new.txt'
        new_file = open(new_fname, 'w')

        new_file.write("Time,State,action,reward,visualization,subtask,raw_actions\n")
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
                #This line uses the raw actions (brush, pan, zoom, range select) from the raw_interactions file.
                # new_file.write("{},({}+{}),{},{},{},{}\n".format(lines[1], data[idx][1], lines[3], lines[2], data[idx][2], lines[3], data[idx][3]))
                #This line uses the abstract actions (Observation, Generalization, Explanation, Steer) from the excel sheet and merges them with the raw interaction file
                new_file.write("{},({}+{}),{},{},{},{},{}\n".format(lines[1], data[idx][0], lines[3], data[idx][1], data[idx][2], lines[3], data[idx][3], lines[2]))

            else:
                if idx + 1 < len(data):
                    idx += 1
                # This line uses the raw actions (brush, pan, zoom, range select) from the raw_interactions file.
                # new_file.write("{},({}+{}),{},{},{},{}\n".format(lines[1], data[idx][1], lines[3], lines[2], data[idx][2], lines[3], data[idx][3]))
                #This line uses the abstract actions (Observation, Generalization, Explanation, Steer) from the excel sheet and merges them with the raw interaction file
                new_file.write("{},({}+{}),{},{},{},{},{}\n".format(lines[1], data[idx][0], lines[3], data[idx][1], data[idx][2], lines[3], data[idx][3], lines[2]))

        raw_interaction.close()
        new_file.close()

class reform:
    def __init__(self):
        # self.raw_files = glob.glob('*_raw.csv') imMensEvt.txt
        self.path = 'D:\\imMens Learning\\Faa_neew\\merged_new\\'
        # self.raw_files = glob.glob(self.path + '*_new.txt', recursive=True)
        #Finding the files from a different subdirectory because the glob cannot do that
        self.raw_files = [os.path.join(dirpath, f)
                          for dirpath, dirnames, files in os.walk(self.path)
                          for f in files if f.endswith('_new.txt')]
        # self.approved_actions = ['brush', 'range select', 'pan', 'zoom', 'clear']
        self.approved_actions = ['brush', 'range select', 'pan', 'zoom'] # Without using the action clear.

    def run_reform(self):
        for files in self.raw_files:
            raw_interaction = open(files, "r")

            temp = files.split("\\")
            user = temp[len(temp) - 1].split('_')[0]
            new_fname = self.path + user + '_reform.csv'
            new_file = open(new_fname, 'w')

            self.reform(raw_interaction, new_file)
            raw_interaction.close()
            new_file.close()
            # break


    def reform(self, old, new):
        # print("{} {}".format(old, new))
        flag = True
        prev_time = None
        prev_state = None
        # print(old)
        for lines in old.readlines():
            curline = lines.strip()
            curline = curline.split(',')
            if flag:
                new.write(lines)
                flag = False
                continue
            # pdb.set_trace()
            sz = len(curline)
            # pdb.set_trace()
            if curline[sz - 1] in self.approved_actions and (curline[0] != prev_time or curline[1] != prev_state):
                # pdb.set_trace()
                # print("{} {} {}".format(lines, prev_time, prev_state))
                new.write(lines)

                prev_time = curline[0]
                prev_state = curline[1]


if __name__ == "__main__":

    #The Integrate class helps to integrate the raw interaction files with feedback (.excel) files.
    obj = integrate()
    obj.get_files()
    # print(obj.temp)

    #This class reform cleans the data a bit, removing concurrent similar actions in the same time step / visualization
    #We can use this because for now we are working in a high level and the differentiating factors on those interactions
    # are not being used for now.
    obj = reform()
    obj.run_reform()
