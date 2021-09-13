#Let's use this file for reading the imMens interaction log

import subprocess
import os
import fnmatch
from collections import defaultdict

class ReadData:
    def __init__(self):
        #using the init to read directory information
        # self.approved_actions = ['brush', 'range select', 'pan', 'zoom']
        listoffiles = os.listdir('Brightkite-0ms-state/')
        pattern = "*0-imMensEvt.txt"
        self.filelist = []
        for entry in listoffiles:
            if fnmatch.fnmatch(entry, pattern):
                self.filelist.append(entry)
        # print(len(self.filelist))
        # self.test = defaultdict()

    def read_file(self, fname):
        filename = 'Brightkite-0ms-state/' + fname
        file = open(filename, 'r')
        return file
        # for lines in file:
        #     lines = lines.strip()
        #     line = lines.split(', ')
        #     self.test[(line[2], line[1])] = 1


if __name__ == "__main__":
    obj = ReadData()
    # for filename in obj.filelist:
    #     if "-1-" in filename:
    #         obj.read_file(filename)
    #     else:
    #         continue
    #     # break
    #     # obj.read_file(filename)
    # actions1 = list()
    # for keys in obj.test:
    #     # print(keys)
    #     actions1.append(keys)
    # print(actions1)