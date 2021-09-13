#Let's use this file for reading the imMens interaction log

import subprocess
import os
import fnmatch
from collections import defaultdict

class read_user:
    def __init__(self):
        #using the init to read directory information
        self.approved_actions = ['brush', 'range select', 'pan', 'zoom']
        listoffiles = os.listdir('triggered-evt-logs/')
        pattern = "*-0-imMensEvt.txt"
        # cnt = 0
        self.filelist = []
        for entry in listoffiles:
            if fnmatch.fnmatch(entry, pattern):
                self.filelist.append(entry)
                # cnt += 1
                # print(entry)
        # print(cnt)

        #Testing which actions are available
        self.test = defaultdict()

    def approve(self, action, viewport, prev_action, prev_viewport):
        if prev_action == action and prev_viewport == viewport:
            return False
        if action not in self.approved_actions:
            return False
        return True

    def read_file(self, fname):
        filename = 'triggered-evt-logs/' + fname
        file = open(filename, 'r')
        write_filename = 'cleaned-triggered-evt-logs-new/' + fname
        new_file = open(write_filename, 'w')
        prev_action = prev_viewport = None
        cnt = 0
        flag = False
        StartTime = 0
        for lines in file:
            lines = lines.strip()
            line = lines.split(',')
            # print(line)
            timestamp = line[1]
            action = line[2]
            viewport = line[3]
            if not self.approve(action, viewport, prev_action, prev_viewport):
                continue
            # print(timestamp, action, viewport)
            cnt += 1
            self.test[action] = True

            CurTime = None
            if flag:
                CurTime = int(timestamp) - StartTime
            else:
                StartTime = int(timestamp)
                CurTime = 0
                flag = True
            mins = int((CurTime / 1000) / 60)
            seconds = int(CurTime / 1000) - (mins * 60)
            new_file.write("{:02d}:{:02d}, {}, {}\n".format(mins, seconds, action, viewport))
            prev_action = action
            prev_viewport = viewport
        print("{}".format(cnt), end=' ')


if __name__ == "__main__":
    obj = read_user()
    for filename in obj.filelist:
        obj.read_file(filename)
    for actions in obj.test:
        print(actions)
