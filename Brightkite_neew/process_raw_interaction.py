import os
import glob
import numpy
import pdb

class process_raw_interaction:
    def __init__(self):
        self.file_names = None
        self.approved_actions = ['brush', 'range select', 'pan', 'zoom', 'clear']
        self.map = set()

    def get_files(self):
        #read raw interaction file names
        self.file_names = glob.glob('*imMensEvt.txt')
        # files = sorted(os.listdir('.'))
        for names in self.file_names:
            file = open(names, 'r')
            new_fname = names.split('-')[0]
            new_fname += '_raw.csv'
            self.viz(file, new_fname)
            file.close()
            # break

    def read_file(self, file, new_fname):
        # print(new_fname)
        new_file = open(new_fname, 'w')
        for lines in file:
            lines = lines.strip()
            interaction = lines.split(',')
            new_file.write(str(interaction))
            new_file.write('\n')
        new_file.close()

    def viz(self, file, new_fname):
        # print(new_fname)
        # new_file = open("all_users.csv", 'a')
        new_file = open(new_fname, 'w')
        new_file.write("no, time, user, action, count_cum_action, fre_cum_action\n")
        flag = False
        cnt = 0
        sum = 0
        cummulative = []
        start_time = None
        for lines in file:
            lines = lines.strip()
            interaction = lines.split(',')
            user = interaction[0]
            time = interaction[1]
            action = interaction[2]
            if cnt == 0:
                start_time = time
            if action not in self.approved_actions:
                continue
            self.map.add(action)
            viewport = interaction[3]
            if flag:
                if action == prev_action:
                    cnt += 1
                    sum += 1
                else:
                    if cnt == 0:
                        cnt = 1
                    cummulative.append((time, prev_action, cnt))
                    cnt = 0

            prev_action = action
            flag = True
            # pdb.set_trace()
            # print(action)

        cnt = 0
        for time, a, c in cummulative:
            new_file.write("{},{},{},{},{},{:.5f}\n".format(cnt, time, user, a, c, c /sum))
            cnt += 1
        # print(map)
        # new_file.write("{} {}\n".format(action, viewport))
        # new_file.write('\n')
        new_file.close()


if __name__ == "__main__":
    obj = process_raw_interaction()
    obj.get_files()