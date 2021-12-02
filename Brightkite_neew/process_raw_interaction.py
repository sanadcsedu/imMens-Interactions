import os
import glob
import numpy

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
            new_fname += '.csv'
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
        new_file = open("all_users.csv", 'a')
        flag = False
        cnt = 0
        sum = 0
        cummulative = []
        for lines in file:
            lines = lines.strip()
            interaction = lines.split(',')
            user = interaction[0]
            action = interaction[2]
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
                    cummulative.append((prev_action, cnt))
                    cnt = 0

            prev_action = action
            flag = True
            # print(action)
        cnt = 0
        for a, c in cummulative:
            new_file.write("{}, {}, {}, {}, {:.5f}\n".format(cnt, user, a, c, c /sum))
            cnt += 1
        # print(map)
        # new_file.write("{} {}\n".format(action, viewport))
        # new_file.write('\n')
        new_file.close()


if __name__ == "__main__":
    obj = process_raw_interaction()
    obj.get_files()