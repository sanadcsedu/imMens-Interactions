import csv
import os
import glob
import numpy
import pdb
# import pandas

class time_states:
    def __init__(self):
        self.file_names = None

    def get_files(self):
        #read raw interaction file names
        self.file_names = glob.glob("*.csv")
        # files = sorted(os.listdir('.'))
        for names in self.file_names:
            file = open(names, 'r')
            csv_reader = csv.reader(file)
            # pdb.set_trace()
            user = names.split('.')[0]
            print(user)
            # new_fname = user + '_raw.csv'
            self.viz(user, csv_reader)
            file.close()
            # break

    def viz(self, user, csv_reader):
        # print(new_fname)
        new_file = open("all_users.csv", 'a')
        # new_file = open(new_fname, 'w')
        # new_file.write("user, state, time(seconds)\n")
        flag = False
        prev_state = None
        start_time = None
        time = None
        state = None
        sum = 0
        store = []
        reward = 0.0
        for interaction in csv_reader:
            # lines = lines.strip()
            # interaction = lines.split(',')
            # pdb.set_trace()
            time_hms = interaction[0].split(":")
            # pdb.set_trace()
            time = int(time_hms[1]) * 60 + int(time_hms[2])
            # pdb.set_trace()
            state = interaction[2]
            # pdb.set_trace()
            reward += float(interaction[3])
            if state == 'None':
                continue
            # pdb.set_trace()
            if flag:
                if prev_state != state:
                    duration = time - start_time
                    # new_file.write("{},{},{}\n".format(user, prev_state, max(1, duration)))
                    store.append((prev_state, max(1, duration), reward))
                    sum += max(1, duration)
                    reward = 0
                    # pdb.set_trace()
                    start_time = time
            else:
                start_time = time
            prev_state = state
            flag = True

        duration = time - start_time
        sum += max(1, duration)
        # print(sum)
        store.append((prev_state, max(1, duration), reward))
        # new_file.write("{},{},{}\n".format(user, prev_state, max(1, duration)))
        id = 0
        for s, d, r in store:
            new_file.write("{},{},{},{:.2f}\n".format(user, s, d, r))
            id += 1
            # print(d/sum, end=',')
        new_file.close()


if __name__ == "__main__":
    os.system("rm all_users.csv")
    obj = time_states()
    obj.get_files()