import modified_roth_and_erev
import ReadData
import pdb
from collections import defaultdict

class TrainRothAndErev:
    def __init__(self):
        self.states = ["'Hypothesis / Questions'", "'Exploration'", "'Drill-Down'", "'Sensemaking'"]
        # self.threshold = 0.5

    # Roth and Erev algorithm without states
    def run_roth_and_erev(self, cur_data, forgetting):
        rae = modified_roth_and_erev.modified_roth_and_erev()
        # Setting up action set for the Roth and Erev model
        rae.add_prior_strategies(self.states, 0)

        Y = []
        # _dict = defaultdict()
        for row in cur_data:
            row = row.strip()
            row = row.split(', ')
            cur_state = row[len(row) - 1]
            # _dict[cur_state] = 1

            picked_action = rae.make_choice_nostate()
            rae.update_qtable(cur_state, 1, forgetting)

            y = 0
            if picked_action == cur_state:
                y = 1
            Y.append(y)

        #Calculating prediction accuracy of our learning model
        # for items in _dict:
        #     print(items)
        # print(_dict)
        # threshold = int(self.threshold * len(Y))
        num = denom = 0
        for idx in range(len(Y)):
            num += Y[idx]
            denom += 1
        # pdb.set_trace()
        return num / denom


if __name__ == "__main__":
    read = ReadData.ReadData()
    obj = TrainRothAndErev()
    forgetting = 0.05
    num = denom = 0
    for filename in read.filelist:
        data = read.read_file(filename)
        # print(filename, end=" -> ")
        accu = obj.run_roth_and_erev(data, forgetting)
        num += accu
        denom += 1
        # break
    print("Accuracy: {:.2f}".format(num / denom))