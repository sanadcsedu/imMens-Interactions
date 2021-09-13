import EpsilonGreedy
import ReadData
import pdb

class TrainEpsilonGreedy:
    def __init__(self):
        self.states = ["'Hypothesis / Questions'", "'Exploration'", "'Drill-Down'", "'Sensemaking'"]

    # Roth and Erev algorithm without states
    def run_greedy(self, cur_data, epsilon, l, f):
        obj = EpsilonGreedy.EpsilonGreedy(self.states, epsilon, l, f)
        Y = []
        for row in cur_data:
            row = row.strip()
            row = row.split(', ')
            cur_state = row[len(row) - 1]
            # pdb.set_trace()

            # picked_action = obj.make_choice_classic()
            picked_action = obj.make_choice_adaptive()

            obj.update(cur_state, 1)

            y = 0
            # print("Picked: {} Ground:{}".format(picked_action, (viz, action)))
            print("{} {}".format(picked_action, cur_state))
            if picked_action == cur_state:
                y = 1
            Y.append(y)

        # Calculating prediction accuracy of our learning model
        num = denom = 0
        for idx in range(len(Y)):
            num += Y[idx]
            denom += 1
        return num / denom


if __name__ == "__main__":
    read = ReadData.ReadData()
    obj = TrainEpsilonGreedy()
    epsilon = 0.5
    l = 6
    f = 3
    num = denom = 0
    for filename in read.filelist:
        data = read.read_file(filename)
        accu = obj.run_greedy(data, epsilon, l, f)
        num += accu
        denom += 1
        break
    print("Accuracy: {:.2f}".format(num / denom))