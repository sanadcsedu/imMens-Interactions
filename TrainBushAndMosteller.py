import BushMosteller
import ReadData
import pdb

class TrainBushAndMosteller:
    def __init__(self):
        self.states = ["'Hypothesis / Questions'", "'Exploration'", "'Drill-Down'", "'Sensemaking'"]

    # Roth and Erev algorithm without states
    def run_bush_mosteller(self, cur_data, alpha, beta):
        obj = BushMosteller.BushMosteller(alpha, beta)

        # Setting up action set for the Roth and Erev model
        obj.add_prior_strategies(self.states)

        Y = []
        for row in cur_data:
            row = row.strip()
            row = row.split(', ')
            cur_state = row[len(row) - 1]
            # pdb.set_trace()

            picked_action = obj.make_choice_nostate()

            # obj.update((viz, action), 1)

            y = 0
            # print("Picked: {} Ground:{}".format(picked_action, (viz, action)))
            if picked_action == cur_state:
                obj.update(cur_state, 1)
                y = 1
            else:
                obj.update(cur_state, 1)
                obj.update(cur_state, -1)
            Y.append(y)

        #Calculating prediction accuracy of our learning model
        num = denom = 0
        for idx in range(len(Y)):
            num += Y[idx]
            denom += 1
        return num / denom


if __name__ == "__main__":
    read = ReadData.ReadData()
    obj = TrainBushAndMosteller()
    alpha = 0.1
    beta = 0.0
    num = denom = 0
    for filename in read.filelist:
        data = read.read_file(filename)
        accu = obj.run_bush_mosteller(data, alpha, beta)
        num += accu
        denom += 1
        # break
    print("Accuracy: {:.2f}".format(num / denom))