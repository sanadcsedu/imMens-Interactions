from random import Random
import numpy

class LSTDQ(qlearning.qlearning_agent):
    """Least Squares Temporal Difference Learning agent, LSTD-Q.
    This differs from LSTD class in that it holds onto the samples themselves
    and regenerates the matrix and vector (A, b) at each update based upon
    those samples and the current policy.
    From the paper:
    Least-Squares Policy Iteration. 2003.
    Michail Lagoudakis and Ronald Parr.
    """

    name = "LSTD-Q"

    @classmethod
    def agent_parameters(cls):
        param_set = super(LSTDQ, cls).agent_parameters()
        add_parameter(param_set, "lstd_num_samples", default=500, type=int, min=1, max=5000)
        add_parameter(param_set, "lstd_precond", default=0.1)
        remove_parameter(param_set, "alpha")
        return param_set

    def init_parameters(self):
        super(LSTDQ, self).init_parameters()
        self.lstd_gamma = self.gamma
        self.num_samples = int(self.params.setdefault('lstd_num_samples', 500))
        self.precond = self.params.setdefault('lstd_precond', 0.1)
        self.gamma = 1.0

    def init_stepsize(self, weights_shape, params):
        """Initializes the step-size variables, in this case meaning the A matrix and b vector.
        Args:
            weights_shape: Shape of the weights array
            params: Additional parameters.
        """
        # Data samples should hold num_samples, and each sample should
        # contain phi_t (|weights_shape|), state_t+1 (numStates), discState_t+1 (1), and reward_t (1)
        self.samples = numpy.zeros((self.num_samples, numpy.prod(weights_shape) + self.numStates + 2))
        self.lstd_counter = 0

    def shouldUpdate(self):
        self.lstd_counter += 1
        return self.lstd_counter % self.num_samples == 0

    def extractSample(self, sample):
        s = sample[:self.weights.size]
        state = sample[self.weights.size:self.weights.size+self.numStates]
        discState = sample[-2]
        qvalues = self.getActionValues(state, discState)
        a_p = self.getAction(state, discState)#values.argmax()
        s_p = numpy.zeros(self.weights.shape)
        s_p[discState, :, a_p] = self.basis.computeFeatures(state)
        return s, s_p.flatten(), sample[-1]

    def updateWeights(self):
        B = numpy.eye(self.weights.size) * self.precond
        b = numpy.zeros(self.weights.size)
        for sample in self.samples[:self.lstd_counter]:
            s, s_p, r = self.extractSample(sample)
            # B = matrix.SMInv(B, s, (s - self.lstd_gamma * s_p), 1.0)
            b += s * r
        self.weights = numpy.dot(B, b).reshape(self.weights.shape)

    def update(self, phi_t, state, discState, reward):
        index = self.lstd_counter % self.num_samples
        self.samples[index, :phi_t.size] = phi_t.flatten()
        self.samples[index, phi_t.size:phi_t.size + self.numStates] = state.copy() if state is not None else numpy.zeros((self.numStates,))
        self.samples[index, -2] = discState
        self.samples[index, -1] = reward
        if self.shouldUpdate():
            self.updateWeights()
