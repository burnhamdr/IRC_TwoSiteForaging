from __future__ import division
from MDPclass import *
from scipy.linalg import block_diag
from boxtask_func import *
from HMMtwoboxCol import *


# we need five different transition matrices, one for each of the following actions:
a0 = 0    # a0 = do nothing
g0 = 1    # g0 = go to location 0
g1 = 2    # g1 = go toward box 1 (via location 0 if from 2)
g2 = 3    # g2 = go toward box 2 (via location 0 if from 1)
pb = 4    # pb  = push button

class twoboxColMDP:
    def __init__(self, discount, nq, nr, na, nl, parameters):
        self.discount = discount
        self.nq = nq# number of belief states
        self.nr = nr# number of reward states
        self.na = na# number of actions
        self.nl = nl   # number of locations
        self.n = self.nq * self.nq * self.nr * self.nl   # total number of states
        self.parameters = parameters  # [beta, gamma, epsilon, rho]
        self.ThA = np.zeros((self.na, self.n, self.n))
        self.R = np.zeros((self.na, self.n, self.n))

    def setupMDP(self):
        """
        Based on the parameters, create transition matrices and reward function.
        Implement the codes in file 'twoboxtask_ini.py'
        :return:
                ThA: transition probability,
                    shape: (# of action) * (# of states, old state) * (# of states, new state)
                R: reward function
                    shape: (# of action) * (# of states, old state) * (# of states, new state)
        """

        beta = 0     # available food dropped back into box after button press
        gamma1 = self.parameters[0]   # reward becomes available in box 1
        gamma2 = self.parameters[1]   # reward becomes available in box 2
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0
        epsilon1 = self.parameters[2] # available food disappears from box 1
        epsilon2 = self.parameters[3] # available food disappears from box 2
        rho = 1      # food in mouth is consumed

        # State rewards
        Reward = 1   # reward per time step with food in mouth
        Groom = self.parameters[4]     # location 0 reward
        # Action costs
        travelCost = self.parameters[5]
        pushButtonCost = self.parameters[6]

        NumCol = np.rint(self.parameters[7]).astype(int)   # number of colors
        Ncol = NumCol - 1  # max value of color
        qmin = self.parameters[8]
        qmax = self.parameters[9]

        # initialize probability distribution over states (belief and world)
        pr0 = np.array([1, 0])  # (r=0, r=1) initially no food in mouth p(R=0)=1.
        pl0 = np.array([1, 0, 0])  # (l=0, l=1, l=2) initial location is at L=0
        pb10 = np.insert(np.zeros(self.nq - 1), 0, 1)  # initial belief states (here, lowest availability)
        pb20 = np.insert(np.zeros(self.nq - 1), 0, 1)  # initial belief states (here, lowest availability)

        ph0 = kronn(pl0, pb10, pr0, pb20)
        # kronecker product of these initial distributions
        # Note that this ordering makes the subsequent products easiest

        # setup single-variable transition matrices
        Tr = np.array([[1, rho], [0, 1 - rho]])  # consume reward
        self.Trans_belief_obs1, self.Obs_emis_trans1, self.den1 = beliefTransitionMatrixGaussianCol(gamma1, epsilon1, qmin, qmax, Ncol,self.nq, sigma = 1 / self.nq / 3)
        Trans_belief1 = np.sum(self.Trans_belief_obs1, axis=0)      # belief transitions, it is  marginalized over observations
        Tb1 = Trans_belief1 / np.tile(np.sum(Trans_belief1, 0), (self.nq, 1))


        self.Trans_belief_obs2, self.Obs_emis_trans2, self.den2 = beliefTransitionMatrixGaussianCol(gamma2, epsilon2, qmin, qmax, Ncol, self.nq, sigma = 1 / self.nq / 3)
        Trans_belief2 = np.sum(self.Trans_belief_obs2, axis=0)      # belief transitions, it is  marginalized over observations
        Tb2 = Trans_belief2 / np.tile(np.sum(Trans_belief2, 0), (self.nq, 1)) #marginalzied over observation


        # ACTION: do nothing
        self.ThA[a0, :, :] = kronn(np.identity(self.nl), Tb1, Tr, Tb2)
        # kronecker product of these transition matrices


        # ACTION: go to location 0/1/2
        Tl0 = np.array(
            [[1, 1 - delta, 1 - delta], [0, delta, 0], [0, 0, delta]])  # go to loc 0 (with error of delta)
        Tl1 = np.array([[delta, 0, 1 - delta - direct], [1 - delta, 1, direct],
                        [0, 0, delta]])  # go to box 1 (with error of delta)
        Tl2 = np.array([[delta, 1 - delta - direct, 0], [0, delta, 0],
                        [1 - delta, direct, 1]])  # go to box 2 (with error of delta)
        self.ThA[g0, :, :] = kronn(Tl0, Tb1, Tr, Tb2)
        self.ThA[g1, :, :] = kronn(Tl1, Tb1, Tr, Tb2)
        self.ThA[g2, :, :] = kronn(Tl2, Tb1, Tr, Tb2)

        # ACTION: push button
        bL = (np.array(range(self.nq)) + 1 / 2) / self.nq

        #I think this is a combination of the Tr and Tb2 matrices because the size
        #would mostly work out. if you were to take the kronecker product of Tr and Tb2.
        #this also seems to be what the name suggests. It also seems to be accounting
        #for the removal of the reward from the box after the button is pressed if there
        #is any reward in the box.
        beta=0.1
        Trb2 = np.concatenate((np.array([np.insert(np.zeros(self.nq), 0, 1 - bL)]),
                               np.zeros((self.nq - 2, 2 * self.nq)),
                               np.array([np.insert([np.zeros(self.nq)], 0, beta * bL)]),
                               np.array([np.insert([(1 - beta) * bL], self.nq, 1 - bL)]),
                               np.zeros(((self.nq - 2), 2 * self.nq)),
                               np.array([np.insert([np.zeros(self.nq)], self.nq, bL)])), axis=0)
        Tb1r = reversekron(Trb2, np.array([2, self.nq])) #Tb1 reverse.. figuring out what Tb1 must have been from Trb2 with the reset
        Th = block_diag(np.identity(self.nq * self.nr * self.nq),
                        np.kron(Tb1r, np.identity(self.nq)),
                        np.kron(np.identity(self.nq), Trb2))
        self.ThA[pb, :, :] = Th.dot(self.ThA[a0, :, :])
        # self.ThA[pb, :, :] = Th

        Reward_h = tensorsumm(np.array([[Groom, 0, 0]]), np.zeros((1, self.nq)), np.array([[0, Reward]]),
                              np.zeros((1, self.nq)))
        Reward_a = - np.array([0, travelCost, travelCost, travelCost, pushButtonCost])

        [R1, R2, R3] = np.meshgrid(Reward_a.T, Reward_h, Reward_h, indexing='ij')
        Reward = R1 + R3
        # R = Reward[:, 0, :].T
        self.R = Reward

        for i in range(self.na):
            self.ThA[i, :, :] = self.ThA[i, :, :].T

        self.Trans_hybrid_obs12 = np.zeros(((NumCol, NumCol, self.na, self.n, self.n)))
        for i in range(NumCol):
            for j in range(NumCol):
                self.Trans_hybrid_obs12[i, j, a0, :, :] = kronn(np.identity(self.nl),
                                                             self.den1[i], Tr, self.den2[j]).T
                self.Trans_hybrid_obs12[i, j, g0, :, :] = kronn(Tl0, self.den1[i], Tr, self.den2[j]).T
                self.Trans_hybrid_obs12[i, j, g1, :, :] = kronn(Tl1, self.den1[i], Tr, self.den2[j]).T
                self.Trans_hybrid_obs12[i, j, g2, :, :] = kronn(Tl2, self.den1[i], Tr, self.den2[j]).T
                self.Trans_hybrid_obs12[i, j, pb, :, :] = (block_diag(np.identity(self.nq * self.nr * self.nq),
                                                                     np.kron(Tb1r, np.identity(self.nq)),
                                                                     np.kron(np.identity(self.nq), Trb2)).dot(
                    kronn(np.identity(self.nl), self.den1[i], Tr, self.den2[j]))).T


    def solveMDP_op(self, epsilon = 0.001, niterations = 10000, initial_value=0):
        vi = ValueIteration_opZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        # optimal policy, stopping criterion changed to "converged Qvalue"
        vi.run()
        self.Q = self._QfromV(vi)
        self.policy = np.array(vi.policy)
        self.Vop = vi.V

    def solveMDP_sfm(self, epsilon = 0.001, niterations = 10000, initial_value=0):

        temperatureQ = self.parameters[10]
        vi = ValueIteration_sfmZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        vi.run(temperatureQ)
        self.Qsfm = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        self.softpolicy = np.array(vi.softpolicy)
        self.Vsfm = vi.V

    def _QfromV(self, ValueIteration):
        Q = np.zeros((ValueIteration.A, ValueIteration.S)) # Q is of shape: na * n
        for a in range(ValueIteration.A):
            Q[a, :] = ValueIteration.R[a] + ValueIteration.discount * \
                                            ValueIteration.P[a].dot(ValueIteration.V)
        return Q

class twoboxColMDPdata(twoboxColMDP):
    def __init__(self, discount, nq, nr, na, nl, parameters, parametersExp,
                 sampleTime, sampleNum):
        twoboxColMDP.__init__(self, discount, nq, nr, na, nl, parameters)

        self.parametersExp = parametersExp
        self.sampleNum = sampleNum
        self.sampleTime = sampleTime

        self.action = np.empty((self.sampleNum, self.sampleTime), int)  # initialize action, assumed to be optimal
        self.hybrid = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hybrid state.
        # Here it is the joint state of reward and belief
        self.location = np.empty((sampleNum, sampleTime), int)  # initialize location state
        self.belief1 = np.empty((self.sampleNum, self.sampleTime), int)
        self.belief2 = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hidden state, belief state
        self.reward = np.empty((self.sampleNum, self.sampleTime), int)  # initialize reward state
        self.trueState1 = np.zeros((self.sampleNum, self.sampleTime))
        self.trueState2 = np.zeros((self.sampleNum, self.sampleTime))
        self.color1 = np.zeros((self.sampleNum, self.sampleTime), dtype=int)
        self.color2 = np.zeros((self.sampleNum, self.sampleTime), dtype=int)

        self.actionDist = np.zeros((self.sampleNum, self.sampleTime, self.na))
        self.belief1Dist = np.zeros((self.sampleNum, self.sampleTime, self.nq))
        self.belief2Dist = np.zeros((self.sampleNum, self.sampleTime, self.nq))

        self.setupMDP()
        self.solveMDP_op()
        self.solveMDP_sfm()


    def dataGenerate_sfm(self):

        ## Parameters
        beta = 0     # available food dropped back into box after button press
        gamma1 = self.parameters[0]   # reward becomes available in box 1
        gamma2 = self.parameters[1]   # reward becomes available in box 2
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0
        epsilon1 = self.parameters[2] # available food disappears from box 1
        epsilon2 = self.parameters[3] # available food disappears from box 2
        rho = 1      # food in mouth is consumed
        # State rewards
        Reward = 1   # reward per time step with food in mouth
        Groom = self.parameters[4]     # location 0 reward
        # Action costs
        travelCost = self.parameters[5]
        pushButtonCost = self.parameters[6]

        NumCol = np.rint(self.parameters[7]).astype(int)   # number of colors
        Ncol = NumCol - 1  # max value of color
        qmin = self.parameters[8]
        qmax = self.parameters[9]

        gamma1_e = self.parametersExp[0]
        gamma2_e = self.parametersExp[1]
        epsilon1_e = self.parametersExp[2]
        epsilon2_e = self.parametersExp[3]
        qmin_e = self.parametersExp[4]
        qmax_e = self.parametersExp[5]

        ## Generate data
        for n in range(self.sampleNum):

            belief1Initial = np.random.randint(self.nq)
            rewInitial = np.random.randint(self.nr)
            belief2Initial = np.random.randint(self.nq)
            locationInitial = np.random.randint(self.nl)

            for t in range(self.sampleTime):
                if t == 0:
                    # Initialize the true world states, sensory information and latent states
                    self.trueState1[n, t] = np.random.binomial(1, gamma1_e)
                    self.trueState2[n, t] = np.random.binomial(1, gamma2_e)
                    q1 = self.trueState1[n, t] * qmin_e + (1 - self.trueState1[n, t]) * qmax_e
                    self.color1[n, t] = np.random.binomial(Ncol, q1)  # color for box 1
                    q2 = self.trueState2[n, t] * qmin_e + (1 - self.trueState2[n, t]) * qmax_e
                    self.color2[n, t] = np.random.binomial(Ncol, q2)  # color for box 2

                    self.location[n, t], self.reward[n, t] = locationInitial, rewInitial

                    self.belief1[n, t], self.belief2[n, t] = belief1Initial, belief2Initial
                    self.belief1Dist[n, t, self.belief1[n, t]] = 1
                    self.belief2Dist[n, t, self.belief2[n, t]] = 1

                    self.hybrid[n, t] = self.location[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) + \
                                        self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing
                    self.actionDist[n, t] = self.softpolicy.T[self.hybrid[n, t]]
                    self.action[n, t] = self._chooseAction(self.actionDist[n, t])

                else:
                    # variables evolve with dynamics
                    if self.action[n, t - 1] != pb:
                        # button not pressed, then true world dynamic is not affected by actions
                        if self.trueState1[n, t - 1] == 0:
                            self.trueState1[n, t] = np.random.binomial(1, gamma1_e)
                        else:
                            self.trueState1[n, t] = 1 - np.random.binomial(1, epsilon1_e)

                        if self.trueState2[n, t - 1] == 0:
                            self.trueState2[n, t] = np.random.binomial(1, gamma2_e)
                        else:
                            self.trueState2[n, t] = 1 - np.random.binomial(1, epsilon2_e)

                        q1 = self.trueState1[n, t] * qmin_e + (1 - self.trueState1[n, t]) * qmax_e
                        self.color1[n, t] = np.random.binomial(Ncol, q1)  # color for box 1
                        q2 = self.trueState2[n, t] * qmin_e + (1 - self.trueState2[n, t]) * qmax_e
                        self.color2[n, t] = np.random.binomial(Ncol, q2)  # color for box 2

                        self.belief1Dist[n, t] = self.den1[self.color1[n, t], :, self.belief1[n, t - 1]]
                        self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))
                        self.belief2Dist[n, t] = self.den2[self.color2[n, t], :, self.belief2[n, t - 1]]
                        self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))

                        if self.reward[n, t - 1] == 0:
                            self.reward[n, t] = 0
                        else:
                            self.reward[n, t] = np.random.binomial(1, 1 - rho)

                        if self.action[n, t - 1] == a0:
                            self.location[n, t] = self.location[n, t - 1]
                        if self.action[n, t - 1] == g0:
                            Tl0 = np.array(
                                [[1, 1 - delta, 1 - delta], [0, delta, 0],
                                 [0, 0, delta]])  # go to loc 0 (with error of delta)
                            self.location[n, t] = np.argmax(np.random.multinomial(1, Tl0[:, self.location[n, t-1]], size  = 1))
                        if self.action[n, t - 1] == g1:
                            Tl1 = np.array([[delta, 0, 1 - delta - direct], [1 - delta, 1, direct],
                                            [0, 0, delta]])  # go to box 1 (with error of delta)
                            self.location[n, t] = np.argmax(np.random.multinomial(1, Tl1[:, self.location[n, t - 1]], size  = 1))
                        if self.action[n, t - 1] == g2:
                            Tl2 = np.array([[delta, 1 - delta - direct, 0], [0, delta, 0],
                                            [1 - delta, direct, 1]])  # go to box 2 (with error of delta)
                            self.location[n, t] = np.argmax(np.random.multinomial(1, Tl2[:, self.location[n, t - 1]], size  = 1))

                    if self.action[n, t - 1] == pb:  # press button
                        self.location[n, t] = self.location[n, t - 1]  # pressing button does not change location

                        if self.location[n, t - 1] == 0:
                            # pressing button at the center does not change anything
                            # then wait an intermediate step (everything evolves as if doing nothing)
                            self.reward[n, t] = 0

                            if self.trueState1[n, t - 1] == 0:
                                self.trueState1[n, t] = np.random.binomial(1, gamma1_e)
                            else:
                                self.trueState1[n, t] = 1 - np.random.binomial(1, epsilon1_e)

                            if self.trueState2[n, t - 1] == 0:
                                self.trueState2[n, t] = np.random.binomial(1, gamma2_e)
                            else:
                                self.trueState2[n, t] = 1 - np.random.binomial(1, epsilon2_e)

                            q1 = self.trueState1[n, t] * qmin_e + (1 - self.trueState1[n, t]) * qmax_e
                            self.color1[n, t] = np.random.binomial(Ncol, q1)  # color for box 1
                            q2 = self.trueState2[n, t] * qmin_e + (1 - self.trueState2[n, t]) * qmax_e
                            self.color2[n, t] = np.random.binomial(Ncol, q2)  # color for box 2

                            self.belief1Dist[n, t] = self.den1[self.color1[n, t], :, self.belief1[n, t - 1]]
                            self.belief2Dist[n, t] = self.den2[self.color2[n, t], :, self.belief2[n, t - 1]]
                            self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t],size=1))
                            self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))

                        if self.location[n, t - 1] == 1:  # consider location 1 case

                            if self.trueState1[n, t - 1] == 0:
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward[n, t] = 0
                                else:
                                    self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food
                            else:
                                self.reward[n, t] = 1  # give some reward

                            self.trueState1[n, t] = np.random.binomial(1, gamma1_e)  # after pressing button, the box is empty,
                            # then during a intermediate waiting time, it follows the dynamic
                            q1 = self.trueState1[n, t] * qmin_e + (1 - self.trueState1[n, t]) * qmax_e
                            self.color1[n, t] = np.random.binomial(Ncol, q1)  # color for box 1

                            # belief on box 2 is independent on box 1
                            if self.trueState2[n, t - 1] == 0:
                                self.trueState2[n, t] = np.random.binomial(1, gamma2_e)
                            else:
                                self.trueState2[n, t] = 1 - np.random.binomial(1, epsilon2_e)
                            q2 = self.trueState2[n, t] * qmin_e + (1 - self.trueState2[n, t]) * qmax_e
                            self.color2[n, t] = np.random.binomial(Ncol, q2)  # color for box 2

                            self.belief1Dist[n, t] = self.den1[self.color1[n, t], :, 0]   # after pressing button, the food is gone,
                            # belief is reset to zero. Based on whatever color it is generated after the intermediate waiting time, belief is updated.
                            self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))
                            self.belief2Dist[n, t] = self.den2[self.color2[n, t], :, self.belief2[n, t - 1]]
                            self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))

                        if self.location[n, t - 1] == 2:  # consider location 2 case

                            if self.trueState2[n, t - 1] == 0:
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward[n, t] = 0
                                else:
                                    self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food
                            else:
                                self.reward[n, t] = 1  # give some reward

                                # belief on box 1 is independent on box 2
                            if self.trueState1[n, t - 1] == 0:
                                self.trueState1[n, t] = np.random.binomial(1, gamma1_e)
                            else:
                                self.trueState1[n, t] = 1 - np.random.binomial(1, epsilon1_e)

                            q1 = self.trueState1[n, t] * qmin_e + (1 - self.trueState1[n, t]) * qmax_e
                            self.color1[n, t] = np.random.binomial(Ncol, q1)  # color for box 2

                            self.trueState2[n, t] = np.random.binomial(1, gamma2_e)
                            q2 = self.trueState2[n, t] * qmin_e + (1 - self.trueState2[n, t]) * qmax_e
                            self.color2[n, t] = np.random.binomial(Ncol, q2)  # color for box 2

                            self.belief1Dist[n, t] = self.den1[self.color1[n, t], :, self.belief1[n, t - 1]]
                            self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))
                            self.belief2Dist[n, t] = self.den2[self.color2[n, t], :, 0]
                            self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))


                    self.hybrid[n, t] = self.location[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) \
                                        + self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing
                    self.action[n, t] = self._chooseAction(self.softpolicy.T[self.hybrid[n, t]])
                    self.actionDist[n, t] = self.softpolicy.T[self.hybrid[n, t]]

    def _chooseAction(self, pvec):
        # Generate action according to multinomial distribution
        stattemp = np.random.multinomial(1, pvec)
        return np.argmax(stattemp)


class twoboxColMDP_der(twoboxColMDP):
    """
    Derivatives of log_likelihood with respect to the parameters
    """

    def __init__(self, discount, nq, nr, na, nl, parameters):
        twoboxColMDP.__init__(self, discount, nq, nr, na, nl, parameters)

        self.setupMDP()
        self.solveMDP_op()
        self.solveMDP_sfm()

    def dloglikelihhod_dpara_sim(self, obs):
        L = len(self.parameters)
        pi = np.ones(self.nq * self.nq) / self.nq / self.nq
        Numcol = np.rint(self.parameters[7]).astype(int) # number of colors
        Ncol = Numcol - 1  # number value: 0 top Numcol-1

        twoboxColHMM = HMMtwoboxCol(self.ThA, self.softpolicy,
                                    self.Trans_hybrid_obs12, self.Obs_emis_trans1,
                                    self.Obs_emis_trans2, pi, Ncol)
        log_likelihood =  twoboxColHMM.computeQaux(obs, self.ThA, self.softpolicy, self.Trans_hybrid_obs12, self.Obs_emis_trans1, self.Obs_emis_trans2) + \
                          twoboxColHMM.latent_entr(obs)

        perturb = 10 ** -6

        dloglikelihhod_dpara = np.zeros(L)

        for i in range(L):
            if i != 7: # if not color number parameter
                para_perturb = np.copy(self.parameters)
                para_perturb[i] = para_perturb[i] + perturb

                twoboxCol_perturb = twoboxColMDP(self.discount, self.nq, self.nr, self.na, self.nl, para_perturb)
                twoboxCol_perturb.setupMDP()
                twoboxCol_perturb.solveMDP_sfm()
                ThA_perturb = twoboxCol_perturb.ThA
                policy_perturb = twoboxCol_perturb.softpolicy
                Trans_hybrid_obs12_perturb = twoboxCol_perturb.Trans_hybrid_obs12
                Obs_emis_trans1_perturb = twoboxCol_perturb.Obs_emis_trans1
                Obs_emis_trans2_perturb = twoboxCol_perturb.Obs_emis_trans2
                twoboxColHMM_perturb = HMMtwoboxCol(ThA_perturb, policy_perturb, Trans_hybrid_obs12_perturb,
                                            Obs_emis_trans1_perturb, Obs_emis_trans2_perturb, pi, Ncol)

                log_likelihood_perturb = twoboxColHMM_perturb.computeQaux(obs, ThA_perturb, policy_perturb, Trans_hybrid_obs12_perturb,
                                            Obs_emis_trans1_perturb, Obs_emis_trans2_perturb) + twoboxColHMM_perturb.latent_entr(obs)

                dloglikelihhod_dpara[i] = (log_likelihood_perturb - log_likelihood) / perturb

        return dloglikelihhod_dpara















