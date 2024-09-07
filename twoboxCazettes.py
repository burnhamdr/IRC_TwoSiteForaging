from __future__ import division
from MDPclass import *
from scipy.linalg import block_diag
from boxtask_func import *
from HMMtwoboxCol import *


# we need five different belief state transition matrices
a0 = 0    # a0 = do nothing
g0 = 1    # g0 = go to location 0
g1 = 2    # g1 = go toward box 1 (via location 0 if from 2)
g2 = 3    # g2 = go toward box 2 (via location 0 if from 1)
pb = 4    # pb  = push button

class twoboxCazettesMDP:
    def __init__(self, discount, nq, nr, na, nl, parameters):
        self.discount = discount
        self.nq = nq #number of belief states
        self.nr = nr #number of reward states, i.e. world states, active or inactive
        self.na = na #number of actions
        self.nl = nl   # number of locations
        self.n = self.nq * self.nq * self.nr * self.nl   # total number of outcomes, or unique system (world and agent) states
        self.parameters = parameters
        #every action has a transition matrix and a reward function.
        # transition matrix, per action each column defines the probability of transitioning to each other unique system state
        self.ThA = np.zeros((self.na, self.n, self.n)) 
        # for each action the probability of receiving a reward in each unique system state
        self.R = np.zeros((self.na, self.n, self.n)) # reward function

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
        rho = 1      # food in mouth is consumed
        prwd = self.parameters[0]   # reward is returned for button press at active location
        psw = self.parameters[1]    # location activity switches after button press
        actions = np.array([a0, g0, g1, g2, pb])
        NumCol = 2  # number of colors

        # State rewards
        Reward = 1   # reward per time step with food in mouth
        Groom = self.parameters[2]     # location 0 reward
        # Action costs
        travelCost = self.parameters[3]
        pushButtonCost = self.parameters[4]
        beta = 0     # available food dropped back into box after button press
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0

        # initialize probability distribution over states (belief and world)
        pr0 = np.array([1, 0])  # (r=0, r=1) initially no food in mouth p(R=0)=1.
        pl0 = np.array([1, 0, 0])  # (l=0, l=1, l=2) initial location is at L=0
        pb10 = np.insert(np.zeros(self.nq - 1), 0, 1)  # initial belief states (here, lowest availability)
        pb20 = np.insert(np.zeros(self.nq - 1), 0, 1)  # initial belief states (here, lowest availability)

        ph0 = kronn(pl0, pb10, pr0, pb20)
        # kronecker product of these initial distributions
        # Note that this ordering makes the subsequent products easiest

        # setup single-variable transition matrices for each action..
        # these calculate \overline{O}(o_{t+1}|b_{t},a_{t}) = \int ds_{t+1} ds_{t} O(o_{t+1}|s_{t+1}, a_{t}) * T(s_{t+1}|s_{t},a_{t})* B(s_{t}|b_{t})
        # and P(b_{t+1}|b_{t},a_{t},o_{t+1})
        # for box 1
        Tr = np.array([[1, rho], [0, 1 - rho]])  # consume reward
        self.Trans_belief_obs1, self.Obs_emis_trans1, self.den1 = beliefTransitionMatrixGaussianCazettes(prwd, psw, self.nq, actions, sigma = 1 / self.nq / 3)
        # for box 2
        self.Trans_belief_obs2, self.Obs_emis_trans2, self.den2 = beliefTransitionMatrixGaussianCazettes(prwd, psw, self.nq, actions, sigma = 1 / self.nq / 3)
        
        Tb2 = {}
        Tb1 = {}
        for action in actions:
            # belief transitions, it is  marginalized over observations, P(b_{t+1}|b_{t},a_{t},o_{t+1})
            Trans_belief1 = np.sum(self.Trans_belief_obs1[action], axis=0)
            Trans_belief2 = np.sum(self.Trans_belief_obs2[action], axis=0)
            Tb1[action] = Trans_belief1 / np.tile(np.sum(Trans_belief1, 0), (self.nq, 1))
            Tb2[action] = Trans_belief2 / np.tile(np.sum(Trans_belief2, 0), (self.nq, 1))
            
        # now fill out ThA i.e. 
        # \overline{T}(b_{t+1}|b_{t},a_{t}) = \int do_{t+1} P(b_{t+1}|b_{t},a_{t},o_{t+1})*\overline{O}(o_{t+1}|b_{t},a_{t})
        # where \overline{O}(o_{t+1}|b_{t},a_{t}) = \int ds_{t+1} ds_{t} O(o_{t+1}|s_{t+1}, a_{t}) * T(s_{t+1}|s_{t},a_{t})* B(s_{t}|b_{t})
        
        # ACTION: do nothing
        self.ThA[a0, :, :] = kronn(np.identity(self.nl), Tb1[a0], Tr, Tb2[a0])
        # Note: the order in which the kronecker products are taken is important because it
        # implies the organization of the ThA matrix. The ThA is organized into blocks, the first
        # are specific to the location.


        # ACTION: go to location 0/1/2
        #create location transition matrix for each unique location.
        #delta is the probability of not going to the target location
        #direct is the probability of going directly to the target location (skipping location 0)
        Tl0 = np.array(
            [[1, 1 - delta, 1 - delta], [0, delta, 0], [0, 0, delta]])  # go to loc 0 (with error of delta)
        Tl1 = np.array([[delta, 0, 1 - delta - direct], [1 - delta, 1, direct],
                        [0, 0, delta]])  # go to box 1 (with error of delta)
        Tl2 = np.array([[delta, 1 - delta - direct, 0], [0, delta, 0],
                        [1 - delta, direct, 1]])  # go to box 2 (with error of delta)
        #each entry in ThA is a product, for each action, of every possible system state combination
        #the kronecker product allows us to efficiently calculate these products by taking the iterative
        #kronecker product of the location, belief, reward consumption states.
        self.ThA[g0, :, :] = kronn(Tl0, Tb1[g0], Tr, Tb2[g0])
        self.ThA[g1, :, :] = kronn(Tl1, Tb1[g1], Tr, Tb2[g1])
        self.ThA[g2, :, :] = kronn(Tl2, Tb1[g2], Tr, Tb2[g2])

        # ACTION: push button.
        Tb1Tr = kronn(Tb1[pb], Tr)
        TrTb2 = kronn(Tr, Tb2[pb])
        Th = block_diag(np.identity(self.nq * self.nr * self.nq),
                        np.kron(Tb1Tr, np.identity(self.nq)),
                        np.kron(np.identity(self.nq), TrTb2))
        self.ThA[pb, :, :] = Th.dot(self.ThA[a0, :, :])

        Reward_h = tensorsumm(np.array([[Groom, 0, 0]]), np.zeros((1, self.nq)), np.array([[0, Reward]]),
                              np.zeros((1, self.nq)))
        Reward_a = - np.array([0, travelCost, travelCost, travelCost, pushButtonCost])

        [R1, R2, R3] = np.meshgrid(Reward_a.T, Reward_h, Reward_h, indexing='ij')
        Reward = R1 + R3
        # R = Reward[:, 0, :].T
        self.R = Reward

        #flip the transition matrices so past states are now rows
        #and future states are columns
        for i in range(self.na):
            self.ThA[i, :, :] = self.ThA[i, :, :].T

        self.Trans_hybrid_obs12 = np.zeros(((NumCol, NumCol, self.na, self.n, self.n)))
        for i in range(NumCol):
            for j in range(NumCol):
                self.Trans_hybrid_obs12[i, j, a0, :, :] = kronn(np.identity(self.nl),
                                                             self.den1[a0][i], Tr, self.den2[a0][j]).T
                self.Trans_hybrid_obs12[i, j, g0, :, :] = kronn(Tl0, self.den1[g0][i], Tr, self.den2[g0][j]).T
                self.Trans_hybrid_obs12[i, j, g1, :, :] = kronn(Tl1, self.den1[g1][i], Tr, self.den2[g1][j]).T
                self.Trans_hybrid_obs12[i, j, g2, :, :] = kronn(Tl2, self.den1[g2][i], Tr, self.den2[g2][j]).T
                self.Trans_hybrid_obs12[i, j, pb, :, :] = kronn(np.identity(self.nl),
                                                                self.den1[pb][i], Tr, self.den2[pb][j]).T
                self.Trans_hybrid_obs12[i, j, pb, :, :] = Th.dot(kronn(np.identity(self.nl), self.den1[pb][i], Tr, self.den2[pb][j])).T

    def solveMDP_op(self, epsilon = 0.001, niterations = 10000, initial_value=0):
        vi = ValueIteration_opZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        # optimal policy, stopping criterion changed to "converged Qvalue"
        vi.run()
        self.Q = self._QfromV(vi)
        self.policy = np.array(vi.policy)
        self.Vop = vi.V

    def solveMDP_sfm(self, epsilon = 0.001, niterations = 10000, initial_value=0):

        temperatureQ = self.parameters[5]
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

class twoboxCazettesMDPdata(twoboxCazettesMDP):
    def __init__(self, discount, nq, nr, na, nl, parameters, parametersExp,
                 sampleTime, sampleNum):
        twoboxCazettesMDP.__init__(self, discount, nq, nr, na, nl, parameters)

        self.parametersExp = parametersExp# parameters for the experiment
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

        self.actionDist = np.zeros((self.sampleNum, self.sampleTime, self.na))
        self.belief1Dist = np.zeros((self.sampleNum, self.sampleTime, self.nq))
        self.belief2Dist = np.zeros((self.sampleNum, self.sampleTime, self.nq))

        self.setupMDP()
        self.solveMDP_op()
        self.solveMDP_sfm()


    def dataGenerate_sfm(self):

        ## Parameters
        beta = 0     # available food dropped back into box after button press
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0
        rho = 1      # food in mouth is consumed
        # State rewards
        Reward = 1   # reward per time step with food in mouth

        Ncol = 1  # max value of color

        prwd_e = self.parametersExp[0]   # reward is returned for button press at active location
        psw_e = self.parametersExp[1]    # location activity switches after button press
        actions = np.array([a0, g0, g1, g2, pb])

        # State rewards
        Groom = self.parametersExp[2]     # location 0 reward
        # Action costs
        travelCost = self.parametersExp[3]
        pushButtonCost = self.parametersExp[4]

        ## Generate data
        for n in range(self.sampleNum):

            belief1Initial = np.random.randint(self.nq)
            rewInitial = np.random.randint(self.nr)
            belief2Initial = np.random.randint(self.nq)
            locationInitial = np.random.randint(self.nl)

            for t in range(self.sampleTime):
                if t == 0:
                    # Initialize the true world states, sensory information and latent states
                    active_site = np.random.randint(2)
                    if active_site == 0:
                        self.trueState1[n, t] = 1
                        self.trueState2[n, t] = 0
                    else:
                        self.trueState2[n, t] = 1
                        self.trueState1[n, t] = 0

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
                        if self.reward[n, t - 1] == 0:
                            self.reward[n, t] = 0
                        else:
                            #handle the case where the reward is not consumed,
                            #deterministically return 0 if the reward is comsumed,
                            #i.e. rho = 1 yields np.random.binomial(1, 0) = 0
                            self.reward[n, t] = np.random.binomial(1, 1 - rho)

                        self.belief1[n, t] = self.belief1[n, t - 1]  # use the previous belief state
                        self.belief2[n, t] = self.belief2[n, t - 1]  # use the previous belief state

                        if self.action[n, t - 1] == a0:#do nothing
                            # if the action is to do nothing, the location remains the same
                            self.location[n, t] = self.location[n, t - 1]
                        # if the action is to go to location 0, i.e. the middle location
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

                        if self.location[n, t] == 1:  # consider location 1 case
                            if self.trueState1[n, t - 1] == 0:#if the box is empty
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward[n, t] = 0
                                else:
                                    self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food
                            else:#if the box is active
                                self.reward[n, t] = np.random.binomial(1, prwd_e)  # give some reward with probability prwd

                            #check how this works with observations as rewards...
                            self.belief1Dist[n, t] = self.den1[pb][self.reward[n, t], :, self.belief1[n, t - 1]]
                            self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))

                            if self.reward[n, t] == 1:#if we received a reward at this time step
                                #then we know the other site is inactive
                                self.belief2Dist[n, t] = self.den2[pb][0, :, 0]
                                self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))

                        if self.location[n, t] == 2:  # consider location 2 case

                            if self.trueState2[n, t - 1] == 0:
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward[n, t] = 0
                                else:
                                    self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food
                            else:
                                self.reward[n, t] = np.random.binomial(1, prwd_e)  # give some reward

                            self.belief2Dist[n, t] = self.den2[pb][self.reward[n, t], :, self.belief2[n, t - 1]]
                            self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))

                            if self.reward[n, t] == 1:#if we received a reward at this time step
                                #then we know the other site is inactive
                                self.belief1Dist[n, t] = self.den1[pb][0, :, 0]
                                self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))

                    self.hybrid[n, t] = self.location[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) \
                                        + self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing
                    self.action[n, t] = self._chooseAction(self.softpolicy.T[self.hybrid[n, t]])
                    self.actionDist[n, t] = self.softpolicy.T[self.hybrid[n, t]]

    def _chooseAction(self, pvec):
        # Generate action according to multinomial distribution
        stattemp = np.random.multinomial(1, pvec)
        return np.argmax(stattemp)


class twoboxColCazettesMDP_der(twoboxCazettesMDP):
    """
    Derivatives of log_likelihood with respect to the parameters
    """

    def __init__(self, discount, nq, nr, na, nl, parameters):
        twoboxCazettesMDP.__init__(self, discount, nq, nr, na, nl, parameters)

        self.setupMDP()
        self.solveMDP_op()
        self.solveMDP_sfm()

    def dloglikelihhod_dpara_sim(self, obs):
        L = len(self.parameters)
        pi = np.ones(self.nq * self.nq) / self.nq / self.nq
        Numcol = 2 # number of colors
        Ncol = 1 # max value of color

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

                twoboxCol_perturb = twoboxCazettesMDP(self.discount, self.nq, self.nr, self.na, self.nl, para_perturb)
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















