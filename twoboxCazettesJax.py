from __future__ import division
from MDPclassJax import *
from jax.scipy.linalg import block_diag
from boxtask_funcJax import *
from HMMtwoboxCazettesJax import *
from itertools import permutations

import jax.numpy as jnp
import jax
from jax import vmap, pmap, grad
from pprint import pprint
from jax.lib import xla_bridge
import jaxlib
from jax import jit
import jax.lax as lax
from tensorflow_probability.substrates import jax as tfp

jax.config.update("jax_enable_x64", True)

import multiprocessing
from joblib import Parallel, delayed

import numpy as np
import jax.numpy as jnp

a0 = 0    # a0 = do nothing
g0 = 1    # g0 = go (i.e. travel/switch sites via location 0)
pb = 2    # pb  = push button

#jax multinomial function uses jax from tesnorflow probability
def multinomial(key, n, p, shape=()):
    return tfp.distributions.Multinomial(n, probs=p).sample(
        seed=key,
        sample_shape=shape,
    )

def create_ordered_tuples(K):
    # Step 1: Generate all possible combinations where i != j
    all_combinations = list(permutations(range(K), 2))
    
    # Step 2: Sort the combinations by the second entry first, then the first entry second
    sorted_combinations = sorted(all_combinations, key=lambda x: (x[1], x[0]))
    
    # Step 3: Keep only the tuples where one of the entries is 0
    filtered_combinations = [tup for tup in sorted_combinations if 0 in tup]
    
    return filtered_combinations

def mirror_column_ranks(matrix):
    # Get the rank of the elements in each column, and mirror them
    ranks = np.argsort(matrix, axis=0)
    mirrored_matrix = np.empty_like(matrix)
    
    # Flip the ranks: max_rank - current rank
    for i in range(matrix.shape[1]):  # Iterate over each column
        mirrored_matrix[:, i] = matrix[:, i][ranks[:, i]]
    
    return mirrored_matrix

class twoboxCazettesMDP:
    def __init__(self, discount, nq, nr, na, nl, parameters, lick_state=False):
        self.lick_state = lick_state
        self.discount = discount
        self.nq = nq #number of belief states
        self.nr = nr #number of reward states, i.e. world states, active or inactive
        self.na = na #number of actions
        self.nl = nl   # number of locations
        self.nlwh = 2*(nl - 1)   # number of locations with location history
        self.nlpbh = 2 # number of previous button press history locations (0 or 1)

        self.n = lax.cond(
            lick_state,
            lambda _: self.nq * self.nq * self.nr * self.nlwh * self.nlpbh,
            lambda _: self.nq * self.nq * self.nr * self.nlwh,
            operand=None
        )
        
        self.parameters = parameters
        #every action has a transition matrix and a reward function.
        # transition matrix, per action each column defines the probability of transitioning to each other unique system state
        self.ThA = jnp.zeros((self.na, self.n, self.n))
        # for each action the probability of receiving a reward in each unique system state
        self.R = jnp.zeros((self.na, self.n, self.n)) # reward function
        #set up location zero action mask
        # Shape and slice size
        shape = (self.na, self.nq*self.nq*self.nr*self.nlwh*self.nlpbh)
        slice_size = 2 * self.nq *  self.nr *  self.nq *  self.nlpbh
        loc0_mask_jax = jnp.ones(shape)
        update_slice = jnp.zeros((1, slice_size))
        self.loc0_mask = jax.lax.dynamic_update_slice(loc0_mask_jax, update_slice, (pb, 0))

    #location and rewards select the possible states
    #in the case with lick states, the previous action is also considered
    def _states_wLickStates(self, r_, l_, a_):
        temp = jnp.reshape(np.array(range(self.nq)), [1, self.nq])
        return jnp.squeeze(l_ * self.nq * self.nq * self.nr * self.nlpbh + a_ * self.nq * self.nq * self.nr + 
                            tensorsum(temp * self.nr * self.nq, r_ * self.nq + temp)).astype(int)


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
        psw = self.parameters[0]   # location activity switches after button press
        prwd = self.parameters[1]    # reward is returned for button press at active location
        Groom = self.parameters[2]   # location 0 reward
        travelCost = self.parameters[3]
        pushButtonCost = self.parameters[4]
        startPushButtonCost = self.parameters[5]
        stopPushButtonCost = 0.0#self.parameters[6]
        portRestReward = self.parameters[6]

        actions = jnp.array([a0, g0, pb])
        locations = jnp.arange(self.nl)
        locations_with_history = create_ordered_tuples(self.nl)
        pb_history = create_ordered_tuples(2)

        NumCol = 2  # number of colors
        # State rewards
        r_val = 1
        doNothingCost = 0.0
        beta = 0     # available food dropped back into box after button press
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0

        # setup single-variable transition matrices for each action..
        # these calculate \overline{O}(o_{t+1}|b_{t},a_{t}) = \int ds_{t+1} ds_{t} O(o_{t+1}|s_{t+1}, a_{t}) * T(s_{t+1}|s_{t},a_{t})* B(s_{t}|b_{t})
        # and P(b_{t+1}|b_{t},a_{t},o_{t+1})
        # for box 1
        Tr = jnp.array([[1, rho], [0, 1 - rho]]).astype(jnp.float32)  # consume reward
        Trpb = jnp.array([[0.5,0.5], [0.5, 0.5]])

        self.Trans_belief_obs, self.Obs_emis_trans, self.den = beliefTransitionMatrixGaussianCazettes(psw, prwd, self.nq, actions, locations, sigma = 1 / self.nq / 3)
        self.Obs_emis_trans1 = self.Obs_emis_trans[1]
        self.Obs_emis_trans2 = self.Obs_emis_trans[2]
        belief_locations = jnp.array([1, 2])
        Tb_unpack = {bloc.item():{oloc.item():{} for oloc in belief_locations} for bloc in belief_locations}
        for bloc in belief_locations:#belief location is the location of the agent
            bloc = bloc.item()
            for oloc in belief_locations:#other locations where the agent is NOT located
                oloc = oloc.item()
                for action in actions:
                    # belief transitions, it is  marginalized over observations, P(b_{t+1}|b_{t},a_{t},o_{t+1})
                    action = action.item()
                    Trans_belief = jnp.sum(self.Trans_belief_obs[bloc][oloc][action], axis=0)
                    Tb_unpack[bloc][oloc][action] = Trans_belief / jnp.tile(jnp.sum(Trans_belief, 0), (self.nq, 1))#normalize each column
        
        # now fill out ThA i.e.
        # \overline{T}(b_{t+1}|b_{t},a_{t}) = \int do_{t+1} P(b_{t+1}|b_{t},a_{t},o_{t+1})*\overline{O}(o_{t+1}|b_{t},a_{t})
        # where \overline{O}(o_{t+1}|b_{t},a_{t}) = \int ds_{t+1} ds_{t} O(o_{t+1}|s_{t+1}, a_{t}) * T(s_{t+1}|s_{t},a_{t})* B(s_{t}|b_{t})
        # Note: the order in which the kronecker products are taken is important because it
        # implies the organization of the ThA matrix. The ThA is organized into blocks, the first
        # are specific to the location.

        # #For debugging and housekeeping, we will also create the labels for the ThA and R matrices
        # #first make the labels for ThA using the kronn written for strings
        # loc_labels = jnp.array([str(loc) for loc in locations])
        # loc_labels_with_history = jnp.array([str(loc) for loc in locations_with_history])
        # ac_history_labels = jnp.array([str(pb) + '_' + str(loc) for loc in range(2)])

        # bel1_labels = jnp.array(['b1_' + str(bel) for bel in range(self.nq)])
        # bel2_labels = jnp.array(['b2_' + str(bel) for bel in range(self.nq)])
        # rew_labels = jnp.array(['r_' + str(rew) for rew in range(self.nr)])
        # #make the transition matrices for the labels using the string meshgrid
        # loc_mesh = string_meshgrid(loc_labels_with_history, loc_labels_with_history)
        # rew_mesh = string_meshgrid(rew_labels, rew_labels)
        # bel1_mesh = string_meshgrid(bel1_labels, bel1_labels)
        # bel2_mesh = string_meshgrid(bel2_labels, bel2_labels)
        # ac_history_mesh = string_meshgrid(ac_history_labels, ac_history_labels)

        # def true_fn_labels(_):
        #     ThA_labels = kronn_str(loc_mesh, ac_history_mesh, bel1_mesh, rew_mesh, bel2_mesh)
        #     R_labels = tensorsumm_str(loc_labels_with_history[:, None], ac_history_labels[:, None],
        #                             bel1_labels[None, :], rew_labels[:, None], bel2_labels[None, :]).flatten()
        #     return ThA_labels, R_labels

        # def false_fn_labels(_):
        #     ThA_labels = kronn_str(loc_mesh, bel1_mesh, rew_mesh, bel2_mesh)
        #     R_labels = tensorsumm_str(loc_labels_with_history[:, None], bel1_labels[None, :],
        #                             rew_labels[:, None], bel2_labels[None, :]).flatten()
        #     return ThA_labels, R_labels

        # self.ThA_labels, self.R_labels = lax.cond(self.lick_state, true_fn_labels, false_fn_labels, operand=None)

        #technically there are only two foraging sites and 1 intermediate site..
        #but to capture the restrictions on the agent's movement, where it cannot transition directly
        #from one foraging site to the other, and it cannot transition from the intermediate site
        #back to the foraging site it came from, we will create a transition matrix for each possible
        #(previous location, current location) combination that is possible.
        #Here these will be (1, 0), (2, 0), (0, 1), (0, 2).
        #keeping with the current state on the columns and next state on the rows convention,
        #the way to read this transition matrix is what previous location and current location tuple
        #is transitioning to the current location and next location tuple.
        #first column: (1, 0) -> (0, 2) has to go to site 2 if coming from site 1, can error to stay at site 0
        #second column: (2, 0) -> (0, 1) has to go to site 1 if coming from site 2, can error to stay at site 0
        #third column: (0, 1) -> (1, 0) has to return to the intermediate site, can error to stay at site 1
        #fourth column: (0, 2) -> (2, 0) has to return to the intermediate site, can error to stay at site 2
        Tl = jnp.array([[delta, 0., 1. - delta, 0.], 
                       [0., delta, 0., 1. - delta],
                       [0., 1. - delta, delta, 0.],
                       [1. - delta, 0., 0., delta]]).astype(jnp.float32)

        def true_fn_trans(_):
            #we only need Tpb0 here because Tlpb is the state transition matrix for the push button
            #action in the middle locations. As such, this push button action should not transition
            #the state space into or out of push button history state.
            Tpb0 = jnp.array([[1, 1],
                              [0, 0]]).astype(jnp.float32)
            Tla0 = kronn(jnp.identity(self.nlwh), Tpb0)
            Tlg0 = kronn(Tl, Tpb0)
            Tlpb = kronn(Tpb0, jnp.identity(self.nq), Tr, jnp.identity(self.nq))
            return Tla0, Tlg0, Tlpb, Tla0.shape, Tlg0.shape, Tlpb.shape

        def false_fn_trans(_):
            Tla0 = jnp.identity(self.nlwh)
            Tlg0 = Tl
            Tlpb = kronn(jnp.identity(self.nq), Tr, jnp.identity(self.nq))
            # Pad the arrays to match the size of the true function's outputs
            Tla0_padded = jnp.pad(Tla0, ((0, Tla0.shape[0]), (0, Tla0.shape[1])), mode='constant')
            Tlg0_padded = jnp.pad(Tlg0, ((0, Tlg0.shape[0]), (0, Tlg0.shape[1])), mode='constant')
            Tlpb_padded = jnp.pad(Tlpb, ((0, Tlpb.shape[0]), (0, Tlpb.shape[1])), mode='constant')
            return Tla0_padded, Tlg0_padded, Tlpb_padded, Tla0.shape, Tlg0.shape, Tlpb.shape

        # Use lax.cond with padded arrays
        Tla0, Tlg0, Tlpb, Tla0_shape, Tlg0_shape, Tlpb_shape = lax.cond(
            self.lick_state, true_fn_trans, false_fn_trans, operand=None
        )

        # Use lax.dynamic_slice to slice the arrays back to their original sizes
        self.Tla0 = lax.dynamic_slice(Tla0, (0, 0), (Tla0_shape[0], Tla0_shape[0]))
        self.Tlg0 = lax.dynamic_slice(Tlg0, (0, 0), (Tlg0_shape[0], Tlg0_shape[0]))
        self.Tlpb = lax.dynamic_slice(Tlpb, (0, 0), (Tlpb_shape[0], Tlpb_shape[0]))


        #make per location belief transition matrices. We will also absorb the reward state
        #into these belief matrices so that we can construct the total state space transition
        #matrices.
        Tb = {bloc.item():{} for bloc in belief_locations}
        for action in actions:
            action = action.item()
            for bloc in belief_locations:#belief location is the location of the agent
                bloc = bloc.item()
                Tb_loc = Tb_unpack[bloc]
                #for any action but push button, thee belief transition matrices should be identity
                Tb_temp_n0 = self.Trans_belief_obs[bloc][bloc][action][0]#observe no reward at current location
                Tb_temp_n1 = self.Trans_belief_obs[bloc][bloc][action][1]#observe reward at current location
                #enforcing this block structure allows for encoding how rewards vs. no rewards
                #transitions the state space into and out of the reward state (Tr) otherwise there is no
                #mechanism for transitioning into the reward state. Each block of Tb_block_temp corresponds
                #to the belief transitions for each reward state.
                Tb_block_temp = jnp.block([
                                    [Tb_temp_n0, Tb_temp_n0],
                                    [Tb_temp_n1, Tb_temp_n1]
                                    ]).astype(jnp.float32)
                #here we use Tpb1, i.e. the push button history transition matrix for the push button action.
                #pushing the button means that either previous push button state will be transitioned into
                #having push button history. So state 0, no push button history, will transition into state 1
                #push button history, and state 1 push button history will transition into state 1 still having 
                #push button history.
                Tpb1 = jnp.array([[0, 0],
                                  [1, 1]]).astype(jnp.float32)
                oloc = int((2 - bloc) + 1)#should only be one other location
                def true_fn_bloc1(_):
                    def true_fn_lick_state(_):
                        out = kronn(Tpb1, Tb_block_temp, Tb_unpack[bloc][oloc][action])
                        return out, out.shape

                    def false_fn_lick_state(_):
                        out = kronn(Tb_block_temp, Tb_unpack[bloc][oloc][action])
                        out_padded = jnp.pad(out, ((0, out.shape[0]), (0, out.shape[1])), mode='constant')
                        return out_padded, out.shape

                    return lax.cond(self.lick_state, true_fn_lick_state, false_fn_lick_state, operand=None)

                def false_fn_bloc1(_):
                    def true_fn_lick_state(_):
                        out = kronn(Tpb1, Tb_unpack[bloc][oloc][action], Tb_block_temp)
                        return out, out.shape

                    def false_fn_lick_state(_):
                        out = kronn(Tb_unpack[bloc][oloc][action], Tb_block_temp)
                        out_padded = jnp.pad(out, ((0, out.shape[0]), (0, out.shape[1])), mode='constant')
                        return out_padded, out.shape

                    return lax.cond(self.lick_state, true_fn_lick_state, false_fn_lick_state, operand=None)

                result, original_shape = lax.cond(bloc == 1, true_fn_bloc1, false_fn_bloc1, operand=None)
                sliced_result = lax.dynamic_slice(result, (0, 0), (original_shape[0], original_shape[1]))

                Tb[bloc][action] = sliced_result
        
        #flip the transition matrices so current states are now rows
        #and future states are columns
        # ACTION: do nothing
        self.ThA = self.ThA.at[a0, :, :].set(jnp.transpose(kronn(self.Tla0, jnp.identity(self.nq), Tr, jnp.identity(self.nq))))
        # self.ThA[a0, :, :] = kronn(Tla0, Tb_unpack[1][1][a0], Tr, Tb_unpack[2][2][a0])

        # ACTION: go to location 0/1/2
        #each entry in ThA is a product, for each action, of every possible system state combination
        #the kronecker product allows us to efficiently calculate these products by taking the iterative
        #kronecker product of the location, belief, reward consumption states.
        self.ThA = self.ThA.at[g0, :, :].set(jnp.transpose(kronn(self.Tlg0, jnp.identity(self.nq), Tr, jnp.identity(self.nq))))
        # self.ThA[g0, :, :] = kronn(Tlg0, Tb_unpack[1][1][g0], Tr, Tb_unpack[2][2][g0])

        # ACTION: push button.
        #the way Th is structured, the first two blocks are for location 0
        #the next block is for location 1 and the last block is for location 2

        # push button (with error of beta) transitions agent from 
        # action history of pb to the next action history, if the pb action
        # is taken and pb was not previously taken, then the agent will
        # transition to pb true as the history with probability 1 - beta.
        # if the pb action is taken and pb was previously taken, then the agent
        # will transition to pb false as the history with probability 1 - beta.
        Th = block_diag(self.Tlpb, self.Tlpb, Tb[1][pb], Tb[2][pb]).astype(jnp.float64)
        column_sums = jnp.sum(Th, axis=0)
        # Compute the absolute differences between the column sums and 1
        differences = jnp.abs(column_sums - 1)
        # Check if all differences are greater than zero and less than 1E-6
        cond_zero = jnp.all(differences == 0.0)
        cond_less_than = jnp.all((differences < 1E-6))
        if not cond_zero and cond_less_than:
            #renormalize the columns of Th
            rep_col_sums = jnp.repeat(column_sums[jnp.newaxis,:], self.n, axis=0)
            Th_norm = Th / rep_col_sums
            column_sums = jnp.sum(Th_norm, axis=0)
            differences = jnp.abs(column_sums - 1.0)
            cond_zero_ = jnp.all(differences < 1E-8)
            assert cond_zero_, 'Columns of ThA are not normalized correctly'
            Th = Th_norm
        else:
            raise ValueError('Columns of ThA do not form valid probability matrix')

        #this dot product is of the ith push button NEXT system state with the jth 
        #do nothing system CURRENT system state.
        self.ThA = self.ThA.at[pb, :, :].set(jnp.transpose(Th))

        # The reward function is the expecation of the reward on the next time step
        # given the current state and action.
        # accumulate rewards for each possible location, belief state 1, reward consumption state, and
        # belief state.         
        #complie the action costs.
        loc_rewards = jnp.array([[Groom, Groom, 0, 0]])
        def true_fn_reward(_):
            #for all other actions, there is a push button cost associated with changing
            #from an action history of just having pushed the button to the next action history
            #of not having pushed the button. By choosing another action this is a choice to no
            #longer push the button, and there is a cost associated with stopping the button push.
            Reward_h = tensorsumm(loc_rewards, jnp.zeros((1, self.nlpbh)), jnp.zeros((1, self.nq)), jnp.array([[0, r_val]]),
                              jnp.zeros((1, self.nq)))
            #the action costs are the same as with no lick state but there is now no global push button cost, 
            #rather the push button cost is assigned contextually of whether the action is taken given a 
            #particular action history.
            Reward_a = - jnp.array([doNothingCost, travelCost, pushButtonCost])
            #create a new push button specific state space reward matrix where the push button cost is
            #levied only on the push button action if the agent had previously not pushed the button.
            loc_rewards_pb = np.array([[0, 0, 0, 0]])
            Reward_h_pb = tensorsumm(loc_rewards_pb, jnp.zeros((1, self.nlpbh)), jnp.zeros((1, self.nq)), jnp.array([[0, r_val]]),
                              jnp.zeros((1, self.nq)))

            [R1, R2, R3] = jnp.meshgrid(jnp.transpose(Reward_a), jnp.squeeze(Reward_h), jnp.squeeze(Reward_h), indexing='ij')
            Reward = R1 + R3 # R1 is the action cost, R3 is the reward for the state

            [R1, R2, R3] = jnp.meshgrid(jnp.transpose(Reward_a), jnp.squeeze(Reward_h_pb), jnp.squeeze(Reward_h_pb), indexing='ij')
            Reward_pb = R1 + R3 # R1 is the action cost, R3 is the reward for the state
            #apply push button cost only to the push button action 
            Reward = Reward.at[pb].set(Reward_pb[pb])

            #now set the cost of the first push button action. We will need to add this cost
            #to all indices where the current push button state is 0, and the next push button state
            #is 1.
            r0_l2_a0 = self._states_wLickStates(0, 2, 0)
            r0_l3_a0 = self._states_wLickStates(0, 3, 0)
            r1_l2_a0 = self._states_wLickStates(1, 2, 0)
            r1_l3_a0 = self._states_wLickStates(1, 3, 0)
            lick_state0 = jnp.concatenate((r0_l2_a0, r0_l3_a0, r1_l2_a0, r1_l3_a0))
            #check that the indices are unique
            assert len(lick_state0) == len(jnp.unique(lick_state0, size=len(lick_state0)))
            lick_state0 = jnp.unique(lick_state0, size=len(lick_state0))#sort the indices
            #now do the same for the stop push button cost
            r0_l2_a1 = self._states_wLickStates(0, 2, 1)
            r0_l3_a1 = self._states_wLickStates(0, 3, 1)
            r1_l2_a1 = self._states_wLickStates(1, 2, 1)
            r1_l3_a1 = self._states_wLickStates(1, 3, 1)
            lick_state1 = jnp.concatenate((r0_l2_a1, r0_l3_a1, r1_l2_a1, r1_l3_a1))
            #check that the indices are unique
            assert len(lick_state1) == len(jnp.unique(lick_state1, size=len(lick_state1)))
            lick_state1 = jnp.unique(lick_state1, size=len(lick_state1))#sort the indices
            row_inds, col_inds = jnp.ix_(lick_state0, lick_state1)
            Reward = Reward.at[pb, row_inds, col_inds].add(-startPushButtonCost)
            row_inds, col_inds = jnp.ix_(lick_state1, lick_state0)
            Reward = Reward.at[a0, row_inds, col_inds].add(-stopPushButtonCost)
            Reward = Reward.at[g0, row_inds, col_inds].add(-stopPushButtonCost)

            #add a reward for taking a rest at the port in between push button actions
            Reward = Reward.at[a0, :, 2 * self.nq * self.nq * self.nr * self.nlpbh:].add(portRestReward)
            # rew_locs = np.concatenate((r1_l0_a0, r1_l0_a1, r1_l1_a0, r1_l1_a1))
            # assert len(rew_locs) == len(np.unique(rew_locs))
            # rew_locs = np.unique(rew_locs)
            # Reward[:, :, rew_locs] -= r_val
            # #apply an extra penalty to all location 0 and 1 states where push button action is taken
            # # Reward[pb, :, rew_locs] += -pushButtonCost

            # Reward[pb, :2*self.nq * self.nq * self.nr * self.nlpbh, :2*self.nq * self.nq * self.nr * self.nlpbh] = 0.0 #zero out costs and reward for these positions
            # Reward[pb, :2*self.nq * self.nq * self.nr * self.nlpbh, :2*self.nq * self.nq * self.nr * self.nlpbh] += -pushButtonCostl0
            return Reward, Reward.shape
        def false_fn_reward(_):
            Reward_h = tensorsumm(loc_rewards, jnp.zeros((1, self.nq)), jnp.array([[0, r_val]]),
                              jnp.zeros((1, self.nq)))
            Reward_a = - jnp.array([doNothingCost, travelCost, pushButtonCost])
            #create a 3D meshgrid which pairs each possible reward state h with each
            #possible action cost a.
            [R1, R2, R3] = jnp.meshgrid(jnp.transpose(Reward_a), jnp.squeeze(Reward_h), jnp.squeeze(Reward_h), indexing='ij')
            Reward = R1 + R3 # R1 is the action cost, R3 is the reward for the state

            #adjust the reward matrix so that the push button action is differentially penalized
            #in the middle position. There is no button to push at that location.
            Reward = Reward.at[pb, :, 2*self.nq * self.nr * self.nq:].add(-pushButtonCost)
            Reward_padded = jnp.pad(Reward, ((0, 0), (0, Reward.shape[1]), (0, Reward.shape[2])), mode='constant')
            return Reward_padded, Reward.shape

        result, result_shape = lax.cond(self.lick_state, true_fn_reward, false_fn_reward, operand=None)
        self.R = lax.dynamic_slice(result, (0, 0, 0), (result_shape[0], result_shape[1], result_shape[2]))

        #prep the density matrices
        den_col = {bloc.item():{} for bloc in belief_locations}
        for action in actions:
            action = action.item()
            for bloc in belief_locations:#belief location is the location of the agent
                bloc = bloc.item()
                obs_dict = {}
                for i in range(NumCol):
                    j = i
                    den_loc = self.den[bloc]
                    den_temp_n0 =  self.den[bloc][bloc][action][0]
                    den_temp_n1 = self.den[bloc][bloc][action][1]

                    den_block_temp = jnp.block([
                                        [den_temp_n0, den_temp_n0],
                                        [den_temp_n1, den_temp_n1]
                                        ])

                    #normalize columns
                    # den_block_temp = den_block_temp / np.sum(den_block_temp, axis=0)
                    #get the oloc
                    oloc = int((2 - bloc) + 1)  # should only be one other location
                    def true_fn_bloc1(_):
                        def true_fn_lick_state(_):
                            out = kronn(Tpb1, den_block_temp, den_loc[oloc][action][j])
                            return out, out.shape

                        def false_fn_lick_state(_):
                            # Pad the arrays to match the size of the true function's outputs
                            out = kronn(den_block_temp, den_loc[oloc][action][j])
                            out_padded = jnp.pad(out, ((0, out.shape[0]), (0, out.shape[1])), mode='constant')
                            return out_padded, out.shape

                        return lax.cond(self.lick_state, true_fn_lick_state, false_fn_lick_state, operand=None)

                    def false_fn_bloc1(_):
                        def true_fn_lick_state(_):
                            out = kronn(Tpb1, den_loc[oloc][action][j], den_block_temp)
                            return out, out.shape 

                        def false_fn_lick_state(_):
                            out = kronn(den_loc[oloc][action][j], den_block_temp)
                            out_padded = jnp.pad(out, ((0, out.shape[0]), (0, out.shape[1])), mode='constant')
                            return out_padded, out.shape

                        return lax.cond(self.lick_state, true_fn_lick_state, false_fn_lick_state, operand=None)

                    result, original_shape = lax.cond(bloc == 1, true_fn_bloc1, false_fn_bloc1, operand=None)
                    sliced_result = lax.dynamic_slice(result, (0, 0), (original_shape[0], original_shape[1]))

                    obs_dict[(i, j)] = sliced_result

                den_col[bloc][action] = obs_dict
                
        # self.Trans_hybrid_obs12 = np.zeros(((NumCol, NumCol, self.na, self.n, self.n)))
        self.Trans_hybrid_obs12 = jnp.zeros(((NumCol, NumCol, self.na, self.n, self.n)))
        for i in range(NumCol):
            j = i
            self.Trans_hybrid_obs12 = self.Trans_hybrid_obs12.at[i, j, a0, :, :].set(jnp.transpose(kronn(self.Tla0, jnp.identity(self.nq), Tr, jnp.identity(self.nq))))
            self.Trans_hybrid_obs12 = self.Trans_hybrid_obs12.at[i, j, g0, :, :].set(jnp.transpose(kronn(self.Tlg0, jnp.identity(self.nq), Tr, jnp.identity(self.nq))))
            self.Trans_hybrid_obs12 = self.Trans_hybrid_obs12.at[i, j, pb, :, :].set(jnp.transpose(block_diag(self.Tlpb, self.Tlpb, den_col[1][pb][(i,j)], den_col[2][pb][(i,j)])))

    def solveMDP_op(self, epsilon = 0.001, niterations = 10000, initial_value=0):
        vi = ValueIteration_opZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value, self.loc0_mask)
        # optimal policy, stopping criterion changed to "converged Qvalue"
        # vi.setVerbose()
        vi.run()
        self.Q = self._QfromV(vi)
        self.policy = jnp.array(vi.policy)
        self.Vop = vi.V

    def solveMDP_sfm(self, epsilon = 0.001, niterations = 10000, initial_value=0):
        temperatureQ = self.parameters[7]
        vi = ValueIteration_sfmZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value, self.loc0_mask)
        # vi.setVerbose()
        vi.run(temperatureQ)
        self.Qsfm = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        self.softpolicy = jnp.array(vi.softpolicy)
        self.Vsfm = vi.V

    def _QfromV(self, ValueIteration):
        Q = jnp.zeros((ValueIteration.A, ValueIteration.S)) # Q is of shape: na * n
        for a in range(ValueIteration.A):
            Q = Q.at[a, :].set(ValueIteration.R[a] + ValueIteration.discount * \
                                            jnp.dot(ValueIteration.P[a], jnp.array(ValueIteration.V)))
        return Q

class twoboxCazettesMDPdata(twoboxCazettesMDP):
    def __init__(self, discount, nq, nr, na, nl, parameters, parametersExp,
                 sampleTime, sampleNum, lick_state=False, seed=0):
        twoboxCazettesMDP.__init__(self, discount, nq, nr, na, nl, parameters, lick_state)

        self.parametersExp = parametersExp# parameters for the experiment
        self.sampleNum = sampleNum
        self.sampleTime = sampleTime
        self.lick_state = lick_state

        self.action = jnp.empty((self.sampleNum, self.sampleTime), int)  # initialize action, assumed to be optimal
        self.hybrid = jnp.empty((self.sampleNum, self.sampleTime), int)  # initialize hybrid state.
        # Here it is the joint state of reward and belief
        self.location = jnp.empty((self.sampleNum, self.sampleTime), int)  # initialize location state, true world location
        self.location_ind = jnp.empty((self.sampleNum, self.sampleTime), int)  # initialize location state, abstract world location with history
        self.prev_location = jnp.empty((self.sampleNum, self.sampleTime), int)  # initialize location state
        self.belief1 = jnp.empty((self.sampleNum, self.sampleTime), int)
        self.belief2 = jnp.empty((self.sampleNum, self.sampleTime), int)  # initialize hidden state, belief state
        self.reward = jnp.empty((self.sampleNum, self.sampleTime), int)  # initialize reward state
        self.trueState1 = jnp.zeros((self.sampleNum, self.sampleTime))
        self.trueState2 = jnp.zeros((self.sampleNum, self.sampleTime))
        self.color1 = jnp.zeros((self.sampleNum, self.sampleTime), dtype=int)#color and true state are the same
        self.color2 = jnp.zeros((self.sampleNum, self.sampleTime), dtype=int)#color and true state are the same

        self.actionDist = jnp.zeros((self.sampleNum, self.sampleTime, self.na))
        self.belief1Dist = jnp.zeros((self.sampleNum, self.sampleTime, self.nq))
        self.belief2Dist = jnp.zeros((self.sampleNum, self.sampleTime, self.nq))

        self.seed = seed

        self.setupMDP()
        jax.jit(self.solveMDP_op())
        jax.jit(self.solveMDP_sfm())

    def dataGenerate_sfm(self):

        ## Parameters
        beta = 0     # available food dropped back into box after button press
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0
        rho = 1      # food in mouth is consumed
        # State rewards
        Reward = 1   # reward per time step with food in mouth

        Ncol = 1  # max value of color

        psw_e = self.parametersExp[0]    # location activity switches after button press
        prwd_e = self.parametersExp[1]   # reward is returned for button press at active location
        actions = jnp.array([a0, g0, pb])

        # State rewards
        Groom = self.parametersExp[2]     # location 0 reward
        # Action costs
        travelCost = self.parametersExp[3]
        pushButtonCost = self.parametersExp[4]
        locations_with_history = create_ordered_tuples(self.nl)
        self.abstract_locations = locations_with_history

        rkey = jax.random.key(self.seed)

        ## Generate data
        for n in range(self.sampleNum):
            _, n_rkey = jax.random.split(rkey)
            belief1Initial = jax.random.randint(n_rkey, shape= (1,), minval=0, maxval=self.nq)[0]
            rewInitial = jax.random.randint(n_rkey, shape= (1,), minval=0, maxval=self.nr)[0]#maybe set 0
            belief2Initial = self.nq - belief1Initial - 1

            locationInitial_ind = jax.random.randint(n_rkey, shape= (1,), minval=0, maxval=self.nlwh)[0]
            locationInitial_prev = locations_with_history[locationInitial_ind][0]
            locationInitial = locations_with_history[locationInitial_ind][1]

            for t in range(self.sampleTime):
                if t == 0:
                    # Initialize the true world states, sensory information and latent states
                    _, n_rkey = jax.random.split(rkey)
                    active_site = jax.random.randint(n_rkey, shape= (1,), minval=1, maxval=3)[0]
                    if active_site == 1:
                        self.trueState1 = self.trueState1.at[n, t].set(1)
                        self.color1 = self.color1.at[n, t].set(1)
                        self.trueState2 = self.trueState2.at[n, t].set(0)
                        self.color2 = self.color2.at[n, t].set(0)
                    else:
                        self.trueState2 = self.trueState2.at[n, t].set(1)
                        self.color2 = self.color2.at[n, t].set(1)
                        self.trueState1 = self.trueState1.at[n, t].set(0)
                        self.color1 = self.color1.at[n, t].set(0)

                    self.location = self.location.at[n, t].set(locationInitial)
                    self.prev_location = self.prev_location.at[n, t].set(locationInitial_prev)
                    self.location_ind = self.location_ind.at[n, t].set(locationInitial_ind)
                    self.reward = self.reward.at[n, t].set(rewInitial)

                    self.belief1 = self.belief1.at[n, t].set(belief1Initial)
                    self.belief2 = self.belief2.at[n, t].set(belief2Initial)
                    self.belief1Dist = self.belief1Dist.at[n, t, belief1Initial].set(1)
                    self.belief2Dist = self.belief2Dist.at[n, t, belief2Initial].set(1)

                    if self.lick_state:
                        pb_ind = 0
                        h = self.location_ind[n, t] * (self.nq * self.nr * self.nq * 2) + \
                                            pb_ind * (self.nq * self.nr * self.nq) + \
                                            self.belief1[n, t] * (self.nr * self.nq) + \
                                            self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing
                    else:
                        h = self.location_ind[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) + \
                                            self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing
                    self.hybrid = self.hybrid.at[n, t].set(h)
                    self.actionDist = self.actionDist.at[n, t].set(self.softpolicy.T[h])
                    _, n_rkey = jax.random.split(rkey)
                    self.action = self.action.at[n, t].set(self._chooseAction(self.actionDist[n, t], n_rkey))

                else:
                    # variables evolve with dynamics
                    if self.action[n, t - 1] != pb:
                        if self.reward[n, t - 1] == 0:
                            self.reward = self.reward.at[n, t].set(0)
                        else:
                            #handle the case where the reward is not consumed,
                            #deterministically return 0 if the reward is comsumed,
                            #i.e. rho = 1 yields np.random.binomial(1, 0) = 0
                            _, n_rkey = jax.random.split(rkey)
                            self.reward = self.reward.at[n, t].set(jax.random.binomial(n_rkey, 1, 1 - rho))

                        if self.action[n, t - 1] == a0:#do nothing
                            # if the action is to do nothing, the location remains the same
                            self.location = self.location.at[n, t].set(self.location[n, t - 1])
                            self.prev_location = self.prev_location.at[n, t].set(self.prev_location[n, t - 1])
                            self.location_ind = self.location_ind.at[n, t].set(self.location_ind[n, t - 1])

                        # if the action is to go to location 0, i.e. the middle location

                        if self.action[n, t - 1] == g0:
                            Tl = jnp.array([[delta, 0., 1. - delta, 0.],
                                            [0., delta, 0., 1. - delta],
                                            [0., 1. - delta, delta, 0.],
                                            [1. - delta, 0., 0., delta]])
                            _, n_rkey = jax.random.split(rkey)
                            self.location_ind = self.location_ind.at[n, t].set(jnp.argmax(multinomial(n_rkey, 1, Tl[:, self.location_ind[n, t - 1]], shape=())))
                            self.location = self.location.at[n, t].set(locations_with_history[self.location_ind[n, t]][1])
                            self.prev_location = self.prev_location.at[n, t].set(locations_with_history[self.location_ind[n, t]][0])

                        self.trueState1 = self.trueState1.at[n, t].set(self.trueState1[n, t - 1])
                        self.trueState2 = self.trueState2.at[n, t].set(self.trueState2[n, t - 1])
                        self.color1 = self.color1.at[n, t].set(self.trueState1[n, t])
                        self.color2 = self.color2.at[n, t].set(self.trueState2[n, t])

                        self.belief1 = self.belief1.at[n, t].set(self.belief1[n, t - 1])
                        self.belief2 = self.belief2.at[n, t].set(self.belief2[n, t - 1])

                    if self.action[n, t - 1] == pb:  # press button
                        self.location = self.location.at[n, t].set(self.location[n, t - 1])  # pressing button does not change location
                        self.prev_location = self.prev_location.at[n, t].set(self.prev_location[n, t - 1])
                        self.location_ind = self.location_ind.at[n, t].set(self.location_ind[n, t - 1])

                        if self.location[n, t - 1] == 0:
                            # pressing button at the center does not change anything.. and should be
                            # technically impossible given the action masking on the policy
                            self.reward = self.reward.at[n, t].set(0)
                            self.trueState1 = self.trueState1.at[n, t].set(self.trueState1[n, t - 1])
                            self.trueState2 = self.trueState2.at[n, t].set(self.trueState2[n, t - 1])
                            self.color1 = self.color1.at[n, t].set(self.trueState1[n, t])
                            self.color2 = self.color2.at[n, t].set(self.trueState2[n, t])
                            self.belief1 = self.belief1.at[n, t].set(self.belief1[n, t - 1])
                            self.belief2 = self.belief2.at[n, t].set(self.belief2[n, t - 1])

                        if self.location[n, t] == 1:  # consider location 1 case
                            if self.trueState1[n, t - 1] == 0:#if the box is inactive
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward = self.reward.at[n, t].set(0)
                                else:
                                    _, n_rkey = jax.random.split(rkey)
                                    self.reward[n, t] = jax.random.binomial(n_rkey, 1, 1 - rho)  # have not consumed food

                                self.trueState1 = self.trueState1.at[n, t].set(0)
                                self.color1 = self.color1.at[n, t].set(self.trueState1[n, t])
                                self.trueState2 = self.trueState2.at[n, t].set(1)
                                self.color2 = self.color2.at[n, t].set(self.trueState2[n, t])

                            else:#if the box is active
                                _, n_rkey = jax.random.split(rkey)
                                self.reward[n, t] = jax.random.binomial(n_rkey, 1, prwd_e)  # give some reward with probability prwd
                                #there is now a chance psw_e that the box will switch off
                                _, n_rkey = jax.random.split(rkey)
                                self.trueState1[n, t] = jax.random.binomial(n_rkey, 1, 1 - psw_e)
                                self.color1 = self.color1.at[n, t].set(self.trueState1[n, t])
                                self.trueState2 = self.trueState2.at[n, t].set(abs(1 - self.trueState1[n, t]))
                                self.color2 = self.color2.at[n, t].set(self.trueState2[n, t])

                            #check how this works with observations as rewards...
                            self.belief1Dist = self.belief1Dist.at[n, t].set(self.den[1][1][pb][self.reward[n, t], :, self.belief1[n, t - 1]])
                            _, n_rkey = jax.random.split(rkey)
                            self.belief1 = self.belief1.at[n, t].set(jnp.argmax(multinomial(n_rkey, 1, self.belief1Dist[n, t], shape=())))
                            
                            self.belief2Dist = self.belief2Dist.at[n, t].set(self.den[1][2][pb][self.reward[n, t], :, self.belief2[n, t - 1]])
                            _, n_rkey = jax.random.split(rkey)
                            self.belief2 = self.belief2.at[n, t].set(jnp.argmax(multinomial(n_rkey, 1, self.belief2Dist[n, t], shape=())))
                            
                        if self.location[n, t] == 2:  # consider location 2 case
                            if self.trueState2[n, t - 1] == 0:
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward = self.reward.at[n, t].set(0)
                                else:
                                    _, n_rkey = jax.random.split(rkey)
                                    self.reward = self.reward.at[n, t].set(jax.random.binomial(n_rkey, 1, 1 - rho))  # have not consumed food
                                
                                self.trueState2 = self.trueState2.at[n, t].set(0)
                                self.color2 = self.color2.at[n, t].set(self.trueState2[n, t])
                                self.trueState1 = self.trueState1.at[n, t].set(1)
                                self.color1 = self.color1.at[n, t].set(self.trueState1[n, t])

                            else:
                                _, n_rkey = jax.random.split(rkey)
                                self.reward = self.reward.at[n, t].set(jax.random.binomial(n_rkey, 1, prwd_e))  # give some reward
                                #there is now a chance psw_e that the box will switch off
                                _, n_rkey = jax.random.split(rkey)
                                self.trueState2 = self.trueState2.at[n, t].set(jax.random.binomial(n_rkey, 1, 1 - psw_e))
                                self.color2 = self.color2.at[n, t].set(self.trueState2[n, t])
                                self.trueState1 = self.trueState1.at[n, t].set(abs(1 - self.trueState2[n, t]))
                                self.color1 = self.color1.at[n, t].set(self.trueState1[n, t])

                            self.belief1Dist = self.belief1Dist.at[n, t].set(self.den[2][1][pb][self.reward[n, t], :, self.belief1[n, t - 1]])
                            _, n_rkey = jax.random.split(rkey)
                            self.belief1 = self.belief1.at[n, t].set(jnp.argmax(multinomial(n_rkey, 1, self.belief1Dist[n, t], shape=())))
                            self.belief2Dist = self.belief2Dist.at[n, t].set(self.den[2][2][pb][self.reward[n, t], :, self.belief2[n, t - 1]])
                            _, n_rkey = jax.random.split(rkey)
                            self.belief2 = self.belief2.at[n, t].set(jnp.argmax(multinomial(n_rkey, 1, self.belief2Dist[n, t], shape=())))
                    
                    if self.lick_state:
                        pb_ind = int(self.action[n, t - 1] == pb)#if the action is to push the button or not
                        h = self.location_ind[n, t] * (self.nq * self.nr * self.nq * 2) + \
                                            pb_ind * (self.nq * self.nr * self.nq) + \
                                            self.belief1[n, t] * (self.nr * self.nq) + \
                                            self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing
                    else:
                        h = self.location_ind[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) + \
                                            self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing

                    self.hybrid = self.hybrid.at[n, t].set(h)
                    self.actionDist = self.actionDist.at[n, t].set(self.softpolicy.T[h])
                    _, n_rkey = jax.random.split(rkey)
                    self.action = self.action.at[n, t].set(self._chooseAction(self.actionDist[n, t], n_rkey))

    def _chooseAction(self, pvec, key):
        # Generate action according to multinomial distribution
        # toss the 3 sided coin one time to sample the action
        stattemp = multinomial(key, 1, pvec, shape=())
        return jnp.argmax(stattemp)

# class twoboxCazettesMDP_der(twoboxCazettesMDP):
#     """
#     Derivatives of log_likelihood with respect to the parameters
#     """

#     def __init__(self, discount, nq, nr, na, nl, parameters, lick_state=False):
#         twoboxCazettesMDP.__init__(self, discount, nq, nr, na, nl, parameters, lick_state)
#         self.setupMDP()
#         self.solveMDP_op()
#         self.solveMDP_sfm()

#     def dloglikelihhod_dpara_sim(self, obs):
#         L = len(self.parameters)
#         pi = jnp.ones(self.nq * self.nq) / self.nq / self.nq
#         Numcol = 2 # number of colors
#         Ncol = 1 # max value of color

#         twoboxColHMM = HMMtwoboxCazettes(self.ThA, self.softpolicy,
#                                     self.Trans_hybrid_obs12, self.Obs_emis_trans1,
#                                     self.Obs_emis_trans2, pi, Ncol, self.lick_state)
#         log_likelihood =  twoboxColHMM.computeQaux(obs, self.ThA, self.softpolicy, self.Trans_hybrid_obs12, self.Obs_emis_trans1, self.Obs_emis_trans2) + \
#                           twoboxColHMM.latent_entr(obs)

#         perturb = 10 ** -6

#         def per_param_deriv(i):
#             para_perturb = np.copy(self.parameters)
#             para_perturb[i] = para_perturb[i] + perturb

#             twoboxCol_perturb = twoboxCazettesMDP(self.discount, self.nq, self.nr, self.na, self.nl, para_perturb, self.lick_state)
#             twoboxCol_perturb.setupMDP()
#             twoboxCol_perturb.solveMDP_sfm()
#             ThA_perturb = twoboxCol_perturb.ThA
#             policy_perturb = twoboxCol_perturb.softpolicy
#             Trans_hybrid_obs12_perturb = twoboxCol_perturb.Trans_hybrid_obs12
#             Obs_emis_trans1_perturb = twoboxCol_perturb.Obs_emis_trans1
#             Obs_emis_trans2_perturb = twoboxCol_perturb.Obs_emis_trans2
#             twoboxColHMM_perturb = HMMtwoboxCazettes(ThA_perturb, policy_perturb, Trans_hybrid_obs12_perturb,
#                                         Obs_emis_trans1_perturb, Obs_emis_trans2_perturb, pi, Ncol, self.lick_state)

#             log_likelihood_perturb = twoboxColHMM_perturb.computeQaux(obs, ThA_perturb, policy_perturb, Trans_hybrid_obs12_perturb,
#                                         Obs_emis_trans1_perturb, Obs_emis_trans2_perturb) + twoboxColHMM_perturb.latent_entr(obs)

#             return (log_likelihood_perturb - log_likelihood) / perturb

#         NumThread = len(self.parameters)
#         results = Parallel(n_jobs=NumThread)(delayed(per_param_deriv)(i) for i in range(L))
#         #for debugging
#         # results = [per_param_deriv(i) for i in range(L)]
#         dloglikelihhod_dpara = np.array(results)

#         return dloglikelihhod_dpara

class twoboxCazettesMDP_der(twoboxCazettesMDP):
    """
    Derivatives of log_likelihood with respect to the parameters
    """

    def __init__(self, discount, nq, nr, na, nl, parameters, lick_state=False):
        twoboxCazettesMDP.__init__(self, discount, nq, nr, na, nl, parameters, lick_state)
        self.setupMDP()
        self.solveMDP_op()
        self.solveMDP_sfm()

    def log_likelihood_fn(self, parameters, obs):
        pi = jnp.ones(self.nq * self.nq) / self.nq / self.nq
        Numcol = 2  # number of colors
        Ncol = 1  # max value of color

        twoboxColHMM = HMMtwoboxCazettes(self.ThA, self.softpolicy,
                                         self.Trans_hybrid_obs12, self.Obs_emis_trans1,
                                         self.Obs_emis_trans2, pi, Ncol, self.lick_state)
        log_likelihood = twoboxColHMM.computeQaux(obs, self.ThA, self.softpolicy, self.Trans_hybrid_obs12, self.Obs_emis_trans1, self.Obs_emis_trans2) + \
                         twoboxColHMM.latent_entr(obs)
        return log_likelihood

    def dloglikelihhod_dpara_sim(self, obs):
        # Define the gradient function
        grad_fn = grad(self.log_likelihood_fn)

        # Compute the gradient with respect to the parameters
        dloglikelihhod_dpara = grad_fn(self.parameters, obs)

        return dloglikelihhod_dpara

class twoboxCazettesIndependentMDP:
    def __init__(self, discount, nq, nr, na, nl, parameters):
        self.discount = discount
        self.nq = nq #number of belief states
        self.nr = nr #number of reward states, i.e. world states, active or inactive
        self.na = na #number of actions
        self.nl = nl   # number of locations
        self.nlwh = 2*(nl - 1)   # number of locations with history
        self.n = self.nq * self.nq * self.nr * self.nl   # total number of outcomes, or unique system (world and agent) states
        self.nwh = self.nq * self.nq * self.nr * self.nlwh   # total number of outcomes, or unique system (world and agent) states
        self.parameters = parameters
        #every action has a transition matrix and a reward function.
        # transition matrix, per action each column defines the probability of transitioning to each other unique system state
        # self.ThA = np.zeros((self.na, self.n, self.n))
        self.ThA = np.zeros((self.na, self.nwh, self.nwh)) 

        # for each action the probability of receiving a reward in each unique system state
        # self.R = np.zeros((self.na, self.n, self.n)) # reward function
        self.R = np.zeros((self.na, self.nwh, self.nwh)) # reward function

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
        p_sw = self.parameters[0]   # location activity switches after button press
        p_rwd = self.parameters[1]    # reward is returned for button press at active location
        actions = np.array([a0, g0, pb])
        locations = np.arange(self.nl)
        locations_with_history = create_ordered_tuples(self.nl)
        
        NumCol = 2  # number of colors
        # State rewards
        Reward = 1   # reward per time step with food in mouth
        Groom = self.parameters[2]   # location 0 reward
        doNothingCost = 0.005
        pushButtonCost_l0 = 100#cost of pushing the button at location 0
        # Action costs
        travelCost = self.parameters[3]
        pushButtonCost = self.parameters[4]
        beta = 0     # available food dropped back into box after button press
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0

        # kronecker product of these initial distributions
        # Note that this ordering makes the subsequent products easiest

        # setup single-variable transition matrices for each action..
        # these calculate \overline{O}(o_{t+1}|b_{t},a_{t}) = \int ds_{t+1} ds_{t} O(o_{t+1}|s_{t+1}, a_{t}) * T(s_{t+1}|s_{t},a_{t})* B(s_{t}|b_{t})
        # and P(b_{t+1}|b_{t},a_{t},o_{t+1})
        # for box 1
        Tr = np.array([[1, rho], [0, 1 - rho]])  # consume reward
        self.Trans_belief_obs, self.Obs_emis_trans, self.den = beliefTransitionMatrixGaussianCazettesIndependent(p_sw, p_rwd, self.nq, actions, locations, sigma = 1 / self.nq / 3)
        belief_locations = [loc for loc in locations if loc != 0]
        Tb_unpack = {bloc:{oloc:{} for oloc in locations} for bloc in locations}
        for bloc in locations:#bloc is the location of the agent
            for oloc in locations:#other locations where the agent is NOT located
                for action in actions:
                    # belief transitions, it is  marginalized over observations, P(b_{t+1}|b_{t},a_{t},o_{t+1})
                    Trans_belief = np.sum(self.Trans_belief_obs[bloc][oloc][action], axis=0)
                    Tb_unpack[bloc][oloc][action] = Trans_belief / np.tile(np.sum(Trans_belief, 0), (self.nq, 1))#normalize each column
        
        #make per location belief transition matrices
        Tb = {bloc:{} for bloc in locations}
        for action in actions:
            for bloc in belief_locations:#belief location is the location of the agent
                Tb_loc = Tb_unpack[bloc]
                tbs = []
                for loc in belief_locations:#all locations in order.
                    tb = Tb_loc[loc][action]
                    tbs.append(tb)
                #insert Tr at the second position
                tbs.insert(1, Tr)
                Tb[bloc][action] = kronn(*tbs)#kronecker product of the transition matrices
            Tb[0][action] = kronn(Tb_unpack[0][1][action], Tr, Tb_unpack[0][2][action])
        
        # now fill out ThA i.e.
        # \overline{T}(b_{t+1}|b_{t},a_{t}) = \int do_{t+1} P(b_{t+1}|b_{t},a_{t},o_{t+1})*\overline{O}(o_{t+1}|b_{t},a_{t})
        # where \overline{O}(o_{t+1}|b_{t},a_{t}) = \int ds_{t+1} ds_{t} O(o_{t+1}|s_{t+1}, a_{t}) * T(s_{t+1}|s_{t},a_{t})* B(s_{t}|b_{t})
        # Note: the order in which the kronecker products are taken is important because it
        # implies the organization of the ThA matrix. The ThA is organized into blocks, the first
        # are specific to the location.

        #For debugging and housekeeping, we will also create the labels for the ThA and R matrices
        #first make the labels for ThA using the kronn written for strings
        loc_labels = np.array([str(loc) for loc in locations])
        loc_labels_with_history = np.array([str(loc) for loc in locations_with_history])
        bel1_labels = np.array(['b1_' + str(bel) for bel in range(self.nq)])
        bel2_labels = np.array(['b2_' + str(bel) for bel in range(self.nq)])
        rew_labels = np.array(['r_' + str(rew) for rew in range(self.nr)])
        #make the transition matrices for the labels using the string meshgrid
        # loc_mesh = string_meshgrid(loc_labels, loc_labels)
        loc_mesh = string_meshgrid(loc_labels_with_history, loc_labels_with_history)
        rew_mesh = string_meshgrid(rew_labels, rew_labels)
        bel1_mesh = string_meshgrid(bel1_labels, bel1_labels)
        bel2_mesh = string_meshgrid(bel2_labels, bel2_labels)
        self.ThA_labels = kronn_str(loc_mesh, bel1_mesh, rew_mesh, bel2_mesh)
        #next make the labels for Reward function using the tensorsumm written for strings
        self.R_labels = tensorsumm_str(loc_labels_with_history[:, None], bel1_labels[None, :], rew_labels[:, None], bel2_labels[None, :]).flatten()

        # ACTION: do nothing
        # self.ThA[a0, :, :] = kronn(np.identity(self.nl), np.identity(self.nq), Tr, np.identity(self.nq))
        self.ThA[a0, :, :] = block_diag(Tb[0][a0], Tb[0][a0], Tb[1][a0], Tb[2][a0])
        # ACTION: go to location 0/1/2
        #create location transition matrix for each unique location.
        #delta is the probability of not going to the target location
        #direct is the probability of going directly to the target location (skipping location 0)
        # Tl0 = np.array([[1, 1 - delta, 1 - delta], 
        #                 [0, delta, 0], 
        #                 [0, 0, delta]])  # go to loc 0 (with error of delta)
        # Tl1 = np.array([[delta, 0, 1 - delta - direct], 
        #                 [1 - delta, 1, direct],
        #                 [0, 0, delta]])  # go to box 1 (with error of delta)
        # Tl2 = np.array([[delta, 1 - delta - direct, 0], 
        #                 [0, delta, 0],
        #                 [1 - delta, direct, 1]])  # go to box 2 (with error of delta)

        #technically there are only two foraging sites and 1 intermediate site..
        #but to capture the restrictions on the agent's movement, where it cannot transition directly
        #from one foraging site to the other, and it cannot transition from the intermediate site
        #back to the foraging site it came from, we will create a transition matrix for each possible
        #(previous location, current location) combination that is possible.
        #Here these will be (1, 0), (2, 0), (0, 1), (0, 2).
        #keeping with the current state on the columns and next state on the rows convention,
        #the way to read this transition matrix is what previous location and current location tuple
        #is transitioning to the current location and next location tuple.
        #first column: (1, 0) -> (0, 2) has to go to site 2 if coming from site 1, can error to stay at site 0
        #second column: (2, 0) -> (0, 1) has to go to site 1 if coming from site 2, can error to stay at site 0
        #third column: (0, 1) -> (1, 0) has to return to the intermediate site, can error to stay at site 1
        #fourth column: (0, 2) -> (2, 0) has to return to the intermediate site, can error to stay at site 2
        Tl = np.array([[delta, 0., 1. - delta, 0.],
                        [0., delta, 0., 1. - delta],
                        [0., 1. - delta, delta, 0.],
                        [1. - delta, 0., 0., delta]])

        #each entry in ThA is a product, for each action, of every possible system state combination
        #the kronecker product allows us to efficiently calculate these products by taking the iterative
        #kronecker product of the location, belief, reward consumption states.
        # l0g0 = kronn(Tl0[0, :], Tb[0][g0])
        # l0g1 = kronn(Tl1[0, :], Tb[0][g1])
        # l0g2 = kronn(Tl2[0, :], Tb[0][g2])
        
        # l1g0 = kronn(Tl0[1, :], Tb[1][g0])
        # l1g1 = kronn(Tl1[1, :], Tb[1][g1])
        # l1g2 = kronn(Tl2[1, :], Tb[1][g2])

        # l2g0 = kronn(Tl0[2, :], Tb[2][g0])
        # l2g1 = kronn(Tl1[2, :], Tb[2][g1])
        # l2g2 = kronn(Tl2[2, :], Tb[2][g2])

        l01g0 = kronn(Tl[0, :], Tb[0][g0])
        l02g0 = kronn(Tl[1, :], Tb[0][g0])
        l10g0 = kronn(Tl[2, :], Tb[1][g0])
        l20g0 = kronn(Tl[3, :], Tb[2][g0])

        self.ThA[g0, :, :] = np.row_stack((l01g0, l02g0, l10g0, l20g0))
        # self.ThA[g1, :, :] = np.row_stack((l0g1, l1g1, l2g1))
        # self.ThA[g2, :, :] = np.row_stack((l0g2, l1g2, l2g2))

        # ACTION: push button.
        # Th = block_diag(Tb[0][pb], Tb[1][pb], Tb[2][pb])
        Th = block_diag(Tb[0][pb], Tb[0][pb], Tb[1][pb], Tb[2][pb])
        
        #this dot product is of the ith push button NEXT system state with the jth 
        #do nothing system CURRENT system state.
        self.ThA[pb, :, :] = Th#.dot(self.ThA[a0, :, :])

        # The reward function is the expecation of the reward on the next time step
        # given the current state and action.
        # accumulate rewards for each possible location, belief state 1, reward consumption state, and
        # belief state. 
        loc_rewards = np.array([[Groom, Groom, 0, 0]])
        Reward_h = tensorsumm(loc_rewards, np.zeros((1, self.nq)), np.array([[0, Reward]]),
                              np.zeros((1, self.nq)))
        # reward_l0 = np.ones((1, self.nq*self.nr*self.nq)) * Groom
        # reward_l1_l2 = tensorsumm(np.zeros((1, 2)),np.zeros((1, self.nq)), np.array([[0, Reward]]),
        #                       np.zeros((1, self.nq)))
        # Reward_h = np.column_stack((reward_l0, reward_l1_l2))
        #complie the action costs.
        Reward_a = - np.array([doNothingCost, travelCost, pushButtonCost])

        #create a 3D meshgrid which pairs each possible reward state h with each
        #possible action cost a.
        [R1, R2, R3] = np.meshgrid(Reward_a.T, Reward_h, Reward_h, indexing='ij')
        Reward = R1 + R3 # R1 is the action cost, R3 is the reward for the state
        #adjust the reward matrix so that the push button action is differentially penalized
        #in the middle position. There is no button to push at that location.
        Reward[pb, :, :2*self.nq * self.nr * self.nq] += pushButtonCost_l0*-1
        
        # R = Reward[:, 0, :].T
        self.R = Reward

        #flip the transition matrices so past states are now rows
        #and future states are columns
        for i in range(self.na):
            self.ThA[i, :, :] = self.ThA[i, :, :].T
            # self.R[i, :, :] = self.R[i, :, :].T

        #prep the density matrices
        den_col = {bloc:{} for bloc in locations}
        for action in actions:
            for bloc in belief_locations:#belief location is the location of the agent
                obs_dict = {}
                for i in range(NumCol):
                    j = i
                    den_loc = self.den[bloc]
                    dens = [den_loc[1][action][i], Tr, den_loc[2][action][j]]
                    obs_dict[(i, j)] = kronn(*dens)#kronecker product of the observation matrices
                den_col[bloc][action] = obs_dict
            den_col[0][action] = {(i, i):kronn(self.den[0][1][action][0], Tr, self.den[0][2][action][0]) for i in range(NumCol)}
                
        self.Trans_hybrid_obs12 = np.zeros(((NumCol, NumCol, self.na, self.nwh, self.nwh)))
        for i in range(NumCol):
            j = i     
            self.Trans_hybrid_obs12[i, j, a0, :, :] =  block_diag(den_col[0][a0][(i,j)], den_col[0][a0][(i,j)], 
                                                                  den_col[1][a0][(i,j)], den_col[2][a0][(i,j)]).T
            
            # l0g0 = kronn(Tl0[0, :], den_col[0][g0][(i,j)])
            # l0g1 = kronn(Tl1[0, :], den_col[0][g1][(i,j)])
            # l0g2 = kronn(Tl2[0, :], den_col[0][g2][(i,j)])
            
            # l1g0 = kronn(Tl0[1, :], den_col[1][g0][(i,j)])
            # l1g1 = kronn(Tl1[1, :], den_col[1][g1][(i,j)])
            # l1g2 = kronn(Tl2[1, :], den_col[1][g2][(i,j)])

            # l2g0 = kronn(Tl0[2, :], den_col[2][g0][(i,j)])
            # l2g1 = kronn(Tl1[2, :], den_col[2][g1][(i,j)])
            # l2g2 = kronn(Tl2[2, :], den_col[2][g2][(i,j)])

            # self.Trans_hybrid_obs12[i, j, g0, :, :] = np.row_stack((l0g0, l1g0, l2g0)).T
            # self.Trans_hybrid_obs12[i, j, g1, :, :] = np.row_stack((l0g1, l1g1, l2g1)).T
            # self.Trans_hybrid_obs12[i, j, g2, :, :] = np.row_stack((l0g2, l1g2, l2g2)).T

            l01g0 = kronn(Tl[0, :], den_col[0][g0][(i,j)])
            l02g0 = kronn(Tl[1, :], den_col[0][g0][(i,j)])
            l10g0 = kronn(Tl[2, :], den_col[1][g0][(i,j)])
            l20g0 = kronn(Tl[3, :], den_col[2][g0][(i,j)])

            self.Trans_hybrid_obs12[i, j, g0, :, :] = np.row_stack((l01g0, l02g0, l10g0, l20g0))

            self.Trans_hybrid_obs12[i, j, pb, :, :] = block_diag(den_col[0][pb][(i,j)],
                                                                 den_col[0][pb][(i,j)],
                                                                 den_col[1][pb][(i,j)], 
                                                                 den_col[2][pb][(i,j)]).T

    def solveMDP_op(self, epsilon = 0.001, niterations = 10000, initial_value=0):
        vi = ValueIteration_opZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        # optimal policy, stopping criterion changed to "converged Qvalue"
        vi.setVerbose()
        vi.run()
        self.Q = self._QfromV(vi)
        self.policy = np.array(vi.policy)
        self.Vop = vi.V

    def solveMDP_sfm(self, epsilon = 0.001, niterations = 10000, initial_value=0):

        temperatureQ = self.parameters[5]
        vi = ValueIteration_sfmZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        vi.setVerbose()
        vi.run(temperatureQ)
        self.Qsfm = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        self.softpolicy = np.array(vi.softpolicy)
        self.Vsfm = vi.V

    def _QfromV(self, ValueIteration):
        Q = np.zeros((ValueIteration.A, ValueIteration.S)) # Q is of shape: na * n
        for a in range(ValueIteration.A):
            Q[a, :] = ValueIteration.R[a] + ValueIteration.discount * \
                                            jnp.dot(ValueIteration.P[a], ValueIteration.V)
        return Q

class twoboxCazettesIndependentMDPdata(twoboxCazettesIndependentMDP):
    def __init__(self, discount, nq, nr, na, nl, parameters, parametersExp,
                 sampleTime, sampleNum):
        twoboxCazettesIndependentMDP.__init__(self, discount, nq, nr, na, nl, parameters)

        self.parametersExp = parametersExp# parameters for the experiment
        self.sampleNum = sampleNum
        self.sampleTime = sampleTime

        self.action = np.empty((self.sampleNum, self.sampleTime), int)  # initialize action, assumed to be optimal
        self.hybrid = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hybrid state.
        # Here it is the joint state of reward and belief
        self.location = np.empty((self.sampleNum, self.sampleTime), int)  # initialize location state
        self.location_ind = np.empty((self.sampleNum, self.sampleTime), int)  # initialize location state, abstract world location with history
        self.prev_location = np.empty((self.sampleNum, self.sampleTime), int)  # initialize location state

        self.belief1 = np.empty((self.sampleNum, self.sampleTime), int)
        self.belief2 = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hidden state, belief state
        self.reward = np.empty((self.sampleNum, self.sampleTime), int)  # initialize reward state
        self.trueState1 = np.zeros((self.sampleNum, self.sampleTime))
        self.trueState2 = np.zeros((self.sampleNum, self.sampleTime))
        self.color1 = np.zeros((self.sampleNum, self.sampleTime), dtype=int)#color and true state are the same
        self.color2 = np.zeros((self.sampleNum, self.sampleTime), dtype=int)#color and true state are the same

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

        p_sw_e = self.parametersExp[0]    # location activity switches after button press
        p_rwd_e = self.parametersExp[1]   # reward is returned for button press at active location
        actions = np.array([a0, g0, pb])
        locations_with_history = create_ordered_tuples(self.nl)

        # State rewards
        Groom = self.parametersExp[2]     # location 0 reward
        # Action costs
        travelCost = self.parametersExp[3]
        pushButtonCost = self.parametersExp[4]

        ## Generate data
        for n in range(self.sampleNum):

            belief1Initial = np.random.randint(self.nq)
            rewInitial = np.random.randint(self.nr)#maybe set 0
            belief2Initial = np.random.randint(self.nq)
            locationInitial = np.random.randint(self.nl)
            locationInitial_ind = np.random.randint(self.nlwh)
            locationInitial_prev = locations_with_history[locationInitial_ind][0]
            locationInitial = locations_with_history[locationInitial_ind][1]

            for t in range(self.sampleTime):
                if t == 0:
                    # Initialize the true world states, sensory information and latent states
                    active_site = np.random.randint(1,3)
                    self.trueState1[n, t] = 1#np.random.randint(2)
                    self.color1[n, t] = self.trueState1[n, t]
                    self.trueState2[n, t] = 1#np.random.randint(2)
                    self.color2[n, t] = self.trueState2[n, t]
                
                    self.location[n, t], self.reward[n, t] = locationInitial, rewInitial
                    self.prev_location[n, t] = locationInitial_prev
                    self.location_ind[n, t] = locationInitial_ind

                    self.belief1[n, t], self.belief2[n, t] = belief1Initial, belief2Initial
                    self.belief1Dist[n, t, self.belief1[n, t]] = 1
                    self.belief2Dist[n, t, self.belief2[n, t]] = 1

                    self.hybrid[n, t] = self.location_ind[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) + \
                                        self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing
                    self.actionDist[n, t] = self.softpolicy.T[self.hybrid[n, t]]
                    self.action[n, t] = self._chooseAction(self.actionDist[n, t])

                else:
                    # variables evolve with dynamics
                    if self.trueState1[n, t - 1] == 0:
                        self.trueState1[n, t] = np.random.binomial(1, p_sw_e)
                    else:
                        self.trueState1[n, t] = np.random.binomial(1, 1 - p_sw_e)
                    
                    if self.trueState2[n, t - 1] == 0:
                        self.trueState2[n, t] = np.random.binomial(1, p_sw_e)
                    else:
                        self.trueState2[n, t] = np.random.binomial(1, 1 - p_sw_e)

                    self.color1[n, t] = self.trueState1[n, t]
                    self.color2[n, t] = self.trueState2[n, t]

                    if self.action[n, t - 1] != pb:
                        if self.reward[n, t - 1] == 0:
                            self.reward[n, t] = 0
                        else:
                            #handle the case where the reward is not consumed,
                            #deterministically return 0 if the reward is comsumed,
                            #i.e. rho = 1 yields np.random.binomial(1, 0) = 0
                            self.reward[n, t] = np.random.binomial(1, 1 - rho)

                        # self.belief1[n, t] = self.belief1[n, t - 1]  # use the previous belief state
                        # self.belief2[n, t] = self.belief2[n, t - 1]  # use the previous belief state
                        self.belief1Dist[n, t] = self.den[self.location[n, t - 1]][1][self.action[n, t - 1]][self.reward[n, t], :, self.belief1[n, t - 1]]
                        self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))
                        self.belief2Dist[n, t] = self.den[self.location[n, t - 1]][2][self.action[n, t - 1]][self.reward[n, t], :, self.belief2[n, t - 1]]
                        self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))

                        if self.action[n, t - 1] == a0:#do nothing
                            # if the action is to do nothing, the location remains the same
                            self.location[n, t] = self.location[n, t - 1]
                            self.location_ind[n, t] = self.location_ind[n, t - 1]
                            self.prev_location[n, t] = self.prev_location[n, t - 1]
                        # if the action is to go to location 0, i.e. the middle location
                        if self.action[n, t - 1] == g0:
                            # Tl0 = np.array(
                            #     [[1, 1 - delta, 1 - delta], [0, delta, 0],
                            #      [0, 0, delta]])  # go to loc 0 (with error of delta)
                            Tl = np.array([[delta, 0., 1. - delta, 0.],
                                            [0., delta, 0., 1. - delta],
                                            [0., 1. - delta, delta, 0.],
                                            [1. - delta, 0., 0., delta]])
                            self.location_ind[n, t] = np.argmax(np.random.multinomial(1, Tl[:, self.location_ind[n, t - 1]], size  = 1))
                            self.location[n, t] = locations_with_history[self.location_ind[n, t]][1]
                            self.prev_location[n, t] = locations_with_history[self.location_ind[n, t]][0]
                        # if self.action[n, t - 1] == g1:
                        #     Tl1 = np.array([[delta, 0, 1 - delta - direct], [1 - delta, 1, direct],
                        #                     [0, 0, delta]])  # go to box 1 (with error of delta)
                        #     self.location[n, t] = np.argmax(np.random.multinomial(1, Tl1[:, self.location[n, t - 1]], size  = 1))
                        # if self.action[n, t - 1] == g2:
                        #     Tl2 = np.array([[delta, 1 - delta - direct, 0], [0, delta, 0],
                        #                     [1 - delta, direct, 1]])  # go to box 2 (with error of delta)
                        #     self.location[n, t] = np.argmax(np.random.multinomial(1, Tl2[:, self.location[n, t - 1]], size  = 1))
                        
                        # if self.trueState1[n, t - 1] == 0:
                        #     self.trueState1[n, t] = np.random.binomial(1, 1 - p_disappear_e)
                        # else:
                        #     self.trueState1[n, t] = np.random.binomial(1, p_appear_e)

                        # if self.trueState2[n, t - 1] == 0:
                        #     self.trueState2[n, t] = np.random.binomial(1, 1 - p_disappear_e)
                        # else:
                        #     self.trueState2[n, t] = np.random.binomial(1, p_appear_e)

                    if self.action[n, t - 1] == pb:  # press button
                        self.location[n, t] = self.location[n, t - 1]  # pressing button does not change location
                        self.prev_location[n, t] = self.prev_location[n, t - 1]
                        self.location_ind[n, t] = self.location_ind[n, t - 1]
                        
                        if self.location[n, t - 1] == 0:
                            # pressing button at the center does not change anything
                            # then wait an intermediate step (everything evolves as if doing nothing)
                            self.reward[n, t] = 0.0
                            # self.belief1[n, t] = self.belief1[n, t - 1]
                            # self.belief2[n, t] = self.belief2[n, t - 1]
                            self.belief1Dist[n, t] = self.den[0][1][pb][self.reward[n, t], :, self.belief1[n, t - 1]]
                            self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))
                            # self.belief2[n, t] = self.belief2[n, t - 1]
                            self.belief2Dist[n, t] = self.den[0][2][pb][self.reward[n, t], :, self.belief2[n, t - 1]]
                            self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))
                            

                        if self.location[n, t] == 1:  # consider location 1 case
                            if self.trueState1[n, t - 1] == 0:#if the box is inactive
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward[n, t] = 0
                                else:
                                    self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food

                            else:#if the box is active
                                self.reward[n, t] = np.random.binomial(1, p_rwd_e)  # give some reward with probability prwd

                            #check how this works with observations as rewards...
                            self.belief1Dist[n, t] = self.den[1][1][pb][self.reward[n, t], :, self.belief1[n, t - 1]]
                            self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))
                            # self.belief2[n, t] = self.belief2[n, t - 1]
                            self.belief2Dist[n, t] = self.den[1][2][pb][self.reward[n, t], :, self.belief2[n, t - 1]]
                            self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))
                            
                            # if self.reward[n, t - 1] == 1:#if we received a reward at the previous time step
                            #     #then we know the other site is inactive
                            #     self.belief2Dist[n, t] = self.den2[pb][0, :, 0]
                            #     self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))
                            # else:
                            #     #get the mirror of belief 1 index around
                            #     #the center of the belief state space (nq/2)
                            #     self.belief2[n, t] = self.nq - self.belief1[n, t] - 1#self.belief2[n, t - 1]#

                        if self.location[n, t] == 2:  # consider location 2 case
                            if self.trueState2[n, t - 1] == 0:
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward[n, t] = 0
                                else:
                                    self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food

                            else:
                                self.reward[n, t] = np.random.binomial(1, p_rwd_e)

                            # self.belief1[n, t] = self.belief1[n, t - 1]
                            self.belief1Dist[n, t] = self.den[2][1][pb][self.reward[n, t], :, self.belief1[n, t - 1]]
                            self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))
                            self.belief2Dist[n, t] = self.den[2][2][pb][self.reward[n, t], :, self.belief2[n, t - 1]]
                            self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))
                            
                            # if self.reward[n, t - 1] == 1:#if we received a reward at this time step
                            #     #then we know the other site is inactive
                            #     self.belief1Dist[n, t] = self.den1[pb][0, :, 0]
                            #     self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))
                            # else:
                            #     self.belief1[n, t] = self.nq - self.belief2[n, t] - 1
                            #     # self.belief1[n, t] = self.belief1[n, t - 1]

                    self.hybrid[n, t] = self.location_ind[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) \
                                        + self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing

                    # self.action[n, t] = self._chooseAction(self.softpolicy.T[self.hybrid[n, t]])
                    # self.actionDist[n, t] = self.softpolicy.T[self.hybrid[n, t]]
                    self.actionDist[n, t] = self.softpolicy.T[self.hybrid[n, t]]
                    self.action[n, t] = self._chooseAction(self.actionDist[n, t])

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

class twoboxCazettesIndependentDependentMDP:
    def __init__(self, discount, nq, nr, na, nl, parameters):
        self.discount = discount
        self.nq = nq #number of belief states
        self.nr = nr #number of reward states, i.e. world states, active or inactive
        self.na = na #number of actions
        self.nl = nl   # number of locations
        self.nlwh = 2*(nl - 1)   # number of locations with history
        self.n = self.nq * self.nq * self.nr * self.nl   # total number of outcomes, or unique system (world and agent) states
        self.nwh = self.nq * self.nq * self.nr * self.nlwh   # total number of outcomes, or unique system (world and agent) states
        self.parameters = parameters
        #every action has a transition matrix and a reward function.
        # transition matrix, per action each column defines the probability of transitioning to each other unique system state
        # self.ThA = np.zeros((self.na, self.n, self.n))
        self.ThA = np.zeros((self.na, self.nwh, self.nwh)) 

        # for each action the probability of receiving a reward in each unique system state
        # self.R = np.zeros((self.na, self.n, self.n)) # reward function
        self.R = np.zeros((self.na, self.nwh, self.nwh)) # reward function

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
        p_sw = self.parameters[0]   # location activity switches after button press
        p_rwd = self.parameters[1]    # reward is returned for button press at active location
        actions = np.array([a0, g0, pb])
        locations = np.arange(self.nl)
        locations_with_history = create_ordered_tuples(self.nl)
        
        NumCol = 2  # number of colors
        # State rewards
        Reward = 1   # reward per time step with food in mouth
        Groom = self.parameters[2]   # location 0 reward
        doNothingCost = 0.005
        pushButtonCost_l0 = 100#cost of pushing the button at location 0
        # Action costs
        travelCost = self.parameters[3]
        pushButtonCost = self.parameters[4]
        beta = 0     # available food dropped back into box after button press
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0

        # kronecker product of these initial distributions
        # Note that this ordering makes the subsequent products easiest

        # setup single-variable transition matrices for each action..
        # these calculate \overline{O}(o_{t+1}|b_{t},a_{t}) = \int ds_{t+1} ds_{t} O(o_{t+1}|s_{t+1}, a_{t}) * T(s_{t+1}|s_{t},a_{t})* B(s_{t}|b_{t})
        # and P(b_{t+1}|b_{t},a_{t},o_{t+1})
        # for box 1
        Tr = jnp.array([[1, rho], [0, 1 - rho]])  # consume reward
        self.Trans_belief_obs, self.Obs_emis_trans, self.den = beliefTransitionMatrixGaussianCazettesIndependentDependent(p_sw, p_rwd, self.nq, actions, locations, sigma = 1 / self.nq / 3)
        belief_locations = [loc for loc in locations if loc != 0]
        Tb_unpack = {bloc:{oloc:{} for oloc in locations} for bloc in locations}
        for bloc in locations:#bloc is the location of the agent
            for oloc in locations:#other locations where the agent is NOT located
                for action in actions:
                    # belief transitions, it is  marginalized over observations, P(b_{t+1}|b_{t},a_{t},o_{t+1})
                    Trans_belief = jnp.sum(self.Trans_belief_obs[bloc][oloc][action], axis=0)
                    Tb_unpack[bloc][oloc][action] = Trans_belief / jnp.tile(np.sum(Trans_belief, 0), (self.nq, 1))#normalize each column
        
        #make per location belief transition matrices
        Tb = {bloc:{} for bloc in locations}
        for action in actions:
            for bloc in belief_locations:#belief location is the location of the agent
                Tb_loc = Tb_unpack[bloc]
                tbs = []
                for loc in belief_locations:#all locations in order.
                    tb = Tb_loc[loc][action]
                    tbs.append(tb)
                #insert Tr at the second position
                tbs.insert(1, Tr)
                Tb[bloc][action] = kronn(*tbs)#kronecker product of the transition matrices
            Tb[0][action] = kronn(Tb_unpack[0][1][action], Tr, Tb_unpack[0][2][action])
        
        # now fill out ThA i.e.
        # \overline{T}(b_{t+1}|b_{t},a_{t}) = \int do_{t+1} P(b_{t+1}|b_{t},a_{t},o_{t+1})*\overline{O}(o_{t+1}|b_{t},a_{t})
        # where \overline{O}(o_{t+1}|b_{t},a_{t}) = \int ds_{t+1} ds_{t} O(o_{t+1}|s_{t+1}, a_{t}) * T(s_{t+1}|s_{t},a_{t})* B(s_{t}|b_{t})
        # Note: the order in which the kronecker products are taken is important because it
        # implies the organization of the ThA matrix. The ThA is organized into blocks, the first
        # are specific to the location.

        #For debugging and housekeeping, we will also create the labels for the ThA and R matrices
        #first make the labels for ThA using the kronn written for strings
        loc_labels = np.array([str(loc) for loc in locations])
        loc_labels_with_history = np.array([str(loc) for loc in locations_with_history])

        bel1_labels = np.array(['b1_' + str(bel) for bel in range(self.nq)])
        bel2_labels = np.array(['b2_' + str(bel) for bel in range(self.nq)])
        rew_labels = np.array(['r_' + str(rew) for rew in range(self.nr)])
        #make the transition matrices for the labels using the string meshgrid
        loc_mesh = string_meshgrid(loc_labels_with_history, loc_labels_with_history)
        rew_mesh = string_meshgrid(rew_labels, rew_labels)
        bel1_mesh = string_meshgrid(bel1_labels, bel1_labels)
        bel2_mesh = string_meshgrid(bel2_labels, bel2_labels)
        self.ThA_labels = kronn_str(loc_mesh, bel1_mesh, rew_mesh, bel2_mesh)
        #next make the labels for Reward function using the tensorsumm written for strings
        self.R_labels = tensorsumm_str(loc_labels_with_history[:, None], bel1_labels[None, :], rew_labels[:, None], bel2_labels[None, :]).flatten()

        # ACTION: do nothing
        self.ThA[a0, :, :] = kronn(jnp.identity(self.nlwh), jnp.identity(self.nq), Tr, jnp.identity(self.nq))

        # self.ThA[a0, :, :] = kronn(np.identity(self.nl), np.identity(self.nq), Tr, np.identity(self.nq))
        # ACTION: go to location 0/1/2
        #create location transition matrix for each unique location.
        #delta is the probability of not going to the target location
        #direct is the probability of going directly to the target location (skipping location 0)
        # Tl0 = np.array([[1, 1 - delta, 1 - delta], 
        #                 [0, delta, 0], 
        #                 [0, 0, delta]])  # go to loc 0 (with error of delta)
        # Tl1 = np.array([[delta, 0, 1 - delta - direct], 
        #                 [1 - delta, 1, direct],
        #                 [0, 0, delta]])  # go to box 1 (with error of delta)
        # Tl2 = np.array([[delta, 1 - delta - direct, 0], 
        #                 [0, delta, 0],
        #                 [1 - delta, direct, 1]])  # go to box 2 (with error of delta)

        # self.ThA[g0, :, :] = kronn(Tl0, np.identity(self.nq), Tr, np.identity(self.nq))
        # self.ThA[g1, :, :] = kronn(Tl1, np.identity(self.nq), Tr, np.identity(self.nq))
        # self.ThA[g2, :, :] = kronn(Tl2, np.identity(self.nq), Tr, np.identity(self.nq))
        #technically there are only two foraging sites and 1 intermediate site..
        #but to capture the restrictions on the agent's movement, where it cannot transition directly
        #from one foraging site to the other, and it cannot transition from the intermediate site
        #back to the foraging site it came from, we will create a transition matrix for each possible
        #(previous location, current location) combination that is possible.
        #Here these will be (1, 0), (2, 0), (0, 1), (0, 2).
        #keeping with the current state on the columns and next state on the rows convention,
        #the way to read this transition matrix is what previous location and current location tuple
        #is transitioning to the current location and next location tuple.
        #first column: (1, 0) -> (0, 2) has to go to site 2 if coming from site 1, can error to stay at site 0
        #second column: (2, 0) -> (0, 1) has to go to site 1 if coming from site 2, can error to stay at site 0
        #third column: (0, 1) -> (1, 0) has to return to the intermediate site, can error to stay at site 1
        #fourth column: (0, 2) -> (2, 0) has to return to the intermediate site, can error to stay at site 2
        Tl = np.array([[delta, 0., 1. - delta, 0.], 
                        [0., delta, 0., 1. - delta],
                        [0., 1. - delta, delta, 0.],
                        [1. - delta, 0., 0., delta]])

        Tl_other = np.identity(self.nlwh)
        
        #each entry in ThA is a product, for each action, of every possible system state combination
        #the kronecker product allows us to efficiently calculate these products by taking the iterative
        #kronecker product of the location, belief, reward consumption states.
        self.ThA[g0, :, :] = kronn(Tl, np.identity(self.nq), Tr, np.identity(self.nq))

        # ACTION: push button.
        Th = block_diag(kronn(np.identity(self.nq), Tr, np.identity(self.nq)), 
                        kronn(np.identity(self.nq), Tr, np.identity(self.nq)),
                        Tb[1][pb], Tb[2][pb])
        
        #this dot product is of the ith push button NEXT system state with the jth 
        #do nothing system CURRENT system state.
        self.ThA[pb, :, :] = Th#.dot(self.ThA[a0, :, :])

        # The reward function is the expecation of the reward on the next time step
        # given the current state and action.
        # accumulate rewards for each possible location, belief state 1, reward consumption state, and
        # belief state.
        loc_rewards = np.array([[Groom, Groom, 0, 0]])
        Reward_h = tensorsumm(loc_rewards, np.zeros((1, self.nq)), np.array([[0, Reward]]),
                              np.zeros((1, self.nq)))
        # reward_l0 = np.ones((1, self.nq*self.nr*self.nq)) * Groom
        # reward_l1_l2 = tensorsumm(np.zeros((1, 2)),np.zeros((1, self.nq)), np.array([[0, Reward]]),
        #                       np.zeros((1, self.nq)))
        # Reward_h = np.column_stack((reward_l0, reward_l1_l2))
        #complie the action costs.
        Reward_a = - np.array([doNothingCost, travelCost, pushButtonCost])

        #create a 3D meshgrid which pairs each possible reward state h with each
        #possible action cost a.
        [R1, R2, R3] = np.meshgrid(Reward_a.T, Reward_h, Reward_h, indexing='ij')
        Reward = R1 + R3 # R1 is the action cost, R3 is the reward for the state
        #adjust the reward matrix so that the push button action is differentially penalized
        #in the middle position. There is no button to push at that location.
        Reward[pb, :, :2*self.nq * self.nr * self.nq] += pushButtonCost_l0*-1
        
        # R = Reward[:, 0, :].T
        self.R = Reward

        #flip the transition matrices so past states are now rows
        #and future states are columns
        for i in range(self.na):
            self.ThA[i, :, :] = self.ThA[i, :, :].T
            # self.R[i, :, :] = self.R[i, :, :].T

        #prep the density matrices
        den_col = {bloc:{} for bloc in locations}
        for action in actions:
            for bloc in belief_locations:#belief location is the location of the agent
                obs_dict = {}
                for i in range(NumCol):
                    j = i
                    den_loc = self.den[bloc]
                    dens = [den_loc[1][action][i], Tr, den_loc[2][action][j]]
                    obs_dict[(i, j)] = kronn(*dens)#kronecker product of the observation matrices
                den_col[bloc][action] = obs_dict
            den_col[0][action] = {(i, i):kronn(self.den[0][1][action][0], Tr, self.den[0][2][action][0]) for i in range(NumCol)}
                
        self.Trans_hybrid_obs12 = np.zeros(((NumCol, NumCol, self.na, self.nwh, self.nwh)))
        for i in range(NumCol):
            j = i
            # self.Trans_hybrid_obs12[i, j, a0, :, :] = kronn(np.identity(self.nl), np.identity(self.nq), Tr, np.identity(self.nq)).T
            # self.Trans_hybrid_obs12[i, j, g0, :, :] = kronn(Tl, np.identity(self.nq), Tr, np.identity(self.nq)).T
            # self.Trans_hybrid_obs12[i, j, g1, :, :] = kronn(Tl1, np.identity(self.nq), Tr, np.identity(self.nq)).T
            # self.Trans_hybrid_obs12[i, j, g2, :, :] = kronn(Tl2, np.identity(self.nq), Tr, np.identity(self.nq)).T
            self.Trans_hybrid_obs12[i, j, a0, :, :] = kronn(np.identity(self.nlwh), np.identity(self.nq), Tr, np.identity(self.nq)).T
            self.Trans_hybrid_obs12[i, j, g0, :, :] = kronn(Tl, np.identity(self.nq), Tr, np.identity(self.nq)).T
            self.Trans_hybrid_obs12[i, j, pb, :, :] = block_diag(kronn(np.identity(self.nq), Tr, np.identity(self.nq)), 
                                                                        kronn(np.identity(self.nq), Tr, np.identity(self.nq)),
                                                                        den_col[1][pb][(i,j)], den_col[2][pb][(i,j)]).T


    def solveMDP_op(self, epsilon = 0.001, niterations = 10000, initial_value=0):
        vi = ValueIteration_opZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        # optimal policy, stopping criterion changed to "converged Qvalue"
        vi.setVerbose()
        vi.run()
        self.Q = self._QfromV(vi)
        self.policy = np.array(vi.policy)
        self.Vop = vi.V

    def solveMDP_sfm(self, epsilon = 0.001, niterations = 10000, initial_value=0):

        temperatureQ = self.parameters[5]
        vi = ValueIteration_sfmZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        vi.setVerbose()
        vi.run(temperatureQ)
        self.Qsfm = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        self.softpolicy = np.array(vi.softpolicy)
        self.Vsfm = vi.V

    def _QfromV(self, ValueIteration):
        Q = np.zeros((ValueIteration.A, ValueIteration.S)) # Q is of shape: na * n
        for a in range(ValueIteration.A):
            Q[a, :] = ValueIteration.R[a] + ValueIteration.discount * \
                                            jnp.dot(ValueIteration.P[a], ValueIteration.V)
        return Q

class twoboxCazettesIndependentDependentMDPdata(twoboxCazettesIndependentDependentMDP):
    def __init__(self, discount, nq, nr, na, nl, parameters, parametersExp,
                 sampleTime, sampleNum):
        twoboxCazettesIndependentDependentMDP.__init__(self, discount, nq, nr, na, nl, parameters)

        self.parametersExp = parametersExp# parameters for the experiment
        self.sampleNum = sampleNum
        self.sampleTime = sampleTime

        self.action = np.empty((self.sampleNum, self.sampleTime), int)  # initialize action, assumed to be optimal
        self.hybrid = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hybrid state.
        # Here it is the joint state of reward and belief
        self.location = np.empty((self.sampleNum, self.sampleTime), int)  # initialize location state, true world location
        self.location_ind = np.empty((self.sampleNum, self.sampleTime), int)  # initialize location state, abstract world location with history
        self.prev_location = np.empty((self.sampleNum, self.sampleTime), int)  # initialize location state
        self.belief1 = np.empty((self.sampleNum, self.sampleTime), int)
        self.belief2 = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hidden state, belief state
        self.reward = np.empty((self.sampleNum, self.sampleTime), int)  # initialize reward state
        self.trueState1 = np.zeros((self.sampleNum, self.sampleTime))
        self.trueState2 = np.zeros((self.sampleNum, self.sampleTime))
        self.color1 = np.zeros((self.sampleNum, self.sampleTime), dtype=int)#color and true state are the same
        self.color2 = np.zeros((self.sampleNum, self.sampleTime), dtype=int)#color and true state are the same

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

        p_rwd_e = self.parametersExp[1]    # location activity switches after button press
        p_sw_e = self.parametersExp[0]   # reward is returned for button press at active location
        actions = np.array([a0, g0, pb])
        locations_with_history = create_ordered_tuples(self.nl)

        # State rewards
        Groom = self.parametersExp[2]     # location 0 reward
        # Action costs
        travelCost = self.parametersExp[3]
        pushButtonCost = self.parametersExp[4]

        ## Generate data
        for n in range(self.sampleNum):

            belief1Initial = np.random.randint(self.nq)
            rewInitial = np.random.randint(self.nr)#maybe set 0
            belief2Initial = np.random.randint(self.nq)
            locationInitial_ind = np.random.randint(self.nlwh)
            locationInitial_prev = locations_with_history[locationInitial_ind][0]
            locationInitial = locations_with_history[locationInitial_ind][1]

            for t in range(self.sampleTime):
                if t == 0:
                    # Initialize the true world states, sensory information and latent states
                    active_site = np.random.randint(1,3)
                    self.trueState1[n, t] = 1#np.random.randint(2)
                    self.color1[n, t] = self.trueState1[n, t]
                    self.trueState2[n, t] = 1#np.random.randint(2)
                    self.color2[n, t] = self.trueState2[n, t]
                
                    self.location[n, t], self.reward[n, t] = locationInitial, rewInitial
                    self.prev_location[n, t] = locationInitial_prev
                    self.location_ind[n, t] = locationInitial_ind

                    self.belief1[n, t], self.belief2[n, t] = belief1Initial, belief2Initial
                    self.belief1Dist[n, t, self.belief1[n, t]] = 1
                    self.belief2Dist[n, t, self.belief2[n, t]] = 1

                    self.hybrid[n, t] = self.location_ind[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) + \
                                        self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing
                    self.actionDist[n, t] = self.softpolicy.T[self.hybrid[n, t]]
                    self.action[n, t] = self._chooseAction(self.actionDist[n, t])

                else:
                    if self.action[n, t - 1] != pb:
                        # variables evolve with dynamics
                        self.trueState1[n, t] = self.trueState1[n, t - 1]
                        self.trueState2[n, t] = self.trueState2[n, t - 1]
                        self.color1[n, t] = self.trueState1[n, t]
                        self.color2[n, t] = self.trueState2[n, t]

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
                            self.prev_location[n, t] = self.prev_location[n, t - 1]
                            self.location_ind[n, t] = self.location_ind[n, t - 1]
                        # if the action is to go to location 0, i.e. the middle location
                        if self.action[n, t - 1] == g0:
                            # Tl0 = np.array(
                            #     [[1, 1 - delta, 1 - delta], [0, delta, 0],
                            #      [0, 0, delta]])  # go to loc 0 (with error of delta)
                            # self.location[n, t] = np.argmax(np.random.multinomial(1, Tl0[:, self.location[n, t - 1]], size  = 1))
                            Tl = np.array([[delta, 0., 1. - delta, 0.],
                                            [0., delta, 0., 1. - delta],
                                            [0., 1. - delta, delta, 0.],
                                            [1. - delta, 0., 0., delta]])
                            self.location_ind[n, t] = np.argmax(np.random.multinomial(1, Tl[:, self.location_ind[n, t - 1]], size  = 1))
                            self.location[n, t] = locations_with_history[self.location_ind[n, t]][1]
                            self.prev_location[n, t] = locations_with_history[self.location_ind[n, t]][0]
                        # if self.action[n, t - 1] == g1:
                        #     Tl1 = np.array([[delta, 0, 1 - delta - direct], [1 - delta, 1, direct],
                        #                     [0, 0, delta]])  # go to box 1 (with error of delta)
                        #     self.location[n, t] = np.argmax(np.random.multinomial(1, Tl1[:, self.location[n, t - 1]], size  = 1))
                        # if self.action[n, t - 1] == g2:
                        #     Tl2 = np.array([[delta, 1 - delta - direct, 0], [0, delta, 0],
                        #                     [1 - delta, direct, 1]])  # go to box 2 (with error of delta)
                        #     self.location[n, t] = np.argmax(np.random.multinomial(1, Tl2[:, self.location[n, t - 1]], size  = 1))

                    if self.action[n, t - 1] == pb:  # press button
                        # self.location[n, t] = self.location[n, t - 1]  # pressing button does not change location
                        # variables evolve with dynamics
                        self.location[n, t] = self.location[n, t - 1]  # pressing button does not change location
                        self.prev_location[n, t] = self.prev_location[n, t - 1]
                        self.location_ind[n, t] = self.location_ind[n, t - 1]

                        if self.location[n, t - 1] == 0:
                            # pressing button at the center does not change anything
                            # then wait an intermediate step (everything evolves as if doing nothing)
                            self.reward[n, t] = 0.0
                            self.belief1[n, t] = self.belief1[n, t - 1]
                            self.belief2[n, t] = self.belief2[n, t - 1]

                        if self.location[n, t] == 1:  # consider location 1 case
                            if self.trueState1[n, t - 1] == 0:
                                self.trueState1[n, t] = np.random.binomial(1, p_sw_e)
                            else:
                                self.trueState1[n, t] = np.random.binomial(1, 1 - p_sw_e)
                            self.color1[n, t] = self.trueState1[n, t]
                            self.trueState2[n, t] = self.trueState2[n, t - 1]
                            self.color2[n, t] = self.trueState2[n, t]

                            if self.trueState1[n, t - 1] == 0:#if the box is inactive
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward[n, t] = 0
                                else:
                                    self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food

                            else:#if the box is active
                                self.reward[n, t] = np.random.binomial(1, p_rwd_e)  # give some reward with probability prwd

                            #check how this works with observations as rewards...
                            self.belief1Dist[n, t] = self.den[1][1][pb][self.reward[n, t], :, self.belief1[n, t - 1]]
                            self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))
                            # self.belief2[n, t] = self.belief2[n, t - 1]
                            self.belief2Dist[n, t] = self.den[1][2][pb][self.reward[n, t], :, self.belief2[n, t - 1]]
                            self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))

                            # if self.reward[n, t - 1] == 1:#if we received a reward at the previous time step
                            #     #then we know the other site is inactive
                            #     self.belief2Dist[n, t] = self.den2[pb][0, :, 0]
                            #     self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))
                            # else:
                            #     #get the mirror of belief 1 index around
                            #     #the center of the belief state space (nq/2)
                            #     self.belief2[n, t] = self.nq - self.belief1[n, t] - 1#self.belief2[n, t - 1]#

                        if self.location[n, t] == 2:  # consider location 2 case
                            if self.trueState2[n, t - 1] == 0:
                                self.trueState2[n, t] = np.random.binomial(1, p_sw_e)
                            else:
                                self.trueState2[n, t] = np.random.binomial(1, 1 - p_sw_e)
                            self.color2[n, t] = self.trueState2[n, t]
                            self.trueState1[n, t] = self.trueState1[n, t - 1]
                            self.color1[n, t] = self.trueState1[n, t]

                            if self.trueState2[n, t - 1] == 0:
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward[n, t] = 0
                                else:
                                    self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food

                            else:
                                self.reward[n, t] = np.random.binomial(1, p_rwd_e)

                            # self.belief1[n, t] = self.belief1[n, t - 1]
                            self.belief1Dist[n, t] = self.den[2][1][pb][self.reward[n, t], :, self.belief1[n, t - 1]]
                            self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))
                            self.belief2Dist[n, t] = self.den[2][2][pb][self.reward[n, t], :, self.belief2[n, t - 1]]
                            self.belief2[n, t] = np.argmax(np.random.multinomial(1, self.belief2Dist[n, t], size=1))
                            
                            # if self.reward[n, t - 1] == 1:#if we received a reward at this time step
                            #     #then we know the other site is inactive
                            #     self.belief1Dist[n, t] = self.den1[pb][0, :, 0]
                            #     self.belief1[n, t] = np.argmax(np.random.multinomial(1, self.belief1Dist[n, t], size=1))
                            # else:
                            #     self.belief1[n, t] = self.nq - self.belief2[n, t] - 1
                            #     # self.belief1[n, t] = self.belief1[n, t - 1]

                    self.hybrid[n, t] = self.location_ind[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) \
                                        + self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing

                    # self.action[n, t] = self._chooseAction(self.softpolicy.T[self.hybrid[n, t]])
                    # self.actionDist[n, t] = self.softpolicy.T[self.hybrid[n, t]]
                    self.actionDist[n, t] = self.softpolicy.T[self.hybrid[n, t]]
                    self.action[n, t] = self._chooseAction(self.actionDist[n, t])

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




























