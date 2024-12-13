import numpy as np
from boxtask_funcJax import *

import jax.numpy as jnp
import jax
from jax import vmap, pmap
from pprint import pprint
from jax.lib import xla_bridge
import jaxlib
from jax import jit

a0 = 0  # a0 = do nothing
g0 = 1  # g0 = go to location 0
pb = 2  # pb  = push button


class HMMtwoboxCazettes:
    def __init__(self, A, B, C, D1, D2, pi, Ncol, lick_state=False):
        self.A = A # transition matrix action x state x state
        self.B = B # policy matrix action x state x state
        self.C = C  # (Trans_hybrid_obs12, belief transition given observation and action)
        self.D1 = D1  # (box1, Obs_emis.dot(Trans_state, to calculate oberservation emission)
        self.D2 = D2  # (box2, Obs_emis.dot(Trans_state, to calculate oberservation emission)
        self.pi = pi
        self.S = len(self.pi)  # number of possible values of the hidden state (hybrid for two boxes)
        self.R = 2 # number of reward consumption states 
        self.L = 3 # number of possible physical locations
        self.Linds = 2*(self.L - 1)#number of abstract locations considering location history
        self.Pb = 2#number of possible previous push button action states
        self.Ss = int(sqrt(self.S))
        self.Ncol = Ncol#number of observable colors 0 no observation , 1 no reward, 2 reward
        self.lick_state = lick_state#whether to consider lick states in the model, i.e. use push button action

    #location and rewards select the possible states
    #in the case with lick states, the previous action is also considered
    def _states_wLickStates(self, r, l, a):
        temp = jnp.reshape(jnp.array(range(self.Ss)), [1, self.Ss])
        return jnp.squeeze(l * self.S * self.R * self.Pb + a * self.S * self.R + 
                            tensorsum(temp * self.R * self.Ss, r * self.Ss + temp)).astype(int)
    
    def _states_noLickStates(self, r, l):
        temp = jnp.reshape(jnp.array(range(self.Ss)), [1, self.Ss])
        return jnp.squeeze(l * self.S * self.R + tensorsum(temp * self.R * self.Ss, r * self.Ss + temp)).astype(int)

    def forward_scale(self, obs):
        # probability of latent at time t given observations from the beginning to time t
        T = obs.shape[0]  # length of a sample sequence
        act = obs[:, 0]  # action, two possible values: 0: doing nothing; 1: press button
        #get act array indicating push button trials as 1s
        act_pb = jnp.zeros_like(act)
        mask = (act == pb)
        act_pb = jnp.where(mask, 1, act_pb)
        rew = obs[:, 1]  # observable, two possible values: 0 : not have; 1: have
        loc = obs[:, 2]  # location index
        col1 = obs[:, 3]  # color of the 1st box
        col2 = obs[:, 4]  # color of the 2nd box

        alpha = jnp.zeros((self.S, T))  # initialize alpha value for each belief value
        scale = jnp.zeros(T)

        def init_vals_fn(lick_state):
            return lax.cond(
                lick_state,
                lambda _: self.pi * self.B[act[0], self._states_wLickStates(rew[0], loc[0], 0)],
                lambda _: self.pi * self.B[act[0], self._states_noLickStates(rew[0], loc[0])],
                operand=None
            )

        init_vals = init_vals_fn(self.lick_state)

        alpha = alpha.at[:, 0].set(init_vals)
        scale = scale.at[0].set(jnp.sum(alpha[:, 0]))
        alpha = alpha.at[:, 0].set(alpha[:, 0] / (scale[0]))

        belief_vector = jnp.array(
            [jnp.arange(0, 1, 1 / self.Ss) + 1 / self.Ss / 2, 1 - jnp.arange(0, 1, 1 / self.Ss) - 1 / self.Ss / 2])

        def body_fn(carry, t):
            alpha, scale = carry

            def alpha_update_fn(lick_state):
                return lax.cond(
                    lick_state,
                    lambda _: jnp.dot(alpha[:, t - 1],
                                      self.C[col1[t]][col2[t]][act[t - 1]][
                                          jnp.ix_(self._states_wLickStates(rew[t - 1], loc[t - 1], act_pb[t-2] if t > 1 else 0), 
                                                  self._states_wLickStates(rew[t], loc[t], act_pb[t-1]))]) \
                              * self.B[act[t], self._states_wLickStates(rew[t], loc[t], act_pb[t-1])],
                    lambda _: jnp.dot(alpha[:, t - 1],
                                      self.C[col1[t]][col2[t]][act[t - 1]][
                                          jnp.ix_(self._states_noLickStates(rew[t - 1], loc[t - 1]), 
                                                  self._states_noLickStates(rew[t], loc[t]))]) \
                              * self.B[act[t], self._states_noLickStates(rew[t], loc[t])],
                    operand=None
                )

            alpha_update = alpha_update_fn(self.lick_state)
            alpha = alpha.at[:, t].set(alpha_update)
            scale = scale.at[t].set(jnp.sum(alpha[:, t]))
            alpha = alpha.at[:, t].set(alpha[:, t] / scale[t])

            return (alpha, scale), None

        (alpha, scale), _ = lax.scan(body_fn, (alpha, scale), jnp.arange(1, T))

        return alpha, scale

    def backward_scale(self, obs, scale):
        # likelihood of latent at t given observations from time t+1 till the end
        # in other words, the probability of observations from time t+1 till the end given latent at t

        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        #get act array indicating push button trials as 1s
        act_pb = jnp.zeros_like(act)
        act_pb = jnp.where(mask, 1, act_pb)
        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values
        col1 = obs[:, 3]  # color of the 1st box
        col2 = obs[:, 4]  # color of the 2nd box

        beta = jnp.zeros((self.S, T))
        beta = beta.at[:, T - 1].set(1)

        belief_vector = jnp.array(
            [jnp.arange(0, 1, 1 / self.Ss) + 1 / self.Ss / 2, 1 - jnp.arange(0, 1, 1 / self.Ss) - 1 / self.Ss / 2])

        def body_fn(carry, t):
            beta = carry

            act_t1 = lax.cond(t > 0, lambda _: act_pb[t-1], lambda _: 0, operand=None)
            act_t = act_pb[t]

            beta_update = lax.cond(
                self.lick_state,
                lambda _: jnp.dot(self.C[col1[t + 1]][col2[t + 1]][act[t]][
                    jnp.ix_(self._states_wLickStates(rew[t], loc[t], act_t1), 
                            self._states_wLickStates(rew[t + 1], loc[t + 1], act_t))],
                    (beta[:, t + 1]) * self.B[act[t + 1], self._states_wLickStates(rew[t + 1], loc[t + 1], act_t)]),
                lambda _: jnp.dot(self.C[col1[t + 1]][col2[t + 1]][act[t]][
                    jnp.ix_(self._states_noLickStates(rew[t], loc[t]), 
                            self._states_noLickStates(rew[t + 1], loc[t + 1]))],
                    beta[:, t + 1] * self.B[act[t + 1], self._states_noLickStates(rew[t + 1], loc[t + 1])]),
                operand=None
            )

            beta = beta.at[:, t].set(beta_update / scale[t + 1])

            return beta, None

        beta, _ = lax.scan(body_fn, beta, jnp.arange(T - 2, -1, -1))

        return beta

    def compute_gamma(self, alpha, beta):
        # The posterior  of latent variables given the whole sequence of observations

        gamma = alpha * beta
        gamma = gamma / (jnp.sum(gamma, 0))

        return gamma

    def compute_xi(self, alpha, beta, obs):
        # joint probability of latent at time t and t+1 given the whole sequence of observations

        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        #get act array indicating push button trials as 1s
        act_pb = jnp.zeros_like(act)
        
        mask = (act == pb)
        act_pb = jnp.where(mask, 1, act_pb)

        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values
        col1 = obs[:, 3]  # color of the 1st box
        col2 = obs[:, 4]  # color of the 2nd box

        xi = jnp.zeros((T - 1, self.S, self.S))

        belief_vector = np.array(
            [jnp.arange(0, 1, 1 / self.Ss) + 1 / self.Ss / 2, 1 - jnp.arange(0, 1, 1 / self.Ss) - 1 / self.Ss / 2])

        def body_fn(carry, t):
            xi, alpha, beta = carry

            act_t1 = lax.cond(t > 0, lambda _: act_pb[t-1], lambda _: 0, operand=None)
            act_t = act_pb[t]

            def xi_update_fn(lick_state):
                return lax.cond(
                    lick_state,
                    lambda _: jnp.diag(alpha[:, t]).dot(
                        self.C[col1[t + 1]][col2[t + 1]][act[t]][
                            jnp.ix_(self._states_wLickStates(rew[t], loc[t], act_t1), 
                                    self._states_wLickStates(rew[t + 1], loc[t + 1], act_t))]).dot(
                        jnp.diag(beta[:, t + 1] * self.B[act[t + 1], self._states_wLickStates(rew[t + 1], loc[t + 1], act_t)])
                    ),
                    lambda _: jnp.diag(alpha[:, t]).dot(
                        self.C[col1[t + 1]][col2[t + 1]][act[t]][
                            jnp.ix_(self._states_noLickStates(rew[t], loc[t]), 
                                    self._states_noLickStates(rew[t + 1], loc[t + 1]))]).dot(
                        jnp.diag(beta[:, t + 1] * self.B[act[t + 1], self._states_noLickStates(rew[t + 1], loc[t + 1])])
                    ),
                    operand=None
                )

            xi_update = xi_update_fn(self.lick_state)
            xi = xi.at[t].set(xi_update / jnp.sum(xi_update))

            return (xi, alpha, beta), None

        (xi, _, _), _ = lax.scan(body_fn, (xi, alpha, beta), jnp.arange(T - 1))

        return xi

    def latent_entr(self, obs):
        # the entropy of the sequence of latent variables
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        #get act array indicating push button trials as 1s
        act_pb = jnp.zeros_like(act)
        mask = (act == pb)
        act_pb = act_pb.at[mask].set(1)
        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values
        col1 = obs[:, 3]  # color of the 1st box
        col2 = obs[:, 4]  # color of the 2nd box

        # Entropy of all path that leads to a certain state at t certain time
        Hpath = jnp.zeros((self.S, T))
        # P(state at time t-1 | state at time t, observations up to time t)
        lat_cond = jnp.zeros((T - 1, self.S, self.S))

        alpha_scaled, _ = self.forward_scale(obs)

        belief_vector = jnp.array(
            [jnp.arange(0, 1, 1 / self.Ss) + 1 / self.Ss / 2, 1 - jnp.arange(0, 1, 1 / self.Ss) - 1 / self.Ss / 2])

        Hpath = Hpath.at[:, 0].set(0)

        def body_fn(carry, t):
            lat_cond, Hpath = carry

            act_t2 = lax.cond(t == 1, lambda _: 0, lambda _: act_pb[t-2], operand=None)
            act_t1 = act_pb[t-1]

            lat_cond_update = lax.cond(
                self.lick_state,
                lambda _: jnp.diag(alpha_scaled[:, t - 1]).dot(
                    self.C[col1[t]][col2[t]][act[t - 1]][
                        jnp.ix_(self._states_wLickStates(rew[t - 1], loc[t - 1], act_t2), 
                                self._states_wLickStates(rew[t], loc[t], act_t1))]),
                lambda _: jnp.diag(alpha_scaled[:, t - 1]).dot(
                    self.C[col1[t]][col2[t]][act[t - 1]][
                        jnp.ix_(self._states_noLickStates(rew[t - 1], loc[t - 1]), 
                                self._states_noLickStates(rew[t], loc[t]))]),
                operand=None
            )

            lat_cond = lat_cond.at[t - 1].set(lat_cond_update / (
                jnp.sum(lat_cond[t - 1], axis=0) + 1 * (jnp.sum(lat_cond[t - 1], axis=0) == 0)))

            Hpath = Hpath.at[:, t].set(Hpath[:, t - 1].dot(lat_cond[t - 1]) - jnp.sum(
                lat_cond[t - 1] * jnp.log(lat_cond[t - 1] + 10 ** -13 * (lat_cond[t - 1] == 0)), axis=0))

            return (lat_cond, Hpath), None

        (lat_cond, Hpath), _ = lax.scan(body_fn, (lat_cond, Hpath), jnp.arange(1, T))

        lat_ent = jnp.sum(Hpath[:, -1] * alpha_scaled[:, -1]) - jnp.sum(
            alpha_scaled[:, -1] * jnp.log(alpha_scaled[:, -1] + 10 ** -13 * (alpha_scaled[:, -1] == 0)))

        return lat_ent

    # def likelihood(self, lat, obs, Anew, Bnew):

    def computeQaux(self, obs, Anew, Bnew, Cnew, D1new, D2new):

        '''
        computer the Q auxillary funciton, the expected complete data likelihood
        :param obs: observations
        :param Anew: transition matrix with new parameters
        :param Bnew: policy with new parameters
        :param Cnew: Trans_hybrid_obs12 with new parameters
        :param D1new: box1, Obs_emis.dot(Trans_state, to calculate observation emission, with new parameters
        :param D2new: box2, Obs_emis.dot(Trans_state, to calculate observation emission, with new parameters
        :return: Q auxilary value
        '''
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        #get act array indicating push button trials as 1s
        act_pb = jnp.zeros_like(act)
        mask = (act == pb)
        act_pb = act_pb.at[mask].set(1)
        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values
        col1 = obs[:, 3]  # color of the 1st box
        col2 = obs[:, 4]  # color of the 2nd box

        alpha, scale = self.forward_scale(obs)
        beta = self.backward_scale(obs, scale)

        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(alpha, beta, obs)
        Qaux1 = jnp.sum(jnp.log(self.pi) * gamma[:, 0])
        Qaux2 = 0
        Qaux3 = 0

        def body_fn_Qaux2(carry, t):
            Qaux2 = carry

            act_t1 = lax.cond(t == 0, lambda _: 0, lambda _: act_pb[t-1], operand=None)
            act_t = act_pb[t]

            Trantemp = lax.cond(
                self.lick_state,
                lambda _: Cnew[col1[t + 1]][col2[t + 1]][act[t]][
                    jnp.ix_(self._states_wLickStates(rew[t], loc[t], act_t1), self._states_wLickStates(rew[t + 1], loc[t + 1], act_t))],
                lambda _: Cnew[col1[t + 1]][col2[t + 1]][act[t]][
                    jnp.ix_(self._states_noLickStates(rew[t], loc[t]), self._states_noLickStates(rew[t + 1], loc[t + 1]))],
                operand=None
            )

            Qaux2 += jnp.sum(jnp.log(Trantemp + 10 ** -13 * (Trantemp == 0)) * xi[t, :, :])
            return Qaux2, None

        Qaux2, _ = lax.scan(body_fn_Qaux2, Qaux2, jnp.arange(T - 1))

        def body_fn_Qaux3(carry, t):
            Qaux3 = carry

            act_t1 = lax.cond(t == 0, lambda _: 0, lambda _: act_pb[t-1], operand=None)
            act_t = act_pb[t]

            Btemp = lax.cond(
                self.lick_state,
                lambda _: Bnew[act[t], self._states_wLickStates(rew[t], loc[t], act_t1)],
                lambda _: Bnew[act[t], self._states_noLickStates(rew[t], loc[t])],
                operand=None
            )

            Qaux3 += jnp.sum(jnp.log(Btemp + 10 ** -13 * (Btemp == 0)) * gamma[:, t])
            return Qaux3, None

        Qaux3, _ = lax.scan(body_fn_Qaux3, Qaux3, jnp.arange(T))


        Qaux = 1 * (Qaux1 + Qaux2) + 1 * Qaux3

        return Qaux