import math as _math
import time as _time

import numpy as _np
import jax
import jax.numpy as _jnp
from jax.experimental import checkify
import jax.debug as debug
# jax.config.update("jax_enable_x64", True)
import scipy.sparse as _sp

# import mdptoolbox.util as _util
from mdp_utils_jax import check as check_
from mdp_utils_jax import getSpan

_MSG_STOP_MAX_ITER = "Iterating stopped due to maximum number of iterations condition."
_MSG_STOP_EPSILON_OPTIMAL_POLICY = "Iterating stopped, epsilon-optimal policy found."
_MSG_STOP_EPSILON_OPTIMAL_VALUE = "Iterating stopped, epsilon-optimal value function found."
_MSG_STOP_UNCHANGING_POLICY = "Iterating stopped, unchanging policy found."


def _computeDimensions(transition):
    A = len(transition)
    # Simplified: Assuming transition is always (A, S, S) based on your usage
    S = transition.shape[1]
    return S, A


class MDP(object):

    def __init__(self, transitions, reward, discount, epsilon, max_iter, mask):
        # Initialise a MDP based on the input parameters.
        
        # --- FIX: Direct assignment for JIT compatibility ---
        # Arguments passed to a JIT function are tracers. 
        self.discount = discount
        self.epsilon = epsilon
        self.max_iter = max_iter

        # We assume transitions and reward are already arrays or tracers
        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)
        self.R = self._computeReward(reward, transitions)
        
        # Defaults
        self.verbose = False
        self.time = None
        self.iter = 0
        self.V = None
        self.policy = None
        self.actionMask = mask

    def __repr__(self):
        P_repr = "P: \n"
        R_repr = "R: \n"
        for aa in range(self.A):
            P_repr += repr(self.P[aa]) + "\n"
            R_repr += repr(self.R[aa]) + "\n"
        return(P_repr + "\n" + R_repr)

    def _bellmanOperator(self, V=None):
        # Use Python control flow for static arguments like V and actionMask
        if V is None:
            V_curr = self.V
        else:
            V_curr = V
        
        if self.actionMask is None:
            mask = _jnp.ones((self.A, self.S))
        else:
            mask = self.actionMask

        def compute_Q(P_a, R_a):
            return R_a + self.discount * _jnp.dot(P_a, V_curr)

        Q = jax.vmap(compute_Q)(self.P, self.R)

        # apply mask: set masked actions to -inf
        masked_Q = _jnp.where(mask, Q, -_jnp.inf)
        
        return (masked_Q.argmax(axis=0).astype(_jnp.int32), masked_Q.max(axis=0))

    #############################################
    #  Bellman equation for softmax (Vectorized & JIT Optimized)
    #############################################
    def _bellmanOperator_softmax(self, temperature, V=None):
        if V is None:
            V_curr = self.V
        else:
            V_curr = V
        
        if self.actionMask is None:
            mask = _jnp.ones((self.A, self.S))
        else:
            mask = self.actionMask

        # 1. Compute Q-values (Vectorized)
        def compute_Q(P_a, R_a):
            return R_a + self.discount * _jnp.dot(P_a, V_curr)

        Q = jax.vmap(compute_Q)(self.P, self.R)

        # 2. Apply Mask
        masked_Q = _jnp.where(mask, Q, -_jnp.inf)

        # 3. Vectorized Softmax
        # Subtract max for numerical stability
        max_Q = _jnp.max(masked_Q, axis=0, keepdims=True)
        
        # Calculate exponentials. 
        exp_vals = _jnp.exp((masked_Q - max_Q) / temperature) * mask
        
        # Sum exponentials per state
        exp_sum = _jnp.sum(exp_vals, axis=0, keepdims=True)
        
        # Normalize
        softpolicy = exp_vals / (exp_sum + 1e-10)

        # 4. Compute Value Function
        new_V = _jnp.sum(Q * softpolicy, axis=0)

        return (softpolicy, new_V)

    def _computeTransition(self, transition):
        # Already correct shape, no loop needed
        return transition
    
    def _computeReward(self, reward, transition):
        # Vectorized reward computation
        def compute_matrix_reward(_):
            r = jax.vmap(self._computeMatrixReward)(reward, transition)
            return r

        result = compute_matrix_reward(None)
        return result

    def _computeMatrixReward(self, reward, transition):
        return _jnp.multiply(transition, reward).sum(1).reshape(self.S)

    def run(self):
        raise NotImplementedError("You should create a run() method.")

    def setSilent(self):
        self.verbose = False

    def setVerbose(self):
        self.verbose = True


class ValueIteration_sfmZW(MDP):

    def __init__(self, transitions, reward, discount, epsilon= 10 ** -6,
                 max_iter = 1000, initial_value=0, mask=None):
        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter, mask)

        def init_zero(_):
            self.V = _jnp.zeros(self.S)
            return self.V

        def init_non_zero(_):
            self.V = _jnp.array(initial_value).reshape(self.S)
            return self.V

        if isinstance(initial_value, int) or isinstance(initial_value, float):
             if initial_value == 0:
                 self.V = _jnp.zeros(self.S)
             else:
                 self.V = _jnp.array(initial_value).reshape(self.S)
        else:
             self.V = jax.lax.cond(initial_value == 0, init_zero, init_non_zero, operand=None)

        # --- FIX: Functional Purity inside lax.cond ---
        def discount_less_than_one(_):
            # Calculate new max_iter but DO NOT assign to self.max_iter here
            new_max_iter = self._boundIter(epsilon)
            
            safe_discount = _jnp.where(self.discount == 0, 1.0, self.discount) 
            t_ = epsilon * (1 - self.discount) / safe_discount
            return t_, new_max_iter

        def discount_equal_one(_):
            t_ = epsilon
            # Return current max_iter (casted to array to match type)
            return t_, _jnp.array(self.max_iter, dtype=_jnp.int32)

        # Get values out of cond
        thresh, max_iter = jax.lax.cond(self.discount < 1, discount_less_than_one, discount_equal_one, operand=None)
        
        # Assign to self outside of cond
        self.thresh = thresh
        self.max_iter = max_iter
    
    def _boundIter(self, epsilon):
        # Calculate max_iter without side effects
        k = 0
        h = _jnp.zeros(self.S)

        def inner_loop_body(aa, PP_ss):
            ss, PP = PP_ss
            PP = PP.at[aa].set(self.P[aa][:, ss])
            return ss, PP

        def outer_loop_body(ss, h):
            PP = _jnp.zeros((self.A, self.S))
            _, PP = jax.lax.fori_loop(0, self.A, inner_loop_body, (ss, PP))
            h = h.at[ss].set(PP.min())
            return h

        h = jax.lax.fori_loop(0, self.S, outer_loop_body, h)

        k = 1 - h.sum()
        Vprev = self.V
        null, value = self._bellmanOperator()
        v_diff = self.V - Vprev
        span = v_diff.max()  - v_diff.min()  + 1E-8

        # Safe logs
        log_discount_k = _jnp.log(self.discount * k + 1e-20)
        safe_discount = _jnp.where(self.discount == 0, 1e-20, self.discount)
        
        numerator = _jnp.log((epsilon * (1 - self.discount) / safe_discount) / span)
        max_iter = numerator / log_discount_k
        
        # Return result instead of setting self.max_iter
        return _jnp.ceil(max_iter).astype(_jnp.int32)

    def run(self, temperature):
        self.time = _time.time()
        
        state = {
            'iter': 0,
            'V': self.V,
            'softpolicy': _jnp.zeros_like(self.R),
            'variation': _jnp.inf,
            'continue_loop': True
        }

        def cond_fun(state):
            return state['continue_loop']

        def body_fun(state):
            state['iter'] += 1
            Vprev = state['V']
            
            state['softpolicy'], state['V'] = self._bellmanOperator_softmax(temperature, V=Vprev)

            v_diff = state['V'] - Vprev
            state['variation'] = v_diff.max()  - v_diff.min()  + 1E-8
            
            stop = (state['variation'] < self.thresh) | (state['iter'] == self.max_iter)
            state['continue_loop'] = ~stop
            
            return state

        state = jax.lax.while_loop(cond_fun, body_fun, state)

        self.iter = state['iter']
        self.V = state['V']
        self.softpolicy = state['softpolicy']

        self.time = _time.time() - self.time


class ValueIteration_opZW(MDP):
    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=1000, initial_value=0, mask=None):
        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter, mask)
        
        if isinstance(initial_value, int) or isinstance(initial_value, float):
             if initial_value == 0:
                 self.V = _jnp.zeros(self.S)
             else:
                 self.V = _jnp.array(initial_value).reshape(self.S)
        else:
             self.V = jax.lax.cond(initial_value == 0, 
                                   lambda _: _jnp.zeros(self.S), 
                                   lambda _: _jnp.array(initial_value).reshape(self.S), 
                                   operand=None)

        def discount_less_than_one(_):
            new_max_iter = self._boundIter(epsilon)
            t_ = epsilon * (1 - self.discount) / self.discount
            return t_, new_max_iter

        def discount_equal_one(_):
            t_ = epsilon
            return t_, _jnp.array(self.max_iter, dtype=_jnp.int32)

        thresh, max_iter = jax.lax.cond(self.discount < 1, discount_less_than_one, discount_equal_one, operand=None)
        self.thresh = thresh
        self.max_iter = max_iter
        
    def _boundIter(self, epsilon):
        k = 0
        h = _jnp.zeros(self.S)

        def inner_loop_body(aa, PP_ss):
            ss, PP = PP_ss
            PP = PP.at[aa].set(self.P[aa][:, ss])
            return ss, PP

        def outer_loop_body(ss, h):
            PP = _jnp.zeros((self.A, self.S))
            _, PP = jax.lax.fori_loop(0, self.A, inner_loop_body, (ss, PP))
            h = h.at[ss].set(PP.min())
            return h

        h = jax.lax.fori_loop(0, self.S, outer_loop_body, h)

        k = 1 - h.sum()
        Vprev = self.V
        null, value = self._bellmanOperator()
        v_diff = self.V - Vprev
        span = v_diff.max()  - v_diff.min()  + 1E-8

        log_k = _jnp.log(self.discount * k + 1e-20)
        safe_discount = _jnp.where(self.discount == 0, 1e-20, self.discount)
        
        max_iter = (_jnp.log((epsilon * (1 - self.discount) / safe_discount) / span ) / log_k)

        return _jnp.ceil(max_iter).astype('int32')

    def run(self):
        self.time = _time.time()

        state = {
            'iter': 0,
            'V': self.V,
            'policy': _jnp.zeros(self.S, dtype=_jnp.int32),
            'variation': _jnp.inf,
            'continue_loop': True
        }

        def cond_fun(state):
            return state['continue_loop']

        def body_fun(state):
            state['iter'] += 1
            Vprev = state['V']
            state['policy'], state['V'] = self._bellmanOperator()
            v_diff = state['V'] - Vprev
            state['variation'] = v_diff.max()  - v_diff.min()  + 1E-8
            
            stop = (state['variation'] < self.thresh) | (state['iter'] == self.max_iter)
            state['continue_loop'] = ~stop
            
            return state

        state = jax.lax.while_loop(cond_fun, body_fun, state)

        self.iter = state['iter']
        self.V = state['V']
        self.policy = state['policy']

        self.time = _time.time() - self.time