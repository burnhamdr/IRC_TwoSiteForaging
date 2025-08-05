
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

_MSG_STOP_MAX_ITER = "Iterating stopped due to maximum number of iterations " \
    "condition."
_MSG_STOP_EPSILON_OPTIMAL_POLICY = "Iterating stopped, epsilon-optimal " \
    "policy found."
_MSG_STOP_EPSILON_OPTIMAL_VALUE = "Iterating stopped, epsilon-optimal value " \
    "function found."
_MSG_STOP_UNCHANGING_POLICY = "Iterating stopped, unchanging policy found."


def _computeDimensions(transition):
    A = len(transition)

    def is_ndim_3(_):
        return transition.shape[1]

    def is_not_ndim_3(_):
        return transition[0].shape[0]

    def check_ndim(_):
        return jax.lax.cond(transition.ndim == 3, is_ndim_3, is_not_ndim_3, operand=None)

    def handle_attribute_error(_):
        return transition[0].shape[0]

    # Use host_callback.id_tap to handle the AttributeError
    # S = debug.callback(
    #     lambda: check_ndim(None) if hasattr(transition, 'ndim') else handle_attribute_error(None)
    # )
    S = jax.lax.cond(hasattr(transition, 'ndim'), check_ndim, handle_attribute_error, operand=None)

    return S, A


class MDP(object):

    def __init__(self, transitions, reward, discount, epsilon, max_iter, mask):
        # Initialise a MDP based on the input parameters.

        # Define the function to handle the discount initialization
        def init_discount(_):
            self.discount = float(discount)
            def check_discount():
                checkify.check(0.0 < self.discount <= 1.0, "Discount rate must be in ]0; 1]")
            def warn():
                if self.discount == 1:
                    print("WARNING: check conditions of convergence. With no discount, convergence cannot be assumed.")
            checked_func = checkify.checkify(check_discount)
            checked_func()
            debug.callback(warn)
            return self.discount

        def no_discount(_):
            return self.discount

        # Use lax.cond to handle the discount initialization
        self.discount = jax.lax.cond(discount is not None, init_discount, no_discount, operand=None)

        # Define the function to handle the max_iter initialization
        def init_max_iter(_):
            self.max_iter = int(max_iter)
            def check_max_iter():
                checkify.check(self.max_iter > 0, "The maximum number of iterations must be greater than 0.")
            checked_func = checkify.checkify(check_max_iter)
            checked_func()
            return self.max_iter

        def no_max_iter(_):
            return self.max_iter

        # Use lax.cond to handle the max_iter initialization
        self.max_iter = jax.lax.cond(max_iter is not None, init_max_iter, no_max_iter, operand=None)

        # Define the function to handle the epsilon initialization
        def init_epsilon(_):
            self.epsilon = float(epsilon)
            def check_epsilon():
                checkify.check(self.epsilon > 0, "Epsilon must be greater than 0.")
            checked_func = checkify.checkify(check_epsilon)
            checked_func()
            return self.epsilon

        def no_epsilon(_):
            return self.epsilon

        # Use lax.cond to handle the epsilon initialization
        self.epsilon = jax.lax.cond(epsilon is not None, init_epsilon, no_epsilon, operand=None)
        
        # we run a check on P and R to make sure they are describing an MDP. If
        # an exception isn't raised then they are assumed to be correct.
        transitions_ = _jnp.array(transitions)#.astype(_np.float64)
        reward_ = _jnp.array(reward)
        check_(transitions_, reward_)
        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)
        self.R = self._computeReward(reward, transitions)
        

        # the verbosity is by default turned off
        self.verbose = False
        # Initially the time taken to perform the computations is set to None
        self.time = None
        # set the initial iteration count to zero
        self.iter = 0
        # V should be stored as a vector ie shape of (S,) or (1, S)
        self.V = None
        # policy can also be stored as a vector
        self.policy = None
        # allow action masking
        self.actionMask = mask

    def __repr__(self):
        P_repr = "P: \n"
        R_repr = "R: \n"
        for aa in range(self.A):
            P_repr += repr(self.P[aa]) + "\n"
            R_repr += repr(self.R[aa]) + "\n"
        return(P_repr + "\n" + R_repr)

    def _bellmanOperator(self, V=None):
        # Apply the Bellman operator on the value function.
        #
        # Updates the value function and the Vprev-improving policy.
        #
        # Returns: (policy, value), tuple of new policy and its value
        #
        # If V hasn't been sent into the method, then we assume to be working
        # on the objects V attribute
        # def VisNone(_):
        #     # this V should be a reference to the data rather than a copy
        #     V = self.V
        # def VnotNone(_):
        #     # make sure the user supplied V is of the right shape
        #     def check_shape(_):
        #         def _check_shape():
        #             checkify.check(V.shape in ((self.S,), (1, self.S)), "V is not the right shape (Bellman operator).")
        #         checked_func = checkify.checkify(_check_shape)
        #         checked_func()
        #         return V

        #     def handle_shape_error(_):
        #         debug.callback(lambda: print("V must be a numpy array or matrix."))
        #         return None

        #     V = jax.lax.cond(hasattr(V, 'shape'), check_shape, handle_shape_error, operand=None)
        
        # V = jax.lax.cond(V is None, VisNone, VnotNone, operand=None)
        
        # def get_mask(_):
        #     return _jnp.ones((self.A, self.S))

        # def check_mask_shape(_):
        #     def _check_mask_shape():
        #         checkify.check(mask.shape == (self.A, self.S), "Mask is not the right shape.")
        #     checked_func = checkify.checkify(_check_mask_shape)
        #     checked_func()
        #     return mask

        # mask = jax.lax.cond(self.actionMask is None, get_mask, lambda _: jax.lax.cond(mask.shape == (self.A, self.S), check_mask_shape, lambda _: mask, operand=None), operand=None)
        mask = self.actionMask
        # Looping through each action the the Q-value matrix is calculated.
        # P and V can be any object that supports indexing, so it is important
        # that you know they define a valid MDP before calling the
        # _bellmanOperator method. Otherwise the results will be meaningless.
        # Define the function to compute Q-values for a single action
        def compute_Q(P_a, R_a):
            return R_a + self.discount * _jnp.dot(P_a, self.V)

        # Use vmap to vectorize the computation over actions
        Q = jax.vmap(compute_Q)(self.P, self.R)

        #apply mask
        masked_Q = _jnp.where(mask, Q, -_jnp.inf)  # Replace unavailable actions with -inf
        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)
        return (Q.argmax(axis=0).astype(_jnp.int32), Q.max(axis=0))
        # 2. update self.policy and self.V directly
        # self.V = Q.max(axis=1)
        # self.policy = Q.argmax(axis=1)



    #############################################
    #
    #  Bellman equation for softmax (modified by ZW)
    #
    #############################################

    def _bellmanOperator_softmax(self, temperature, V=None):
        # on the objects V attribute
        def VisNone(_):
            # this V should be a reference to the data rather than a copy
            V = self.V
        def VnotNone(_):
            # make sure the user supplied V is of the right shape
            def check_shape(_):
                def _check_shape():
                    checkify.check(V.shape in ((self.S,), (1, self.S)), "V is not the right shape (Bellman operator).")
                checked_func = checkify.checkify(_check_shape)
                checked_func()
                return V

            def handle_shape_error(_):
                debug.callback(lambda: print("V must be a numpy array or matrix."))
                return None

            V = jax.lax.cond(hasattr(V, 'shape'), check_shape, handle_shape_error, operand=None)
        
        V = jax.lax.cond(V is None, VisNone, VnotNone, operand=None)
        
        def get_mask(_):
            return _jnp.ones((self.A, self.S))

        def check_mask_shape(_):
            def _check_mask_shape():
                checkify.check(mask.shape == (self.A, self.S), "Mask is not the right shape.")
            checked_func = checkify.checkify(_check_mask_shape)
            checked_func()
            return mask

        mask = jax.lax.cond(self.actionMask is None, get_mask, lambda _: jax.lax.cond(mask.shape == (self.A, self.S), check_mask_shape, lambda _: mask, operand=None), operand=None)

        # Looping through each action the the Q-value matrix is calculated.
        # P and V can be any object that supports indexing, so it is important
        # that you know they define a valid MDP before calling the
        # _bellmanOperator method. Otherwise the results will be meaningless.
        # Define the function to compute Q-values for a single action
        def compute_Q(P_a, R_a):
            return R_a + self.discount * _jnp.dot(P_a, V)

        # Use vmap to vectorize the computation over actions
        Q = jax.vmap(compute_Q)(self.P, self.R)

        #apply mask
        masked_Q = _jnp.where(mask, Q, -_np.inf)
        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)

        softpolicy = _jnp.zeros(Q.shape)

        # Define the loop body for computing the soft policy
        def loop_body(i, softpolicy):
            # Apply the mask by only considering valid actions (non -inf)
            valid_Q = masked_Q[:, i]

            # Subtract the maximum to stabilize the softmax calculation (avoid overflow)
            max_Q = _jnp.max(valid_Q[valid_Q > -_jnp.inf])  # Find the max among available actions
            exp_vals = _jnp.exp((valid_Q - max_Q) / temperature) * mask[:, i]  # Apply mask to exponentials

            # Normalize only over the available actions (ignore masked ones)
            exp_sum = _jnp.sum(exp_vals)
            softpolicy = jax.lax.cond(
                exp_sum > 0,
                lambda _: softpolicy.at[:, i].set(exp_vals / exp_sum),  # Apply mask to policy
                lambda _: softpolicy,
                operand=None
            )
            return softpolicy

        # Run the loop using lax.fori_loop
        softpolicy = jax.lax.fori_loop(0, self.S, loop_body, softpolicy)

        return (softpolicy, _jnp.sum(Q * softpolicy, axis = 0))
        # 2. update self.policy and self.V directly
        # self.V = Q.max(axis=1)
        # self.policy = Q.argmax(axis=1)
    #############################################
    #
    # Bellman equation for softmax (modified by ZW)
    #
    #############################################

    def _computeTransition(self, transition):
        # Define the loop body
        def loop_body(a, transition_tup):
            transition_tup = transition_tup.at[a].set(transition[a])
            return transition_tup

        # Initialize an empty array to store the transition matrices
        transition_array = _jnp.empty((self.A, self.S, self.S))

        # Run the loop using lax.fori_loop
        transition_array = jax.lax.fori_loop(0, self.A, loop_body, transition_array)

        # Convert the array to a tuple
        # transition_tuple = tuple(transition_array)

        return transition_array#transition_tuple
    
    def _computeReward(self, reward, transition):
        # Compute the reward for the system in one state choosing an action.
        # Arguments
        # Let S = number of states, A = number of actions
        # P could be an array with 3 dimensions or a cell array (1xA),
        # each cell containing a matrix (SxS) possibly sparse
        # R could be an array with 3 dimensions (SxSxA) or a cell array
        # (1xA), each cell containing a sparse matrix (SxS) or a 2D
        # array(SxA) possibly sparse

        # def compute_vector_reward(_):
        #     return self._computeVectorReward(reward)

        # def compute_array_reward(_):
        #     return self._computeArrayReward(reward)

        def compute_matrix_reward(_):
            r = jax.lax.map(lambda x: self._computeMatrixReward(x[0], x[1]), (reward, transition))
            return r

        # def handle_error(_):
        #     def check_len(_):
        #         r = tuple(map(self._computeMatrixReward, reward, transition))
        #         return r

        #     def handle_len_error(_):
        #         return self._computeVectorReward(reward)

        #     return jax.lax.cond(len(reward) == self.A, check_len, handle_len_error, operand=None)

        # def check_ndim(_):
        #     return jax.lax.cond(reward.ndim == 1, compute_vector_reward,
        #                         lambda _: jax.lax.cond(reward.ndim == 2, compute_array_reward, compute_matrix_reward, operand=None),
        #                         operand=None)

        # Use jax.debug.callback to handle the AttributeError and ValueError
        # result = debug.callback(
        #     lambda: check_ndim(None) if hasattr(reward, 'ndim') else handle_error(None)
        # )
        # result = jax.lax.cond(hasattr(reward, 'ndim'), check_ndim, handle_error, operand=None)
        result = compute_matrix_reward(None)
        return result

    # def _computeVectorReward(self, reward):
    #     r = _jnp.array(reward).reshape(self.S)

    #     # Initialize an empty array to store the rewards
    #     rewards = _jnp.empty((self.A, self.S))

    #     # Define the loop body
    #     def loop_body(a, rewards):
    #         rewards = rewards.at[a].set(r)
    #         return rewards

    #     # Run the loop using lax.fori_loop
    #     rewards = jax.lax.fori_loop(0, self.A, loop_body, rewards)

    #     # Convert the array to a tuple
    #     reward_tuple = tuple(rewards)

    #     return reward_tuple

    # def _computeArrayReward(self, reward):
    #     func = lambda x: _jnp.array(x).reshape(self.S)

    #     # Initialize an empty array to store the rewards
    #     rewards = _jnp.empty((self.A, self.S))

    #     # Define the loop body
    #     def loop_body(a, rewards):
    #         rewards = rewards.at[a].set(func(reward[:, a]))
    #         return rewards

    #     # Run the loop using lax.fori_loop
    #     rewards = jax.lax.fori_loop(0, self.A, loop_body, rewards)

    #     # Convert the array to a tuple
    #     reward_tuple = tuple(rewards)

    #     return reward_tuple

    def _computeMatrixReward(self, reward, transition):
        return _jnp.multiply(transition, reward).sum(1).reshape(self.S)

    def run(self):
        # Raise error because child classes should implement this function.
        raise NotImplementedError("You should create a run() method.")

    def setSilent(self):
        """Set the MDP algorithm to silent mode."""
        self.verbose = False

    def setVerbose(self):
        """Set the MDP algorithm to verbose mode."""
        self.verbose = True
 
'''
Value iteration with softmax
modified by ZW
'''

class ValueIteration_sfmZW(MDP):

    """A discounted MDP solved using the value iteration algorithm.

    Description
    -----------
    ValueIteration applies the value iteration algorithm to solve a
    discounted MDP. The algorithm consists of solving Bellman's equation
    iteratively.
    Iteration is stopped when an epsilon-optimal policy is found or after a
    specified number (``max_iter``) of iterations.
    This function uses verbose and silent modes. In verbose mode, the function
    displays the variation of ``V`` (the value function) for each iteration and
    the condition which stopped the iteration: epsilon-policy found or maximum
    number of iterations reached.

    Parameters
    ----------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    discount : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    epsilon : float, optional
        Stopping criterion. See the documentation for the ``MDP`` class for
        details.  Default: 0.01.
    max_iter : int, optional
        Maximum number of iterations. If the value given is greater than a
        computed bound, a warning informs that the computed bound will be used
        instead. By default, if ``discount`` is not equal to 1, a bound for
        ``max_iter`` is computed, otherwise ``max_iter`` = 1000. See the
        documentation for the ``MDP`` class for further details.
    initial_value : array, optional
        The starting value function. Default: a vector of zeros.

    Data Attributes
    ---------------
    V : tuple
        The optimal value function.
    policy : tuple
        The optimal policy function. Each element is an integer corresponding
        to an action which maximises the value function in that state.
    iter : int
        The number of iterations taken to complete the computation.
    time : float
        The amount of CPU time used to run the algorithm.

    Methods
    -------
    run()
        Do the algorithm iteration.
    setSilent()
        Sets the instance to silent mode.
    setVerbose()
        Sets the instance to verbose mode.

    Notes
    -----
    In verbose mode, at each iteration, displays the variation of V
    and the condition which stopped iterations: epsilon-optimum policy found
    or maximum number of iterations reached.

    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.96)
    >>> vi.verbose
    False
    >>> vi.run()
    >>> expected = (5.93215488, 9.38815488, 13.38815488)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (0, 0, 0)
    >>> vi.iter
    4

    >>> import mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.run()
    >>> expected = (40.048625392716815, 33.65371175967546)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (1, 0)
    >>> vi.iter
    26

    >>> import mdptoolbox
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix as sparse
    >>> P = [None] * 2
    >>> P[0] = sparse([[0.5, 0.5],[0.8, 0.2]])
    >>> P[1] = sparse([[0, 1],[0.1, 0.9]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.run()
    >>> expected = (40.048625392716815, 33.65371175967546)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (1, 0)

    """

    def __init__(self, transitions, reward, discount, epsilon= 10 ** -6,
                 max_iter = 1000, initial_value=0, mask=None):
        # Initialise a value iteration MDP.

        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter, mask)

        # initialization of optional arguments

        # Define the function to handle the initialization of V when initial_value is 0
        def init_zero(_):
            self.V = _jnp.zeros(self.S)
            return self.V

        # Define the function to handle the initialization of V when initial_value is not 0
        def init_non_zero(_):
            def check_shape():
                checkify.check(len(initial_value) == self.S, "The initial value must be a vector of length S.")
            checked_func = checkify.checkify(check_shape)
            checked_func()
            self.V = _jnp.array(initial_value).reshape(self.S)
            return self.V

        # Use lax.cond to handle the initialization of V
        self.V = jax.lax.cond(initial_value == 0, init_zero, init_non_zero, operand=None)

        # Define the function to handle the discount < 1 case
        def discount_less_than_one(_):
            self._boundIter(epsilon)
            self.thresh = epsilon * (1 - self.discount) / self.discount
            return self.thresh

        # Define the function to handle the discount == 1 case
        def discount_equal_one(_):
            self.thresh = epsilon
            return self.thresh

        # Use lax.cond to handle the discount condition
        self.thresh = jax.lax.cond(self.discount < 1, discount_less_than_one, discount_equal_one, operand=None)
    
    def _boundIter(self, epsilon):
        # Compute a bound for the number of iterations.
        #
        # for the value iteration
        # algorithm to find an epsilon-optimal policy with use of span for the
        # stopping criterion
        #
        # Arguments -----------------------------------------------------------
        # Let S = number of states, A = number of actions
        #    epsilon   = |V - V*| < epsilon,  upper than 0,
        #        optional (default : 0.01)
        # Evaluation ----------------------------------------------------------
        #    max_iter  = bound of the number of iterations for the value
        #    iteration algorithm to find an epsilon-optimal policy with use of
        #    span for the stopping criterion
        #    cpu_time  = used CPU time
        #
        # See Markov Decision Processes, M. L. Puterman,
        # Wiley-Interscience Publication, 1994
        # p 202, Theorem 6.6.6
        # k =    max     [1 - S min[ P(j|s,a), p(j|s',a')] ]
        #     s,a,s',a'       j
        k = 0
        h = _jnp.zeros(self.S)

        # Define the inner loop body for constructing PP
        def inner_loop_body(aa, PP_ss):
            ss, PP = PP_ss
            # try:
            #     PP = PP.at[aa].set(self.P[aa][:, ss])
            # except ValueError:
            #     PP = PP.at[aa].set(jax.experimental.sparse.todense(self.P[aa][:, ss].todense()))
            PP = PP.at[aa].set(self.P[aa][:, ss])
            return ss, PP

        # Define the outer loop body for constructing h
        def outer_loop_body(ss, h):
            PP = _jnp.zeros((self.A, self.S))
            _, PP = jax.lax.fori_loop(0, self.A, inner_loop_body, (ss, PP))
            # Minimum of the entire array
            h = h.at[ss].set(PP.min())
            return h

        # Run the outer loop using lax.fori_loop
        h = jax.lax.fori_loop(0, self.S, outer_loop_body, h)

        k = 1 - h.sum()
        Vprev = self.V
        null, value = self._bellmanOperator()
        # p 201, Proposition 6.6.5
        v_diff = self.V - Vprev
        span = v_diff.max()  - v_diff.min()  + 1E-8

        max_iter = (_math.log((epsilon * (1 - self.discount) / self.discount) /
                    span ) / _math.log(self.discount * k))
        #self.V = Vprev

        self.max_iter = int(_math.ceil(max_iter))

    def run(self, temperature):
        # Run the value iteration algorithm.

        # if self.verbose:
        #     print('  Iteration\t\tV-variation')

        self.time = _time.time()

        # Initialize the state
        state = {
            'iter': self.iter,
            'V': self.V,
            'softpolicy': _jnp.zeros_like(self.R),  # Initialize softpolicy with zeros
            'variation': _jnp.inf,
            'continue_loop': True
        }

        # Define the loop condition
        def cond_fun(state):
            return state['continue_loop']

        # Define the loop body
        def body_fun(state):
            state['iter'] += 1
            Vprev = state['V']

            # Bellman Operator: compute policy and value functions
            state['softpolicy'], state['V'] = self._bellmanOperator_softmax(temperature)

            # Compute the variation
            v_diff = self.V - Vprev
            state['variation'] = v_diff.max()  - v_diff.min()  + 1E-8
            
            # Check stopping conditions
            def stop_condition_met(_):
                # if self.verbose:
                #     print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                state['continue_loop'] = False
                return state

            def max_iter_reached(_):
                # if self.verbose:
                #     print(_MSG_STOP_MAX_ITER)
                state['continue_loop'] = False
                return state

            def continue_loop(_):
                return state

            state = jax.lax.cond(state['variation'] < self.thresh, stop_condition_met, continue_loop, operand=None)
            state = jax.lax.cond(state['iter'] == self.max_iter, max_iter_reached, continue_loop, operand=None)

            # if self.verbose:
            #     print(("    %s\t\t  %s" % (state['iter'], state['variation'])))

            return state

        # Run the loop using lax.while_loop
        state = jax.lax.while_loop(cond_fun, body_fun, state)

        # Update the class attributes
        self.iter = state['iter']
        self.V = tuple(state['V'].tolist())
        self.softpolicy = tuple(state['softpolicy'].tolist())

        self.time = _time.time() - self.time


class ValueIteration_opZW(MDP):

    """A discounted MDP solved using the value iteration algorithm.

    Description
    -----------
    ValueIteration applies the value iteration algorithm to solve a
    discounted MDP. The algorithm consists of solving Bellman's equation
    iteratively.
    Iteration is stopped when an epsilon-optimal policy is found or after a
    specified number (``max_iter``) of iterations.
    This function uses verbose and silent modes. In verbose mode, the function
    displays the variation of ``V`` (the value function) for each iteration and
    the condition which stopped the iteration: epsilon-policy found or maximum
    number of iterations reached.

    Parameters
    ----------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    discount : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    epsilon : float, optional
        Stopping criterion. See the documentation for the ``MDP`` class for
        details.  Default: 0.01.
    max_iter : int, optional
        Maximum number of iterations. If the value given is greater than a
        computed bound, a warning informs that the computed bound will be used
        instead. By default, if ``discount`` is not equal to 1, a bound for
        ``max_iter`` is computed, otherwise ``max_iter`` = 1000. See the
        documentation for the ``MDP`` class for further details.
    initial_value : array, optional
        The starting value function. Default: a vector of zeros.

    Data Attributes
    ---------------
    V : tuple
        The optimal value function.
    policy : tuple
        The optimal policy function. Each element is an integer corresponding
        to an action which maximises the value function in that state.
    iter : int
        The number of iterations taken to complete the computation.
    time : float
        The amount of CPU time used to run the algorithm.

    Methods
    -------
    run()
        Do the algorithm iteration.
    setSilent()
        Sets the instance to silent mode.
    setVerbose()
        Sets the instance to verbose mode.

    Notes
    -----
    In verbose mode, at each iteration, displays the variation of V
    and the condition which stopped iterations: epsilon-optimum policy found
    or maximum number of iterations reached.

    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.96)
    >>> vi.verbose
    False
    >>> vi.run()
    >>> expected = (5.93215488, 9.38815488, 13.38815488)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (0, 0, 0)
    >>> vi.iter
    4

    >>> import mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.run()
    >>> expected = (40.048625392716815, 33.65371175967546)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (1, 0)
    >>> vi.iter
    26

    >>> import mdptoolbox
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix as sparse
    >>> P = [None] * 2
    >>> P[0] = sparse([[0.5, 0.5],[0.8, 0.2]])
    >>> P[1] = sparse([[0, 1],[0.1, 0.9]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.run()
    >>> expected = (40.048625392716815, 33.65371175967546)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (1, 0)

    """

    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=1000, initial_value=0, mask=None):
        # Initialise a value iteration MDP.

        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter, mask)

        # # initialization of optional arguments
        # # Define the function to handle the initialization of V when initial_value is 0
        # def init_zero(_):
        #     v_ = _jnp.zeros(self.S)
        #     return v_

        # # Define the function to handle the initialization of V when initial_value is not 0
        # def init_non_zero(_):
        #     def check_shape():
        #         checkify.check(len(initial_value) == self.S, "The initial value must be a vector of length S.")
            
        #     checked_func = checkify.checkify(check_shape)
        #     checked_func()
            
        #     v_ = _jnp.array(initial_value).reshape(self.S)
            
        #     return v_

        # # Use lax.cond to handle the initialization of V
        # self.V = jax.lax.cond(initial_value == 0, init_zero, init_non_zero, operand=None)
        # initialization of optional arguments
        if initial_value == 0:
            self.V = _jnp.zeros(self.S)

        else:
            assert len(initial_value) == self.S, "The initial value must be " \
                "a vector of length S."
            self.V = _jnp.array(initial_value).reshape(self.S)

        # Define the function to handle the discount < 1 case
        def discount_less_than_one(_):
            new_max_iter = self._boundIter(epsilon)
            t_ = epsilon * (1 - self.discount) / self.discount
            return t_, new_max_iter

        # Define the function to handle the discount == 1 case
        def discount_equal_one(_):
            t_ = epsilon
            return t_, self.max_iter.astype('int32')

        # Use lax.cond to handle the discount condition
        thresh, max_iter = jax.lax.cond(self.discount < 1, discount_less_than_one, discount_equal_one, operand=None)
        self.thresh = thresh
        self.max_iter = max_iter
        
    def _boundIter(self, epsilon):
        # Compute a bound for the number of iterations.
        #
        # for the value iteration
        # algorithm to find an epsilon-optimal policy with use of span for the
        # stopping criterion
        #
        # Arguments -----------------------------------------------------------
        # Let S = number of states, A = number of actions
        #    epsilon   = |V - V*| < epsilon,  upper than 0,
        #        optional (default : 0.01)
        # Evaluation ----------------------------------------------------------
        #    max_iter  = bound of the number of iterations for the value
        #    iteration algorithm to find an epsilon-optimal policy with use of
        #    span for the stopping criterion
        #    cpu_time  = used CPU time
        #
        # See Markov Decision Processes, M. L. Puterman,
        # Wiley-Interscience Publication, 1994
        # p 202, Theorem 6.6.6
        # k =    max     [1 - S min[ P(j|s,a), p(j|s',a')] ]
        #     s,a,s',a'       j
        k = 0
        h = _jnp.zeros(self.S)

        # Define the inner loop body for constructing PP
        def inner_loop_body(aa, PP_ss):
            ss, PP = PP_ss
            # try:
            #     PP = PP.at[aa].set(self.P[aa][:, ss])
            # except ValueError:
            #     PP = PP.at[aa].set(jax.experimental.sparse.todense(self.P[aa][:, ss].todense()))
            PP = PP.at[aa].set(self.P[aa][:, ss])
            return ss, PP

        # Define the outer loop body for constructing h
        def outer_loop_body(ss, h):
            PP = _jnp.zeros((self.A, self.S))
            _, PP = jax.lax.fori_loop(0, self.A, inner_loop_body, (ss, PP))
            # Minimum of the entire array
            h = h.at[ss].set(PP.min())
            return h

        # Run the outer loop using lax.fori_loop
        h = jax.lax.fori_loop(0, self.S, outer_loop_body, h)

        k = 1 - h.sum()
        Vprev = self.V
        null, value = self._bellmanOperator()
        # p 201, Proposition 6.6.5
        v_diff = self.V - Vprev
        span = v_diff.max()  - v_diff.min()  + 1E-8

        max_iter = (_jnp.log((epsilon * (1 - self.discount) / self.discount) /
                    span ) / _jnp.log(self.discount * k))
        #self.V = Vprev

        return _jnp.ceil(max_iter).astype('int32')

    def run(self):
        # Run the value iteration algorithm.

        # if self.verbose:
        #     print('  Iteration\t\tV-variation')

        self.time = _time.time()

        # Initialize the state
        state = {
            'iter': self.iter,
            'V': self.V,
            'policy': _jnp.zeros(self.S, dtype=_jnp.int32),  # Initialize policy with zeros
            'variation': _jnp.inf,
            'continue_loop': True
        }

        # Define the loop condition
        def cond_fun(state):
            return state['continue_loop']

        # Define the loop body
        def body_fun(state):
            state['iter'] += 1
            Vprev = state['V']

            # Bellman Operator: compute policy and value functions
            state['policy'], state['V'] = self._bellmanOperator()

            # Compute the variation
            v_diff = self.V - Vprev
            state['variation'] = v_diff.max()  - v_diff.min()  + 1E-8
            
            # Check stopping conditions
            def stop_condition_met(_):
                # if self.verbose:
                #     print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                state['continue_loop'] = False
                return state

            def max_iter_reached(_):
                # if self.verbose:
                #     print(_MSG_STOP_MAX_ITER)
                state['continue_loop'] = False
                return state

            def continue_loop(_):
                return state

            state = jax.lax.cond(state['variation'] < self.thresh, stop_condition_met, continue_loop, operand=None)
            state = jax.lax.cond(state['iter'] == self.max_iter, max_iter_reached, continue_loop, operand=None)

            # if self.verbose:
            #     print(("    %s\t\t  %s" % (state['iter'], state['variation'])))

            return state

        # Run the loop using lax.while_loop
        state = jax.lax.while_loop(cond_fun, body_fun, state)

        # Update the class attributes
        self.iter = state['iter']
        self.V = tuple(state['V'].tolist())
        self.policy = tuple(state['policy'].tolist())

        self.time = _time.time() - self.time