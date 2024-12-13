from __future__ import division
import numpy as np
from jax.scipy.stats import norm, binom
from math import sqrt
from scipy import optimize
import jax.numpy as jnp
import jax
from jax import vmap, pmap
from pprint import pprint
from jax.lib import xla_bridge
import jaxlib
from jax import jit
import jax.lax as lax

def compare_nested_dicts(dict1, dict2, rel_tol=1e-7, abs_tol=1e-8):
    """
    Compare two dictionaries with numpy arrays as values and identical key structures.
    
    Parameters:
    dict1, dict2 : dict
        Dictionaries with the same key structure. Values are numpy arrays of the same shape.
    rel_tol : float
        Relative difference tolerance.
    abs_tol : float
        Absolute difference tolerance.
    
    Returns:
    None
    """
    def recursive_compare(d1, d2, key_path=[]):
        if isinstance(d1, dict):
            for key in d1:
                recursive_compare(d1[key], d2[key], key_path + [key])
        else:  # Base case: d1 and d2 are numpy arrays
            mismatches = np.abs(d1 - d2) > (abs_tol + rel_tol * np.abs(d2))
            if np.any(mismatches):
                print(f"\nArrays are not equal")
                print(" -> ".join(f"key={k}" for k in key_path))
                mismatched_elements = np.sum(mismatches)
                print(f"Mismatched elements: {mismatched_elements} / {d1.size} ({mismatched_elements / d1.size:.2%})")
                max_abs_diff = np.max(np.abs(d1 - d2))
                max_rel_diff = np.max(np.abs(d1 - d2) / (np.abs(d2) + abs_tol))
                print(f"Max absolute difference: {max_abs_diff:e}")
                print(f"Max relative difference: {max_rel_diff:e}")
                print(f" x: {d1}")
                print(f" y: {d2}")

    recursive_compare(dict1, dict2)

def tensorsum_str(A, B):
    ra, ca = A.shape
    rb, cb = B.shape
    C = jnp.empty((ra * rb, ca * cb), dtype=object)  # Use object dtype for strings

    for i in range(ra):
        for j in range(ca):
            for k in range(rb):
                for l in range(cb):
                    # Concatenate with '+' for addition
                    C = C.at[i * rb + k, j * cb + l].set(f"{A[i, j]} + {B[k, l]}")
    
    return C

def tensorsumm_str(*args):
    '''
    :param args: matrices with string entries
    :return: returns multidimensional kronecker sum of all matrices in list (strings with "+" as separator)
    '''
    z = args[0]
    for i in range(1, len(args)):
        z = tensorsum_str(z, args[i])

    return z

def kronn_str(*args):
    """
    Returns the multidimensional Kronecker product of all matrices in the argument list,
    for string-type matrices.
    """
    # Start with the first matrix
    z = args[0]
    
    # Iteratively apply the custom Kronecker product function for strings
    for i in range(1, len(args)):
        z = kronn_str_pair(z, args[i])  # Use the pairwise kronn_str for each step
    
    return z

def kronn_str_pair(A, B):
    """
    Performs the Kronecker product for two matrices A and B with string entries.
    """
    ra, ca = A.shape
    rb, cb = B.shape
    C = jnp.empty((ra * rb, ca * cb), dtype=object)  # Use object dtype to store strings

    for i in range(ra):
        for j in range(ca):
            for k in range(rb):
                for l in range(cb):
                    # Concatenate the strings from A and B with '*' in between
                    C = C.at[i * rb + k, j * cb + l].set(f"{A[i, j]} * {B[k, l]}")

    return C

def string_meshgrid(*arrays):
    # Create meshgrid using jax.numpy
    mesh = jnp.meshgrid(*arrays, indexing='ij')
    
    # Convert the meshgrid arrays to strings
    mesh_str = [jnp.array(arr, dtype=str) for arr in mesh]
    
    # Combine the meshgrid arrays element-wise using string concatenation with a space in between
    concatenated = jnp.char.add(mesh_str[0], ' ')  # Add space after the first string
    concatenated = jnp.char.add(concatenated, mesh_str[1])  # Add the second string
    
    return concatenated

def kronn(*args):
    """
    returns multidimensional kronecker product of all matrices in the argument list
    """
    z = args[0]
    for i in range(1, len(args)):
        z = jnp.kron(z, args[i])
    return z

# @jit
# def fminbound(func, bounds, xatol=1e-5, maxiter=500):
#     # Unpack bounds
#     a, b = bounds
#     golden_mean = 0.5 * (3.0 - jnp.sqrt(5.0))  # Golden ratio
#     sqrt_eps = jnp.sqrt(2.2e-16)  # Machine precision

#     # Initial setup for golden section search
#     x1 = a + golden_mean * (b - a)
#     x2 = b - golden_mean * (b - a)
#     f1 = func(x1)
#     f2 = func(x2)
    
#     # Iterate to narrow down the interval
#     for _ in range(maxiter):
#         # Check stopping criteria based on xatol (absolute tolerance)
#         if jnp.abs(b - a) < xatol:
#             break
        
#         # Choose which side to move based on function values
#         if f1 < f2:
#             # Update the interval to [a, x2]
#             b, x2, f2 = x2, x1, f1
#             x1 = a + golden_mean * (b - a)
#             f1 = func(x1)
#         else:
#             # Update the interval to [x1, b]
#             a, x1, f1 = x1, x2, f2
#             x2 = b - golden_mean * (b - a)
#             f2 = func(x2)
    
#     # Choose the final point with the lowest function value as the optimal x
#     xopt = x1 if f1 < f2 else x2
#     fopt = func(xopt)
    
#     return xopt, fopt
def fminbound(func, bounds, xatol=1e-5, maxiter=500):
    # Unpack bounds
    a, b = bounds
    a = jnp.float32(a)
    b = jnp.float32(b)
    golden_mean = 0.5 * (3.0 - jnp.sqrt(5.0))  # Golden ratio
    sqrt_eps = jnp.sqrt(2.2e-16)  # Machine precision

    # Initial setup for golden section search
    x1 = a + golden_mean * (b - a)
    x2 = b - golden_mean * (b - a)
    f1 = func(x1)
    f2 = func(x2)

    def cond_fun(state):
        a, b, x1, x2, f1, f2, i = state
        return (jnp.abs(b - a) >= xatol) & (i < maxiter)

    def body_fun(state):
        a, b, x1, x2, f1, f2, i = state

        def true_fun(_):
            b_new, x2_new, f2_new = x2, x1, f1
            x1_new = a + golden_mean * (b_new - a)
            f1_new = func(x1_new)
            return a, b_new, x1_new, x2_new, f1_new, f2_new, i + 1

        def false_fun(_):
            a_new, x1_new, f1_new = x1, x2, f2
            x2_new = b - golden_mean * (b - a_new)
            f2_new = func(x2_new)
            return a_new, b, x1_new, x2_new, f1_new, f2_new, i + 1

        return lax.cond(f1 < f2, true_fun, false_fun, operand=None)

    # Initial state
    state = (a, b, x1, x2, f1, f2, 0)

    # Run the while loop
    a, b, x1, x2, f1, f2, _ = lax.while_loop(cond_fun, body_fun, state)

    # Choose the final point with the lowest function value as the optimal x
    xopt = lax.cond(f1 < f2, lambda _: x1, lambda _: x2, operand=None)
    fopt = func(xopt)

    return xopt, fopt

# @jit
def beliefTransitionMatrixGaussianCazettes(p_sw, p_rwd, nq, actions, locations, sigma):
    """
    Create transition matrix between nq belief states q to q' WITH action-dependent observations
    AND location and action-dependent state transitions. In the cazettes foraging task, the boxes
    are dependent which means the belief state updates are coupled and are BOTH constructed
    with a single call to this function.

    Use Gaussian approximation for diffusion

    Note: convention is site is active in state 1. and inactive in state 0.
    """
    def gb(x, k1, k0, p_sw, belief_location, other_location, action):
        # the belief by convention is of whether the box is ACTIVE, 
        # 1 - x is the belief that the box is INACTIVE

        def agent_at_belief_location():
            def push_button_action():
                p_off = 1.0
                p_on = 1.0 - p_sw
                p_sw_on_off = p_sw
                p_sw_off_on = 0.0
                return p_off, p_on, p_sw_on_off, p_sw_off_on

            def other_action():
                p_off = 1.0
                p_on = 1.0
                p_sw_on_off = 0.0
                p_sw_off_on = 0.0
                return p_off, p_on, p_sw_on_off, p_sw_off_on

            return lax.cond(action == 2, push_button_action, other_action)

        def agent_not_at_belief_location():
            def push_button_action():
                p_off = 1.0 - p_sw
                p_on = 1.0
                p_sw_on_off = 0.0
                p_sw_off_on = p_sw
                return p_off, p_on, p_sw_on_off, p_sw_off_on

            def other_action():
                p_off = 1.0
                p_on = 1.0
                p_sw_on_off = 0.0
                p_sw_off_on = 0.0
                return p_off, p_on, p_sw_on_off, p_sw_off_on

            return lax.cond(action == 2, push_button_action, other_action)

        #if the belief location is the other (i.e. not active location)
        p_off, p_on, p_sw_on_off, p_sw_off_on = lax.cond(
            belief_location == other_location, agent_at_belief_location, agent_not_at_belief_location
        )

        # calculate the new belief b_{t+1}(s_{t+1}) = (1/c) * P(o_{t+1} | s_{t+1}, a_t) * bhat_{t+1}(s_{t+1})
        # where bhat_{t+1}(s_{t+1}) = \sum_{s_t} P(s_{t+1} | s_t, a_t) * b_t(s_t)

        # the probability of the next observation given 
        # that the box is active at that next time step
        Pot1 = k1
        # P(s_{t+1}=1 | s_t=1, a_t) * b_t(s_t=1) 
        # the probability of the box being active at the next time step given 
        # that it is active at the current time step, times the belief that the box is active
        Pst1 = p_on * x
        # P(s_{t+1}=1 | s_t=0, a_t) * b_t(s_t=0)
        # the probability of the box being inactive at the next time step given 
        # that it is active at the current time step times the belief that the box is inactive
        Pst0 = p_sw_off_on * (1 - x)

        numerator = Pot1 * (Pst0 + Pst1)  # the numerator of the belief update equation

        # now calculate the normalization constant..
        # \frac{1}{\sum_{s_{t+1}} P(o_{t+1} | s_{t+1}, a_t) * \sum_{s_t} P(s_{t+1} | s_t, a_t) * b_t(s_t)}
        c = k1 * p_on * x + k0 * p_sw_on_off * x + k0 * p_off * (1 - x) + k1 * p_sw_off_on * (1 - x)

        bst1 = lax.cond(
            (c == 0.0) & (numerator == 0.0),
            lambda _: 0.0,
            lambda _: numerator / c,
            operand=None
        )

        return bst1

    mu = 0
    dq = 1 / nq # belief resolution
    Ncol = 1  # max color value.. [0, Ncol]
    
    #belief locations are the locations where the agent can make observations
    #about the state of the boxes. So these are locations != 0
    belief_locations = jnp.array([1, 2])

    # Define action and location dependent observation emission matrices
    # Fill Obs_emis based on location-specific actions
    # by convention obs_emis[action] is NCol x N states.
    # so the probability of each possible observation given the state
    def set_push_button_emissions(bloc, oloc, action):
        Obs_emis_bloc_bloc = jnp.empty((2, 2))
        Obs_emis_bloc_oloc = jnp.empty((2, 2))

        Obs_emis_bloc_bloc = Obs_emis_bloc_bloc.at[1, 0].set(0.0)
        Obs_emis_bloc_bloc = Obs_emis_bloc_bloc.at[1, 1].set(p_rwd)
        Obs_emis_bloc_bloc = Obs_emis_bloc_bloc.at[0, 0].set(1.0)
        Obs_emis_bloc_bloc = Obs_emis_bloc_bloc.at[0, 1].set(1.0 - p_rwd)

        Obs_emis_bloc_oloc = Obs_emis_bloc_oloc.at[1, 0].set(p_rwd)
        Obs_emis_bloc_oloc = Obs_emis_bloc_oloc.at[1, 1].set(0.0)
        Obs_emis_bloc_oloc = Obs_emis_bloc_oloc.at[0, 0].set(1.0 - p_rwd)
        Obs_emis_bloc_oloc = Obs_emis_bloc_oloc.at[0, 1].set(1.0)

        return Obs_emis_bloc_bloc, Obs_emis_bloc_oloc

    def set_other_emissions(bloc, oloc, action):
        Obs_emis_bloc_bloc = jnp.ones((Ncol + 1, 2)) / (Ncol + 1)
        Obs_emis_bloc_oloc = jnp.ones((Ncol + 1, 2)) / (Ncol + 1)
        return Obs_emis_bloc_bloc, Obs_emis_bloc_oloc

    def update_obs_emis(Obs_emis, action, bloc, oloc):
        Obs_emis = Obs_emis.copy()
        Obs_emis_bloc_bloc, Obs_emis_bloc_oloc = lax.cond(
            action == 2,
            lambda _: set_push_button_emissions(bloc, oloc, action),
            lambda _: set_other_emissions(bloc, oloc, action),
            operand=None
        )
        Obs_emis[bloc][bloc][action] = Obs_emis_bloc_bloc
        Obs_emis[bloc][oloc][action] = Obs_emis_bloc_oloc
        return Obs_emis
        
    Obs_emis = {bloc.item():{oloc.item():{action.item(): jnp.empty((2,2)) for action in actions} for oloc in belief_locations} for bloc in belief_locations}
    for action in actions:
        action = action.item()
        for bloc in belief_locations:  # belief locations
            bloc = bloc.item()
            oloc = int((2 - bloc) + 1)
            Obs_emis = update_obs_emis(Obs_emis, action, bloc, oloc)

    # Define transition probabilities for states.. but now it's action-dependent AND location-dependent
    # so state transitions only probabilistically happen based on what action is taken and 
    # where the action is taken.
    def set_push_button_transitions(bloc, oloc, action):
        Trans_state_bloc_bloc = jnp.array([[1.0, p_sw],  # Probability of staying on/off or switching
                                        [0.0, 1 - p_sw]])
        Trans_state_bloc_oloc = jnp.array([[1. - p_sw, 0.0],
                                        [p_sw, 1.0]])
        return Trans_state_bloc_bloc.astype(jnp.float32), Trans_state_bloc_oloc.astype(jnp.float32)

    def set_other_transitions(bloc, oloc, action):
        Trans_state_bloc_oloc = jnp.array([[1, 0],  # Identity matrix (no transition)
                                        [0, 1]])
        return Trans_state_bloc_oloc.astype(jnp.float32), Trans_state_bloc_oloc.astype(jnp.float32)

    def update_trans_state(Trans_state, action, bloc, oloc):
        Trans_state = Trans_state.copy()
        Trans_state_bloc_bloc, Trans_state_bloc_oloc = lax.cond(
            action == 2,
            lambda _: set_push_button_transitions(bloc, oloc, action),
            lambda _: set_other_transitions(bloc, oloc, action),
            operand=None
        )
        Trans_state[bloc][bloc][action] = Trans_state_bloc_bloc
        Trans_state[bloc][oloc][action] = Trans_state_bloc_oloc
        return Trans_state

    Trans_state = {bloc.item():{oloc.item():{} for oloc in belief_locations} for bloc in belief_locations}
    for action in actions:
        action = action.item()
        for bloc in belief_locations:  # belief locations
            bloc = bloc.item()
            #set oloc to the other location
            oloc = int((2 - bloc) + 1)
            Trans_state = update_trans_state(Trans_state, action, bloc, oloc)

    d = jnp.zeros((len(belief_locations), len(belief_locations),len(actions), Ncol + 1, nq, nq))  # distance between q and q' for each action
    xopt = jnp.zeros((len(belief_locations), len(belief_locations), len(actions), Ncol + 1, nq, nq))  # optimal x for each action
    height = jnp.zeros((len(belief_locations), len(belief_locations),len(actions), Ncol + 1, nq, nq))  # height of the density
    
    # approximate belief transition matrix
    Trans_belief_obs_approx = {bloc.item():{oloc.item():{action.item(): jnp.zeros((Ncol + 1, nq, nq)) for action in actions} for oloc in belief_locations} for bloc in belief_locations}
    # obseration emission transition matrix
    Obs_emis_trans = {bloc.item():{oloc.item():{} for oloc in belief_locations} for bloc in belief_locations}
    # gaussian belief state densities
    den = {bloc.item():{oloc.item():{action.item(): jnp.zeros((Ncol + 1, nq, nq)) for action in actions} for oloc in belief_locations} for bloc in belief_locations}

    def compute_xopt_d_height(i, j, n, k0, k1, bloc, oloc, action):
        # Set up q and qq based on indices
        q = i * dq + dq / 2
        qq = j * dq + dq / 2

        # Define `dist` for use in `fminbound`
        def dist(x):
            return jnp.sqrt((q - x) ** 2 + (qq - gb(x, k1, k0, p_sw, bloc, oloc, action)) ** 2)

        # Find xopt and d using `fminbound`
        bounds = (0, 1)
        xopt_temp, d_temp = fminbound(dist, bounds)

        # Compute the density as a Gaussian approximation
        den_temp = norm.pdf(d_temp, mu, sigma)

        # Calculate `height` as a dot product of observation emissions and transition state
        height_temp = jnp.dot(jnp.dot(Obs_emis[bloc][oloc][action][n, :],
                                    Trans_state[bloc][oloc][action]),
                            jnp.array([1 - q, q]))
        
        return xopt_temp, d_temp, den_temp, height_temp

    # Vectorize the function across indices `i` and `j`
    vectorized_compute = vmap(
        vmap(compute_xopt_d_height, in_axes=(None, 0, None, None, None, None, None, None)),
        in_axes=(0, None, None, None, None, None, None, None)
    )
    # for each action and next observation pair the goal is to project or translate the 
    # joint probability of observations and state transitions into a discrete space of belief states.
    for b_index, bloc in enumerate(belief_locations):#belief states
        bloc = bloc.item()
        for o_index, oloc in enumerate(belief_locations):#all action locations
            oloc = oloc.item()
            for a_index, action in enumerate(actions):#for each possible current action
                action = action.item()
                # if action == 4:  # push button action
                for n in range(Ncol + 1):# for each possible next observation
                    k0 = Obs_emis[bloc][oloc][action][n, 0]# probability of observing n given that the the box is INACTIVE 
                    k1 = Obs_emis[bloc][oloc][action][n, 1]# probability of observing n given that the the box is ACTIVE

                    # Apply vectorized function
                    xopt_temp, d_temp, den_temp, height_temp = vectorized_compute(
                        jnp.arange(nq), jnp.arange(nq), n, k0, k1, bloc, oloc, action
                    )
                    xopt_temp = jnp.transpose(xopt_temp)
                    d_temp = jnp.transpose(d_temp)
                    den_temp = jnp.transpose(den_temp)
                    height_temp = jnp.transpose(height_temp)

                    # Set results in the arrays
                    xopt = xopt.at[b_index, o_index, a_index, n].set(xopt_temp)
                    d = d.at[b_index, o_index, a_index, n].set(d_temp)
                    height = height.at[b_index, o_index, a_index, n].set(height_temp)
                                
                    #this divides every element in the density by its column sum which normalizes
                    #the density to a transition probability.. this is P(b_t+1 | b_t, a_t, o_t+1)
                    sum_vals = jnp.sum(den_temp, axis=0)
                    den_temp /= np.tile(sum_vals, (nq, 1))
                    # Perform element-wise division
                    den[bloc][oloc][action] = den[bloc][oloc][action].at[n].set(den_temp)

                    #trans_belief_obs_approx is the approximate belief transition matrix for each observation
                    #which is the product of P(b_t+1 | b_t, a_t, o_t+1) and O(o_t+1 | b_t, a_t). In the math
                    #to get the belief transition matrix, we need to sum over all possible next observations.
                    # if action == 4:
                    Trans_belief_obs_approx_temp = jnp.multiply(den_temp, height_temp)
                    Trans_belief_obs_approx[bloc][oloc][action] = Trans_belief_obs_approx[bloc][oloc][action].at[n, :, :].set(Trans_belief_obs_approx_temp)
                    # else:
                        # Trans_belief_obs_approx[action][n, :, :] = np.multiply(den[action][n, :, :], np.identity(nq))

                    Obs_emis_trans[bloc][oloc][action] = jnp.dot(Obs_emis[bloc][oloc][action], Trans_state[bloc][oloc][action])
                # else:
                #     # for all other actions, the observation is always 0
                #     # so optimizing x is not necessary, the belief states deterministically
                #     # transition to the same belief state.
                #     for n in range(Ncol + 1):
                #         den[action][n, :, :] = np.identity(nq)
                #         Trans_belief_obs_approx[action][n, :, :] = np.identity(nq)

    return Trans_belief_obs_approx, Obs_emis_trans, den

def beliefTransitionMatrixGaussianCazettesIndependent(p_sw, p_rwd, nq, actions, locations, sigma):
    """
    create transition matrix between nq belief states q to q' WITH color observation
    use Gaussian approximation for diffusion
    """
    def gb(x, k1, k0, p_sw, belief_location, other_location, action):
        #the belief by convention is of whether the box is ACTIVE, 
        #1 - x is the belief that the box is INACTIVE
        p_off = 1. - p_sw
        p_on = 1. - p_sw
        p_sw_on_off = p_sw
        p_sw_off_on = p_sw

        #calculate the new belief b_{t+1}(s_{t+1}) = (1/c) * P(o_{t+1} | s_{t+1}, a_t) * bhat_{t+1}(s_{t+1})
        #where bhat_{t+1}(s_{t+1}) = \sum_{s_t} P(s_{t+1} | s_t, a_t) * b_t(s_t)

        # the probability of the next observation given 
        # that the box is active at that next time step
        Pot1 = k1
        # P(s_{t+1}=1 | s_t=1, a_t) * b_t(s_t=1) 
        # the probability of the box being active at the next time step given 
        # that it is active at the current time step, times the belief that the box is active
        Pst1 = p_on * x
        # P(s_{t+1}=1 | s_t=0, a_t) * b_t(s_t=0)
        # the probability of the box being inactive at the next time step given 
        # that it is active at the current time step times the belief that the box is inactive
        Pst0 = p_sw_off_on * (1 - x)

        numerator = Pot1 * (Pst0 + Pst1)# the numerator of the belief update equation

        #now calculate the normalization constant..
        # \frac{1}{\sum_{s_{t+1}} P(o_{t+1} | s_{t+1}, a_t) * \sum_{s_t} P(s_{t+1} | s_t, a_t) * b_t(s_t)}
        c = k1 * p_on * x + k0 * p_sw_on_off * x + k0 * p_off * (1 - x) + k1 * p_sw_off_on * (1 - x)

        if c == 0.0 and numerator == 0.0:
            return 0
        else:
            bst1 = numerator / c# the new belief state at the next time step
            return bst1

    mu = 0
    dq = 1 / nq # belief resolution
    Ncol = 1  # max color value.. [0, Ncol]
    
    #belief locations are the locations where the agent can make observations
    #about the state of the boxes. So these are locations != 0
    belief_locations = locations#[l for l in locations if l != 0]

    # Define action and location dependent observation emission matrices
    # Fill Obs_emis based on location-specific actions
    # by convention obs_emis[action] is NCol x N states. 
    # so the probability of each possible observation given the state
    Obs_emis = {bloc:{oloc:{action: np.empty((2,2)) for action in actions} for oloc in belief_locations} for bloc in belief_locations}
    for action in actions:
        for bloc in belief_locations:#belief locations
            if action == 2 and bloc != 0:  # push button at location other than intermediate location 0
                Obs_emis[bloc][bloc][action][1, 0] = 0.0# Probability of observing 1 (i.e. box is ON) when box is actually OFF
                Obs_emis[bloc][bloc][action][1, 1] = p_rwd# Probability of observing 1 (i.e. box is ON) when box is indeed ON
                Obs_emis[bloc][bloc][action][0, 0] = 1.0# Probability of observing 0 (i.e. box is OFF) when box is indeed OFF
                Obs_emis[bloc][bloc][action][0, 1] = 1. - p_rwd# Probability of observing 0 (i.e. box is OFF) when box is actually ON
                
                #for the other location
                #there are no observations made. This means
                #that the emissions from the observation model should carry no information
                #about the state of the box. This equates to a uniform distribution over the
                #possible observations for a given world state.
                other_locs = [l for l in belief_locations if l != bloc]
                for oloc in other_locs:
                    Obs_emis[bloc][oloc][action] = np.ones((Ncol + 1, 2)) / (Ncol + 1)
            else:
                #for all other actions, there are no observations made. This means
                #that the emissions from the observation model should carry no information
                #about the state of the box. This equates to a uniform distribution over the
                #possible observations for a given world state.
                for oloc in belief_locations:#all belief locations..
                    Obs_emis[bloc][oloc][action] = np.ones((Ncol + 1, 2)) / (Ncol + 1)

    # Define transition probabilities for states.. but now it's action-dependent AND location-dependent
    # so state transitions only probabilistically happen based on what action is taken and 
    # where the action is taken.
    Trans_state = {bloc:{oloc:{} for oloc in belief_locations} for bloc in belief_locations}
    for action in actions:
        for bloc in belief_locations:#belief locations
            for oloc in belief_locations:#belief locations
                #the state transition matrix at every location and for every action
                Trans_state[bloc][oloc][action] = np.array([[1 - p_sw, p_sw],  # Probability of staying on/off or switching
                                                            [p_sw, 1 - p_sw]])

    d = np.zeros((len(belief_locations), len(belief_locations),len(actions), Ncol + 1, nq, nq))  # distance between q and q' for each action
    xopt = np.zeros((len(belief_locations), len(belief_locations), len(actions), Ncol + 1, nq, nq))  # optimal x for each action
    height = np.zeros((len(belief_locations), len(belief_locations),len(actions), Ncol + 1, nq, nq))  # height of the density
    
    # approximate belief transition matrix
    Trans_belief_obs_approx = {bloc:{oloc:{action: np.zeros((Ncol + 1, nq, nq)) for action in actions} for oloc in belief_locations} for bloc in belief_locations}
    # obseration emission transition matrix
    Obs_emis_trans = {bloc:{oloc:{} for oloc in belief_locations} for bloc in belief_locations}
    # gaussian belief state densities
    den = {bloc:{oloc:{action: np.zeros((Ncol + 1, nq, nq)) for action in actions} for oloc in belief_locations} for bloc in belief_locations}

    # for each action and next observation pair the goal is to project or translate the 
    # joint probability of observations and state transitions into a discrete space of belief states.
    for b_index, bloc in enumerate(belief_locations):#belief states
        for o_index, oloc in enumerate(belief_locations):#all action locations
            for a_index, action in enumerate(actions):#for each possible current action
                # if action == 4:  # push button action
                for n in range(Ncol + 1):# for each possible next observation
                    k0 = Obs_emis[bloc][oloc][action][n, 0]# probability of observing n given that the the box is INACTIVE 
                    k1 = Obs_emis[bloc][oloc][action][n, 1]# probability of observing n given that the the box is ACTIVE
                    for i in range(nq):
                        for j in range(nq):
                            # Approximate the probability with Gaussian approximation
                            q = i * dq + dq / 2  # past belief (along columns, index with i)
                            qq = j * dq + dq / 2  # new belief (along rows, index with j)

                            def dist(x):
                                # the distance of (x, gb(x)) to the center of each bin
                                return sqrt((q - x) ** 2 + (qq - gb(x, k1, k0, p_sw, bloc, oloc, action)) ** 2)

                            #find a value xopt between 0 and 1 that when advanced to the next time step using
                            #the belief update function gb, (xopt, gb(xopt)) is the closest point to (q, qq). 
                            xopt[b_index, o_index, a_index, n, j, i], d[b_index, o_index, a_index, n, j, i] = optimize.fminbound(dist, 0, 1, full_output=1)[0:2]
                            
                            #den is the density of the distance between the optimal belief state and the actual belief state
                            #this represents the approximation error of the 
                            den[bloc][oloc][action][n, j, i] = norm.pdf(d[b_index, o_index, a_index, n, j, i], mu, sigma)  # use this to approximate delta function with diffusion
                            
                            #height is O(o_t+1 | b_t, a_t) which is the product of
                            #the expectation of the observation over states and the expectation of the state transition over states
                            #and the expectation of the belief over states
                            height[b_index, o_index, a_index, n, j, i] = Obs_emis[bloc][oloc][action][n, :].dot(Trans_state[bloc][oloc][action]).dot(np.array([1 - q, q]))#.dot(np.array([xopt[a_index,n, j, i], 1 - xopt[a_index,n, j, i]]))#

                    #this divides every element in the density by its column sum which normalizes
                    #the density to a transition probability.. this is P(b_t+1 | b_t, a_t, o_t+1)
                    den[bloc][oloc][action][n, :, :] /= np.tile(np.sum(den[bloc][oloc][action][n, :, :], 0), (nq, 1))
                    #trans_belief_obs_approx is the approximate belief transition matrix for each observation
                    #which is the product of P(b_t+1 | b_t, a_t, o_t+1) and O(o_t+1 | b_t, a_t). In the math
                    #to get the belief transition matrix, we need to sum over all possible next observations.
                    # if action == 4:
                    Trans_belief_obs_approx[bloc][oloc][action][n, :, :] = np.multiply(den[bloc][oloc][action][n, :, :], height[b_index, o_index, a_index, n, :, :])
                    # else:
                        # Trans_belief_obs_approx[action][n, :, :] = np.multiply(den[action][n, :, :], np.identity(nq))

                    Obs_emis_trans[bloc][oloc][action] = Obs_emis[bloc][oloc][action].dot(Trans_state[bloc][oloc][action])
                # else:
                #     # for all other actions, the observation is always 0
                #     # so optimizing x is not necessary, the belief states deterministically
                #     # transition to the same belief state.
                #     for n in range(Ncol + 1):
                #         den[action][n, :, :] = np.identity(nq)
                #         Trans_belief_obs_approx[action][n, :, :] = np.identity(nq)

    return Trans_belief_obs_approx, Obs_emis_trans, den

def beliefTransitionMatrixGaussianCazettesIndependentDependent(p_sw, p_rwd, nq, actions, locations, sigma):
    """
    create transition matrix between nq belief states q to q' WITH color observation
    use Gaussian approximation for diffusion
    """
    def gb(x, k1, k0, p_sw, belief_location, other_location, action):
        #the belief by convention is of whether the box is ACTIVE, 
        #1 - x is the belief that the box is INACTIVE
        if belief_location == other_location:#the agent is at this location
            if action == 2:  # push button action
                p_off = 1. - p_sw  # Probability of staying off is 1
                p_on = 1. - p_sw  # Probability of staying on is 1 - p_sw
                p_sw_on_off = p_sw  # Probability of switching from on to off
                p_sw_off_on = p_sw
            else:
                p_off = 1.
                p_on = 1.
                p_sw_on_off = 0.0
                p_sw_off_on = 0.0
        else:#the agent is not at this location
                p_off = 1.
                p_on = 1.
                p_sw_on_off = 0.0
                p_sw_off_on = 0.0

        #calculate the new belief b_{t+1}(s_{t+1}) = (1/c) * P(o_{t+1} | s_{t+1}, a_t) * bhat_{t+1}(s_{t+1})
        #where bhat_{t+1}(s_{t+1}) = \sum_{s_t} P(s_{t+1} | s_t, a_t) * b_t(s_t)

        # the probability of the next observation given
        # that the box is active at that next time step
        Pot1 = k1
        # P(s_{t+1}=1 | s_t=1, a_t) * b_t(s_t=1)
        # the probability of the box being active at the next time step given
        # that it is active at the current time step, times the belief that the box is active
        Pst1 = p_on * x
        # P(s_{t+1}=1 | s_t=0, a_t) * b_t(s_t=0)
        # the probability of the box being inactive at the next time step given
        # that it is active at the current time step times the belief that the box is inactive
        Pst0 = p_sw_off_on * (1 - x)

        numerator = Pot1 * (Pst0 + Pst1)# the numerator of the belief update equation

        #now calculate the normalization constant..
        # \frac{1}{\sum_{s_{t+1}} P(o_{t+1} | s_{t+1}, a_t) * \sum_{s_t} P(s_{t+1} | s_t, a_t) * b_t(s_t)}
        c = k1 * p_on * x + k0 * p_sw_on_off * x + k0 * p_off * (1 - x) + k1 * p_sw_off_on * (1 - x)

        if c == 0.0 and numerator == 0.0:
            return 0
        else:
            bst1 = numerator / c# the new belief state at the next time step
            return bst1

    mu = 0
    dq = 1 / nq # belief resolution
    Ncol = 1  # max color value.. [0, Ncol]
    
    #belief locations are the locations where the agent can make observations
    #about the state of the boxes. So these are locations != 0
    belief_locations = locations#[l for l in locations if l != 0]

    # Define action and location dependent observation emission matrices
    # Fill Obs_emis based on location-specific actions
    # by convention obs_emis[action] is NCol x N states. 
    # so the probability of each possible observation given the state
    Obs_emis = {bloc:{oloc:{action: np.empty((2,2)) for action in actions} for oloc in belief_locations} for bloc in belief_locations}
    for action in actions:
        for bloc in belief_locations:#belief locations
            if action == 2 and bloc != 0:  # push button at location other than intermediate location 0
                p_diff = 2*p_rwd - 1
                Obs_emis[bloc][bloc][action][1, 0] = (1 - p_diff)/2# Probability of observing 1 (i.e. box is ON) when box is actually OFF
                Obs_emis[bloc][bloc][action][1, 1] = (1 + p_diff)/2# Probability of observing 1 (i.e. box is ON) when box is indeed ON
                Obs_emis[bloc][bloc][action][0, 0] = (1 + p_diff)/2# Probability of observing 0 (i.e. box is OFF) when box is indeed OFF
                Obs_emis[bloc][bloc][action][0, 1] = (1 - p_diff)/2# Probability of observing 0 (i.e. box is OFF) when box is actually ON
                
                #for the other location
                #there are no observations made. This means
                #that the emissions from the observation model should carry no information
                #about the state of the box. This equates to a uniform distribution over the
                #possible observations for a given world state.
                other_locs = [l for l in belief_locations if l != bloc]
                for oloc in other_locs:
                    Obs_emis[bloc][oloc][action] = np.ones((Ncol + 1, 2)) / (Ncol + 1)
            else:
                #for all other actions, there are no observations made. This means
                #that the emissions from the observation model should carry no information
                #about the state of the box. This equates to a uniform distribution over the
                #possible observations for a given world state.
                for oloc in belief_locations:#all belief locations..
                    Obs_emis[bloc][oloc][action] = np.ones((Ncol + 1, 2)) / (Ncol + 1)

    # Define transition probabilities for states.. but now it's action-dependent AND location-dependent
    # so state transitions only probabilistically happen based on what action is taken and 
    # where the action is taken.
    Trans_state = {bloc:{oloc:{} for oloc in belief_locations} for bloc in belief_locations}
    for action in actions:
        for bloc in belief_locations:#belief locations
            if action == 2:  # push button action
                #the state transition matrix for the location where the button is pressed
                Trans_state[bloc][bloc][action] = np.array([[1 - p_sw, p_sw],
                                                            [p_sw, 1 - p_sw]])
                #for all other locations...
                #the probability of transitioning between states is the inverse 
                #of the transition probability matrix of the location where the button is pressed
                other_locs = [l for l in belief_locations if l != bloc]
                for oloc in other_locs:
                    Trans_state[bloc][oloc][action] = np.array([[1., 0.0],
                                                                [0.0, 1.0]])
            else:
                for oloc in belief_locations:
                    Trans_state[bloc][oloc][action] = np.array([[1., 0.],  # Identity matrix (no transition)
                                                                [0., 1.]])

    d = np.zeros((len(belief_locations), len(belief_locations),len(actions), Ncol + 1, nq, nq))  # distance between q and q' for each action
    xopt = np.zeros((len(belief_locations), len(belief_locations), len(actions), Ncol + 1, nq, nq))  # optimal x for each action
    height = np.zeros((len(belief_locations), len(belief_locations),len(actions), Ncol + 1, nq, nq))  # height of the density
    
    # approximate belief transition matrix
    Trans_belief_obs_approx = {bloc:{oloc:{action: np.zeros((Ncol + 1, nq, nq)) for action in actions} for oloc in belief_locations} for bloc in belief_locations}
    # obseration emission transition matrix
    Obs_emis_trans = {bloc:{oloc:{} for oloc in belief_locations} for bloc in belief_locations}
    # gaussian belief state densities
    den = {bloc:{oloc:{action: np.zeros((Ncol + 1, nq, nq)) for action in actions} for oloc in belief_locations} for bloc in belief_locations}

    # for each action and next observation pair the goal is to project or translate the 
    # joint probability of observations and state transitions into a discrete space of belief states.
    for b_index, bloc in enumerate(belief_locations):#belief states
        for o_index, oloc in enumerate(belief_locations):#all action locations
            for a_index, action in enumerate(actions):#for each possible current action
                # if action == 4:  # push button action
                for n in range(Ncol + 1):# for each possible next observation
                    k0 = Obs_emis[bloc][oloc][action][n, 0]# probability of observing n given that the the box is INACTIVE 
                    k1 = Obs_emis[bloc][oloc][action][n, 1]# probability of observing n given that the the box is ACTIVE
                    for i in range(nq):
                        for j in range(nq):
                            # Approximate the probability with Gaussian approximation
                            q = i * dq + dq / 2  # past belief (along columns, index with i)
                            qq = j * dq + dq / 2  # new belief (along rows, index with j)

                            def dist(x):
                                # the distance of (x, gb(x)) to the center of each bin
                                return sqrt((q - x) ** 2 + (qq - gb(x, k1, k0, p_sw, bloc, oloc, action)) ** 2)

                            #find a value xopt between 0 and 1 that when advanced to the next time step using
                            #the belief update function gb, (xopt, gb(xopt)) is the closest point to (q, qq). 
                            xopt[b_index, o_index, a_index, n, j, i], d[b_index, o_index, a_index, n, j, i] = optimize.fminbound(dist, 0, 1, full_output=1)[0:2]
                            
                            #den is the density of the distance between the optimal belief state and the actual belief state
                            #this represents the approximation error of the 
                            den[bloc][oloc][action][n, j, i] = norm.pdf(d[b_index, o_index, a_index, n, j, i], mu, sigma)  # use this to approximate delta function with diffusion
                            
                            #height is O(o_t+1 | b_t, a_t) which is the product of
                            #the expectation of the observation over states and the expectation of the state transition over states
                            #and the expectation of the belief over states
                            height[b_index, o_index, a_index, n, j, i] = Obs_emis[bloc][oloc][action][n, :].dot(Trans_state[bloc][oloc][action]).dot(np.array([1 - q, q]))#.dot(np.array([xopt[a_index,n, j, i], 1 - xopt[a_index,n, j, i]]))#

                    #this divides every element in the density by its column sum which normalizes
                    #the density to a transition probability.. this is P(b_t+1 | b_t, a_t, o_t+1)
                    den[bloc][oloc][action][n, :, :] /= np.tile(np.sum(den[bloc][oloc][action][n, :, :], 0), (nq, 1))
                    #trans_belief_obs_approx is the approximate belief transition matrix for each observation
                    #which is the product of P(b_t+1 | b_t, a_t, o_t+1) and O(o_t+1 | b_t, a_t). In the math
                    #to get the belief transition matrix, we need to sum over all possible next observations.
                    # if action == 4:
                    Trans_belief_obs_approx[bloc][oloc][action][n, :, :] = np.multiply(den[bloc][oloc][action][n, :, :], height[b_index, o_index, a_index, n, :, :])
                    # else:
                        # Trans_belief_obs_approx[action][n, :, :] = np.multiply(den[action][n, :, :], np.identity(nq))

                    Obs_emis_trans[bloc][oloc][action] = Obs_emis[bloc][oloc][action].dot(Trans_state[bloc][oloc][action])
                # else:
                #     # for all other actions, the observation is always 0
                #     # so optimizing x is not necessary, the belief states deterministically
                #     # transition to the same belief state.
                #     for n in range(Ncol + 1):
                #         den[action][n, :, :] = np.identity(nq)
                #         Trans_belief_obs_approx[action][n, :, :] = np.identity(nq)

    return Trans_belief_obs_approx, Obs_emis_trans, den

def beliefTransitionMatrixGaussian(p_appear, p_disappear, nq, sigma = 0.1):
    """
    create transition matrix between nq belief states q to q' without color observation
    use Gaussian approximation for diffusion
    """
    mu = 0

    d = np.zeros((nq, nq))
    Trrr = np.zeros((nq, nq))
    dq = 1 / nq
    a = 1 - p_disappear - p_appear

    for i in range(nq):
        for j in range(nq):
            q = i * dq + dq / 2
            qq = j * dq + dq / 2

            d[j, i] = abs(a * q - qq + p_appear) / sqrt(a ** 2 + 1)
            Trrr[j, i] = norm.pdf(d[j, i], mu, sigma)

    Tb = Trrr / np.tile(np.sum(Trrr, 0), (nq, 1))

    return Tb

def beliefTransitionMatrixGaussianCol(p_appear, p_disappear, qmin, qmax, Ncol, nq, sigma):
    """
    create transition matrix between nq belief states q to q' WITH color observation
    use Gaussian approximation for diffusion
    """
    def gb(x, k1, k0, p_appear, p_disappear):
        #he we are calculating the belief update for possible future states P(b_t+1 | b_t, a_t, o_t+1)
        a = 1 - p_disappear - p_appear # a is the probability of staying in the same state
        return (k1 * a * x + k1 * p_appear) / ((k1 - k0) * a * x + k1 * p_appear + k0 * (1 - p_appear))

    def gbinv(y, k1, k0, p_appear, p_disappear):
        a = 1 - p_disappear - p_appear
        return (y * (k1 * p_appear + k0 * (1 - p_appear)) - k1 * p_appear) / (k1 * a - y * (k1 - k0) * a)

    mu = 0
    # Define transition probabilities for states, this is an approximation
    # because the actual transition probabilities are subtly action-dependent
    # in that the box can transition from on to off if the button is pressed.
    Trans_state = np.array([[1 - p_appear, p_disappear],
                            [p_appear, 1 - p_disappear]])
    Obs_emis = np.zeros((Ncol + 1, 2))  # Observation(color) generation,

    #from the paper:
    #Color values for both boxes are drawn independently at each time from a binomial distribution 
    #with five states, with mean q∗1 = 0.4 when food is available in the box, and q∗2 = 0.6 otherwise.
    #this means that Obs_emis[:,0] are emissions for the box being inactive and Obs_emis[:,1] are 
    #emissions for the box being active.
    Obs_emis[:, 0] = binom.pmf(range(Ncol + 1), Ncol, qmax)# Probabilities when the box is INACTIVE
    Obs_emis[:, 1] = binom.pmf(range(Ncol + 1), Ncol, qmin)# Probabilities when the box is ACTIVE

    dq = 1 / nq

    d = np.zeros((Ncol + 1, nq, nq))# distance between q and q' for each color
    den = np.zeros((Ncol + 1, nq, nq))# density of the distance
    xopt = np.zeros((Ncol + 1, nq, nq))# optimal x for each color
    height = np.zeros((Ncol + 1, nq, nq))# height of the density
    Trans_belief_obs_approx = np.zeros((Ncol + 1, nq, nq))# approximate belief transition matrix

    for n in range(Ncol + 1):
        k0 = Obs_emis[n, 0]# probability of observing the box is INACTIVE
        k1 = Obs_emis[n, 1]# probability of observing the box is ACTIVE

        for i in range(nq):
            for j in range(nq):
                # Approximate the probability with Gaussian approxiamtion
                q = i * dq + dq / 2   #past belief(along columns, index with i)
                qq = j * dq + dq / 2  # new belief(along rows, index with j)

                def dist(x):
                    # the distance of (x, gb(x)) to the center of each bin
                    return sqrt((q - x) ** 2 + (qq - gb(x, k1, k0, p_appear, p_disappear)) ** 2)

                #xopt is the optimal belief state and d is the distance between the optimal belief state and the actual belief state
                xopt[n, j, i], d[n, j, i] = optimize.fminbound(dist, 0, 1, full_output=1)[0:2]
                
                den[n, j, i] = norm.pdf(d[n, j, i], mu, sigma)   # use this to approximate delta function with diffusion
                
                #height is O(o_t+1 | b_t, a_t) which is the product of
                #the expectation of the observation over states and the expectation of the state transition over states
                #and the expectation of the belief over states
                height[n, j, i] = Obs_emis[n, :].dot(Trans_state).dot(np.array([1 - q, q]))
                #height[n, j, i] = Obs_emis[n, :].dot(Trans_state).dot(np.array([1 - xopt[n, j, i], xopt[n, j, i]]))
        
        #marginalize the density over the observations and tile it nq times,
        #dividing the density by this tiled array effectively gives the 
        #this divides every element in the density by its column sum which normalizes
        #the density to a transition probability.. this is P(b_t+1 | b_t, a_t, o_t+1)
        den[n] = den[n] / np.tile(np.sum(den[n], 0), (nq, 1))
        #trans_belief_obs_approx is the approximate belief transition matrix for each observation
        #which is the product of P(b_t+1 | b_t, a_t, o_t+1) and O(o_t+1 | b_t, a_t). In the math
        #to get the belief transition matrix, we need to sum over all possible next observations.
        Trans_belief_obs_approx[n] = np.multiply(den[n], height[n])

    return Trans_belief_obs_approx, Obs_emis.dot(Trans_state), den


def beliefTransitionMatrixGaussianDerivative(p_appear, p_disappear, nq, sigma=0.1):
    mu = 0

    d = np.zeros((nq, nq))
    pdpgamma = np.zeros((nq, nq))  # derivative with respect to the appear rate, p_appear
    pdpepsilon = np.zeros((nq, nq)) # derivative with respect to the disappear rate, p_disappear
    dq = 1 / nq
    a = 1 - p_disappear - p_appear

    for i in range(nq):
        for j in range(nq):
            q = i * dq + dq / 2
            qq = j * dq + dq / 2

            d[j, i] = abs(a * q - qq + p_appear) / sqrt(a ** 2 + 1)
            pdpepsilon[j, i] = np.sign(a * q - qq + p_appear) * \
                               (-q * sqrt(a ** 2 + 1) + a/sqrt(a ** 2 + 1) * (a * q - qq + p_appear)) / (a ** 2 + 1)
            pdpgamma[j, i] = np.sign(a * q - qq + p_appear) * \
                             ((1-q) * sqrt(a ** 2 + 1) + a/sqrt(a ** 2 + 1) * (a * q - qq + p_appear)) / (a ** 2 + 1)

    Trrr = norm.pdf(d, mu, sigma)  # probability from Gaussian distribution
    pTrrrpd = Trrr * d * (-1) / (sigma ** 2)  # partial derivative of Trrr with respect to d

    dTbdgamma = np.zeros((nq, nq))  # derivative of Tb with respect to the p_appear(gamma) rate
    dTbdepsilon = np.zeros((nq, nq))  # derivative of Tb with respect to the p_disappear(epsilon) rate

    for i in range(nq):
        for j in range(nq):
            dTbdgamma[j, i] = 1 / np.sum(Trrr[:, i]) * pTrrrpd[j, i] * pdpgamma[j, i] - \
                             Trrr[j, i] / (np.sum(Trrr[:, i]) ** 2) * np.sum(pTrrrpd[:, i] * pdpgamma[:, i])
            dTbdepsilon[j, i] = 1 / np.sum(Trrr[:, i]) * pTrrrpd[j, i] * pdpepsilon[j, i] - \
                             Trrr[j, i] / (np.sum(Trrr[:, i]) ** 2) * np.sum(pTrrrpd[:, i] * pdpepsilon[:, i])
    Tb = Trrr / np.tile(np.sum(Trrr, 0), (nq, 1))

    return dTbdgamma, dTbdepsilon


def _im2col_distinct(A, size):
    A = A.T
    dy, dx = size[::-1]
    assert A.shape[0] % dy == 0
    assert A.shape[1] % dx == 0

    ncol = (A.shape[0]//dy) * (A.shape[1]//dx)
    R = np.empty((ncol, dx*dy), dtype=A.dtype)
    k = 0
    for i in range(0, A.shape[0], dy):
        for j in range(0, A.shape[1], dx):
            R[k, :] = A[i:i+dy, j:j+dx].ravel()
            k += 1
    return R.T


def _im2col_sliding(A, size):
    A = A.T
    dy, dx = size[::-1]
    xsz = A.shape[1]-dx+1
    ysz = A.shape[0]-dy+1
    R = np.empty((xsz*ysz, dx*dy), dtype=A.dtype)

    for i in range(ysz):
        for j in range(xsz):
            R[i*xsz+j, :] = A[i:i+dy, j:j+dx].ravel()
    return R.T


def im2col(A, size, type='sliding'):
    """This function behaves similar to *im2col* in MATLAB.

    Parameters
    ----------
    A : 2-D ndarray
        Image from which windows are obtained.
    size : 2-tuple
        Shape of each window.
    type : {'sliding', 'distinct'}, optional
        The type of the windows.

    Returns
    -------
    windows : 2-D ndarray
        The flattened windows stacked vertically.

    """

    if type == 'sliding':
        return _im2col_sliding(A, size)
    elif type == 'distinct':
        return _im2col_distinct(A, size)
    raise ValueError("invalid type of window")


def _col2im_distinct(R, size, width):
    R = R.T
    dy, dx = size[::-1]

    assert width % dx == 0
    nwinx = width//dx
    xsz = nwinx*dx

    assert R.shape[0] % nwinx == 0
    nwiny = R.shape[0]//nwinx
    ysz = nwiny*dy

    A = np.empty((ysz, xsz), dtype=R.dtype)
    k = 0
    for i in range(0, ysz, dy):
        for j in range(0, xsz, dx):
            A[i:i+dy, j:j+dx] = R[k].reshape(size[::-1])
            k += 1
    return A.T


def _col2im_sliding(R, size, width):
    '*********** This is not the same in Matlab*****************'
    R = R.T
    dy, dx = size
    xsz = width-dx+1
    ysz = R.shape[0]//xsz

    A = np.empty((ysz+(dy-1), width), dtype = R.dtype)
    for i in range(ysz):
        for j in range(xsz):
            A[i:i+dy, j:j+dx] = R[i*xsz+j, :].reshape(dy, dx)
    return A.T


def col2im(R, size, width, type='sliding'):
    """This function behaves similar to *col2im* in MATLAB.

    It is the inverse of :func:`im2col`::

            A == col2im(im2col(A, size), size, A.shape[1])

    `R` is what `im2col` returns. `Size` and `type` are the same as
    in `im2col`. `Width` is the number of columns in `A`.

    Examples
    --------
    import numpy as np
    a = np.arange(12).reshape(3,4)
    a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    b = im2col(a, (2,2))
    b
    array([[ 0,  1,  4,  5],
           [ 1,  2,  5,  6],
           [ 2,  3,  6,  7],
           [ 4,  5,  8,  9],
           [ 5,  6,  9, 10],
           [ 6,  7, 10, 11]])
    col2im(b, (2,2), a.shape[1])
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])

    """
    if type == 'sliding':
        return _col2im_sliding(R, size, width)
    elif type == 'distinct':
        return _col2im_distinct(R, size, width)
    raise ValueError("invalid type of window")

'''
The im2col and col2im functions referred to the code on
http://fhs1.bitbucket.org/glasslab_cluster/_modules/glasslab_cluster/utils.html
######################################################################################################
'''


def reversekron(AB, n):
    BA = col2im(im2col(AB, tuple(n[1] * np.array([1, 1])), 'distinct').T, tuple(n[0] * np.array([1, 1])),
         np.prod(n), 'distinct')
    return BA


def tensorsum(A, B):
    ra, ca = A.shape
    rb, cb = B.shape
    C = jnp.empty((ra * rb, ca * cb))

    for i in range(ra):
        for j in range(ca):
            C = C.at[i*rb : (i+1)*rb, j*cb : (j+1)*cb].set(A[i, j] + B)

    return C


def tensorsumm(*args):
    '''
    :param args:
    :return: returns multidimensional kronecker sum of all matrices in list
    Note that the args must be two-dimensional. When any of the ars is a vector, need to pass in a
i    '''
    z = args[0]
    for i in range(1, len(args)):
        z = tensorsum(z, args[i])

    return z


def softmax(x, t):
    """
    transform the value of a vector x to softmax
    beta is the temperature parameter
    """
    z = np.exp(x/t)
    z = z / np.max(z)
    return z / np.sum(z)



def QfromV(ValueIteration):
    Q = np.zeros((ValueIteration.A, ValueIteration.S))
    for a in range(ValueIteration.A):
        Q[a, :] = ValueIteration.R[a] + ValueIteration.discount * \
                                        ValueIteration.P[a].dot(ValueIteration.V)

    return Q


def QfromV_pi(PolicyIteration):
    Q = np.zeros((PolicyIteration.A, PolicyIteration.S))
    for a in range(PolicyIteration.A):
        Q[a, :] = PolicyIteration.R[a] + PolicyIteration.discount * \
                                        PolicyIteration.P[a].dot(PolicyIteration.V)
    return Q



def find_closest(array, value):
    """
    # array is vector
    # value is scalar
    # find the closest point to value in the array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_closest_array(array, values):
    """
    both array and values are vectors, find the elements in the array that are closest to elements in the values
    """
    output = np.zeros(values.shape)
    for i in range(values.shape[0]):
        output[i] = find_closest(array, values[i])

    return output


def rmv_dup_arrary(x):
    """
    remove duplicate elements from array
    """
    uniques = []
    for arr in x:
        if not any(np.array_equal(arr, unique_arr) for unique_arr in uniques):
            uniques.append(arr)
    return uniques

