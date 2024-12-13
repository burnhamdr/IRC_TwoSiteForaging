import jax.numpy as jnp
import jax
import jax.debug as debug

_MDPERR = {
    "mat_nonneg": "Transition probabilities must be non-negative.",
    "mat_square": "A transition probability matrix must be square, with dimensions S×S.",
    "mat_stoch": "Each row of a transition probability matrix must sum to one (1).",
    "obj_shape": "Object arrays for transition probabilities and rewards must have only 1 dimension: the number of actions A. Each element of the object array contains an SxS ndarray or matrix.",
    "obj_square": "Each element of an object array for transition probabilities and rewards must contain an SxS ndarray or matrix; i.e. P[a].shape = (S, S) or R[a].shape = (S, S).",
    "P_type": "The transition probabilities must be in a numpy array; i.e. type(P) is ndarray.",
    "P_shape": "The transition probability array must have the shape (A, S, S) with S : number of states greater than 0 and A : number of actions greater than 0. i.e. R.shape = (A, S, S)",
    "PR_incompat": "Incompatibility between P and R dimensions.",
    "R_type": "The rewards must be in a numpy array; i.e. type(R) is ndarray, or numpy matrix; i.e. type(R) is matrix.",
    "R_shape": "The reward matrix R must be an array of shape (A, S, S) or (S, A) with S : number of states greater than 0 and A : number of actions greater than 0. i.e. R.shape = (S, A) or (A, S, S)."
}

def _checkDimensionsListLike(arrays):
    """Check that each array in a list of arrays has the same size."""
    dim1 = len(arrays)
    dim2, dim3 = arrays[0].shape

    def check_shape(_):
        for aa in range(1, dim1):
            dim2_aa, dim3_aa = arrays[aa].shape
            if (dim2_aa != dim2) or (dim3_aa != dim3):
                debug.callback(lambda: print(_MDPERR["obj_square"]))
                return False
        return True

    if not check_shape(None):
        return 0, 0, 0

    return dim1, dim2, dim3

def _checkRewardsListLike(reward, n_actions, n_states):
    """Check that a list-like reward input is valid."""
    def check_lenR(_):
        lenR = len(reward)
        def check_actions(_):
            return _checkDimensionsListLike(reward)
        def check_states(_):
            return n_actions, n_states, n_states
        return jax.lax.cond(lenR == n_actions, check_actions, check_states, operand=None)
    
    def handle_error(_):
        debug.callback(lambda: print(_MDPERR["R_shape"]))
        return 0, 0, 0

    # return debug.callback(
    #     lambda: check_lenR(None) if hasattr(reward, '__len__') else handle_error(None)
    # )
    return jax.lax.cond(hasattr(reward, '__len__'), check_lenR, handle_error, operand=None)

def isSquare(matrix):
    """Check that ``matrix`` is square."""
    def check_shape(_):
        dim1, dim2 = matrix.shape
        return dim1 == dim2

    def handle_error(_):
        matrix_ = jnp.array(matrix)
        dim1, dim2 = matrix.shape
        return dim1 == dim2

    # return debug.callback(
    #     lambda: check_shape(None) if hasattr(matrix, 'shape') else handle_error(None)
    # )
    return jax.lax.cond(hasattr(matrix, 'shape'), check_shape, handle_error, operand=None)

def isStochastic(matrix):
    """Check that ``matrix`` is row stochastic."""
    def check_stochastic(_):
        absdiff = jnp.abs(matrix.sum(axis=1) - jnp.ones(matrix.shape[0]))
        return absdiff.max() <= 10 * jnp.finfo(matrix.dtype).eps

    def handle_error(_):
        matrix_ = jnp.array(matrix)
        absdiff = jnp.abs(matrix_.sum(axis=1) - jnp.ones(matrix_.shape[0]))
        return absdiff.max() <= 10 * jnp.finfo(matrix.dtype).eps

    # return debug.callback(
    #     lambda: check_stochastic(None) if hasattr(matrix, 'sum') else handle_error(None)
    # )
    return jax.lax.cond(hasattr(matrix, 'sum'), check_stochastic, handle_error, operand=None)

def isNonNegative(matrix):
    """Check that ``matrix`` is row non-negative."""
    def check_non_negative(_):
        return (matrix >= 0).all()

    def handle_error(_):
        matrix_ = jnp.array(matrix)
        return (matrix >= 0).all()

    # return debug.callback(
    #     lambda: check_non_negative(None) if hasattr(matrix, 'all') else handle_error(None)
    # )
    return jax.lax.cond(hasattr(matrix, 'all'), check_non_negative, handle_error, operand=None)

def checkSquareStochastic(matrix):
    """Check if ``matrix`` is a square and row-stochastic."""
    def check_square_stochastic(_):
        if not isSquare(matrix):
            debug.callback(lambda: print(_MDPERR["mat_square"]))
            return False
        if not isStochastic(matrix):
            debug.callback(lambda: print(_MDPERR["mat_stoch"]))
            return False
        if not isNonNegative(matrix):
            debug.callback(lambda: print(_MDPERR["mat_nonneg"]))
            return False
        return True

    return check_square_stochastic(None)

def check(P, R):
    """Check if ``P`` and ``R`` define a valid Markov Decision Process (MDP)."""
    def check_P(_):
        if P.ndim == 3:
            aP, sP0, sP1 = P.shape
        elif P.ndim == 1:
            aP, sP0, sP1 = _checkDimensionsListLike(P)
        else:
            debug.callback(lambda: print(_MDPERR["P_shape"]))
            return 0, 0, 0
        return aP, sP0, sP1

    def handle_P_error(_):
        aP, sP0, sP1 = _checkDimensionsListLike(P)
        return aP, sP0, sP1

    # aP, sP0, sP1 = debug.callback(
    #     lambda: check_P(None) if hasattr(P, 'ndim') else handle_P_error(None)
    # )
    aP, sP0, sP1 = jax.lax.cond(hasattr(P, 'ndim'), check_P, handle_P_error, operand=None)

    def check_msg(aP, sP0):
        msg = ""
        if aP <= 0:
            msg = "The number of actions in P must be greater than 0."
        elif sP0 <= 0:
            msg = "The number of states in P must be greater than 0."
        if msg:
            debug.callback(lambda: print(msg))
            return False
        return True

    if not check_msg(aP, sP0):
        return

    def check_R(_):
        ndimR = R.ndim
        if ndimR == 1:
            aR, sR0, sR1 = _checkRewardsListLike(R, aP, sP0)
        elif ndimR == 2:
            sR0, aR = R.shape
            sR1 = sR0
        elif ndimR == 3:
            aR, sR0, sR1 = R.shape
        else:
            debug.callback(lambda: print(_MDPERR["R_shape"]))
            return 0, 0, 0
        return aR, sR0, sR1

    def handle_R_error(_):
        aR, sR0, sR1 = _checkRewardsListLike(R, aP, sP0)
        return aR, sR0, sR1

    # aR, sR0, sR1 = debug.callback(
    #     lambda: check_R(None) if hasattr(R, 'ndim') else handle_R_error(None)
    # )
    aR, sR0, sR1 = jax.lax.cond(hasattr(R, 'ndim'), check_R, handle_R_error, operand=None)

    def check_msg_R(aR, sR0, sR1):
        msg = ""
        if sR0 <= 0:
            msg = "The number of states in R must be greater than 0."
        elif aR <= 0:
            msg = "The number of actions in R must be greater than 0."
        elif sR0 != sR1:
            msg = "The matrix R must be square with respect to states."
        elif sP0 != sR0:
            msg = "The number of states must agree in P and R."
        elif aP != aR:
            msg = "The number of actions must agree in P and R."
        if msg:
            debug.callback(lambda: print(msg))
            return False
        return True

    if not check_msg_R(aR, sR0, sR1):
        return

    # Check that the P's are square, stochastic and non-negative
    for aa in range(aP):
        checkSquareStochastic(P[aa])
        
def getSpan(array):
    """Return the span of `array`

    span(array) = max array(s) - min array(s)

    """
    return array.max() - array.min()