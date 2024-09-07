from __future__ import division
import numpy as np
from scipy.linalg import toeplitz, expm
from scipy.stats import norm, binom
from math import sqrt
from scipy.integrate import quad
from scipy import optimize

def kronn(*args):
    """
    returns multidimensional kronecker product of all matrices in the argument list
    """
    z = args[0]
    for i in range(1, len(args)):
        z = np.kron(z, args[i])
    return z


def beliefTransitionMatrix(p_appear, p_disappear, nq, w):
    """
    create transition matrix between nq belief states q to q' without color observation
    diffusion is added
    """
    Tqqq = np.zeros((nq, nq))
    dq = 1 / nq
    a = 1 - p_disappear - p_appear

    for i in range(nq):
        for j in range(nq):
            q = i * dq
            qq = j * dq

            bm = (qq - p_appear) / a
            bp = (qq + dq - p_appear) / a

            Tqqq[j, i] = max(0, min(q + dq, bp) - max(q, bm) )
            Tqqq[j, i] = Tqqq[j, i] / (bp - bm) * sqrt(dq ** 2 + (bp - bm) ** 2)
    Tqqq = Tqqq / np.tile(np.sum(Tqqq, 0), (nq, 1))

    nt = 20
    d = w / nt
    dD = toeplitz(np.insert(np.zeros(nq - 2), 0, np.array([-2 * d, d])))
    dD[1, 0] = 2 * d
    dD[-2, -1] = 2 * d
    D = expm(dD * nt)
    D = D / np.tile(np.sum(D, 0), (nq, 1))

    Tqqq = np.dot(D, Tqqq)

    return Tqqq

def beliefTransitionMatrixGaussianCazettes(p_rwd, p_sw, nq, actions, sigma):
    """
    Create transition matrix between nq belief states q to q' WITH action-dependent observations
    AND action-dependent state transitions.
    Use Gaussian approximation for diffusion

    Note: convention is site is active in state 0. and inactive in state 1.

    """
    def gb(x, k1, k0, p_rwd, p_sw, action):
        #the belief by convention is of whether the box is ACTIVE, 
        #1 - x is the belief that the box is INACTIVE
        if action == 4:  # push button action
            p_off = 1.  # Probability of staying off is 1
            p_on = 1 - p_sw  # Probability of staying on is 1 - p_sw
            p_sw_on_off = p_sw  # Probability of switching from on to off
        else:
            p_off = 1.
            p_on = 1.
            p_sw_on_off = 0.0
        
        #calculate the new belief b_{t+1}(s_{t+1}) = (1/c) * P(o_{t+1} | s_{t+1}, a_t) * bhat_{t+1}(s_{t+1})
        #where bhat_{t+1}(s_{t+1}) = \sum_{s_t} P(s_{t+1} | s_t, a_t) * b_t(s_t)

        # the probability of the next observation given 
        # that the box is active at that next time step
        Pot1 = k0
        # P(s_{t+1}=0 | s_t=0, a_t) * b_t(s_t=1) 
        # the probability of the box being active at the next time step given 
        # that it is active at the current time step, times the belief that the box is active
        Pst0 = p_on * x
        # P(s_{t+1}=1 | s_t=0, a_t) * b_t(s_t=0) 
        # the probability of the box being inactive at the next time step given 
        # that it is active at the current time step times the belief that the box is inactive
        Pst1 = p_sw_on_off * (1 - x)

        numerator = Pot1 * (Pst0 + Pst1)# the numerator of the belief update equation

        #now calculate the normalization constant..
        # \frac{1}{\sum_{s_{t+1}} P(o_{t+1} | s_{t+1}, a_t) * \sum_{s_t} P(s_{t+1} | s_t, a_t) * b_t(s_t)}
        # the P(o_{t+1} | s_{t+1}=0, a_t) * P(s_{t+1}=0 | s_t=1, a_t) * b_t(s_t=1)} is 
        # omitted because the probability of the box switching from off to on is 0.
        c = k0 * p_on * x + k1 * p_sw_on_off * x + k1 * p_off * (1 - x)

        bst1 = numerator / c# the new belief state at the next time step
        return bst1

    mu = 0
    dq = 1 / nq # belief resolution
    Ncol = 1  # max color value.. [0, Ncol]

    d = np.zeros((len(actions), Ncol + 1, nq, nq))  # distance between q and q' for each action
    den = {action: np.zeros((Ncol + 1, nq, nq)) for action in actions}  # density of the distance
    xopt = np.zeros((len(actions), Ncol + 1, nq, nq))  # optimal x for each action
    height = np.zeros((len(actions), Ncol + 1, nq, nq))  # height of the density
    Trans_belief_obs_approx = {action: np.zeros((Ncol + 1, nq, nq)) for action in actions}  # approximate belief transition matrix

    # Define action-dependent observation emission matrices
    Obs_emis = {action: np.zeros((Ncol + 1, 2)) for action in actions}

    # Fill Obs_emis based on location-specific actions
    # by convention obs_emis[action] is NCol x N states. so the probability of each possible observation given the state
    for action in actions:
        if action == 4:  # push button
            Obs_emis[action][1, 0] = 1 - p_rwd# Probability of observing 1 (i.e. box is OFF) when box is actually ON
            Obs_emis[action][1, 1] = 1.0# Probability of observing 1 (i.e. box is OFF) when box is indeed OFF
            Obs_emis[action][0, 0] = p_rwd# Probability of observing 0 (i.e. box is ON) when box is indeed ON
            Obs_emis[action][0, 1] = 0.0# Probability of observing 0 (i.e. box is ON) when box is actually OFF
        else:
            Obs_emis[action][:, :] = 0.0  # No observations for other actions

    # Define transition probabilities for states.. but now it's action-dependent
    # so state transitions only probabilistically happen based on what action is taken
    Trans_state = {}
    for action in actions:
        if action == 4:  # push button action
            Trans_state[action] = np.array([[1 - p_sw, 0.0],  # Probability of staying on/off or switching
                                            [p_sw, 1.0]])
        else:
            Trans_state[action] = np.array([[1, 0],  # Identity matrix (no transition)
                                            [0, 1]])
    # for each action and next observation pair the goal is to project or translate the 
    # joint probability of observations and state transitions into a discrete space of belief states.
    for a_index, action in enumerate(actions):#for each possible current action
        if action == 4:  # push button action
            for n in range(Ncol + 1):# for each possible next observation
                k0 = Obs_emis[action][n, 0]# probability of observing n given that the the box is active 
                k1 = Obs_emis[action][n, 1]# probability of observing n given that the the box is inactive
                for i in range(nq):
                    for j in range(nq):
                        # Approximate the probability with Gaussian approximation
                        q = i * dq + dq / 2  # past belief (along columns, index with i)
                        qq = j * dq + dq / 2  # new belief (along rows, index with j)

                        def dist(x):
                            # the distance of (x, gb(x)) to the center of each bin
                            return sqrt((q - x) ** 2 + (qq - gb(x, k1, k0, p_rwd, p_sw, action)) ** 2)

                        xopt[a_index, n, j, i], d[a_index, n, j, i] = optimize.fminbound(dist, 0, 1, full_output=1)[0:2]
                        den[action][n, j, i] = norm.pdf(d[a_index, n, j, i], mu, sigma)  # use this to approximate delta function with diffusion

                        height[a_index, n, j, i] = Obs_emis[action][n, :].dot(Trans_state[action]).dot(np.array([1 - q, q]))

                den[action][n, :, :] /= np.tile(np.sum(den[action][n, :, :], 0), (nq, 1))
                Trans_belief_obs_approx[action][n, :, :] = np.multiply(den[action][n, :, :], height[a_index, n, :, :])
        else:
            # for all other actions, the observation is always 0
            # so optimizing x is not necessary, the belief states deterministically
            # transition to the same belief state.
            for n in range(Ncol + 1):
                den[action][n, :, :] = np.identity(nq)
                Trans_belief_obs_approx[action][n, :, :] = np.identity(nq)

    return Trans_belief_obs_approx, {action: Obs_emis[action].dot(Trans_state[action]) for action in actions}, den


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
    Obs_emis[:, 0] = binom.pmf(range(Ncol + 1), Ncol, qmax)# Probabilities when the box is active
    Obs_emis[:, 1] = binom.pmf(range(Ncol + 1), Ncol, qmin)# Probabilities when the box is inactive

    dq = 1 / nq

    d = np.zeros((Ncol + 1, nq, nq))# distance between q and q' for each color
    den = np.zeros((Ncol + 1, nq, nq))# density of the distance
    xopt = np.zeros((Ncol + 1, nq, nq))# optimal x for each color
    height = np.zeros((Ncol + 1, nq, nq))# height of the density
    Trans_belief_obs_approx = np.zeros((Ncol + 1, nq, nq))# approximate belief transition matrix

    for n in range(Ncol + 1):
        k0 = Obs_emis[n, 0]# probability of observing the box is active
        k1 = Obs_emis[n, 1]# probability of observing the box is inactive

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
    C = np.empty((ra * rb, ca * cb))

    for i in range(ra):
        for j in range(ca):
            C[i*rb : (i+1)*rb, j*cb : (j+1)*cb] = A[i, j] + B

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

