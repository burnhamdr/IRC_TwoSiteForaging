from __future__ import division
from MDPclassJax import *
from jax.scipy.linalg import block_diag
from boxtask_funcJax import *
from HMMtwoboxCazettesJax import *
from itertools import permutations

import jax.numpy as jnp
import jax
from jax import vmap, pmap, grad, jit
import jax.lax as lax
from tensorflow_probability.substrates import jax as tfp

jax.config.update("jax_enable_x64", True)

import numpy as np

a0 = 0    # a0 = do nothing
g0 = 1    # g0 = go (i.e. travel/switch sites via location 0)
pb = 2    # pb  = push button

# --- HELPER FUNCTIONS ---

def multinomial(key, n, p, shape=()):
    return tfp.distributions.Multinomial(n, probs=p).sample(
        seed=key,
        sample_shape=shape,
    )

def create_ordered_tuples(K):
    all_combinations = list(permutations(range(K), 2))
    sorted_combinations = sorted(all_combinations, key=lambda x: (x[1], x[0]))
    filtered_combinations = [tup for tup in sorted_combinations if 0 in tup]
    return filtered_combinations

def _solveMDP_op_pure(ThA, R, discount, epsilon, niterations, initial_value, mask):
    vi = ValueIteration_opZW(ThA, R, discount, epsilon, niterations, initial_value, mask)
    vi.run()
    def compute_Q_action(r, p):
        return r + vi.discount * jnp.dot(p, vi.V)
    Q = jax.vmap(compute_Q_action)(vi.R, vi.P)
    return Q, vi.policy, vi.V

def _solveMDP_sfm_pure(ThA, R, discount, epsilon, niterations, initial_value, mask, temperature):
    vi = ValueIteration_sfmZW(ThA, R, discount, epsilon, niterations, initial_value, mask)
    vi.run(temperature)
    def compute_Q_action(r, p):
        return r + vi.discount * jnp.dot(p, vi.V)
    Q = jax.vmap(compute_Q_action)(vi.R, vi.P)
    return Q, vi.softpolicy, vi.V

def _flatten_den_dict(den_dict, nl, nq, nr):
    # Flatten nested dictionary den[loc][box][action] -> tensor(nl, nl, 3, nr, nq, nq)
    # Dimensions: [current_loc, box_idx, action, reward, next_belief, prev_belief]
    den_array = jnp.zeros((nl, 3, 3, nr, nq, nq)) 
    
    for loc in den_dict: # current location
        for box in den_dict[loc]: # box we are updating belief about
            for act in den_dict[loc][box]:
                val = den_dict[loc][box][act]
                val_stacked = jnp.stack(val) if isinstance(val, (list, tuple)) else val
                den_array = den_array.at[loc, box, act].set(val_stacked)
                
    return den_array

# --- JIT-COMPILED SIMULATION KERNEL ---
def _generate_single_trial(key, parametersExp, softpolicy, 
                          nq, nr, nl, nlwh, nlpbh, 
                          den_tensor, Tl, locations_with_history, sampleTime, lick_state):
    
    psw_e = parametersExp[0]
    prwd_e = parametersExp[1]

    k0, k_scan = jax.random.split(key)
    
    # --- Initialization (t=0) ---
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(k0, 7)
    
    # Cast all integers to int32 for JAX loop consistency
    # belief1 = jax.random.randint(k1, shape=(), minval=0, maxval=nq).astype(jnp.int32)
    # belief2 = jax.random.randint(k2, shape=(), minval=0, maxval=nq).astype(jnp.int32)
    belief1 = jax.random.randint(k1, shape=(), minval=0, maxval=nq, dtype=jnp.int32)
    belief2 = (nq - 1 - belief1).astype(jnp.int32)

    rew = jax.random.randint(k3, shape=(), minval=0, maxval=nr).astype(jnp.int32)
    
    loc_ind = jax.random.randint(k4, shape=(), minval=0, maxval=nlwh).astype(jnp.int32)
    loc = locations_with_history[loc_ind, 1].astype(jnp.int32)
    prev_loc = locations_with_history[loc_ind, 0].astype(jnp.int32)
    
    active_site = jax.random.randint(
                                        k5, shape=(), minval=1, maxval=3, dtype=jnp.int32
                                    )
    true1 = jnp.where(active_site == 1, 1, 0).astype(jnp.int32)
    true2 = jnp.where(active_site == 1, 0, 1).astype(jnp.int32)
    
    # Hybrid State Helper
    def get_hybrid(li, b1, b2, r, act_prev_is_pb):
        if lick_state:
            pb_ind = jnp.where(act_prev_is_pb, 1, 0)
            return (li * (nq * nr * nq * 2) + 
                    pb_ind * (nq * nr * nq) + 
                    b1 * (nr * nq) + 
                    r * nq + b2).astype(jnp.int32)
        else:
            return (li * (nq * nr * nq) + 
                    b1 * (nr * nq) + 
                    r * nq + b2).astype(jnp.int32)

    h_init = get_hybrid(loc_ind, belief1, belief2, rew, False)
    
    action_dist_init = softpolicy.T[h_init]
    action = jnp.argmax(multinomial(k6, 1, action_dist_init, shape=())).astype(jnp.int32)

    # Initial distributions
    b1_dist_init = jnp.eye(nq)[belief1]
    b2_dist_init = jnp.eye(nq)[belief2]

    # --- Scan Step Function ---
    def step_fn(carry, _):
        (rng, c_loc_ind, c_loc, c_prev_loc,
        c_bel1, c_bel2, c_rew,
        c_active_site, c_last_action) = carry

        # derive current true states from active site (exclusive by construction)
        c_true1 = (c_active_site == 1).astype(jnp.int32)
        c_true2 = (c_active_site == 2).astype(jnp.int32)

        # advance RNG properly
        k_next, k_step, k_act, k_env, k_bel1, k_bel2 = jax.random.split(rng, 6)

        # -------------------------
        # 1) Reward logic (PB only)
        # -------------------------
        r_not_pb = jnp.array(0, dtype=jnp.int32)

        def get_pb_reward(curr_loc, active_site, key):
            # reward only possible at boxes (1 or 2) and only if that box is active
            p = jnp.where(curr_loc == active_site, prwd_e, 0.0)
            return jax.random.bernoulli(key, p).astype(jnp.int32)

        is_pb = (c_last_action == pb)
        r_pb = get_pb_reward(c_loc, c_active_site, k_act)
        n_r = jnp.where(is_pb, r_pb, r_not_pb)

        # ---------------------------------------------
        # 2) Dependent world-state update (Python parity)
        # ---------------------------------------------
        # Switch only if:
        #   - action was PB
        #   - you are at an actual box (loc != 0)
        #   - and that box is currently the active one
        is_box = (c_loc != 0)
        pb_at_active = is_pb & is_box & (c_loc == c_active_site)

        def maybe_switch_active_site(key):
            # if switch happens: toggle 1<->2, else stay
            do_switch = jax.random.bernoulli(key, psw_e)
            switched = (3 - c_active_site)  # 1->2, 2->1
            return jnp.where(do_switch, switched, c_active_site).astype(jnp.int32)

        n_active_site = lax.cond(
            pb_at_active,
            lambda k: maybe_switch_active_site(k),
            lambda k: c_active_site.astype(jnp.int32),
            k_env
        )

        n_true1 = (n_active_site == 1).astype(jnp.int32)
        n_true2 = (n_active_site == 2).astype(jnp.int32)

        # -------------------------
        # 3) Belief update (PB only)
        # -------------------------
        # Your “strict parity” rule matches Python:
        # belief updates only when PB and loc != 0
        do_belief_update = is_pb & is_box

        safe_loc = jnp.maximum(c_loc, 1)  # avoid indexing loc=0

        dist_b1_raw = den_tensor[safe_loc, 1, c_last_action, n_r, :, c_bel1]
        dist_b2_raw = den_tensor[safe_loc, 2, c_last_action, n_r, :, c_bel2]

        n_b1_sampled = jnp.argmax(multinomial(k_bel1, 1, dist_b1_raw, shape=())).astype(jnp.int32)
        n_b2_sampled = jnp.argmax(multinomial(k_bel2, 1, dist_b2_raw, shape=())).astype(jnp.int32)

        n_b1 = jnp.where(do_belief_update, n_b1_sampled, c_bel1)
        n_b2 = jnp.where(do_belief_update, n_b2_sampled, c_bel2)

        dist_b1 = jnp.where(do_belief_update, dist_b1_raw, jnp.eye(nq)[c_bel1])
        dist_b2 = jnp.where(do_belief_update, dist_b2_raw, jnp.eye(nq)[c_bel2])

        # -------------------------
        # 4) Location update (g0 only)
        # -------------------------
        def act_go0(v, k):
            probs = Tl[:, v]
            new_ind = jnp.argmax(multinomial(k, 1, probs, shape=())).astype(jnp.int32)
            new_l = locations_with_history[new_ind, 1].astype(jnp.int32)
            new_p = locations_with_history[new_ind, 0].astype(jnp.int32)
            return new_ind, new_l, new_p

        do_move = (c_last_action == g0)
        n_lind, n_lloc, n_lprev = lax.cond(
            do_move,
            lambda x: act_go0(x[0], x[3]),
            lambda x: (x[0], x[1], x[2]),
            (c_loc_ind, c_loc, c_prev_loc, k_act)
        )

        # -------------------------
        # 5) Choose next action
        # -------------------------
        act_is_pb = is_pb
        h_idx = get_hybrid(n_lind, n_b1, n_b2, n_r, act_is_pb)

        probs_act = softpolicy.T[h_idx]
        n_act = jnp.argmax(multinomial(k_step, 1, probs_act, shape=())).astype(jnp.int32)

        # Carry forward: NOTE active_site, not (true1,true2)
        new_carry = (k_next, n_lind, n_lloc, n_lprev,
                    n_b1, n_b2, n_r,
                    n_active_site, n_act)

        output = (n_act, n_r, n_lloc, n_true1, n_true2,
                n_b1, n_b2, probs_act, n_lind, dist_b1, dist_b2, h_idx)
        
        return new_carry, output

    # Run Scan
     
    init_carry = (k_scan, loc_ind, loc, prev_loc, belief1, belief2, rew, active_site, action)
    final_carry, stack_out = lax.scan(step_fn, init_carry, None, length=sampleTime-1)
    
    # Unpack stack
    s_act, s_rew, s_loc, s_t1, s_t2, s_b1, s_b2, s_adist, s_lind, s_b1dist, s_b2dist, s_h = stack_out
    
    # Concatenate results
    full_act = jnp.concatenate([action[None], s_act])
    full_rew = jnp.concatenate([rew[None], s_rew])
    full_loc = jnp.concatenate([loc[None], s_loc])
    full_t1 = jnp.concatenate([true1[None], s_t1])
    full_t2 = jnp.concatenate([true2[None], s_t2])
    full_b1 = jnp.concatenate([belief1[None], s_b1])
    full_b2 = jnp.concatenate([belief2[None], s_b2])
    full_lind = jnp.concatenate([loc_ind[None], s_lind])
    full_adist = jnp.concatenate([action_dist_init[None, :], s_adist], axis=0)
    full_b1dist = jnp.concatenate([b1_dist_init[None, :], s_b1dist], axis=0)
    full_b2dist = jnp.concatenate([b2_dist_init[None, :], s_b2dist], axis=0)
    full_hybrid = jnp.concatenate([h_init[None], s_h], axis=0)

    return (full_act, full_rew, full_loc, full_t1, full_t2, full_b1, full_b2, full_adist, full_lind, full_b1dist, full_b2dist, full_hybrid)


class twoboxCazettesMDP:
    def __init__(self, discount, nq, nr, na, nl, parameters, lick_state=False):
        self.lick_state = lick_state
        self.discount = discount
        self.nq = nq
        self.nr = nr
        self.na = na
        self.nl = nl
        self.nlwh = 2*(nl - 1)
        self.nlpbh = 2

        self.n = lax.cond(
            lick_state,
            lambda _: self.nq * self.nq * self.nr * self.nlwh * self.nlpbh,
            lambda _: self.nq * self.nq * self.nr * self.nlwh,
            operand=None
        )
        
        self.parameters = parameters
        self.ThA = jnp.zeros((self.na, self.n, self.n))
        self.R = jnp.zeros((self.na, self.n, self.n))
        
        # Loc 0 mask
        shape = (self.na, self.nq*self.nq*self.nr*self.nlwh*self.nlpbh)
        slice_size = 2 * self.nq * self.nr * self.nq * self.nlpbh
        loc0_mask_jax = jnp.ones(shape)
        update_slice = jnp.zeros((1, slice_size))
        self.loc0_mask = jax.lax.dynamic_update_slice(loc0_mask_jax, update_slice, (pb, 0))

    def _states_wLickStates(self, r_, l_, a_):
        temp = jnp.reshape(np.array(range(self.nq)), [1, self.nq])
        return jnp.squeeze(l_ * self.nq * self.nq * self.nr * self.nlpbh + a_ * self.nq * self.nq * self.nr + 
                            tensorsum(temp * self.nr * self.nq, r_ * self.nq + temp)).astype(int)

    def setupMDP(self):
        rho = 1
        psw = self.parameters[0]
        prwd = self.parameters[1]
        Groom = self.parameters[2]
        travelCost = self.parameters[3]
        pushButtonCost = self.parameters[4]
        startPushButtonCost = self.parameters[5]
        stopPushButtonCost = 0.0
        portRestReward = self.parameters[6]

        actions = jnp.array([a0, g0, pb])
        locations = jnp.arange(self.nl)
        locations_with_history = create_ordered_tuples(self.nl)
        pb_history = create_ordered_tuples(2)

        NumCol = 2
        r_val = 1
        doNothingCost = 0.0
        beta = 0
        delta = 0
        direct = 0

        Tr = jnp.array([[1, rho], [0, 1 - rho]]).astype(jnp.float32)
        Trpb = jnp.array([[0.5,0.5], [0.5, 0.5]])

        self.Trans_belief_obs, self.Obs_emis_trans, self.den = beliefTransitionMatrixGaussianCazettes(psw, prwd, self.nq, actions, locations, sigma = 1 / self.nq / 3)
        self.Obs_emis_trans1 = self.Obs_emis_trans[1]
        self.Obs_emis_trans2 = self.Obs_emis_trans[2]
        belief_locations = jnp.array([1, 2])
        Tb_unpack = {bloc.item():{oloc.item():{} for oloc in belief_locations} for bloc in belief_locations}
        for bloc in belief_locations:
            bloc = bloc.item()
            for oloc in belief_locations:
                oloc = oloc.item()
                for action in actions:
                    action = action.item()
                    Trans_belief = jnp.sum(self.Trans_belief_obs[bloc][oloc][action], axis=0)
                    Tb_unpack[bloc][oloc][action] = Trans_belief / jnp.tile(jnp.sum(Trans_belief, 0), (self.nq, 1))

        Tl = jnp.array([[delta, 0., 1. - delta, 0.], 
                       [0., delta, 0., 1. - delta],
                       [0., 1. - delta, delta, 0.],
                       [1. - delta, 0., 0., delta]]).astype(jnp.float32)
        self.Tl = Tl 

        def true_fn_trans(_):
            Tpb0 = jnp.array([[1, 1], [0, 0]]).astype(jnp.float32)
            Tla0 = kronn(jnp.identity(self.nlwh), Tpb0)
            Tlg0 = kronn(Tl, Tpb0)
            Tlpb = kronn(Tpb0, jnp.identity(self.nq), Tr, jnp.identity(self.nq))
            return Tla0, Tlg0, Tlpb, Tla0.shape, Tlg0.shape, Tlpb.shape

        def false_fn_trans(_):
            Tla0 = jnp.identity(self.nlwh)
            Tlg0 = Tl
            Tlpb = kronn(jnp.identity(self.nq), Tr, jnp.identity(self.nq))
            Tla0_padded = jnp.pad(Tla0, ((0, Tla0.shape[0]), (0, Tla0.shape[1])), mode='constant')
            Tlg0_padded = jnp.pad(Tlg0, ((0, Tlg0.shape[0]), (0, Tlg0.shape[1])), mode='constant')
            Tlpb_padded = jnp.pad(Tlpb, ((0, Tlpb.shape[0]), (0, Tlpb.shape[1])), mode='constant')
            return Tla0_padded, Tlg0_padded, Tlpb_padded, Tla0.shape, Tlg0.shape, Tlpb.shape

        Tla0, Tlg0, Tlpb, Tla0_shape, Tlg0_shape, Tlpb_shape = lax.cond(
            self.lick_state, true_fn_trans, false_fn_trans, operand=None
        )

        self.Tla0 = lax.dynamic_slice(Tla0, (0, 0), (Tla0_shape[0], Tla0_shape[0]))
        self.Tlg0 = lax.dynamic_slice(Tlg0, (0, 0), (Tlg0_shape[0], Tlg0_shape[0]))
        self.Tlpb = lax.dynamic_slice(Tlpb, (0, 0), (Tlpb_shape[0], Tlpb_shape[0]))

        Tb = {bloc.item():{} for bloc in belief_locations}
        for action in actions:
            action = action.item()
            for bloc in belief_locations:
                bloc = bloc.item()
                Tb_temp_n0 = self.Trans_belief_obs[bloc][bloc][action][0]
                Tb_temp_n1 = self.Trans_belief_obs[bloc][bloc][action][1]
                Tb_block_temp = jnp.block([[Tb_temp_n0, Tb_temp_n0], [Tb_temp_n1, Tb_temp_n1]]).astype(jnp.float32)
                Tpb1 = jnp.array([[0, 0], [1, 1]]).astype(jnp.float32)
                oloc = int((2 - bloc) + 1)
                
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
        
        self.ThA = self.ThA.at[a0, :, :].set(jnp.transpose(kronn(self.Tla0, jnp.identity(self.nq), Tr, jnp.identity(self.nq))))
        self.ThA = self.ThA.at[g0, :, :].set(jnp.transpose(kronn(self.Tlg0, jnp.identity(self.nq), Tr, jnp.identity(self.nq))))
        Th = block_diag(self.Tlpb, self.Tlpb, Tb[1][pb], Tb[2][pb]).astype(jnp.float64)
        
        column_sums = jnp.sum(Th, axis=0)
        rep_col_sums = jnp.repeat(column_sums[jnp.newaxis,:], self.n, axis=0)
        safe_div = jnp.where(rep_col_sums == 0, 1.0, rep_col_sums)
        Th_norm = Th / safe_div
        self.ThA = self.ThA.at[pb, :, :].set(jnp.transpose(Th_norm))

        loc_rewards = jnp.array([[Groom, Groom, 0, 0]])
        def true_fn_reward(_):
            Reward_h = tensorsumm(loc_rewards, jnp.zeros((1, self.nlpbh)), jnp.zeros((1, self.nq)), jnp.array([[0, r_val]]), jnp.zeros((1, self.nq)))
            Reward_a = - jnp.array([doNothingCost, travelCost, pushButtonCost])
            loc_rewards_pb = np.array([[0, 0, 0, 0]])
            Reward_h_pb = tensorsumm(loc_rewards_pb, jnp.zeros((1, self.nlpbh)), jnp.zeros((1, self.nq)), jnp.array([[0, r_val]]), jnp.zeros((1, self.nq)))
            [R1, R2, R3] = jnp.meshgrid(jnp.transpose(Reward_a), jnp.squeeze(Reward_h), jnp.squeeze(Reward_h), indexing='ij')
            Reward = R1 + R3
            [R1, R2, R3] = jnp.meshgrid(jnp.transpose(Reward_a), jnp.squeeze(Reward_h_pb), jnp.squeeze(Reward_h_pb), indexing='ij')
            Reward_pb = R1 + R3
            Reward = Reward.at[pb].set(Reward_pb[pb])
            
            r0_l2_a0 = self._states_wLickStates(0, 2, 0)
            r0_l3_a0 = self._states_wLickStates(0, 3, 0)
            r1_l2_a0 = self._states_wLickStates(1, 2, 0)
            r1_l3_a0 = self._states_wLickStates(1, 3, 0)
            lick_state0 = jnp.concatenate((r0_l2_a0, r0_l3_a0, r1_l2_a0, r1_l3_a0))
            lick_state0 = jnp.unique(lick_state0, size=len(lick_state0))
            
            r0_l2_a1 = self._states_wLickStates(0, 2, 1)
            r0_l3_a1 = self._states_wLickStates(0, 3, 1)
            r1_l2_a1 = self._states_wLickStates(1, 2, 1)
            r1_l3_a1 = self._states_wLickStates(1, 3, 1)
            lick_state1 = jnp.concatenate((r0_l2_a1, r0_l3_a1, r1_l2_a1, r1_l3_a1))
            lick_state1 = jnp.unique(lick_state1, size=len(lick_state1))
            
            row_inds, col_inds = jnp.ix_(lick_state0, lick_state1)
            Reward = Reward.at[pb, row_inds, col_inds].add(-startPushButtonCost)
            row_inds, col_inds = jnp.ix_(lick_state1, lick_state0)
            Reward = Reward.at[a0, row_inds, col_inds].add(-stopPushButtonCost)
            Reward = Reward.at[g0, row_inds, col_inds].add(-stopPushButtonCost)
            Reward = Reward.at[a0, :, 2 * self.nq * self.nq * self.nr * self.nlpbh:].add(portRestReward)
            return Reward, Reward.shape

        def false_fn_reward(_):
            Reward_h = tensorsumm(loc_rewards, jnp.zeros((1, self.nq)), jnp.array([[0, r_val]]), jnp.zeros((1, self.nq)))
            Reward_a = - jnp.array([doNothingCost, travelCost, pushButtonCost])
            [R1, R2, R3] = jnp.meshgrid(jnp.transpose(Reward_a), jnp.squeeze(Reward_h), jnp.squeeze(Reward_h), indexing='ij')
            Reward = R1 + R3
            Reward = Reward.at[pb, :, 2*self.nq * self.nr * self.nq:].add(-pushButtonCost)
            Reward_padded = jnp.pad(Reward, ((0, 0), (0, Reward.shape[1]), (0, Reward.shape[2])), mode='constant')
            return Reward_padded, Reward.shape

        result, result_shape = lax.cond(self.lick_state, true_fn_reward, false_fn_reward, operand=None)
        self.R = lax.dynamic_slice(result, (0, 0, 0), (result_shape[0], result_shape[1], result_shape[2]))

        den_col = {bloc.item():{} for bloc in belief_locations}
        for action in actions:
            action = action.item()
            for bloc in belief_locations:
                bloc = bloc.item()
                obs_dict = {}
                for i in range(NumCol):
                    j = i
                    den_loc = self.den[bloc]
                    den_temp_n0 =  self.den[bloc][bloc][action][0]
                    den_temp_n1 = self.den[bloc][bloc][action][1]
                    den_block_temp = jnp.block([[den_temp_n0, den_temp_n0], [den_temp_n1, den_temp_n1]])
                    oloc = int((2 - bloc) + 1)
                    
                    def true_fn_bloc1(_):
                        def true_fn_lick_state(_):
                            out = kronn(Tpb1, den_block_temp, den_loc[oloc][action][j])
                            return out, out.shape
                        def false_fn_lick_state(_):
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

        self.Trans_hybrid_obs12 = jnp.zeros(((NumCol, NumCol, self.na, self.n, self.n)))
        for i in range(NumCol):
            j = i
            self.Trans_hybrid_obs12 = self.Trans_hybrid_obs12.at[i, j, a0, :, :].set(jnp.transpose(kronn(self.Tla0, jnp.identity(self.nq), Tr, jnp.identity(self.nq))))
            self.Trans_hybrid_obs12 = self.Trans_hybrid_obs12.at[i, j, g0, :, :].set(jnp.transpose(kronn(self.Tlg0, jnp.identity(self.nq), Tr, jnp.identity(self.nq))))
            self.Trans_hybrid_obs12 = self.Trans_hybrid_obs12.at[i, j, pb, :, :].set(jnp.transpose(block_diag(self.Tlpb, self.Tlpb, den_col[1][pb][(i,j)], den_col[2][pb][(i,j)])))

    def solveMDP_op(self, epsilon = 0.001, niterations = 10000, initial_value=0, mask=None):
        Q, policy, Vop = _solveMDP_op_pure(self.ThA, self.R, self.discount, epsilon, niterations, initial_value, mask)
        self.Q = Q
        self.policy = policy
        self.Vop = Vop

    def solveMDP_sfm(self, epsilon = 0.001, niterations = 10000, initial_value=0, mask=None):
        temperatureQ = self.parameters[7]
        Q, softpolicy, V = _solveMDP_sfm_pure(self.ThA, self.R, self.discount, epsilon, niterations, initial_value, mask, temperatureQ)
        self.Qsfm = Q
        self.softpolicy = softpolicy
        self.Vsfm = V

class twoboxCazettesMDPdata(twoboxCazettesMDP):
    def __init__(self, discount, nq, nr, na, nl, parameters, parametersExp,
                 sampleTime, sampleNum, lick_state=False, seed=0):
        twoboxCazettesMDP.__init__(self, discount, nq, nr, na, nl, parameters, lick_state)

        self.parametersExp = parametersExp
        self.sampleNum = sampleNum
        self.sampleTime = sampleTime
        self.lick_state = lick_state
        self.seed = seed
        self.abstract_locations = jnp.array(create_ordered_tuples(self.nl)).astype(jnp.int32)

        self.setupMDP()

        # JIT the solvers with pure functions
        self.jitted_solveMDP_op = jax.jit(_solveMDP_op_pure, static_argnames=['initial_value'])
        Q, policy, Vop = self.jitted_solveMDP_op(
            self.ThA, self.R, self.discount, 0.001, 10000, 0, self.loc0_mask
        )
        self.Q = Q
        self.policy = policy
        self.Vop = Vop

        self.jitted_solveMDP_sfm = jax.jit(_solveMDP_sfm_pure, static_argnames=['initial_value'])
        temperatureQ = self.parameters[7]
        Qsfm, softpolicy, Vsfm = self.jitted_solveMDP_sfm(
            self.ThA, self.R, self.discount, 0.001, 10000, 0, self.loc0_mask, temperatureQ
        )
        self.Qsfm = Qsfm
        self.softpolicy = softpolicy
        self.Vsfm = Vsfm

    def dataGenerate_sfm(self):
        # FIX: Ensure abstract locations array is int32 to match scan expectation
        locations_with_history = jnp.array(create_ordered_tuples(self.nl)).astype(jnp.int32)
        
        rng = jax.random.PRNGKey(self.seed)
        keys = jax.random.split(rng, self.sampleNum)
        
        den_tensor = _flatten_den_dict(self.den, self.nl, self.nq, self.nr)
        
        generate_vmap = jax.vmap(_generate_single_trial, 
                                 in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None))
        
        (self.action, self.reward, self.location, 
         self.trueState1, self.trueState2, 
         self.belief1, self.belief2, 
         self.actionDist, self.location_ind,
         self.belief1Dist, self.belief2Dist, self.hybrid) = generate_vmap(
             keys, self.parametersExp, self.softpolicy,
             self.nq, self.nr, self.nl, self.nlwh, self.nlpbh,
             den_tensor, self.Tl, locations_with_history, self.sampleTime, self.lick_state
         )
        
        self.color1 = self.trueState1
        self.color2 = self.trueState2

class twoboxCazettesMDP_der(twoboxCazettesMDP):
    def __init__(self, discount, nq, nr, na, nl, parameters, lick_state=False):
        twoboxCazettesMDP.__init__(self, discount, nq, nr, na, nl, parameters, lick_state)
        self.setupMDP()
        self.solveMDP_op()
        self.solveMDP_sfm()

    def log_likelihood_fn(self, parameters, obs):
        pi = jnp.ones(self.nq * self.nq) / self.nq / self.nq
        Numcol = 2
        Ncol = 1
        twoboxColHMM = HMMtwoboxCazettes(self.ThA, self.softpolicy,
                                         self.Trans_hybrid_obs12, self.Obs_emis_trans1,
                                         self.Obs_emis_trans2, pi, Ncol, self.lick_state)
        log_likelihood = twoboxColHMM.computeQaux(obs, self.ThA, self.softpolicy, self.Trans_hybrid_obs12, self.Obs_emis_trans1, self.Obs_emis_trans2) + \
                         twoboxColHMM.latent_entr(obs)
        return log_likelihood

    def dloglikelihhod_dpara_sim(self, obs):
        grad_fn = grad(self.log_likelihood_fn)
        dloglikelihhod_dpara = grad_fn(self.parameters, obs)
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




























