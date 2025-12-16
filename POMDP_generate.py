from twoboxCol import *
from datetime import datetime
import os
import pickle
import numpy as np
import jax.numpy as jnp
from twoboxCazettesJax import twoboxCazettesMDPdata as twoboxCazettesMDPdataJax
from twoboxCazettesJax import twoboxCazettesIndependentDependentMDPdata as twoboxCazettesIndependentDependentMDPdataJax
from twoboxCazettes import twoboxCazettesMDPdata, twoboxCazettesIndependentMDPdata, twoboxCazettesIndependentDependentMDPdata

import jax

path = os.getcwd()


def twoboxColGenerate(parameters, parametersExp, sample_length, sample_number, nq, nr = 2, nl = 3, na = 5,
                      discount = 0.99, save = True):
    """
    Generate data of the teacher POMDPS
    """

    datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M%S)')  # current time used to set file name

    print("\nSet the parameters of the model... \n")

    beta = 0  # available food dropped back into box after button press
    gamma1 = parameters[0]  # reward becomes available in box 1
    gamma2 = parameters[1]  # reward becomes available in box 2
    delta = 0  # animal trips, doesn't go to target location
    direct = 0  # animal goes right to target, skipping location 0
    epsilon1 = parameters[2]  # available food disappears from box 1
    epsilon2 = parameters[3]  # available food disappears from box 2
    rho = 1  # food in mouth is consumed
    # State rewards
    Reward = 1  # reward per time step with food in mouth
    groom = parameters[4]  # location 0 reward
    # Action costs
    travelCost = parameters[5]
    pushButtonCost = parameters[6]

    NumCol = np.rint(parameters[7]).astype(int)  # number of colors
    Ncol = NumCol - 1  # max value of color
    qmin = parameters[8]
    qmax = parameters[9]
    temperatureQ = parameters[10]

    gamma1_e = parametersExp[0]
    gamma2_e = parametersExp[1]
    epsilon1_e = parametersExp[2]
    epsilon2_e = parametersExp[3]
    qmin_e = parametersExp[4]
    qmax_e = parametersExp[5]


    print("Generating data...")
    T = sample_length
    N = sample_number
    twoboxColdata = twoboxColMDPdata(discount, nq, nr, na, nl, parameters, parametersExp, T, N)
    twoboxColdata.dataGenerate_sfm()

    hybrid = twoboxColdata.hybrid
    action = twoboxColdata.action
    location = twoboxColdata.location
    belief1 = twoboxColdata.belief1
    belief2 = twoboxColdata.belief2
    reward = twoboxColdata.reward
    trueState1 = twoboxColdata.trueState1
    trueState2 = twoboxColdata.trueState2
    color1 = twoboxColdata.color1
    color2 = twoboxColdata.color2

    actionDist = twoboxColdata.actionDist
    belief1Dist = twoboxColdata.belief1Dist
    belief2Dist = twoboxColdata.belief2Dist

    # sampleNum * sampleTime * dim of observations(=3 here, action, reward, location)
    # organize data
    obsN = np.dstack([action, reward, location, color1, color2, actionDist])  # includes the action and the observable states
    latN = np.dstack([belief1, belief2, belief1Dist, belief2Dist])
    truthN = np.dstack([trueState1, trueState2])
    dataN = np.dstack([obsN, latN, truthN])

    ### write data to file
    data_dict = {'observations': obsN,
                 'beliefs': latN,
                 'trueStates': truthN,
                 'allData': dataN}

    ### write all model parameters to file
    para_dict = {'discount': discount,
                 'nq': nq,
                 'nr': nr,
                 'nl': nl,
                 'na': na,
                 'foodDrop': beta,
                 'appRate1': gamma1,
                 'appRate2': gamma2,
                 'disappRate1': epsilon1,
                 'disappRate2': epsilon2,
                 'consume': rho,
                 'reward': Reward,
                 'groom': groom,
                 'travelCost': travelCost,
                 'pushButtonCost': pushButtonCost,
                 'ColorNumber': NumCol,
                 'qmin': qmin,
                 'qmax': qmax,
                 'appRateExperiment1': gamma1_e,
                 'disappRateExperiment1': epsilon1_e,
                 'appRateExperiment2': gamma2_e,
                 'disappRateExperiment2': epsilon2_e,
                 'qminExperiment': qmin_e,
                 'qmaxExperiment': qmax_e,
                 'temperature': temperatureQ,
                 'sample_length': sample_length,
                 'sample_number': sample_number
                 }

    if save:
        # create a file that saves the parameter dictionary using pickle
        para_output = open(path + '/Results/' + datestring + '_para_twoboxCol' + '.pkl', 'wb')
        pickle.dump(para_dict, para_output)
        para_output.close()

        data_output = open(path + '/Results/' + datestring + '_dataN_twoboxCol' + '.pkl', 'wb')
        pickle.dump(data_dict, data_output)
        data_output.close()

        print('Data stored in files')

    return obsN, latN, truthN, datestring


def twoboxCazettesGenerate(exp_type, parameters, parametersExp, sample_length, sample_number, nq, nr = 2, nl = 3, na = 5,
                           discount = 0.99, save = True, lick_state=False, use_jax=False):
    """
    Generate data of the teacher POMDPS
    """

    datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M%S)')  # current time used to set file name

    print("\nSet the parameters of the model... \n")
    
    psw = parameters[0]  # reward becomes available in box 1
    prwd = parameters[1]  # reward becomes available in box 2
    psw_e = parametersExp[0]
    prwd_e = parametersExp[1]

    beta = 0  # available food dropped back into box after button press
    delta = 0  # animal trips, doesn't go to target location
    direct = 0  # animal goes right to target, skipping location 0
    rho = 1  # food in mouth is consumed
    # State rewards
    Reward = 100  # reward per time step with food in mouth
    groom = parameters[2]  # location 0 reward
    # Action costs
    travelCost = parameters[3]
    pushButtonCost = parameters[4]
    startPushButtonCost = parameters[5]
    temperatureQ = parameters[6]

    groom_e = parametersExp[2]
    travelCost_e = parametersExp[3]
    pushButtonCost_e = parametersExp[4]
    startPushButtonCost_e = parametersExp[5]

    print("Generating data...")
    T = sample_length
    N = sample_number
    if use_jax:
        parameters = jnp.array(parameters)
        parametersExp = jnp.array(parametersExp)
        if exp_type == 'dependent':
            with jax.check_tracer_leaks():
                twoboxColdata = twoboxCazettesMDPdataJax(discount, nq, nr, na, nl, parameters, parametersExp, T, N, lick_state)
        # elif exp_type == 'independent':
        #     twoboxColdata = twoboxCazettesIndependentMDPdataJax(discount, nq, nr, na, nl, parameters, parametersExp, T, N)
        elif exp_type == 'independentDependent':
            twoboxColdata = twoboxCazettesIndependentDependentMDPdataJax(discount, nq, nr, na, nl, parameters, parametersExp, T, N)
    else:
        if exp_type == 'dependent':
            twoboxColdata = twoboxCazettesMDPdata(discount, nq, nr, na, nl, parameters, parametersExp, T, N, lick_state)
        elif exp_type == 'independent':
            twoboxColdata = twoboxCazettesIndependentMDPdata(discount, nq, nr, na, nl, parameters, parametersExp, T, N)
        elif exp_type == 'independentDependent':
            twoboxColdata = twoboxCazettesIndependentDependentMDPdata(discount, nq, nr, na, nl, parameters, parametersExp, T, N)
        
    twoboxColdata.dataGenerate_sfm()

    hybrid = np.array(twoboxColdata.hybrid)
    action = np.array(twoboxColdata.action)
    location = np.array(twoboxColdata.location)
    belief1 = np.array(twoboxColdata.belief1)
    belief2 = np.array(twoboxColdata.belief2)
    reward = np.array(twoboxColdata.reward)
    trueState1 = np.array(twoboxColdata.trueState1)
    trueState2 = np.array(twoboxColdata.trueState2)
    color1 = np.array(twoboxColdata.color1)
    color2 = np.array(twoboxColdata.color2)
    actionDist = np.array(twoboxColdata.actionDist)
    belief1Dist = np.array(twoboxColdata.belief1Dist)
    belief2Dist = np.array(twoboxColdata.belief2Dist)


    # sampleNum * sampleTime * dim of observations(=3 here, action, reward, location)
    # organize data    
    obsN = {'action': action, 
            'reward': reward, 
            'site_location': location, 
            'color1': color1, 
            'color2': color2, 
            'actionDist': actionDist}
    latN = {'belief1': belief1,
            'belief2': belief2,
            'belief1Dist': belief1Dist,
            'belief2Dist': belief2Dist}

    truthN = {'trueState1': trueState1, 'trueState2': trueState2}
    dataN = {'observations': obsN, 'beliefs': latN, 'trueStates': truthN}

    if exp_type == 'dependent':
        locationInd = np.array(twoboxColdata.location_ind)
        abstractLocations = np.array(twoboxColdata.abstract_locations)
        obsN['locationInd'] = locationInd
        obsN['abstractLocations'] = abstractLocations
    if lick_state:
        pb_trials = np.zeros_like(action)
        pb_trials[action == 2] = 1
        obsN['pb_trials'] = pb_trials
    
    ### write data to file
    data_dict = {'observations': obsN,
                 'beliefs': latN,
                 'trueStates': truthN,
                 'allData': dataN}

    ### write all model parameters to file
    para_dict = {'discount': discount,
                 'nq': nq,
                 'nr': nr,
                 'nl': nl,
                 'na': na,
                 'psw': psw,
                 'prwd': prwd,
                 'psw_Experiment': psw_e,
                 'prwd_Experiment': prwd_e,
                 'foodDrop': beta,
                 'consume': rho,
                 'reward': Reward,
                 'groom': groom,
                 'travelCost': travelCost,
                 'pushButtonCost': pushButtonCost,
                 'startPushButtonCost': startPushButtonCost, 
                 'startPushButtonCost_e': startPushButtonCost_e,
                 'temperature': temperatureQ,
                 'sample_length': sample_length,
                 'sample_number': sample_number
                 }

    if save:
        # create a file that saves the parameter dictionary using pickle
        para_output = open(path + '/Results/' + datestring + '_para_twoboxCazettes_' + exp_type + '.pkl', 'wb')
        pickle.dump(para_dict, para_output)
        para_output.close()

        data_output = open(path + '/Results/' + datestring + '_dataN_twoboxCazettes_' + exp_type +  '.pkl', 'wb')
        pickle.dump(data_dict, data_output)
        data_output.close()

        print('Data stored in files')

    return obsN, latN, truthN, datestring



