### MyFramework: Python module for model identification
"""
Created on Thursday  14 July 2022

@author: Theresa Tillmann at UCL
"""

# ## import#########################
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as stats
import random
import time
# import matplotlib as mpl
import seaborn as sb
import matplotlib.colors as colors
import threading
import itertools
import dill

from scipy.integrate import odeint
from scipy.optimize import minimize
from openpyxl import Workbook, load_workbook
from numpy.linalg import LinAlgError
from scipy.integrate import solve_ivp
from scipy.optimize import shgo
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from FunctionStorage import in_silico_exp, PE, PE_uncertainty, tval_reform, Model_prob, model_selection
from FunctionStorage import lof_fun, sens_fun, FIM_fun, obs_COR, tvalue_fun, dynFIM_fun, PE_table_fun
from FunctionStorage import mbdoemd_HR, mbdoemd_HR_post, mbdoemd_BH, mbdoemd_BH_post, mbdoemd_BF, mbdoemd_BF_post, mbdoemd_ADC, mbdoemd_ADC_post, mbdoemd_JRD, mbdoemd_JRD_post
from FunctionStorage import MBDoE_PE, tval_reform_PE
from FunctionStorage import end_MBDoE, pie_chart, plot_y, plot_D_map_prior, plot_PE, plot_cov, d_map_prior, d_map_FIM, plot_D_map_FIM, plot_paraCov, plot_paraCorr, plot_T
from FunctionStorage import postprocessing_results, store_results

# ## function 'onlinembdoe'


def onlinembdoe(model_set_0, x0, t, y_meas, DS, var_exp, budget, DC, DC_PE, pe_guess, PE_bounds=(0, np.inf), prelim_exp=None, lambda_chi=np.inf, true_model=None, prob_calc=1, prob_rej_th=0.03, prob_th=0.999, tval_th=0.01, precis_th=0.2, newDC=None, lambda_f=0.1):
    '''
    This function should return the model which approximates experimental data best.

    :param model_set_0: [km1, km2, km3, km4] with defines models km1, km2 ..
    :param x0: function defining the initial state
    :param t: np.linspace(0, 40, 5): array with samples along integration variable (start, stop, #samples+1) initial sample will be removed
    :param y_meas: function defining the measurement variables from the states (not identifiable part of model)
    :param DS: [[0.05, 0.2], [5, 35], [1, 10]]: design space bounds with dimensions for x0 at the end
    :param var_exp: np.array([[(0.012)**2, 0.0], [0.0, (0.019)**2]]) measurement uncertainty variance matrix
    :param budget: (int) 20: number of samples (Nexp_max*Nsp)
    :param DC: (int) from list ['HR', 'BH', 'BF', 'ADC', 'JRD', 'R']: defines the discrimination criterion (DC)
    :param DC_PE: (int) from list ['D', 'E', 'modE']: defines the discrimination criterion (DC) for parameter precision
    :param pe_guess: (list) [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]: initial guess for all parameters (e.g. from literature or as guaranteed parameters)
    :param PE_bounds: (0, np.inf) (default): tuple of min and max bounds for parameter estimates. Bound the estimates to prevent numerical errors from from high or low estimates.
    :param prelim_exp: [phi, y_exp]: array of experimental designs and measurements
    :param lambda_chi: np.inf (default): rejects models with SSR > lambda_chi * chi2ref based on preliminary data
    :param true_model: None (default): LabView connected, in-silico exp: [1, np.array([14.86,  5.2,  0.2])] = [true model position in 'model_set_0', true_model_parameter]
    :param prob_calc: (int) 1: 1/chi2 (default), 2: 1-F_DOF(chi2), 3: akaikes weights, 4: gaussian posteriors
    :param prob_rej_th: 0.03 (default) = 3%: threshold for model rejection based on probability, None: no model rejection due to low probabilities
    :param prob_th: 0.999 (default) = 99.9%: threshold for model selection based on probability
    :param tval_th: 0.01 (default): threshold for model reformulation for models with at least one smaller t-value
    :param precis_th: 0.2 (default): minimum required model prediction for each measurement variable (float or array)
    :param newDC: None (default), True (forcing design space exploration and penalising according to information value of previous experiments for model discrimination)
    :param lambda_f: 0.1 (default): stregth of information penalty for new design criterion
    :return: Best model among 'models' with at least a prediction threshold of 'precis_th'
    '''

    # predefined inputs
    DC_set = ['HR', 'BH', 'BF', 'ADC', 'JRD', 'R']

    # definitions from inputs
    phi_ini = np.average(DS, axis=1)  # set initial guess for MBDoE
    budget = budget/(np.shape(t)[0]-1)

    # define infinite as bounds for parameter estimation
    param_bounds = []
    for num_m in range(len(model_set_0)):
        p_bound = []
        for num_p in range(len(pe_guess[num_m])):
            p_bound += [PE_bounds]
        param_bounds += [p_bound]

    #  input checks
    if model_set_0 is None or x0 is None or y_meas is None or t is None or t is None or DS is None or budget is None or pe_guess is None or DC is None:
        print("Missing input")

    if isinstance(model_set_0, list) == False:
        print("Wrong models input")

    if (DC in DC_set) == False:
            print("Wrong DC input")

    if pe_guess == None:
        print("Wrong pe_guess input")

    if true_model != None:
        [true_km, theta_t] = true_model
        true_m_num = np.where([model_set_0[s] == true_km for s in range(len(model_set_0))])[0][0]

    # ## preliminary parameter estimation
    if prelim_exp != None:
        [u, y_exp] = prelim_exp

        # ##  parameter estimates based on preliminary data
        try:
            theta, chisq, dof = PE(u, y_exp, model_set_0, x0, y_meas, t, var_exp, [pe_guess], param_bounds)  # x,u,t,y,models,var_exp,parameter
        except:
            print("Error due to PE")
            return 'E', 0, 5, locals()

        # ## remove underfitting models with SSR > 2* chi2ref
        if np.shape(np.where(chisq[:, 0] > lambda_chi * chisq[:, 1])[0])[0] != 0:
            underfitting = np.where(chisq[:, 0] > lambda_chi * chisq[:, 1])[0]
            print("Removed model(s) due to underfitting (SSR > " + str(lambda_chi) + " * chi2ref): " + str([model_set_0[s].__name__ for s in underfitting]))
            model_set_0 = np.delete(model_set_0, underfitting)
            true_m_num = np.where([model_set_0[s] == true_km for s in range(len(model_set_0))])[0][0]
            print("New model set: " + str([model_set_0[s].__name__ for s in range(len(model_set_0))]))
            theta = np.delete(theta, underfitting)
            param_bounds = np.delete(param_bounds, underfitting)
            dof = np.delete(dof, underfitting)
            chisq = np.delete(chisq, underfitting, 0)

        theta = [theta]
        chisq = np.expand_dims(chisq, axis=0)
        dof = np.expand_dims(dof, axis=0)

        # ## evaluate uncertainty of parameter estimates
        try:
            para_cov, conf_int, t_val, priorFIM = PE_uncertainty(u, t, x0, y_meas, model_set_0, var_exp, theta, dof)
        except ValueError:
            print("Error due to PE_uncertainty")
            return 'E', 0, 6, locals()

        para_cov = np.expand_dims(para_cov, axis=0)
        conf_int = np.expand_dims(conf_int, axis=0)
        t_val = np.expand_dims(t_val, axis=0)
        priorFIM = [priorFIM]
        PE_table = [PE_table_fun(model_set_0, theta, conf_int, priorFIM, t_val, dof)]

        # ## calculate model probability distribution
        prob = Model_prob(prob_calc, model_set_0, chisq[-1], dof[-1], y_exp, t, x0, y_meas, theta, u, var_exp)
        prob = np.expand_dims(prob, axis=0)

        # ## evaluate model selection and termination of onlineMBDoE
        model_set, termination, final_model_num = model_selection(prob_rej_th, model_set_0, prob[-1], prob_th, 0, budget)
        model_set = np.expand_dims(model_set, 0)

        if termination != None:
            print()
            if termination == 'budget':
                return 'D', np.max(prob[-1]), model_set_0[final_model_num], theta[-1][final_model_num], locals()
            if final_model_num == None:
                return 'E', 0, 8, locals()
            print("Model discrimination done. Identified model: ", model_set_0[final_model_num].__name__)
            if final_model_num == true_m_num:
                return onlineMBDoE_PE(locals(), DC_PE, model_set_0[final_model_num], theta_t, [[theta[-1][final_model_num]]],
                                      [param_bounds[final_model_num]], [[dof[-1][final_model_num]]],
                                      [[para_cov[-1][final_model_num]]], [[conf_int[-1][final_model_num]]],
                                      [[t_val[-1][final_model_num]]], [[PE_table[-1][final_model_num]]], u, y_exp,
                                      var_exp, [[priorFIM[-1][final_model_num]]], 0, budget, DS, x0, y_meas, t,
                                      precis_th, tval_th)
            return 'F', 0, model_set_0[final_model_num], theta[-1][final_model_num], locals()

        # ## check for t-values and reformulate models with t-values below tval_th
        model_set, t_val, theta, chisq, dof, para_cov, conf_int, priorFIM, PE_table, prob, param_bounds = tval_reform(prob_calc, model_set, t_val, tval_th, theta, chisq, dof, para_cov, conf_int, priorFIM, PE_table, prob, param_bounds, u, y_exp, model_set_0, x0, y_meas, t, var_exp, prob_th)


        # ## evaluate model selection and termination of onlineMBDoE again (after reformulation)
        model_set, termination, final_model_num = model_selection(prob_rej_th, model_set[-1], prob[-1], prob_th, 0, budget)
        model_set = np.expand_dims(model_set, 0)

        if termination != None:
            print()
            if termination == 'budget':
                return 'D', np.max(prob[-1]), model_set_0[final_model_num], theta[-1][final_model_num], locals()
            if final_model_num == None:
                return 'E', 0, 8, locals()
            print("Model discrimination done. Identified model: ", model_set_0[final_model_num].__name__)
            if final_model_num == true_m_num:
                return onlineMBDoE_PE(locals(), DC_PE, model_set_0[final_model_num], theta_t, [[theta[-1][final_model_num]]],
                                      [param_bounds[final_model_num]], [[dof[-1][final_model_num]]],
                                      [[para_cov[-1][final_model_num]]], [[conf_int[-1][final_model_num]]],
                                      [[t_val[-1][final_model_num]]], [[PE_table[-1][final_model_num]]], u, y_exp,
                                      var_exp, [[priorFIM[-1][final_model_num]]], 0, budget, DS, x0, y_meas, t,
                                      precis_th, tval_th)
            return 'F', 0, model_set_0[final_model_num], theta[-1][final_model_num], locals()

    else:
        [u, y_exp] = [[], []]
        theta = [list(pe_guess)]
        chisq = np.zeros((1, len(model_set_0), 2))
        dof = np.zeros((1, len(model_set_0)))
        para_cov = np.zeros((1, len(model_set_0)))
        conf_int = np.zeros((1, len(model_set_0)))
        t_val = np.ones((1, len(model_set_0), 2))
        priorFIM = [np.zeros((len(model_set_0)))]
        PE_table = [np.zeros((len(model_set_0)))]
        prob = [np.ones((len(model_set_0), 1))/len(model_set_0)]
        model_set = [list(model_set_0)]

    # ## MBDoE of design-vector phi within space DS defined by bounds
    # MBDoE MD
    if DC == 'HR':
        md_sol1 = minimize(mbdoemd_HR, phi_ini, method='Nelder-Mead', bounds=DS, args=(t, x0, y_meas, model_set[-1], theta[-1], newDC, lambda_f, u, y_exp, DS, var_exp))
        md_sol = minimize(mbdoemd_HR, md_sol1.x, method='SLSQP', bounds=DS, args=(t, x0, y_meas, model_set[-1], theta[-1], newDC, lambda_f, u, y_exp, DS, var_exp))
    elif DC == 'BH':
        md_sol1 = minimize(mbdoemd_BH, phi_ini, method='Nelder-Mead', bounds=DS, args=(t, x0, y_meas, model_set[-1], prob[-1], theta[-1], var_exp, priorFIM[-1], y_exp, newDC, lambda_f, u, DS))
        md_sol = minimize(mbdoemd_BH, md_sol1.x, method='SLSQP', bounds=DS, args=(t, x0, y_meas, model_set[-1], prob[-1], theta[-1], var_exp, priorFIM[-1], y_exp, newDC, lambda_f, u, DS))
    elif DC == 'BF':
        md_sol1 = minimize(mbdoemd_BF, phi_ini, method='Nelder-Mead', bounds=DS, args=(t, x0, y_meas, model_set[-1], theta[-1], priorFIM[-1], var_exp, newDC, lambda_f, u, y_exp, DS))
        md_sol = minimize(mbdoemd_BF, md_sol1.x, method='SLSQP', bounds=DS, args=(t, x0, y_meas, model_set[-1], theta[-1], priorFIM[-1], var_exp, newDC, lambda_f, u, y_exp, DS))
        # if -1 * md_sol.fun <= np.shape(y_exp)[-1]:
        #    return 'E', 0, 4, locals()
    elif DC == 'ADC':
        md_sol1 = minimize(mbdoemd_ADC, phi_ini, method='Nelder-Mead', bounds=DS, args=(t, x0, y_meas, model_set[-1], theta[-1], var_exp, priorFIM[-1], newDC, lambda_f, u, y_exp, DS))
        md_sol = minimize(mbdoemd_ADC, md_sol1.x, method='SLSQP', bounds=DS, args=(t, x0, y_meas, model_set[-1], theta[-1], var_exp, priorFIM[-1], newDC, lambda_f, u, y_exp, DS))
    elif DC == 'JRD':
        md_sol1 = minimize(mbdoemd_JRD, phi_ini, method='Nelder-Mead', bounds=DS, args=(t, x0, y_meas, model_set[-1], prob[-1], theta[-1], var_exp, priorFIM[-1], y_exp, newDC, lambda_f, u, DS))
        md_sol = minimize(mbdoemd_JRD, md_sol1.x, method='SLSQP', bounds=DS, args=(t, x0, y_meas, model_set[-1], prob[-1], theta[-1], var_exp, priorFIM[-1], y_exp, newDC, lambda_f, u, DS))

    if DC == 'R':
        md_val_prior = None
        md_design = [random.uniform(DS[s][0], DS[s][1]) for s in range(len(DS))]
    else:
        md_val_prior = -1 * md_sol.fun
        md_design = md_sol.x

    md_val_prior = np.expand_dims(md_val_prior, axis=0)
    # print(md_design)

    md_val_posterior = np.array(0)

    # MBDoE iterations
    it = 0
    while termination == None and it <= budget+1:
        it += 1
        u = np.vstack((u, md_design))

        y_exp_temp = in_silico_exp(model_set_0[true_m_num], var_exp, u[-1], x0, y_meas, t, theta_t)
        y_exp = np.append(y_exp, y_exp_temp)
        y_exp = np.reshape(y_exp, (np.shape(u)[0], np.shape(t)[0]-1, -1))

        # ## parameter estimation
        try:
            theta_temp, chisq_temp, dof_temp = PE(u, y_exp, model_set_0, x0, y_meas, t, var_exp, theta, param_bounds)
        except:
            print("Error due to PE")
            return 'E', it, 5, locals()

        theta += [theta_temp]
        chisq = np.append(chisq, chisq_temp)
        chisq = np.reshape(chisq, (-1, len(model_set_0), 2))
        dof = np.append(dof, dof_temp)
        dof = np.reshape(dof, (-1, len(model_set_0)))

        # ## evaluate uncertainty of parameter estimates
        try:
            para_cov_temp, conf_int_temp, t_val_temp, priorFIM_temp = PE_uncertainty(u, t, x0, y_meas, model_set_0, var_exp, theta, dof, model_set, prob, conf_int, para_cov, t_val)
            if len(para_cov_temp) != len(model_set_0):
                print(para_cov_temp[model_set_0])  # = call except
        except:
            print("Error due to PE_uncertainty")
            return 'E', it, 6, locals()

        para_cov = np.append(para_cov, para_cov_temp)
        para_cov = np.reshape(para_cov, (len(theta), -1))
        conf_int += [conf_int_temp]
        t_val += [t_val_temp]
        priorFIM += [priorFIM_temp]
        PE_table += [PE_table_fun(model_set_0, theta, conf_int, priorFIM, t_val, dof)]

        # ## evaluate posterior MD criterion
        if DC == 'HR':
            md_val_posterior = np.vstack((md_val_posterior, mbdoemd_HR_post(md_design, t, x0, y_meas, model_set[-1], theta[-1])))
        elif DC == 'BH':
            md_val_posterior = np.vstack(
                (md_val_posterior, mbdoemd_BH_post(md_design, t, x0, y_meas, model_set[-1], prob[-1], theta[-1], var_exp,
                                                   priorFIM[-1], y_exp)))
        elif DC == 'BF':
            md_val_posterior = np.vstack(
                (md_val_posterior, mbdoemd_BF_post(md_design, t, x0, y_meas, model_set[-1], theta[-1], priorFIM[-1], var_exp)))
        elif DC == 'ADC':
            md_val_posterior = np.vstack(
                (md_val_posterior, mbdoemd_ADC_post(md_design, t, x0, y_meas, model_set[-1], theta[-1], var_exp, priorFIM[-1])))
        elif DC == 'JRD':
            md_val_posterior = np.vstack(
                (md_val_posterior, mbdoemd_JRD_post(md_design, t, x0, y_meas, model_set[-1], prob[-1], theta[-1],
                                                    var_exp, priorFIM[-1], y_exp)))

        # ## calculate model probability distribution
        prob_temp = Model_prob(prob_calc, model_set[-1], chisq[-1], dof[-1], y_exp, t, x0, y_meas, theta, u, var_exp)
        prob = np.append(prob, prob_temp)
        prob = np.reshape(prob, (-1, len(model_set_0)))

        # ## evaluate model selection and termination of onlineMBDoE
        model_set_temp, termination, final_model_num = model_selection(prob_rej_th, model_set[-1], prob[-1], prob_th, it, budget)
        model_set = np.vstack((model_set, model_set_temp))

        if termination != None:
            print()
            if termination == 'budget':
                return 'D', np.max(prob[-1]), model_set_0[final_model_num], theta[-1][final_model_num], locals()
            if final_model_num == None:
                return 'E', it, 8, locals()
            print("Model discrimination done. Identified model: ", model_set_0[final_model_num].__name__)
            if final_model_num == true_m_num:
                return onlineMBDoE_PE(locals(), DC_PE, model_set_0[final_model_num], theta_t, [[theta[-1][final_model_num]]], [param_bounds[final_model_num]], [[dof[-1][final_model_num]]], [[para_cov[-1][final_model_num]]], [[conf_int[-1][final_model_num]]], [[t_val[-1][final_model_num]]], [[PE_table[-1][final_model_num]]], u, y_exp, var_exp, [[priorFIM[-1][final_model_num]]], it, budget, DS, x0, y_meas, t, precis_th, tval_th)
            return 'F', it, model_set_0[final_model_num], theta[-1][final_model_num], locals()

        # ## check for t-values and reformulate models with t-values below tval_th
        model_set, t_val, theta, chisq, dof, para_cov, conf_int, priorFIM, PE_table, prob, param_bounds = tval_reform(prob_calc,
            model_set, t_val, tval_th, theta, chisq, dof, para_cov, conf_int, priorFIM, PE_table, prob, param_bounds, u,
            y_exp, model_set_0, x0, y_meas, t, var_exp, prob_th)

        # ## evaluate model selection and termination of onlineMBDoE
        model_set_temp, termination, final_model_num = model_selection(prob_rej_th, model_set[-1], prob[-1], prob_th, it, budget)
        model_set = np.vstack((model_set, model_set_temp))

        if termination != None:
            print()
            if termination == 'budget':
                return 'D', np.max(prob[-1]), model_set_0[final_model_num], theta[-1][final_model_num], locals()
            if final_model_num == None:
                return 'E', it, 8, locals()
            print("Model discrimination done. Identified model: ", model_set_0[final_model_num].__name__)
            if final_model_num == true_m_num:
                return onlineMBDoE_PE(locals(), DC_PE, model_set_0[final_model_num], theta_t, [[theta[-1][final_model_num]]],
                                      [param_bounds[final_model_num]], [[dof[-1][final_model_num]]],
                                      [[para_cov[-1][final_model_num]]], [[conf_int[-1][final_model_num]]],
                                      [[t_val[-1][final_model_num]]], [[PE_table[-1][final_model_num]]], u, y_exp,
                                      var_exp, [[priorFIM[-1][final_model_num]]], it, budget, DS, x0, y_meas, t,
                                      precis_th, tval_th)
            return 'F', it, model_set_0[final_model_num], theta[-1][final_model_num], locals()


        # ## MBDoE MD
        if DC == 'HR':
            md_sol1 = minimize(mbdoemd_HR, phi_ini, method='Nelder-Mead', bounds=DS,
                               args=(t, x0, y_meas, model_set[-1], theta[-1], newDC, lambda_f, u, y_exp, DS, var_exp))
            md_sol = minimize(mbdoemd_HR, md_sol1.x, method='SLSQP', bounds=DS,
                              args=(t, x0, y_meas, model_set[-1], theta[-1], newDC, lambda_f, u, y_exp, DS, var_exp))
        elif DC == 'BH':
            md_sol1 = minimize(mbdoemd_BH, phi_ini, method='Nelder-Mead', bounds=DS, args=(
            t, x0, y_meas, model_set[-1], prob[-1], theta[-1], var_exp, priorFIM[-1], y_exp, newDC, lambda_f, u, DS))
            md_sol = minimize(mbdoemd_BH, md_sol1.x, method='SLSQP', bounds=DS, args=(
            t, x0, y_meas, model_set[-1], prob[-1], theta[-1], var_exp, priorFIM[-1], y_exp, newDC, lambda_f, u, DS))
        elif DC == 'BF':
            md_sol1 = minimize(mbdoemd_BF, phi_ini, method='Nelder-Mead', bounds=DS, args=(
            t, x0, y_meas, model_set[-1], theta[-1], priorFIM[-1], var_exp, newDC, lambda_f, u, y_exp, DS))
            md_sol = minimize(mbdoemd_BF, md_sol1.x, method='SLSQP', bounds=DS, args=(
            t, x0, y_meas, model_set[-1], theta[-1], priorFIM[-1], var_exp, newDC, lambda_f, u, y_exp, DS))
            # if -1 * md_sol.fun <= np.shape(y_exp)[-1]:
            #    return 'E', it, 4, locals()
        elif DC == 'ADC':
            md_sol1 = minimize(mbdoemd_ADC, phi_ini, method='Nelder-Mead', bounds=DS, args=(
            t, x0, y_meas, model_set[-1], theta[-1], var_exp, priorFIM[-1], newDC, lambda_f, u, y_exp, DS))
            md_sol = minimize(mbdoemd_ADC, md_sol1.x, method='SLSQP', bounds=DS, args=(
            t, x0, y_meas, model_set[-1], theta[-1], var_exp, priorFIM[-1], newDC, lambda_f, u, y_exp, DS))
        elif DC == 'JRD':
            md_sol1 = minimize(mbdoemd_JRD, phi_ini, method='Nelder-Mead', bounds=DS, args=(
            t, x0, y_meas, model_set[-1], prob[-1], theta[-1], var_exp, priorFIM[-1], y_exp, newDC, lambda_f, u, DS))
            md_sol = minimize(mbdoemd_JRD, md_sol1.x, method='SLSQP', bounds=DS, args=(
            t, x0, y_meas, model_set[-1], prob[-1], theta[-1], var_exp, priorFIM[-1], y_exp, newDC, lambda_f, u, DS))

        if DC == 'R':
            md_val_prior = None
            md_design = [random.uniform(DS[s][0], DS[s][1]) for s in range(len(DS))]
        else:
            md_val_prior = np.vstack((md_val_prior, -1 * md_sol.fun))
            md_design = md_sol.x


    return 'E', it, 7, locals()


def onlineMBDoE_PE(local, DC_PE, model, theta_t, theta_m, param_bounds_m, dof_m, para_cov_m, conf_int_m, t_val_m, PE_table_m, u, y_exp, var_exp, priorFIM_m, itMD, budget, DS, x0, y_meas, t, precis_th, tval_th):
    it = int(itMD)

    PE_table_m = []
    termination = all(conf_int_m[-1][0] <= precis_th)

    while termination == False and it <= budget:
        it += 1
        phi_ini = np.average(DS, axis=1)
        pe_design = MBDoE_PE(DC_PE, DS, phi_ini, t, x0, y_meas, model, theta_m, var_exp, priorFIM_m)
        u = np.vstack((u, pe_design))

        y_exp_temp = in_silico_exp(model, var_exp, u[-1], x0, y_meas, t, theta_t)
        y_exp = np.append(y_exp, y_exp_temp)
        y_exp = np.reshape(y_exp, (np.shape(u)[0], np.shape(t)[0] - 1, -1))

        # ## parameter estimation
        try:
            theta_temp, _, dof_temp = PE(u, y_exp, [model], x0, y_meas, t, var_exp, theta_m, param_bounds_m)
        except:
            print("Error due to PE")
            return 'E', it, 5, locals()

        theta_m += [theta_temp]
        dof_m = np.append(dof_m, dof_temp)
        dof_m = np.reshape(dof_m, (-1, 1))

        # ## evaluate uncertainty of parameter estimates
        try:
            para_cov_temp, conf_int_temp, t_val_temp, priorFIM_temp = PE_uncertainty(u, t, x0, y_meas, [model],
                                                                                     var_exp, theta_m, dof_m)
        except:
            print("Error due to PE_uncertainty")
            return 'E', it, 6, locals()

        para_cov_m += [para_cov_temp]
        conf_int_m += [conf_int_temp]
        t_val_m += [t_val_temp]
        priorFIM_m += [priorFIM_temp]
        PE_table_m += [PE_table_fun([model], theta_m, conf_int_m, priorFIM_m, t_val_m, dof_m)]

        t_val_m, theta_m, dof_m, para_cov_m, conf_int_m, priorFIM_m, PE_table_m, param_bounds_m = tval_reform_PE(model, t_val_m, tval_th, theta_m, dof_m, para_cov_m, conf_int_m, priorFIM_m, PE_table_m,
                       param_bounds_m, u, y_exp, x0, y_meas, t, var_exp)

        termination = all(conf_int_m[-1][0] <= precis_th)
    if termination == True:
        return 'I', itMD, it, model, theta_m[-1], locals()

    return 'P', itMD, conf_int_m[-1], model, theta_m[-1], locals()
