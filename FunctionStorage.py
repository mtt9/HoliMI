### MyFramework: Python module for model identification
"""
Created on Thursday  14 July 2022

@author: Theresa Tillmann at UCL
"""

# ## import#########################
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib as mpl
import seaborn as sb
import matplotlib.colors as colors
import itertools

from scipy.integrate import odeint
from scipy.optimize import minimize
from numpy.linalg import LinAlgError

# data generation with in-silico experiments (based on a true model with true parameters and noise)
def in_silico_exp(model, var_exp, u, x0, y_meas, t, theta_t):
    y_hat = y_meas(odeint(model, x0(u), t, args=(u, theta_t))[1:])
    noise = np.random.multivariate_normal((0, 0), var_exp, np.shape(y_hat)[0])
    y = y_hat + noise  # disturbance with noise (zero mean, var_exp cov-var matrix) # results in 21x2 matrix (21 sample points of 2 measured variables)
    return y


# PE with LOF
def PE(u, y_exp, model_set_0, x0, y_meas, t, var_exp, theta, param_bounds):  # default parameter = initial guess of parameters
    parameter_temp = list(theta[-1])
    num_models = len(model_set_0)
    alpha = 0.05  # for chi and t
    chisq_temp = np.zeros((num_models, 2))
    dof_temp = np.zeros(num_models)

    for km_num in range(num_models):
        km = model_set_0[km_num]
        if np.all(param_bounds[km_num] == (0, 0)):
            chisq_temp[km_num, 0] = lof_fun(parameter_temp[km_num], u, y_exp, km, x0, y_meas, t, var_exp)
        else:
            chisq0 = lof_fun(parameter_temp[km_num], u, y_exp, km, x0, y_meas, t, var_exp)
            pe_sol1 = minimize(lof_fun, parameter_temp[km_num], method='Nelder-Mead', args=(u, y_exp, km, x0, y_meas, t, var_exp), bounds=param_bounds[km_num])
            if pe_sol1.fun < chisq0:
                parameter_temp[km_num] = pe_sol1.x  # new theta
                chisq_temp[km_num, 0] = pe_sol1.fun  # new theta + chi^2 value
            pe_sol2 = minimize(lof_fun, parameter_temp[km_num], method='SLSQP', args=(u, y_exp, km, x0, y_meas, t, var_exp), bounds=param_bounds[km_num])  # necessary change of bounds here? bounds = [(0,21)] * np.shape(theta0_m1)[0],
            if pe_sol2.fun < lof_fun(parameter_temp[km_num], u, y_exp, km, x0, y_meas, t, var_exp):
                parameter_temp[km_num] = pe_sol2.x  # new theta
                chisq_temp[km_num, 0] = pe_sol2.fun  # new theta + chi^2 value
            elif pe_sol1.fun >= chisq0:
                print("PE optimisation failed. Continue with previous PE for "+model_set_0[km_num].__name__)
                chisq_temp[km_num, 0] = chisq0

        dof_temp[km_num] = np.shape(y_exp)[0] * np.shape(y_exp)[1] * np.shape(y_exp)[2] - (np.shape(parameter_temp[km_num])[0]-param_bounds[km_num].count((0, 0)))  # num_exp * num_response_var * num_sampling_times - num_para # num_para is num theta - num rixed theta from reformulation with theta=0

        chisq_temp[km_num, 1] = stats.chi2.ppf((1 - alpha), dof_temp[km_num])  # chi^2 ref value

    return parameter_temp, chisq_temp, dof_temp


# evaluate uncertainty of parameter estimates
def PE_uncertainty(u, t, x0, y_meas, model_set_0, var_exp, theta, dof, model_set=None, prob=None, conf_int=None, para_cov=None, t_val=None):
    num_models = len(model_set_0)
    alpha = 0.05  # for chi and t
    n_exp = np.shape(u)[0]
    para_cov_var = []
    t_val_var = []
    conf_int_var = []
    priorFIM_var = []

    for km_num in range(num_models):
        km = model_set_0[km_num]
        if km == None:
            para_cov_var += [None]  # parameter variance-covariance matrix
            conf_int_var += [None]
            t_val_var += [None]
            priorFIM_var += [None]
            continue
        obsFIM = np.zeros((np.shape(t)[0]-1, len(theta[-1][km_num]), len(theta[-1][km_num])))
        for exp in range(n_exp):
            obsFIM += dynFIM_fun(u[exp], t, km, theta[-1][km_num], var_exp, x0, y_meas)

        try:
            index0 = np.where(np.sum(np.vstack((np.diag(np.sum(obsFIM, axis=0)) == 0, np.isnan(np.diag(np.sum(obsFIM, axis=0))))), axis =0))[0]
            redFIM = np.delete(np.delete(np.sum(obsFIM, axis=0), index0, -1), index0, -2)

            try:
                obscov = np.linalg.inv(redFIM)
                priorFIM_var += [obsFIM]
                if index0.size != 0:
                    obscov = [obscov := np.insert(np.insert(obscov, s, 0, -1), s, 0, -2) for s in index0][-1]
                if redFIM.size == 0:
                    priorFIM_var[-1] = np.expand_dims(
                        np.diag(np.diag(np.ones((len(theta[-1][km_num]), len(theta[-1][km_num]))))), 0)

            except:
                obscov = np.zerros((len(theta[-1][km_num]), len(theta[-1][km_num])))
                priorFIM_var += [obscov]
                priorFIM_var[-1] = np.expand_dims(np.diag(np.diag(np.ones((len(theta[-1][km_num]), len(theta[-1][km_num]))))), 0)
            para_cov_var += [obs_COR(obscov)]  # parameter variance-covariance matrix
            conf_int_var_temp, t_val_var_temp = tvalue_fun(alpha, theta[-1][km_num], obscov, dof[-1][km_num])
            conf_int_var += [conf_int_var_temp]
            t_val_var += [t_val_var_temp]

            if np.sort(np.absolute(para_cov_var[km_num]).reshape(-1))[-(para_cov_var[km_num].shape[0] + 1)] > 0.998:
                print("Problem with correlated parameter estimates. The t test is not reliable for model " + km.__name__ + ".")
                # print(para_cov[km_num])
        except:
            print(np.sum(obsFIM, axis=0))
            if redFIM is not None:
                print(redFIM)
                print(theta[-1][km_num])

            if model_set is None:
                print("Ill-conditioned FIM(-sum) of preliminary experiment(s) is not invertible.")
                print("Problem candidate is model:")
                print(km)
                raise ValueError
            elif model_set[-1][km_num] is None:
                para_cov_var += [para_cov[-1,km_num]]
                conf_int_var += [conf_int[-1,km_num]]
                t_val_var += [t_val[-1,km_num]]
            else:
                print("Ill-conditioned FIM(-sum) is not invertible. Current probability distribution is:")
                print(prob[-1])
                print("Problem candidate is model:")
                print(km)
                raise ValueError
    return para_cov_var, conf_int_var, t_val_var, priorFIM_var


def PE_table_fun(model_set_0, theta, conf_int, priorFIM, t_val, dof):
    PE_tab = []
    for km_num in range(len(model_set_0)):
        param_est = theta[-1][km_num]
        ci = conf_int[-1][km_num]
        index0 = np.where(np.sum(np.vstack((np.diag(np.sum(priorFIM[-1][km_num], axis=0)) == 0, np.isnan(np.diag(np.sum(priorFIM[-1][km_num], axis=0))))),
                   axis=0))[0]
        redFIM = np.delete(np.delete(priorFIM[-1][km_num], index0, -1), index0, -2)
        obscov = np.linalg.inv(np.sum(redFIM, axis=0))
        if index0.size != 0:
            obscov = [obscov := np.insert(np.insert(obscov, s, 0, -1), s, 0, -2) for s in index0][-1]
        sd = np.sqrt(np.diag(obscov))
        t_i = t_val[-1][km_num]

        try:
            PE_tab += list([np.vstack((np.transpose(np.reshape(np.hstack((param_est, ci, sd, t_i)), (4, -1))),
                   [0, 0, 0, stats.t.ppf((1 - (0.05 / 2)), dof[-1][km_num])]))])
        except:
            print(param_est.shape)
            print(ci.shape)
            print(sd.shape)
            print(t_i.shape)
            print()
            print(km_num)
            print(model_set_0)
            print(theta[-1])
            print(conf_int[-1])
            print(priorFIM[-1])
            print(t_val[-1])

    return PE_tab


# Leak of Fit function also used for chi test (PE)
def lof_fun(theta_est, u, y_exp, model, x0, y_meas, t, var_exp):
    n_exp = np.shape(y_exp)[0]
    var_exp_inv = np.array(np.linalg.inv(var_exp), dtype=np.float64)  # var_exp-1
    lof = 0
    for exp in range(n_exp):
        y_hat = y_meas(odeint(model, x0(u[exp]), t, args=(u[exp], theta_est))[1:])
        resi = np.expand_dims(y_exp[exp] - y_hat, 2)  # e
        lof_temp = np.sum(np.trace(resi*var_exp_inv*resi, axis1=1, axis2=2))
        if np.isnan(lof_temp):
            # print("Error in LOF. LOF is 0 or NaN. Experiment no "+str(exp))
            lof_temp = 999999999999999999
        lof += lof_temp

    return lof


def sens_fun(u, t, model, theta, x0, y_meas):  # in shape (t,theta,y)
    epsilon = 0.001

    p_matrix = np.zeros([np.shape(theta)[0] + 1, np.shape(theta)[0]])
    for i in range(np.shape(theta)[0] + 1):
        p_matrix[i] = theta  # store thetas several times in p
    for i in range(np.shape(theta)[0]):
        p_matrix[i][i] = theta[i] * (1 + epsilon)
    y_hat = []
    for theta in p_matrix:
        y_hat += [y_meas(odeint(model, x0(u), t, args=(u, theta))[1:])]  # calculate model response y_hat with slightly disturbed thetas
    s_matrix = np.zeros([np.shape(t)[0]-1, np.shape(theta)[0], np.shape(y_hat[0])[1]])
    # s_matrix[:] = np.NaN
    for i in range(np.shape(y_hat[0])[0]):
        for j in range(np.shape(theta)[0]):
            if theta[j] != 0:
                s_matrix[i][j] = (y_hat[j][i] - y_hat[-1][i]) / (epsilon * theta[j])  # calculation of sensitivity matrix based on disturbed model paramters
    return s_matrix


def dynFIM_fun(u, t, model, theta, var_exp, x0, y_meas):
    s_matrix = sens_fun(u, t, model, theta, x0, y_meas)
    var_exp_inv = np.linalg.inv(var_exp)  # var_exp-1
    dynFIM = np.matmul(s_matrix, np.matmul(var_exp_inv, s_matrix.transpose((0, 2, 1))))
    return dynFIM


def FIM_fun(u, t, model, theta, var_exp, x0, y_meas):
    s_matrix = sens_fun(u, t, model, theta, x0, y_meas)
    var_exp_inv = np.linalg.inv(var_exp)  # var_exp-1
    FIM = np.matmul(s_matrix, np.matmul(var_exp_inv, s_matrix.transpose((0, 2, 1))))
    if np.ndim(FIM)==3:
        FIM = np.sum(FIM, axis=0)
    return FIM


def obs_COR(obscov):
    obsCOR = np.zeros([np.shape(obscov)[0], np.shape(obscov)[0]])
    for i in range(np.shape(obscov)[0]):
        for j in range(np.shape(obscov)[0]):
            obsCOR[i, j] = obscov[i, j] / (np.sqrt(obscov[i, i] * obscov[j, j]))
    return obsCOR


def tvalue_fun(alpha, pe, obscov, dof):
    conf_interval = np.zeros(np.shape(pe)[0])
    t_value = np.zeros(np.shape(pe)[0])
    for j in range(np.shape(pe)[0]):
        conf_interval[j] = np.sqrt(np.absolute(obscov[j, j])) * stats.t.ppf((1 - (alpha / 2)), dof)
        if conf_interval[j] != 0:
            t_value[j] = np.absolute(pe[j]) / (conf_interval[j])

    return conf_interval, t_value


def tval_reform(prob_calc, model_set, t_val, tval_th, theta, chisq, dof, para_cov, conf_int, priorFIM, PE_table, prob, param_bounds, u, y_exp, model_set_0, x0, y_meas, t, var_exp, prob_th):
    num_models = len(model_set[-1])
    if np.amin(dof[-1]) > 0:
        for km in range(num_models):
            # test for t_val and remove one parameter after the other
            if np.sum(t_val[-1][km] <= tval_th) > np.sum([param_bounds[km][s] == (0, 0) for s in range(len(param_bounds[km]))]):
                while np.sum(t_val[-1][km] <= tval_th) > np.sum([param_bounds[km][s] == (0, 0) for s in range(len(param_bounds[km]))]):
                    print("Reformulation of: "+model_set_0[km].__name__)
                    for index_ref in np.where([t_val[-1][km][s] <= tval_th and param_bounds[km][s] != (0, 0) for s in range(len(param_bounds[km]))])[0]:
                        param_bounds[km][index_ref] = (0, 0)
                        theta[-1][km][index_ref] = 0

                    theta_temp, chisq_temp, dof_temp = PE(u, y_exp, model_set_0, x0, y_meas, t, var_exp, theta, param_bounds)
                    # check for doubled model
                    if [np.allclose(chisq_temp[km], chisq_temp[s], rtol=1e-3) for s in range(num_models)].count(True) > 1 and [np.isclose(theta_temp[km].sum(), theta_temp[s].sum(), rtol=1e-3) for s in np.where([dof_temp[km] == dof_temp[s2] and km != s2 for s2 in range(num_models)])[0]].count(True) > 0:
                        model_set[-1][km] = None
                        print("Reformulated model removed due to doubling with another model.")

                    if theta_temp[km].sum() == 0:
                        model_set[-1][km] = None
                        print("Reformulated model removed due to constraining all parameters.")

                    # update latest ... for that model except PE, here: add entry with entry for new model only
                    theta[-1], chisq[-1], dof[-1] = theta_temp, chisq_temp, dof_temp

                    para_cov[-1], conf_int[-1], t_val[-1], priorFIM[-1] = PE_uncertainty(u, t, x0, y_meas, model_set_0,
                                                                                             var_exp, theta, dof, model_set,
                                                                                             prob, conf_int, para_cov,
                                                                                             t_val)

                    prob[-1] = Model_prob(prob_calc, model_set[-1], chisq[-1], dof[-1], y_exp, t, x0, y_meas, theta, u, var_exp)

                PE_table += [PE_table_fun(model_set_0, theta, conf_int, priorFIM, t_val, dof)]

    return model_set, t_val, theta, chisq, dof, para_cov, conf_int, priorFIM, PE_table, prob, param_bounds


def tval_reform_PE(model, t_val_m, tval_th, theta_m, dof_m, para_cov_m, conf_int_m, priorFIM_m, PE_table_m, param_bounds_m, u, y_exp, x0, y_meas, t, var_exp):
    if np.amin(dof_m[-1]) > 0:
            # test for t_val and remove one parameter after the other
            if np.sum(t_val_m[-1][0] <= tval_th) > np.sum([param_bounds_m[0][s] == (0, 0) for s in range(len(param_bounds_m[0]))]):
                while np.sum(t_val_m[-1][0] <= tval_th) > np.sum([param_bounds_m[0][s] == (0, 0) for s in range(len(param_bounds_m[0]))]):
                    print("Reformulation of: "+model.__name__)
                    for index_ref in np.where([t_val_m[-1][0][s] <= tval_th and param_bounds_m[0][s] != (0, 0) for s in range(len(param_bounds_m[0]))])[0]:
                        param_bounds_m[0][index_ref] = (0, 0)
                        theta_m[-1][0][index_ref] = 0

                    theta_temp, chisq_temp, dof_temp = PE(u, y_exp, [model], x0, y_meas, t, var_exp, theta_m, param_bounds_m)

                    # update latest ... for that model except PE, here: add entry with entry for new model only
                    theta_m[-1], dof_m[-1] = theta_temp, dof_temp

                    para_cov_m[-1], conf_int_m[-1], t_val_m[-1], priorFIM_m[-1] = PE_uncertainty(u, t, x0, y_meas, [model], var_exp, theta_m, dof_m)

                PE_table_m += [PE_table_fun([model], theta_m, conf_int_m, priorFIM_m, t_val_m, dof_m)]

    return t_val_m, theta_m, dof_m, para_cov_m, conf_int_m, priorFIM_m, PE_table_m, param_bounds_m


def av_FIM(phi, t, model_set, theta, prior, var_exp, x0, y_meas):  # y dimension has changed, so need the index of y and its usage

    num_models = len(model_set)
    exp_detFIM = []

    for km_num_a in range(num_models-1):
        km_a = model_set[km_num_a]
        if km_a != None:
            pe_a = theta[km_num_a]
            exp_FIM_a = FIM_fun(phi, t, km_a, pe_a, var_exp, x0, y_meas) + np.sum(prior[km_num_a], axis=0)
            index0 = np.where(np.sum(np.vstack((np.diag(exp_FIM_a) == 0, np.isnan(np.diag(exp_FIM_a)))), axis=0))[0]
            exp_detFIM = np.hstack((exp_detFIM, np.linalg.det(np.delete(np.delete(exp_FIM_a, index0, -1), index0, -2))))

    exp_detFIM = np.average(exp_detFIM)
    return exp_detFIM


# ## PE criteria
def MBDoE_PE(DC_PE, DS, phi_ini, t, x0, y_meas, model, theta_m, var_exp, priorFIM_m):
    if DC_PE == 'D':
        pe_sol1 = minimize(mbdoepe_D, phi_ini, method='Nelder-Mead', bounds=DS, args=(t, x0, y_meas, model, theta_m, var_exp, priorFIM_m))
        pe_sol = minimize(mbdoepe_D, pe_sol1.x, method='SLSQP', bounds=DS, args=(t, x0, y_meas, model, theta_m, var_exp, priorFIM_m))
    elif DC_PE == 'E':
        pe_sol1 = minimize(mbdoepe_E, phi_ini, method='Nelder-Mead', bounds=DS, args=(t, x0, y_meas, model, theta_m, var_exp, priorFIM_m))
        pe_sol = minimize(mbdoepe_E, pe_sol1.x, method='SLSQP', bounds=DS, args=(t, x0, y_meas, model, theta_m, var_exp, priorFIM_m))
    elif DC_PE == 'modE':
        pe_sol1 = minimize(mbdoepe_modE, phi_ini, method='Nelder-Mead', bounds=DS, args=(t, x0, y_meas, model, theta_m, var_exp, priorFIM_m))
        pe_sol = minimize(mbdoepe_modE, pe_sol1.x, method='SLSQP', bounds=DS, args=(t, x0, y_meas, model, theta_m, var_exp, priorFIM_m))

    return pe_sol.x


def mbdoepe_D(phi, t, x0, y_meas, model, theta_m, var_exp, priorFIM_m):
    FIM = FIM_fun(phi, t, model, theta_m[-1][0], var_exp, x0, y_meas) + np.sum(priorFIM_m[-1][0], axis=0)
    index0 = np.where(np.sum(np.vstack((np.diag(FIM) == 0, np.isnan(np.diag(FIM)))), axis=0))[0]
    objective = -np.log(np.linalg.det(np.delete(np.delete(FIM, index0, -1), index0, -2))+1e-10)

    return objective


def mbdoepe_E(phi, t, x0, y_meas, model, theta_m, var_exp, priorFIM_m):
    FIM = FIM_fun(phi, t, model, theta_m[-1][0], var_exp, x0, y_meas) + np.sum(priorFIM_m[-1][0], axis=0)
    Eig, _ = np.linalg.eig(FIM)
    objective = -np.min(Eig)

    return objective


def mbdoepe_modE(phi, t, x0, y_meas, model, theta_m, var_exp, priorFIM_m):
    FIM = FIM_fun(phi, t, model, theta_m[-1][0], var_exp, x0, y_meas) + np.sum(priorFIM_m[-1][0], axis=0)
    Eig, _ = np.linalg.eig(FIM)
    objective = np.max(Eig)/np.min(Eig)

    return objective


# ## MD criteria
# calculate HR
def mbdoemd_HR(phi, t, x0, y_meas, model_set, theta, newDC=None, lambda_f=None, u=None, y_exp=None, DS=None, var_exp=None):

    num_models = len(model_set)
    dc = 0

    for km_num_a in range(num_models-1):
        km_a = model_set[km_num_a]
        if km_a != None:
            for km_num_b in range(km_num_a, num_models):  # iteration from num_model_a + 1 up to total num_models
                km_b = model_set[km_num_b]
                if km_b != None:
                    y_hat_ma = y_meas(odeint(km_a, x0(phi), t, args=(phi, theta[km_num_a]))[1:])
                    y_hat_mb = y_meas(odeint(km_b, x0(phi), t, args=(phi, theta[km_num_b]))[1:])
                    dc += np.sum((np.subtract(y_hat_ma, y_hat_mb))**2, axis=(0, 1))

    if newDC == True:
        return -1*mbdoe_new(phi, dc, lambda_f, u, y_exp, model_set, DS, theta, x0, y_meas, t, var_exp)
    return -1*dc


def mbdoemd_HR_post(phi, t, x0, y_meas, model_set, theta):
    dc = mbdoemd_HR(phi, t, x0, y_meas,model_set, theta)
    return -1*dc


# calculate BH
def mbdoemd_BH(phi, t, x0, y_meas, model_set, prob, theta, var_exp, prior, y_exp, newDC=None, lambda_f=None, u=None, DS=None):  # all models or only not None models?

    num_models = len(model_set)
    n_y = np.shape(y_exp)[-1]
    if n_y == 0:
        n_y = np.shape(in_silico_exp(model_set[0], var_exp, phi, x0, y_meas, t, theta[0]))[-1]
    dc = 0

    for km_num_a in range(num_models-1):
        km_a = model_set[km_num_a]
        if km_a != None:

            pe_a = theta[km_num_a]
            y_hat_ma = y_meas(odeint(km_a, x0(phi), t, args=(phi, pe_a))[1:])

            exp_sen_a = sens_fun(phi, t, km_a, pe_a, x0, y_meas)
            exp_FIM_a = dynFIM_fun(phi, t, km_a, pe_a, var_exp, x0, y_meas) + prior[km_num_a]
            try:
                index0 = np.where(np.sum(np.vstack((np.diag(np.sum(exp_FIM_a, axis=0)) == 0, np.isnan(np.diag(np.sum(exp_FIM_a, axis=0))))), axis=0))[0]
                redFIM = np.delete(np.delete(exp_FIM_a, index0, -1), index0, -2)
                invFIM = np.linalg.inv(redFIM)
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov_a = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp
            except:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(exp_FIM_a, axis=0)) == 0, np.isnan(np.diag(np.sum(exp_FIM_a, axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(exp_FIM_a, index0, -1), index0, -2)
                invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov_a = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp

            index0 = np.where(np.sum(
                np.vstack((np.diag(np.sum(exp_pcov_a, axis=0)) == 0, np.isnan(np.diag(np.sum(exp_pcov_a, axis=0))))),
                axis=0))[0]

            redFIM = np.delete(np.delete(exp_pcov_a, index0, -1), index0, -2)
            exp_pcov_ai = np.linalg.inv(redFIM)
            if index0.size != 0:
                exp_pcov_ai = [exp_pcov_ai := np.insert(np.insert(exp_pcov_ai, s, 0, -1), s, 0, -2) for s in index0][-1]

            for km_num_b in range(km_num_a, num_models):
                km_b = model_set[km_num_b]
                if km_b != None:

                    pe_b = theta[km_num_b]
                    y_hat_mb = y_meas(odeint(km_b, x0(phi), t, args=(phi, pe_b))[1:])

                    exp_sen_b = sens_fun(phi, t, km_b, pe_b, x0, y_meas)
                    exp_FIM_b = dynFIM_fun(phi, t, km_b, pe_b, var_exp, x0, y_meas) + prior[km_num_b]
                    try:
                        index0 = np.where(np.sum(
                            np.vstack((np.diag(np.sum(exp_FIM_b, axis=0)) == 0,
                                       np.isnan(np.diag(np.sum(exp_FIM_b, axis=0))))),
                            axis=0))[0]
                        redFIM = np.delete(np.delete(exp_FIM_b, index0, -1), index0, -2)
                        invFIM = np.linalg.inv(redFIM)
                        if index0.size != 0:
                            invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                        exp_pcov_b = np.matmul(exp_sen_b.transpose((0, 2, 1)), np.matmul(invFIM, exp_sen_b)) + var_exp
                    except:
                        index0 = np.where(np.sum(
                            np.vstack((np.diag(np.sum(exp_FIM_b, axis=0)) == 0,
                                       np.isnan(np.diag(np.sum(exp_FIM_b, axis=0))))),
                            axis=0))[0]
                        redFIM = np.delete(np.delete(exp_FIM_b, index0, -1), index0, -2)
                        invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                        if index0.size != 0:
                            invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                        exp_pcov_b = np.matmul(exp_sen_b.transpose((0, 2, 1)),
                                               np.matmul(invFIM, exp_sen_b)) + var_exp
                    index0 = np.where(np.sum(
                        np.vstack((np.diag(np.sum(exp_pcov_b, axis=0)) == 0,
                                   np.isnan(np.diag(np.sum(exp_pcov_b, axis=0))))),
                        axis=0))[0]
                    redFIM = np.delete(np.delete(exp_pcov_b, index0, -1), index0, -2)
                    exp_pcov_bi = np.linalg.inv(redFIM)
                    if index0.size != 0:
                        exp_pcov_bi = [exp_pcov_bi := np.insert(np.insert(exp_pcov_bi, s, 0, -1), s, 0, -2) for s in index0][-1]

                    resi = np.expand_dims(np.subtract(y_hat_ma, y_hat_mb), 2)
                    t1 = np.trace(np.matmul(exp_pcov_a, exp_pcov_bi) + np.matmul(exp_pcov_b, exp_pcov_ai) - 2 * np.eye(n_y), axis1=1, axis2=2)
                    t2 = np.sum(resi * np.matmul(exp_pcov_ai + exp_pcov_bi, resi), axis=(1, 2))
                    dc += np.sum(prob[km_num_a] * prob[km_num_b] * (t1 + t2), 0)
    if newDC == True:
        return -1*mbdoe_new(phi, dc, lambda_f, u, y_exp, model_set, DS, theta, x0, y_meas, t, var_exp)

    return -1*dc


def mbdoemd_BH_post(phi, t, x0, y_meas, model_set, prob, theta, var_exp, priorFIM, y_exp):

    num_models = len(model_set)
    n_y = np.shape(y_exp)[-1]
    if n_y == 0:
        n_y = np.shape(in_silico_exp(model_set[0], var_exp, phi, x0, y_meas, t, theta[0]))[-1]
    dc = 0

    for km_num_a in range(num_models-1):
        km_a = model_set[km_num_a]
        if km_a != None:

            pe_a = theta[km_num_a]
            y_hat_ma = y_meas(odeint(km_a, x0(phi), t, args=(phi, pe_a))[1:])

            exp_sen_a = sens_fun(phi, t, km_a, pe_a, x0, y_meas)
            try:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(priorFIM[km_num_a], axis=0)) == 0,
                               np.isnan(np.diag(np.sum(priorFIM[km_num_a], axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(priorFIM[km_num_a], index0, -1), index0, -2)
                invFIM = np.linalg.inv(redFIM)
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov_a = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp
            except:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(priorFIM[km_num_a], axis=0)) == 0,
                               np.isnan(np.diag(np.sum(priorFIM[km_num_a], axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(priorFIM[km_num_a], index0, -1), index0, -2)
                invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov_a = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp

            index0 = np.where(np.sum(
                np.vstack((np.diag(np.sum(exp_pcov_a, axis=0)) == 0,
                           np.isnan(np.diag(np.sum(exp_pcov_a, axis=0))))),
                axis=0))[0]
            redFIM = np.delete(np.delete(exp_pcov_a, index0, -1), index0, -2)
            exp_pcov_ai = np.linalg.inv(redFIM)
            if index0.size != 0:
                exp_pcov_ai = [exp_pcov_ai := np.insert(np.insert(exp_pcov_ai, s, 0, -1), s, 0, -2) for s in index0][-1]

            for km_num_b in range(km_num_a, num_models):
                km_b = model_set[km_num_b]
                if km_b != None:

                    pe_b = theta[km_num_b]
                    y_hat_mb = y_meas(odeint(km_b, x0(phi), t, args=(phi, pe_b))[1:])

                    exp_sen_b = sens_fun(phi, t, km_b, pe_b, x0, y_meas)
                    try:
                        index0 = np.where(np.sum(
                            np.vstack((np.diag(np.sum(priorFIM[km_num_b], axis=0)) == 0,
                                       np.isnan(np.diag(np.sum(priorFIM[km_num_b], axis=0))))),
                            axis=0))[0]
                        redFIM = np.delete(np.delete(priorFIM[km_num_b], index0, -1), index0, -2)
                        invFIM = np.linalg.inv(redFIM)
                        if index0.size != 0:
                            invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                        exp_pcov_b = np.matmul(exp_sen_b.transpose((0, 2, 1)), np.matmul(invFIM, exp_sen_b)) + var_exp
                    except:
                        index0 = np.where(np.sum(
                            np.vstack((np.diag(np.sum(priorFIM[km_num_b], axis=0)) == 0,
                                       np.isnan(np.diag(np.sum(priorFIM[km_num_b], axis=0))))),
                            axis=0))[0]
                        redFIM = np.delete(np.delete(priorFIM[km_num_b], index0, -1), index0, -2)
                        invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                        if index0.size != 0:
                            invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                        exp_pcov_b = np.matmul(exp_sen_b.transpose((0, 2, 1)),
                                               np.matmul(invFIM, exp_sen_b)) + var_exp
                    index0 = np.where(np.sum(
                        np.vstack((np.diag(np.sum(exp_pcov_b, axis=0)) == 0,
                                   np.isnan(np.diag(np.sum(exp_pcov_b, axis=0))))),
                        axis=0))[0]
                    redFIM = np.delete(np.delete(exp_pcov_b, index0, -1), index0, -2)
                    exp_pcov_bi = np.linalg.inv(redFIM)
                    if index0.size != 0:
                        exp_pcov_bi = [exp_pcov_bi := np.insert(np.insert(exp_pcov_bi, s, 0, -1), s, 0, -2) for s in index0][-1]

                    resi = np.expand_dims(np.subtract(y_hat_ma, y_hat_mb), 2)
                    t1 = np.trace(np.matmul(exp_pcov_a, exp_pcov_bi) + np.matmul(exp_pcov_b, exp_pcov_ai) - 2 * np.eye(n_y), axis1=1, axis2=2)
                    t2 = np.sum(resi * np.matmul(exp_pcov_ai + exp_pcov_bi, resi), axis=(1, 2))
                    dc += np.sum(prob[km_num_a] * prob[km_num_b] * (t1 + t2), 0)

    return dc


# BuzziFerraris-Forzatti criterion from 1990
def mbdoemd_BF(phi, t, x0, y_meas, model_set, theta, prior, var_exp, newDC=None, lambda_f=None, u=None, y_exp=None, DS=None):  # y dimension has changed, so need the index of y and its usage

    num_models = len(model_set)
    dc = 0

    for km_num_a in range(num_models-1):
        km_a = model_set[km_num_a]
        if km_a != None:

            pe_a = theta[km_num_a]
            y_hat_ma = y_meas(odeint(km_a, x0(phi), t, args=(phi, pe_a))[1:])

            exp_sen_a = sens_fun(phi, t, km_a, pe_a, x0, y_meas)
            exp_FIM_a = dynFIM_fun(phi, t, km_a, pe_a, var_exp, x0, y_meas) + prior[km_num_a]
            try:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(exp_FIM_a, axis=0)) == 0,
                               np.isnan(np.diag(np.sum(exp_FIM_a, axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(exp_FIM_a, index0, -1), index0, -2)
                invFIM = np.linalg.inv(redFIM)
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov_a = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp
            except:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(exp_FIM_a, axis=0)) == 0,
                               np.isnan(np.diag(np.sum(exp_FIM_a, axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(exp_FIM_a, index0, -1), index0, -2)
                invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov_a = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp

            for km_num_b in range(km_num_a, num_models):
                km_b = model_set[km_num_b]
                if km_b != None:

                    pe_b = theta[km_num_b]
                    y_hat_mb = y_meas(odeint(km_b, x0(phi), t, args=(phi, pe_b))[1:])

                    exp_sen_b = sens_fun(phi, t, km_b, pe_b, x0, y_meas)
                    exp_FIM_b = dynFIM_fun(phi, t, km_b, pe_b, var_exp, x0, y_meas) + prior[km_num_b]
                    try:
                        index0 = np.where(np.sum(
                            np.vstack((np.diag(np.sum(exp_FIM_b, axis=0)) == 0,
                                       np.isnan(np.diag(np.sum(exp_FIM_b, axis=0))))),
                            axis=0))[0]
                        redFIM = np.delete(np.delete(exp_FIM_b, index0, -1), index0, -2)
                        invFIM = np.linalg.inv(redFIM)
                        if index0.size != 0:
                            invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                        exp_pcov_b = np.matmul(exp_sen_b.transpose((0, 2, 1)), np.matmul(invFIM, exp_sen_b)) + var_exp
                    except:
                        index0 = np.where(np.sum(
                            np.vstack((np.diag(np.sum(exp_FIM_b, axis=0)) == 0,
                                       np.isnan(np.diag(np.sum(exp_FIM_b, axis=0))))),
                            axis=0))[0]
                        redFIM = np.delete(np.delete(exp_FIM_b, index0, -1), index0, -2)
                        invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                        if index0.size != 0:
                            invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                        exp_pcov_b = np.matmul(exp_sen_b.transpose((0, 2, 1)),
                                               np.matmul(invFIM, exp_sen_b)) + var_exp
                    pcov_ab = exp_pcov_a + exp_pcov_b
                    index0 = np.where(np.sum(
                        np.vstack((np.diag(np.sum(pcov_ab, axis=0)) == 0,
                                   np.isnan(np.diag(np.sum(pcov_ab, axis=0))))),
                        axis=0))[0]
                    redFIM = np.delete(np.delete(pcov_ab, index0, -1), index0, -2)
                    invFIM = np.linalg.inv(redFIM)
                    if index0.size != 0:
                        pcov_ab = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                    resi = np.expand_dims(np.subtract(y_hat_ma, y_hat_mb), 2)
                    dc += np.sum(resi*np.matmul(pcov_ab, resi), axis=(0, 1, 2)) + np.sum(np.trace(2 * np.matmul(var_exp, pcov_ab), axis1=1, axis2=2))
    if newDC == True:
        return -1*mbdoe_new(phi, dc, lambda_f, u, y_exp, model_set, DS, theta, x0, y_meas, t, var_exp)

    return -1.0 * dc


def mbdoemd_BF_post(phi, t, x0, y_meas, model_set, theta, priorFIM, var_exp):
    # # this must be done before new probs (models None not changed)
    # execute experiment (in-silico)
    # new PE

    num_models = len(model_set)
    dc = 0

    for km_num_a in range(num_models-1):
        km_a = model_set[km_num_a]
        if km_a != None:

            pe_a = theta[km_num_a]
            y_hat_ma = y_meas(odeint(km_a, x0(phi), t, args=(phi, pe_a))[1:])
            exp_sen_a = sens_fun(phi, t, km_a, pe_a, x0, y_meas)
            try:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(priorFIM[km_num_a], axis=0)) == 0,
                               np.isnan(np.diag(np.sum(priorFIM[km_num_a], axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(priorFIM[km_num_a], index0, -1), index0, -2)
                invFIM = np.linalg.inv(redFIM)
                if index0.size != 0:
                   invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov_a = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp
            except:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(priorFIM[km_num_a], axis=0)) == 0,
                               np.isnan(np.diag(np.sum(priorFIM[km_num_a], axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(priorFIM[km_num_a], index0, -1), index0, -2)
                invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov_a = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp

            for km_num_b in range(km_num_a, num_models):
                km_b = model_set[km_num_b]
                if km_b != None:

                    pe_b = theta[km_num_b]
                    y_hat_mb = y_meas(odeint(km_b, x0(phi), t, args=(phi, pe_b))[1:])

                    exp_sen_b = sens_fun(phi, t, km_b, pe_b, x0, y_meas)
                    try:
                        index0 = np.where(np.sum(
                            np.vstack((np.diag(np.sum(priorFIM[km_num_b], axis=0)) == 0,
                                       np.isnan(np.diag(np.sum(priorFIM[km_num_b], axis=0))))),
                            axis=0))[0]
                        redFIM = np.delete(np.delete(priorFIM[km_num_b], index0, -1), index0, -2)
                        invFIM = np.linalg.inv(redFIM)
                        if index0.size != 0:
                            invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                        exp_pcov_b = np.matmul(exp_sen_b.transpose((0, 2, 1)), np.matmul(invFIM, exp_sen_b)) + var_exp
                    except:
                        index0 = np.where(np.sum(
                            np.vstack((np.diag(np.sum(priorFIM[km_num_b], axis=0)) == 0,
                                       np.isnan(np.diag(np.sum(priorFIM[km_num_b], axis=0))))),
                            axis=0))[0]
                        redFIM = np.delete(np.delete(priorFIM[km_num_b], index0, -1), index0, -2)
                        invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                        if index0.size != 0:
                            invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                        exp_pcov_b = np.matmul(exp_sen_b.transpose((0, 2, 1)),
                                               np.matmul(invFIM, exp_sen_b)) + var_exp

                    pcov_ab = exp_pcov_a + exp_pcov_b
                    index0 = np.where(np.sum(
                        np.vstack((np.diag(np.sum(pcov_ab, axis=0)) == 0,
                                   np.isnan(np.diag(np.sum(pcov_ab, axis=0))))),
                        axis=0))[0]
                    redFIM = np.delete(np.delete(pcov_ab, index0, -1), index0, -2)
                    invFIM = np.linalg.inv(redFIM)
                    if index0.size != 0:
                        pcov_ab = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                    resi = np.expand_dims(np.subtract(y_hat_ma, y_hat_mb), 2)
                    dc += np.sum(resi * np.matmul(pcov_ab, resi), axis=(0, 1, 2)) + np.sum(np.trace(2 * np.matmul(var_exp, pcov_ab), axis1=1, axis2=2))  # t0 does not count as

    return dc


# Akaikes DC (not probability weighted)
def mbdoemd_ADC(phi, t, x0, y_meas, model_set, theta, var_exp, prior, newDC=None, lambda_f=None, u=None, y_exp=None, DS=None):
    num_models = len(model_set)

    delta = np.zeros((num_models, num_models, np.shape(t)[0]-1))

    for km_num_a in range(num_models):
        km_a = model_set[km_num_a]
        if km_a != None:

            pe_a = theta[km_num_a]
            y_hat_ma = y_meas(odeint(km_a, x0(phi), t, args=(phi, pe_a))[1:])

            exp_sen_a = sens_fun(phi, t, km_a, pe_a, x0, y_meas)
            exp_FIM_a = dynFIM_fun(phi, t, km_a, pe_a, var_exp, x0, y_meas) + prior[km_num_a]
            try:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(exp_FIM_a, axis=0)) == 0,
                               np.isnan(np.diag(np.sum(exp_FIM_a, axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(exp_FIM_a, index0, -1), index0, -2)
                invFIM = np.linalg.inv(redFIM)
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov_a = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp
            except:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(exp_FIM_a, axis=0)) == 0,
                               np.isnan(np.diag(np.sum(exp_FIM_a, axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(exp_FIM_a, index0, -1), index0, -2)
                invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov_a = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp

            for km_num_b in range(num_models):
                km_b = model_set[km_num_b]
                if km_b != None:

                    pe_b = theta[km_num_b]
                    y_hat_mb = y_meas(odeint(km_b, x0(phi), t, args=(phi, pe_b))[1:])
                    '''
                    exp_sen_b = sens_fun(phi, t, km_b, pe_b, x0, y_meas)
                    exp_FIM_b = dynFIM_fun(phi, t, km_b, pe_b, var_exp, x0, y_meas) + prior[km_num_b]
                    try:
                        index0 = np.where(np.sum(
                            np.vstack((np.diag(np.sum(exp_FIM_b, axis=0)) == 0,
                                       np.isnan(np.diag(np.sum(exp_FIM_b, axis=0))))),
                            axis=0))[0]
                        redFIM = np.delete(np.delete(exp_FIM_b, index0, -1), index0, -2)
                        invFIM = np.linalg.inv(redFIM)
                        if index0.size != 0:
                            invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                        exp_pcov_b = np.matmul(exp_sen_b.transpose((0, 2, 1)), np.matmul(invFIM, exp_sen_b)) + var_exp
                    except:
                        index0 = np.where(np.sum(
                            np.vstack((np.diag(np.sum(exp_FIM_b, axis=0)) == 0,
                                       np.isnan(np.diag(np.sum(exp_FIM_b, axis=0))))),
                            axis=0))[0]
                        redFIM = np.delete(np.delete(exp_FIM_b, index0, -1), index0, -2)
                        invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                        if index0.size != 0:
                            invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                        exp_pcov_b = np.matmul(exp_sen_b.transpose((0, 2, 1)),
                                               np.matmul(invFIM, exp_sen_b)) + var_exp
                    '''
                    pcov_ab = exp_pcov_a #+exp_pcov_b
                    index0 = np.where(np.sum(
                        np.vstack((np.diag(np.sum(pcov_ab, axis=0)) == 0,
                                   np.isnan(np.diag(np.sum(pcov_ab, axis=0))))),
                        axis=0))[0]
                    redFIM = np.delete(np.delete(pcov_ab, index0, -1), index0, -2)
                    invFIM = np.linalg.inv(redFIM)
                    if index0.size != 0:
                        pcov_ab = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                    resi = np.expand_dims(np.subtract(y_hat_ma, y_hat_mb), 2)
                    delta[km_num_a, km_num_b] = -0.5*np.sum(resi * np.matmul(pcov_ab, resi), axis=(1, 2)) + (np.shape(pe_a)[0]-np.shape(pe_b)[0])  # delta: M, M, t-1

    w = 1/np.sum(np.prod(np.exp(delta), axis=2), axis=1)
    dc = np.sum(w)
    if newDC == True:
        return -1*mbdoe_new(phi, dc, lambda_f, u, y_exp, model_set, DS, theta, x0, y_meas, t, var_exp)

    return -1 * dc


def mbdoemd_ADC_post(phi, t, x0, y_meas, model_set, theta, var_exp, priorFIM):
    num_models = len(model_set)

    delta = np.zeros((num_models, num_models, np.shape(t)[0]-1))

    for km_num_a in range(num_models):
        km_a = model_set[km_num_a]
        if km_a != None:

            pe_a = theta[km_num_a]
            y_hat_ma = y_meas(odeint(km_a, x0(phi), t, args=(phi, pe_a))[1:])

            exp_sen_a = sens_fun(phi, t, km_a, pe_a, x0, y_meas)
            try:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(priorFIM[km_num_a], axis=0)) == 0,
                               np.isnan(np.diag(np.sum(priorFIM[km_num_a], axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(priorFIM[km_num_a], index0, -1), index0, -2)
                invFIM = np.linalg.inv(redFIM)
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov_a = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp
            except:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(priorFIM[km_num_a], axis=0)) == 0,
                               np.isnan(np.diag(np.sum(priorFIM[km_num_a], axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(priorFIM[km_num_a], index0, -1), index0, -2)
                invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov_a = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp

            for km_num_b in range(num_models):
                km_b = model_set[km_num_b]
                if km_b != None:

                    pe_b = theta[km_num_b]
                    y_hat_mb = y_meas(odeint(km_b, x0(phi), t, args=(phi, pe_b))[1:])

                    '''
                    exp_sen_b = sens_fun(phi, t, km_b, pe_b, x0, y_meas)
                    try:
                        index0 = np.where(np.sum(
                            np.vstack((np.diag(np.sum(priorFIM[km_num_b], axis=0)) == 0,
                                       np.isnan(np.diag(np.sum(priorFIM[km_num_b], axis=0))))),
                            axis=0))[0]
                        redFIM = np.delete(np.delete(priorFIM[km_num_b], index0, -1), index0, -2)
                        invFIM = np.linalg.inv(redFIM)
                        if index0.size != 0:
                            invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                        exp_pcov_b = np.matmul(exp_sen_b.transpose((0, 2, 1)), np.matmul(invFIM, exp_sen_b)) + var_exp
                    except:
                        index0 = np.where(np.sum(
                            np.vstack((np.diag(np.sum(priorFIM[km_num_b], axis=0)) == 0,
                                       np.isnan(np.diag(np.sum(priorFIM[km_num_b], axis=0))))),
                            axis=0))[0]
                        redFIM = np.delete(np.delete(priorFIM[km_num_b], index0, -1), index0, -2)
                        invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                        if index0.size != 0:
                            invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                        exp_pcov_b = np.matmul(exp_sen_b.transpose((0, 2, 1)),
                                               np.matmul(invFIM, exp_sen_b)) + var_exp
                    '''
                    pcov_ab = exp_pcov_a #+ exp_pcov_b
                    index0 = np.where(np.sum(
                        np.vstack((np.diag(np.sum(pcov_ab, axis=0)) == 0,
                                   np.isnan(np.diag(np.sum(pcov_ab, axis=0))))),
                        axis=0))[0]
                    redFIM = np.delete(np.delete(pcov_ab, index0, -1), index0, -2)
                    invFIM = np.linalg.inv(redFIM)
                    if index0.size != 0:
                        pcov_ab = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                    resi = np.expand_dims(np.subtract(y_hat_ma, y_hat_mb), 2)
                    delta[km_num_a, km_num_b] = -0.5*np.sum(resi * np.matmul(pcov_ab, resi), axis=(1, 2)) + (np.shape(pe_a)[0]-np.shape(pe_b)[0])  # delta: M, M, t-1

    w = 1/np.sum(np.prod(np.exp(delta), axis=2), axis=1)
    dc = np.sum(w)

    return dc


# calculate Jensen-Rényi divergence
def mbdoemd_JRD(phi, t, x0, y_meas, model_set, proba, theta, var_exp, prior, y_exp, newDC=None, lambda_f=None, u=None, DS=None):
    num_models = len(model_set)
    n_y = np.shape(y_exp)[-1]
    if n_y == 0:
        n_y = np.shape(in_silico_exp(model_set[0], var_exp, phi, x0, y_meas, t, theta[0]))[-1]

    y_hat_models0 = np.zeros((num_models, np.shape(t)[0]-1, n_y))
    exp_pcov0 = np.zeros((num_models, np.shape(t)[0]-1, n_y, n_y))
    det_exp_pcov0 = np.zeros((num_models, np.shape(t)[0]-1))

    ind = np.array(-1)

    for km_num_a in range(num_models):
        km_a = model_set[km_num_a]
        if km_a == None:
            ind = np.append(ind, km_num_a)
        else:
            y_temp = y_meas(odeint(km_a, x0(phi), t, args=(phi, theta[km_num_a]))[1:])
            y_hat_models0[km_num_a] = y_temp

            exp_sen_a = sens_fun(phi, t, km_a, theta[km_num_a], x0, y_meas)
            exp_FIM_a = dynFIM_fun(phi, t, km_a, theta[km_num_a], var_exp, x0, y_meas) + prior[km_num_a]

            try:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(exp_FIM_a, axis=0)) == 0,
                               np.isnan(np.diag(np.sum(exp_FIM_a, axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(exp_FIM_a, index0, -1), index0, -2)
                invFIM = np.linalg.inv(redFIM)
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov0[km_num_a] = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp
            except:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(exp_FIM_a, axis=0)) == 0,
                               np.isnan(np.diag(np.sum(exp_FIM_a, axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(exp_FIM_a, index0, -1), index0, -2)
                invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov0[km_num_a] = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp

            index0 = np.where(np.sum(
                np.vstack((np.diag(np.sum(exp_pcov0[km_num_a], axis=0)) == 0,
                           np.isnan(np.diag(np.sum(exp_pcov0[km_num_a], axis=0))))),
                axis=0))[0]
            det_exp_pcov0[km_num_a] = np.linalg.det(np.delete(np.delete(exp_pcov0[km_num_a], index0, -1), index0, -2))

    ind = np.delete(ind, 0)
    y_hat_models = np.delete(y_hat_models0, ind, axis=0)
    exp_pcov = np.delete(exp_pcov0, ind, axis=0)
    det_exp_pcov = np.delete(det_exp_pcov0, ind, axis=0)
    prob = np.delete(proba, ind, axis=0)
    prob = prob / np.sum(prob)
    log_exp_pcov = np.log(det_exp_pcov)

    """ Sum of entropies """
    T1 = np.sum(0.5 * prob * (log_exp_pcov.tranpose() + n_y  * np.log(4 * np.pi)), axis=(0, 1))

    """ Entropy of sum """
    # Diagonal elements: (i,i)
    T2 = np.sum(prob * prob / (2 ** (n_y / 2.) * np.sqrt(det_exp_pcov)).transpose(), axis=1)

    # Off-diagonal elements: (i,j)
    for i in range(np.shape(log_exp_pcov)[0]):
        # mu_i^T * inv(Si) * mu_i
        y_hat_i = np.expand_dims(y_hat_models[i], 2)
        iSmi = np.matmul(exp_pcov[i], y_hat_i)
        miiSmi = np.sum(y_hat_i * iSmi, axis=(1, 2))

        for j in range(i + 1, np.shape(log_exp_pcov)[0]):
            # mu_j^T * inv(Sj) * mu_j
            y_hat_j = np.expand_dims(y_hat_models[j], 2)
            iSmj = np.matmul(exp_pcov[j], y_hat_j)
            mjiSmj = np.sum(y_hat_j * iSmj, axis=(1, 2))

            # inv( inv(Si) + inv(Sj) )
            iSiS = exp_pcov[i] + exp_pcov[j]

            index0 = np.where(np.sum(
                np.vstack((np.diag(np.sum(iSiS, axis=0)) == 0,
                           np.isnan(np.diag(np.sum(iSiS, axis=0))))),
                axis=0))[0]
            redFIM = np.delete(np.delete(iSiS, index0, -1), index0, -2)
            iiSiS = np.linalg.inv(np.sum(redFIM, axis=0))
            if index0.size != 0:
                iiSiS = [iiSiS := np.insert(np.insert(iiSiS, s, 0, -1), s, 0, -2) for s in index0][-1]
            liSiS = np.log(np.linalg.det(redFIM))

            # mu_ij^T * inv( inv(Si) + inv(Sj) ) * mu_ij
            mij = iSmi + iSmj
            iiSSj = np.sum(mij * np.matmul(iiSiS, mij), axis=(1, 2))

            phi_JR = miiSmi + mjiSmj - iiSSj + log_exp_pcov[i] + log_exp_pcov[j] + liSiS

            T2 += 2 * prob[i] * prob[j] * np.exp(-0.5 * phi_JR)

    T2 = np.sum(n_y / 2 * np.log(2 * np.pi) - np.log(T2), 0)

    dc = T2 - T1
    if newDC == True:
        return -1*mbdoe_new(phi, dc, lambda_f, u, y_exp, model_set, DS, theta, x0, y_meas, t, var_exp)

    return -1*dc


def mbdoemd_JRD_post(phi, t, x0, y_meas, model_set, proba, theta, var_exp, priorFIM, y_exp):
    num_models = len(model_set)
    n_y = np.shape(y_exp)[-1]
    if n_y == 0:
        n_y = np.shape(in_silico_exp(model_set[0], var_exp, phi, x0, y_meas, t, theta[0]))[-1]

    y_hat_models0 = np.zeros((num_models, np.shape(t)[0]-1, n_y))
    exp_pcov0 = np.zeros((num_models, np.shape(t)[0]-1, n_y, n_y))
    det_exp_pcov0 = np.zeros((num_models, np.shape(t)[0]-1))

    ind = np.array(-1)

    for km_num_a in range(num_models):
        km_a = model_set[km_num_a]
        if km_a == None:
            ind = np.append(ind, km_num_a)
        else:
            y_temp = y_meas(odeint(km_a, x0(phi), t, args=(phi, theta[km_num_a]))[1:])
            y_hat_models0[km_num_a] = y_temp

            exp_sen_a = sens_fun(phi, t, km_a, theta[km_num_a], x0, y_meas)
            try:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(priorFIM[km_num_a], axis=0)) == 0,
                               np.isnan(np.diag(np.sum(priorFIM[km_num_a], axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(priorFIM[km_num_a], index0, -1), index0, -2)
                invFIM = np.linalg.inv(redFIM)
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov0[km_num_a] = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp
            except:
                index0 = np.where(np.sum(
                    np.vstack((np.diag(np.sum(priorFIM[km_num_a], axis=0)) == 0,
                               np.isnan(np.diag(np.sum(priorFIM[km_num_a], axis=0))))),
                    axis=0))[0]
                redFIM = np.delete(np.delete(priorFIM[km_num_a], index0, -1), index0, -2)
                invFIM = np.linalg.inv(np.sum(redFIM, axis=0))
                if index0.size != 0:
                    invFIM = [invFIM := np.insert(np.insert(invFIM, s, 0, -1), s, 0, -2) for s in index0][-1]

                exp_pcov0[km_num_a] = np.matmul(exp_sen_a.transpose((0, 2, 1)),
                                       np.matmul(invFIM, exp_sen_a)) + var_exp

            index0 = np.where(np.sum(
                np.vstack((np.diag(np.sum(exp_pcov0[km_num_a], axis=0)) == 0,
                           np.isnan(np.diag(np.sum(exp_pcov0[km_num_a], axis=0))))),
                axis=0))[0]
            det_exp_pcov0[km_num_a] = np.linalg.det(np.delete(np.delete(exp_pcov0[km_num_a], index0, -1), index0, -2))

    ind = np.delete(ind, 0)
    y_hat_models = np.delete(y_hat_models0, ind, axis=0)
    exp_pcov = np.delete(exp_pcov0, ind, axis=0)
    det_exp_pcov = np.delete(det_exp_pcov0, ind, axis=0)
    prob = np.delete(proba, ind, axis=0)
    prob = prob / np.sum(prob)
    log_exp_pcov = np.log(det_exp_pcov)

    """ Sum of entropies """
    T1 = np.sum(0.5 * log_exp_pcov.transpose() + prob * ((n_y / 2) * np.log(4 * np.pi)), axis=(0, 1))

    """ Entropy of sum """
    # Diagonal elements: (i,i)
    T2 = np.sum(prob * prob / (2 ** (n_y / 2.) * np.sqrt(det_exp_pcov)).transpose(), axis=1)

    # Off-diagonal elements: (i,j)
    for i in range(np.shape(log_exp_pcov)[0]):
        # mu_i^T * inv(Si) * mu_i
        y_hat_i = np.expand_dims(y_hat_models[i], 2)
        iSmi = np.matmul(exp_pcov[i], y_hat_i)
        miiSmi = np.sum(y_hat_i * iSmi, axis=(1, 2))

        for j in range(i + 1, np.shape(log_exp_pcov)[0]):
            # mu_j^T * inv(Sj) * mu_j
            y_hat_j = np.expand_dims(y_hat_models[j], 2)
            iSmj = np.matmul(exp_pcov[j], y_hat_j)
            mjiSmj = np.sum(y_hat_j * iSmj, axis=(1, 2))

            # inv( inv(Si) + inv(Sj) )
            iSiS = exp_pcov[i] + exp_pcov[j]

            index0 = np.where(np.sum(
                np.vstack((np.diag(np.sum(iSiS, axis=0)) == 0,
                           np.isnan(np.diag(np.sum(iSiS, axis=0))))),
                axis=0))[0]
            redFIM = np.delete(np.delete(iSiS, index0, -1), index0, -2)
            iiSiS = np.linalg.inv(np.sum(redFIM, axis=0))
            if index0.size != 0:
                iiSiS = [iiSiS := np.insert(np.insert(iiSiS, s, 0, -1), s, 0, -2) for s in index0][-1]
            liSiS = np.log(np.linalg.det(redFIM))

            # mu_ij^T * inv( inv(Si) + inv(Sj) ) * mu_ij
            mij = iSmi + iSmj
            iiSSj = np.sum(mij * np.matmul(iiSiS, mij), axis=(1, 2))

            phi_JR = miiSmi + mjiSmj - iiSSj + log_exp_pcov[i] + log_exp_pcov[j] + liSiS

            T2 += 2 * prob[i] * prob[j] * np.exp(-0.5 * phi_JR)

    T2 = np.sum(n_y / 2 * np.log(2 * np.pi) - np.log(T2), 0)

    dc = T2 - T1
    return dc


def mbdoe_new(phi, psi_core, lambda_f, u, y_exp, model_set, DS, theta, x0, y_meas, t, var_exp):
    # lambda_f is percentual factor of Information-Exploration penalty, e.g. 0.05 = 5%
    # here: only active models in model_set
    # model_set and theta is already [-1]
    chi_exp = np.empty((np.shape(u)[0], 0))
    for km_num in range(len(model_set)):
        if model_set[km_num] == None:
            continue
        chi_model = []
        for exp in range(np.shape(u)[0]):
            chi_model += [lof_fun(theta[km_num], [u[exp]], [y_exp[exp]], model_set[km_num], x0, y_meas, t, var_exp)]
        chi_model = np.reshape(np.array(chi_model), (-1, 1))
        chi_exp = np.hstack((chi_exp, chi_model))
    I = np.std(chi_exp, axis=1)/np.min(chi_exp, axis=1) - np.average(np.std(chi_exp, axis=1)/np.min(chi_exp, axis=1), axis=0)*(1+1/(np.shape(u)[0])) # dim: 1*N_exp - scalar
    I = np.reshape(I, (1, -1)) # normalisation: I/np.sum(I)
    delta_u = u-phi # dim: N_exp*N_ds
    w = np.zeros((len(DS),1)) # dim: N_ds * 1
    if np.shape(u)[0] > 1:
        for d in range(len(DS)):
            w[d] = stats.linregress(np.interp(u[:,d], DS[d], [0, 1]), I[0])[0] # here: possible dependency on R-value e.g. R<0.2 => w=0
    if np.sum(np.abs(w)) != 0:
        w = w/np.sum(np.abs(w))
    u_rel = np.zeros(np.shape(delta_u))
    for d in range(len(DS)):
        u_rel[:, d] = np.interp(np.abs(delta_u[:, d]), DS[d], [0, 1])
    u_rel = np.average(np.linalg.norm(u_rel, axis=1))
    DC = psi_core*(1 + lambda_f*np.matmul(np.matmul(I, delta_u), w)[0, 0] - np.exp(-u_rel)/(np.shape(u)[0]) )
    return DC


# calculate D maps with BF
def d_map_prior(MBDoE, bounds, t, x0, y_meas, model_set, theta, priorFIM, var_exp, prob, y_exp, num, num3, u, model_set_0, theta_t, param_bounds):

    phi3_val = np.linspace(bounds[2][0], bounds[2][1], num3)
    D_map = np.zeros((num, num, num3))
    xx = 0
    yy = 0
    zz = 0
    for phi1 in np.linspace(bounds[0][0], bounds[0][1], num):
        for phi2 in np.linspace(bounds[1][0], bounds[1][1], num):
            for phi3 in phi3_val:

                if MBDoE == 'HR':
                    DC = mbdoemd_HR([phi1, phi2, phi3], t, x0, y_meas, model_set, theta[-1])
                elif MBDoE == 'BH':
                    DC = mbdoemd_BH([phi1, phi2, phi3], t, x0, y_meas, model_set, prob[-1], theta[-1], var_exp, priorFIM[-1], y_exp)
                elif MBDoE == 'BF':
                    DC = mbdoemd_BF([phi1, phi2, phi3], t, x0, y_meas, model_set, theta[-1], priorFIM[-1], var_exp)
                elif MBDoE == 'ADC':
                    DC = mbdoemd_ADC([phi1, phi2, phi3], t, x0, y_meas, model_set, theta[-1], var_exp, priorFIM[-1])
                elif MBDoE == 'JRD':
                    DC = mbdoemd_JRD([phi1, phi2, phi3], t, x0, y_meas, model_set, prob[-1], theta[-1], var_exp, priorFIM[-1], y_exp)
                elif MBDoE == 'prob':
                    u_t = np.vstack((u, [phi1, phi2, phi3]))

                    y_exp_temp = in_silico_exp(model_set_0[0], var_exp, u_t[-1], x0, y_meas, t, theta_t)
                    y_exp_t = np.append(y_exp, y_exp_temp)
                    y_exp_t = np.reshape(y_exp_t, (np.shape(u_t)[0], np.shape(t)[0] - 1, -1))

                    theta_temp, chisq_temp, dof_temp = PE(u_t, y_exp_t, model_set_0, x0, y_meas, t, var_exp, theta,
                                                          param_bounds)

                    prob_temp = Model_prob(4, model_set, chisq_temp, dof_temp, y_exp_t, t, x0, y_meas, theta, u_t,
                                           var_exp)

                    DC = prob[-1][0] - prob_temp[0] # negative formulation
                else:
                    print("Please chose a valid MBDoE criterion")
                    DC = None
                    exit()

                D_map[xx, yy, zz] = -1*DC
                zz += 1
            yy += 1
            zz = 0
        xx += 1
        yy = 0
        zz = 0
    return D_map


def d_map_FIM(bounds, t, x0, y_meas, model_set, theta, priorFIM, var_exp, num, num3):
    phi3_val = np.linspace(bounds[2][0], bounds[2][1], num3)
    D_map = np.zeros((num, num, num3))
    xx = 0
    yy = 0
    zz = 0
    for phi1 in np.linspace(bounds[0][0], bounds[0][1], num):
        for phi2 in np.linspace(bounds[1][0], bounds[1][1], num):
            for phi3 in phi3_val:
                D_map[xx, yy, zz] = av_FIM([phi1, phi2, phi3], t, model_set, theta[-1], priorFIM[-1], var_exp, x0, y_meas)
                zz += 1
            yy += 1
            zz = 0
        xx += 1
        yy = 0
        zz = 0
    return D_map


# plot D maps
def plot_D_map_prior(MBDoE, bounds, D_map, fixcb=False):

    # 3D surface map (av and highest DC)
    """
    global D_post_map
    D_map = np.zeros((num, num, num))
    xx = 0
    yy = 0
    zz = 0
    for phi1 in np.linspace(bounds[0][0], bounds[0][1], num):
        for phi2 in np.linspace(bounds[1][0], bounds[1][1], num):
            for phi3 in np.linspace(bounds[2][0], bounds[2][1], num):
                D_map[xx, yy, zz] = -1*mbdoemd_BF([phi1, phi2, phi3], t, model_set, theta[-1], priorFIM[-1], n_phi, var_exp)
                zz += 1
            yy += 1
            zz = 0
        xx += 1
        yy = 0
        zz = 0
    fig = plt.figure()
    axes = plt.axes(projection ='3d')
    xx, yy = np.meshgrid(np.linspace(bounds[0][0], bounds[0][1], num), np.linspace(bounds[1][0], bounds[1][1], num))
    scamap = plt.cm.ScalarMappable(cmap='inferno')
    maxDC = scamap.to_rgba(np.amax(D_map, axis=2))
    axes.plot_surface(xx, yy, D_map.sum(axis=2)/num, facecolors=maxDC, cmap='inferno')
    plt.colorbar(scamap)
    plt.show()



    """

    # 2D color map (av DC)
    """
    global D_post_map
    D_map = np.zeros((num, num, num))
    xx = 0
    yy = 0
    zz = 0
    for phi1 in np.linspace(bounds[0][0], bounds[0][1], num):
        for phi2 in np.linspace(bounds[1][0], bounds[1][1], num):
            for phi3 in np.linspace(bounds[2][0], bounds[2][1], num):
                D_map[xx, yy, zz] = -1*mbdoemd_BF([phi1, phi2, phi3], t, model_set, theta[-1], priorFIM[-1], n_phi, var_exp)
                zz += 1
            yy += 1
            zz = 0
        xx += 1
        yy = 0
        zz = 0

    D_map = D_map.sum(axis=2)/num
    plt.figure()
    plt.contourf(D_map)
    plt.xlabel("Dilution rate [0.05, 0.2] 1/h")
    plt.ylabel("Substrate Concentration in Feed [5, 35] g/L")
    plt.title("BF Discrimination Criterion (prior) Averaged over phi3 (Initial Biomass Concentration [1, 10] g/L)")
    plt.axis('scaled')
    plt.colorbar()
    plt.show()
    """

    # 2D color map (for 3 values of phi 3 DC)
    num, num2, num3 = np.shape(D_map)
    num_tick = 4
    phi3_val = np.linspace(bounds[2][0], bounds[2][1], num3)
    D_min = D_map.min()
    D_max = D_map.max()
    D_center = np.amin((D_min * 3, D_max / 2))
    if D_center <= D_min:
        D_center = D_min + 0.5 * (D_max - D_min)
    fig, ax = plt.subplots(1, num3)
    # plt.suptitle("Prior: "+str(MBDoE)+" Discrimination Criterion for Fixed Initial Biomass Concentration [1 g/L, 10 g/L]")
    cmap = mpl.colormaps['viridis']
    if fixcb == True:
        norm = colors.Normalize(vmin=-0.6, vmax=0.6)
    else:
        norm = colors.Normalize(vmin=D_min, vmax=D_max)
    for phi3 in range(num3):
        ax[phi3].contourf(D_map[:, :, phi3].transpose(1, 0), cmap=cmap, norm=norm)
        ax[phi3].set_title(r'$U_{3}$: ' + str(float('%.3g' % phi3_val[phi3])))
        ax[phi3].set_xlabel(r'$U_{1}$')
        ax[phi3].set_ylabel(r'$U_{2}$')
        ax[phi3].axis('scaled')
        ax[phi3].set_xticks(np.linspace(0, (num - 1), num_tick))
        ax[phi3].set_xticklabels([float('%.3g' % s) for s in np.linspace(bounds[0][0], bounds[0][1], num_tick)])
        ax[phi3].set_yticks(np.linspace(0, (num - 1), num_tick))
        ax[phi3].set_yticklabels([float('%.3g' % s) for s in np.linspace(bounds[1][0], bounds[1][1], num_tick)])
        # ax[phi3].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # ax[0].plot(u[0, 0], u[0, 1], marker="x", color="red", mew=10, ms=20)
    fig.subplots_adjust(left=0.04, bottom=0, right=1.1, top=1, wspace=0.15, hspace=0)
    fig.colorbar(mappable=mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.45)
    fig.set_size_inches((15, 10), forward=False)
    plt.savefig('Dmap_prior_' + str(MBDoE) + '.pdf', bbox_inches='tight')
    plt.show(block=True)

    return


def plot_D_map_FIM(MBDoE, bounds, D_map):
    num, num2, num3 = np.shape(D_map)
    num_tick = 4
    phi3_val = np.linspace(bounds[2][0], bounds[2][1], num3)
    D_min = D_map.min()
    D_max = D_map.max()
    D_center = np.amin((D_min * 3, D_max / 2))
    if D_center <= D_min:
        D_center = D_min + 0.5 * (D_max - D_min)
    fig, ax = plt.subplots(1, num3)
    # plt.suptitle("FIM ("+str(MBDoE)+") for Fixed Initial Biomass Concentration [1 g/L, 10 g/L]")
    cmap = mpl.colormaps['viridis']
    norm = colors.Normalize(vmin=D_min, vmax=D_max)
    for phi3 in range(num3):
        ax[phi3].contourf(D_map[:, :, phi3].transpose(1, 0), cmap=cmap, norm=norm)
        ax[phi3].set_title(r'$U_{3}$: ' + str(float('%.3g' % phi3_val[phi3])))
        ax[phi3].set_xlabel(r'$U_{1}$')
        ax[phi3].set_ylabel(r'$U_{2}$')
        ax[phi3].axis('scaled')
        ax[phi3].set_xticks(np.linspace(0, (num - 1), num_tick))
        ax[phi3].set_xticklabels([float('%.3g' % s) for s in np.linspace(bounds[0][0], bounds[0][1], num_tick)])
        ax[phi3].set_yticks(np.linspace(0, (num - 1), num_tick))
        ax[phi3].set_yticklabels([float('%.3g' % s) for s in np.linspace(bounds[1][0], bounds[1][1], num_tick)])
        # ax[phi3].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # ax[0].plot(u[0, 0], u[0, 1], marker="x", color="red", mew=10, ms=20)
    fig.subplots_adjust(left=0.04, bottom=0, right=1.1, top=1, wspace=0.15, hspace=0)
    fig.colorbar(mappable=mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.45)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches((15, 10), forward=False)
    plt.savefig('Dmap_FIM_' + str(MBDoE) + '.pdf', bbox_inches='tight')
    plt.show()

    return


# calculate model probabilities
def Model_prob(prob_calc, model_set, chi, dof, y_exp, t, x0, y_meas, theta, u, var_exp):
    num_models = len(model_set)
    m_prob = np.zeros(num_models)
    indexNaN = np.where(np.array(model_set) == None)[0]

    if np.amin(dof) > 0:
        if prob_calc== 2:  # 1-F
            for km_num in range(num_models):
                if np.isin(km_num, indexNaN):
                    continue
                m_prob[km_num] = stats.chi2.sf(x=chi[km_num, 0], df=dof[km_num])
        elif prob_calc == 3:  # akaikes weights
            logL = []
            for km_num in range(num_models):
                if np.isin(km_num, indexNaN):
                    continue
                L = 0
                for s in range(np.shape(y_exp)[1]):
                    L += np.sum([stats.multivariate_normal.logpdf(y_exp[exp][s], y_meas(odeint(model_set[km_num], x0(u[exp]), t, args=(u[exp], theta[-1][km_num]))[1:])[s], var_exp) for exp in range(np.shape(y_exp)[0])])
                logL.append(2*L-2*dof[km_num]) # AIC
            aic = np.array(logL)
            aws = []
            for a1 in aic:
                aw = 0
                for a2 in aic:
                    aw += np.exp(np.min((np.max((a2 - a1, -1000)), 100)))
                aws.append(1. / aw)
            if indexNaN.size != 0:
                m_prob = [aws := np.insert(np.array(aws), ind, 0) for ind in indexNaN][-1]
            else:
                m_prob = aws
        elif prob_calc == 4:  # gaussian posteriors
            logL = []
            for km_num in range(num_models):
                if np.isin(km_num, indexNaN):
                    continue
                L = 0
                for s in range(np.shape(y_exp)[1]):
                    L += np.sum([stats.multivariate_normal.logpdf(y_exp[exp][s], y_meas(odeint(model_set[km_num], x0(u[exp]), t, args=(u[exp], theta[-1][km_num]))[1:])[s], var_exp) for exp in range(np.shape(y_exp)[0])])
                logL.append(L)
            pis = []
            for p1 in logL:
                pi = 0
                for p2 in logL:
                    pi += np.exp(np.min((np.max((p2 - p1, -1000)), 100)))
                pis.append(1. / pi)
            if indexNaN.size != 0:
                m_prob = [pis := np.insert(np.array(pis), ind, 0) for ind in indexNaN][-1]
            else:
                m_prob = pis
        else:  # 1/chi2
            for km_num in range(num_models):
                if np.isin(km_num, indexNaN):
                    continue
                m_prob[km_num] = 1/chi[km_num, 0]
        m_prob = m_prob/np.sum(m_prob)

    else:
        m_prob = np.ones((num_models, 1))/np.sum([model != None for model in model_set])

    return m_prob


def model_selection(prob_rej_th, model_set, prob, prob_th, it, budget):
    num_models = len(model_set)
    modelSet = list(model_set)
    termination = None
    final_model_num = None
    if prob_rej_th != None:
        for km_num in range(num_models):
            if prob[km_num] <= prob_rej_th:
                modelSet[km_num] = None

    if it >= budget:
        termination = 'budget'
        final_model_num = np.argmax(prob)

    if sum(var != None for var in modelSet) == 1:
            termination = '1 remaining'
            final_model_num = np.where(np.array(modelSet) != None)[0][0]

    if sum(var != None for var in modelSet) == 0:
            termination = '0 remaining'

    if any(prob >= prob_th):
            termination = 'high prob'
            final_model_num = np.argmax(prob)

    return modelSet, termination, final_model_num


# stop run and call result plots
def end_MBDoE(MBDoE, res_x, res_u, res_y_exp, y_exp, res_theta, res_FIM, res_priorDC, md_val_prior, res_posteriorDC, md_val_posterior, res_model_set, model_set, res_t_value, PE_table, prob, theta_t, conf_int, priorFIM, u, theta, var_exp, t_val, dof, y, models, para_cov, true_m_num=0):
    store_results(res_x, None, res_u, u, res_y_exp, y_exp, res_theta, theta, res_FIM, priorFIM, res_priorDC, md_val_prior, res_posteriorDC, md_val_posterior, res_model_set, model_set, res_t_value, PE_table)
    # if prob is not None:
        # pie_chart(prob)
    # plot_PE(MBDoE, theta_t, theta, conf_int, true_m_num)
    # plot_cov(MBDoE, priorFIM, u, theta, var_exp, models)
    # plot_paraCov(MBDoE, priorFIM, theta, true_m_num)
    # plot_paraCorr(MBDoE, para_cov, true_m_num)
    # plot_T(MBDoE, t_val, dof, true_m_num)
    # plot_y(y_exp, y_hat_t, models, theta, x, u, t)
    # exit()
    return


# show results in plots
def pie_chart(prob_dist, model_set_0):
    n_pie = np.shape(prob_dist)[0]
    fig, ax = plt.subplots(1, n_pie)
    if n_pie == 1:
        ax.pie(prob_dist[0], labels=[model_set_0[km_num].__name__ for km_num in range(len(model_set_0))], autopct='%1.1f%%', pctdistance=1.15, labeldistance=1.35)
    else:
        for exp in range(n_pie):
            ax[exp].pie(prob_dist[exp], labels=[model_set_0[km_num].__name__ for km_num in range(len(model_set_0))], autopct='%1.1f%%', pctdistance=1.15, labeldistance=1.35)
    plt.show(block=True)
    return


def res_hist(theta_est, u, y_exp, model_set, x0, y_meas, t, var_exp, km_num):
    e_data = []
    n_exp = np.shape(y_exp)[0]
    for exp in range(n_exp):
        y_hat = y_meas(odeint(model_set[km_num], x0(u[exp]), t, args=(u[exp], theta_est[km_num]))[1:])
        e_data += [np.absolute(y_exp[exp] - y_hat)]
    e_data = np.average(e_data, axis=1)
    fig, ax = plt.subplots(1,np.shape(y_exp)[-1])
    for yi in range(np.shape(y_exp)[-1]):
        mean = np.mean(e_data[:,yi])
        dist = np.max(np.absolute([np.min(e_data[:,yi])-mean, np.max(e_data[:,yi])-mean]))
        std = np.std(e_data[:,yi])
        print("Standard deviations for measurement variable y"+str(yi+1)+":")
        print("Residuals "+str(std))
        print("Assumed measurement uncertainty "+str(var_exp(yi,yi)))
        bins = list(np.arange(mean - 3*std, mean + 4*std, std/2))
        if dist > 2*std:
            bins += [mean-dist, mean+dist]
        ax[yi].hist(e_data[:,yi]-mean, bins=np.sort(bins-mean-std/2))
    plt.show(block=True)
    return


def plot_sim_exp(time, y_exp, true_y):
    diag = plt.figure()
    plt.subplot(121)
    plt.plot(time, y_exp[:, 0], label="True Model")
    plt.scatter(time, true_y[:, 0], marker="x", color="black", label="Measured Data")
    plt.xlabel("Time t [h]")
    plt.ylabel("Biomass Concentration C1 [g/L]")
    plt.legend()
    plt.axis([-0.5, 40.5, -0.5, 20])
    plt.subplot(122)
    plt.plot(time, y_exp[:, 1], label="True Model")
    plt.scatter(time, true_y[:, 1], marker="x", color="black", label="Measured Data")
    plt.xlabel("Time t [h]")
    plt.ylabel("Substrate Concentration C1 [g/L]")
    plt.legend()
    plt.axis([-0.5, 40.5, -0.5, 20])
    plt.show(block=True)
    # plt.savefig("C:\Arbeiten\MTT\Uni\Master\MA\Code\Figures\etwas.pdf", dpi=None, format='png', bbox_inches='tight', pad_inches=0.2, bbox=None, pad=None, dashes=None, loc='upper left', rot=0, vmax='I', vmin='I', hmax='I', hmin='I')
    return diag


def plot_y(y_exp, y_hat_t, model_set_0, theta, x0, y_meas, u, t):
    num_models = len(model_set_0)
    color = ["green", "dimgrey", "darkgrey", "lightgrey"]
    n_exp = np.shape(u)[0]
    for exp in range(n_exp):

        y_hat = []
        for km_num in range(num_models):
            km = model_set_0[km_num]
            y_hat = np.append(y_hat, y_meas(odeint(km, x0(u[exp]), t, args=(u[exp], theta[exp][km_num]))[1:]))
            if km_num == 0:
                y_hat = np.expand_dims(y_hat, axis=0)
            y_hat = np.reshape(y_hat, (-1, np.shape(t)[0] - 1, 2))

        plt.figure()
        plt.suptitle("Data from " + str(
            exp) + ". experiment and resulting model predictions with updated parameter. \n \n Experiment 0 is preliminary all others are MBDoE designed.")

        plt.subplot(121)
        plt.plot(t[1:], y_hat_t[exp][:, 0], label="True Model")
        plt.scatter(t[1:], y_exp[exp][:, 0], marker="x", color="black", label="Measured Data")
        plt.xlabel("Time t [h]")
        plt.ylabel("Biomass Concentration C1 [g/L]")
        plt.axis([-0.5, 40.5, -0.5, 50])
        for km_num in range(num_models):
            plt.plot(t[1:], y_hat[km_num, :, 0], label=model_set_0[km_num].__name__, color=color[km_num],
                     linestyle='dashed')
        plt.legend()

        plt.subplot(122)
        plt.plot(t[1:], y_hat_t[exp][:, 1], label="True Model")
        plt.scatter(t[1:], y_exp[exp][:, 1], marker="x", color="black", label="Measured Data")
        plt.xlabel("Time t [h]")
        plt.ylabel("Substrate Concentration C1 [g/L]")
        plt.axis([-0.5, 40.5, -0.5, 30])
        for km_num in range(num_models):
            plt.plot(t[1:], y_hat[km_num, :, 1], label=model_set_0[km_num].__name__, color=color[km_num],
                     linestyle='dashed')
        plt.legend()

        plt.show(block=True)
    return


def plot_PE(MBDoE, theta_t, theta, conf_int, true_m_num=0):
    n_theta = np.shape(theta_t)[0]

    fig, ax = plt.subplots(1, n_theta)
    plt.suptitle("Variation of Parameter Estimates and Uncertainties")

    delta_y = np.amax(np.add(np.absolute(np.subtract(np.vstack(theta[:, true_m_num]), np.broadcast_to(theta_t, (np.shape(np.vstack(theta[:, true_m_num]))[0], np.shape(theta_t)[0])))), np.vstack(conf_int[:, true_m_num])))  # amax: along all theta

    for it_theta in range(n_theta):
        ax[it_theta].plot(np.arange(np.shape(theta)[0]+1)-0.5, [theta_t[it_theta]]*(np.shape(theta)[0]+1), color='green', label="True Parameter")
        ax[it_theta].errorbar(np.arange(np.shape(theta)[0]), np.vstack(theta[:, true_m_num])[:, it_theta], np.vstack(conf_int[:, true_m_num])[:, it_theta], linestyle='None', marker='s', ecolor='orange', color='black', capsize=6)
        ax[it_theta].set_title("Parameter: theta" + str(it_theta+1))
        ax[it_theta].set_xbound(-0.5, np.shape(theta)[0]-0.5)
        ax[it_theta].set_xticks(np.arange(0, np.shape(theta)[0]))
        ax[it_theta].set_ybound((theta_t[it_theta]-delta_y)*1.1, (theta_t[it_theta]+delta_y)*1.1)
    fig.set_size_inches((15, 10), forward=False)
    plt.savefig('PE_' + str(MBDoE) + '.pdf', bbox_inches='tight')
    plt.show()

    return


def plot_cov(MBDoE, priorFIM, u, x0, y_meas, t, theta, var_exp, model_set_0, true_m_num=0):

    x_cent = 0
    y_cent = 0
    data_num = 1e3
    mass_level = 0.95
    elli_color = np.append(list(itertools.repeat('tab:blue', np.shape(u)[0]-np.shape(theta)[0]+1)), ['tab:orange', 'tab:green', 'tab:brown', 'tab:olive', 'tab:purple', 'tab:cyan',
                  'limegreen',
                  'navy', 'slategrey'])
    elli_style = ['solid', 'dashed']

    plt.figure()
    # plt.title("True model prediction variance confidence interval of 95%. Solid line is the worst-case and dashed line is time-average case.")
    for exp in range(np.shape(u)[0]):
        exp_sen = sens_fun(u[exp], t, model_set_0[true_m_num], theta[0][true_m_num], x0, y_meas)
        try:
            exp_cov = np.matmul(exp_sen.transpose(),
                                np.matmul(np.linalg.inv(priorFIM[0][true_m_num]), exp_sen)) + var_exp
        except:
            exp_cov = np.matmul(exp_sen.transpose(),
                                np.matmul(np.linalg.inv(priorFIM[0][true_m_num]), exp_sen)) + var_exp
        cov = [exp_cov]

        for elli in range(np.shape(cov)[0]):
            eig_vec, eig_val, cov_u = np.linalg.svd(cov[elli])
            # Make sure 0th eigenvector has positive x-coordinate
            if eig_vec[0][0] < 0:
                eig_vec[0] *= -1
            semimaj = np.sqrt(eig_val[0])
            semimin = np.sqrt(eig_val[1])
            distances = np.linspace(0, 20, 20001)
            chi2_cdf = stats.chi2.cdf(distances, df=2)
            multiplier = np.sqrt(
                distances[np.where(np.abs(chi2_cdf - mass_level) == np.abs(chi2_cdf - mass_level).min())[0][0]])
            semimaj *= multiplier
            semimin *= multiplier
            phi = np.arccos(np.dot(eig_vec[0], np.array([1, 0])))
            if eig_vec[0][1] < 0 and phi > 0:
                phi *= -1
            # Generate data for ellipse structure
            ellpsis_data = np.linspace(0, 2 * np.pi, int(data_num))
            r = 1 / np.sqrt((np.cos(ellpsis_data)) ** 2 + (np.sin(ellpsis_data)) ** 2)
            x_elli = r * np.cos(ellpsis_data)
            y_elli = r * np.sin(ellpsis_data)
            data = np.array([x_elli, y_elli])
            S = np.array([[semimaj, 0], [0, semimin]])
            R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            T = np.dot(R, S)
            data = np.dot(T, data)
            # Plot!
            plt.plot(data[0], data[1], color=elli_color[exp], label="Experiment " + str(max(exp-(np.shape(u)[0]-np.shape(theta)[0]),0)))  #
    # plot also the measurement covariance matrix
    eig_vec, eig_val, cov_u = np.linalg.svd(var_exp)
    # Make sure 0th eigenvector has positive x-coordinate
    if eig_vec[0][0] < 0:
        eig_vec[0] *= -1
    semimaj = np.sqrt(eig_val[0])
    semimin = np.sqrt(eig_val[1])
    distances = np.linspace(0, 20, 20001)
    chi2_cdf = stats.chi2.cdf(distances, df=2)
    multiplier = np.sqrt(
        distances[np.where(np.abs(chi2_cdf - mass_level) == np.abs(chi2_cdf - mass_level).min())[0][0]])
    semimaj *= multiplier
    semimin *= multiplier
    phi = np.arccos(np.dot(eig_vec[0], np.array([1, 0])))
    if eig_vec[0][1] < 0 and phi > 0:
        phi *= -1
    # Generate data for ellipse structure
    ellpsis_data = np.linspace(0, 2 * np.pi, int(data_num))
    r = 1 / np.sqrt((np.cos(ellpsis_data)) ** 2 + (np.sin(ellpsis_data)) ** 2)
    x_elli = r * np.cos(ellpsis_data)
    y_elli = r * np.sin(ellpsis_data)
    data = np.array([x_elli, y_elli])
    S = np.array([[semimaj, 0], [0, semimin]])
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    T = np.dot(R, S)
    data = np.dot(T, data)
    # Plot!
    plt.plot(data[0], data[1], color="red", label="Measurement noise")
    plt.axis('equal')
    plt.legend()
    plt.xlabel(r'$Y_{1}$')
    plt.ylabel(r'$Y_{2}$')
    fig = mpl.pyplot.gcf()
    fig.set_size_inches((15, 10), forward=False)
    plt.savefig('COV_' + str(MBDoE) + '.pdf', bbox_inches='tight')
    plt.show()

    return


def plot_paraCov(MBDoE, priorFIM, theta, true_m_num=0):
    data_num = 1e3
    mass_level = 0.95
    elli_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:olive', 'tab:purple', 'tab:cyan',
                  'limegreen',
                  'navy', 'slategrey']
    fig, ax = plt.subplots(np.shape(theta[0][true_m_num])[0] - 1, np.shape(theta[0][true_m_num])[0] - 1)
    for a in ax.flat:  # this will iterate over all 6 axes
        a.axis('off')
    a = []
    # plt.suptitle("True model parameter estimate variance confidence interval of 95%")
    PEcov = np.linalg.inv(priorFIM[0][true_m_num])
    cov = np.zeros((len(priorFIM[0][true_m_num]), 2, 2))
    for combi1 in range(1, np.shape(theta[0][true_m_num])[0]):
        for combi2 in range(combi1):
            cov = [[PEcov[combi2][combi2], PEcov[combi2][combi1]], [PEcov[combi1][combi2], PEcov[combi1][combi1]]]
            for exp in range(np.shape(theta)[0]):
                eig_vec, eig_val, cov_u = np.linalg.svd(cov)
                # Make sure 0th eigenvector has positive x-coordinate
                if eig_vec[0][0] < 0:
                    eig_vec[0] *= -1
                semimaj = np.sqrt(eig_val[0])
                semimin = np.sqrt(eig_val[1])
                distances = np.linspace(0, 20, 20001)
                chi2_cdf = stats.chi2.cdf(distances, df=2)
                multiplier = np.sqrt(
                    distances[np.where(np.abs(chi2_cdf - mass_level) == np.abs(chi2_cdf - mass_level).min())[0][0]])
                semimaj *= multiplier
                semimin *= multiplier
                phi = np.arccos(np.dot(eig_vec[0], np.array([1, 0])))
                if eig_vec[0][1] < 0 and phi > 0:
                    phi *= -1
                # Generate data for ellipse structure
                ellpsis_data = np.linspace(0, 2 * np.pi, int(data_num))
                r = 1 / np.sqrt((np.cos(ellpsis_data)) ** 2 + (np.sin(ellpsis_data)) ** 2)
                x_elli = r * np.cos(ellpsis_data)
                y_elli = r * np.sin(ellpsis_data)
                data = np.array([x_elli, y_elli])
                S = np.array([[semimaj, 0], [0, semimin]])
                R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
                T = np.dot(R, S)
                data = np.dot(T, data)
                data[0] += theta[exp][true_m_num][combi2]
                data[1] += theta[exp][true_m_num][combi1]
                # Plot!
                a = ax[combi1 - 1, combi2].plot(data[0], data[1], color=elli_color[exp], label="Experiment " + str(exp),
                                                linestyle='solid')
                ax[combi1 - 1, combi2].set_ylabel(r'$\Theta_{}$'.format(combi1 + 1))
                ax[combi1 - 1, combi2].set_xlabel(r'$\Theta_{}$'.format(combi2 + 1))
                ax[combi1 - 1, combi2].axis('on')
                ax[combi1 - 1, combi2].axis('equal')
    lines_labels = [ax[0, 0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax[0, np.shape(theta[0][true_m_num])[0] - 2].legend(lines, labels)
    # fig.legend(lines, labels)
    # fig.legend(*_get_legend_handles_and_labels(fig.axes), loc='upper right')
    # ax[np.shape(theta[0][true_m_num])[0]-1, np.shape(theta[0][true_m_num])[0]-1].legend(a.get_legend())
    fig.set_size_inches((15, 10), forward=False)
    plt.savefig('PEcov_' + str(MBDoE) + '.pdf', bbox_inches='tight')
    plt.show()

    return


def plot_paraCorr(MBDoE, para_cov, theta, true_m_num=0):
    plt.figure()
    data = para_cov[-1][true_m_num]
    data = np.where(np.isnan(data), 0, data)

    cmap = mpl.colors.LinearSegmentedColormap.from_list('BWB',
                                                        [[0, 'mediumblue'], [0.1, 'royalblue'], [0.2, 'cornflowerblue'],
                                                         [0.5, 'white'], [0.8, 'cornflowerblue'], [0.9, 'royalblue'],
                                                         [1, 'mediumblue']])
    sb.heatmap(data, cmap=cmap, annot=True, fmt='.4g', vmin=-1, vmax=1, annot_kws={"fontsize": 20}, cbar=False)
    ticks_labels = [r'$\Theta_{1}$', r'$\Theta_{2}$', r'$\Theta_{3}$', r'$\Theta_{4}$', r'$\Theta_{5}$',
                    r'$\Theta_{6}$']  #
    ticks_labels = ticks_labels[:np.shape(theta[0][true_m_num])[0]]
    plt.xticks(np.arange(np.shape(data)[0]) + .5, labels=ticks_labels, fontsize=20)
    plt.yticks(np.arange(np.shape(data)[0]) + .5, labels=ticks_labels, fontsize=20)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches((15, 10), forward=False)
    plt.savefig('PEcorr_' + str(MBDoE) + '.pdf', bbox_inches='tight')
    plt.show()
    return


def plot_T(MBDoE, t_val, dof, true_m_num=0):
    labellist = [r'$\Theta_{1}$', r'$\Theta_{2}$', r'$\Theta_{3}$', r'$\Theta_{4}$']
    fig, ax = plt.subplots()
    for para in range(np.shape(t_val[0, true_m_num])[0]):
        data = [t_val[s, true_m_num][para] for s in np.arange(np.shape(t_val)[0])]
        plt.plot(np.arange(np.shape(t_val)[0]), data, marker="s", label='t-value: ' + labellist[para])
    plt.plot([stats.t.ppf((1 - (0.05 / 2)), dof[s][true_m_num]) for s in np.arange(np.shape(t_val)[0])], marker="d",
             label='Ref. t-value')
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.xlabel("Sequence of designed experiments")
    plt.ylabel("Parameter statistic (t-value)")
    plt.legend()

    fig = mpl.pyplot.gcf()
    fig.set_size_inches((15, 10), forward=False)
    plt.savefig('Tval_' + str(MBDoE) + '.pdf', bbox_inches='tight')
    plt.show()
    return


def postprocessing_results(results_from_study):
    I = np.sum(results_from_study[:, 0] == 'I', axis=0) / np.shape(results_from_study)[0]
    F = np.sum(results_from_study[:, 0] == 'F', axis=0) / np.shape(results_from_study)[0]
    D = np.sum(results_from_study[:, 0] == 'D', axis=0) / np.shape(results_from_study)[0]
    P = np.sum(results_from_study[:, 0] == 'P', axis=0) / np.shape(results_from_study)[0]
    E = np.sum(results_from_study[:, 0] == 'E', axis=0) / np.shape(results_from_study)[0]

    A_T = np.average([results_from_study[s, 2] for s in np.where(results_from_study[:, 0] == 'I')[0]])
    if np.isnan(A_T) == False:
        SD_T = np.std([results_from_study[s, 2] for s in np.where(results_from_study[:, 0] == 'I')[0]])
    else: SD_T = np.NaN
    A_MD = np.average([results_from_study[s, 1] for s in np.where(results_from_study[:, 0] == 'I')[0]])
    if np.isnan(A_MD) == False:
        SD_MD = np.std([results_from_study[s, 1] for s in np.where(results_from_study[:, 0] == 'I')[0]])
    else: SD_MD = np.NaN
    A_F = np.average([results_from_study[s, 1] for s in np.where(results_from_study[:, 0] == 'F')[0]])
    if np.isnan(A_F) == False:
        SD_F = np.std([results_from_study[s, 1] for s in np.where(results_from_study[:, 0] == 'F')[0]])
    else: SD_F = np.NaN
    A_P = np.average([results_from_study[s, 1] for s in np.where(results_from_study[:, 0] == 'P')[0]])
    if np.isnan(A_P) == False:
        SD_P = np.std([results_from_study[s, 1] for s in np.where(results_from_study[:, 0] == 'P')[0]])
    else:
        SD_P = np.NaN
    A_P_conf_t = [np.average([results_from_study[s, 2][0] for s in np.where(results_from_study[:, 0] == 'P')[0]], axis=0), np.average([results_from_study[s, 2][1] for s in np.where(results_from_study[:, 0] == 'P')[0]], axis=0)]
    if np.isnan(A_P_conf_t[0]) == False:
        SD_P_conf_t = [np.std([results_from_study[s, 2][0] for s in np.where(results_from_study[:, 0] == 'P')[0]], axis=0), np.average([results_from_study[s, 2][1] for s in np.where(results_from_study[:, 0] == 'P')[0]], axis=0)]
    else:
        SD_P_conf_t = np.NaN
    A_D = np.average([results_from_study[s, 1] for s in np.where(results_from_study[:, 0] == 'D')[0]])
    if np.isnan(A_D) == False:
        SD_D = np.std([results_from_study[s, 1] for s in np.where(results_from_study[:, 0] == 'D')[0]])
    else:
        SD_D = np.NaN
    A_D_prob = np.average([results_from_study[s, 2] for s in np.where(results_from_study[:, 0] == 'D')[0]])
    if np.isnan(A_D_prob) == False:
        SD_D_prob = np.std([results_from_study[s, 2] for s in np.where(results_from_study[:, 0] == 'D')[0]])
    else:
        SD_D_prob = np.NaN

    return I, F, D, P, E, A_T, SD_T, A_MD, SD_MD, A_F, SD_F, A_D, SD_D, A_P, SD_P, A_P_conf_t, SD_P_conf_t, A_D_prob, SD_D_prob


def store_results(res_x, x, res_u, u, res_y_exp, y_exp, res_theta, theta, res_FIM, priorFIM, res_priorDC, md_val_prior, res_posteriorDC, md_val_posterior, res_model_set, model_set, res_t_value, PE_table):
    try:
        res_x += [x]
    except:
        res_x += [-1]
        pass
    try:
        res_u += [u]
    except:
        res_u += [-1]
        pass
    try:
        res_y_exp += [y_exp]
    except:
        res_y_exp += [-1]
        pass
    try:
        res_theta += [theta]
    except:
        res_theta += [-1]
        pass
    try:
        res_FIM += [priorFIM]
    except:
        res_FIM += [-1]
        pass
    try:
        res_priorDC += [md_val_prior]
    except:
        res_priorDC += [-1]
        pass
    try:
        res_posteriorDC += [md_val_posterior[1:]]
    except:
        res_posteriorDC += [-1]
        pass
    try:
        res_model_set += [model_set]
    except:
        res_model_set += [-1]
        pass
    try:
        res_t_value += [PE_table]
    except:
        res_t_value += [-1]
        pass

    return

