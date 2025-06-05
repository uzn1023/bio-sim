import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Logistic model helpers

def weibull_inhibition(C, K, Cm, a):
    return 1 - K * (1 - np.exp(-np.log(2) * (C / Cm) ** a))

def logistic_params(C, p):
    Xm = p['Xm0'] * weibull_inhibition(C, p['Kx'], p['Cmx'], p['ax'])
    Vm = p['Vm0'] * weibull_inhibition(C, p['Kv'], p['Cmv'], p['av'])
    lam = p['lam0'] * weibull_inhibition(C, p['Kl'], p['Cml'], p['al'])
    return Xm, Vm, lam

def logistic_ode(t, x, C, p):
    Xm, Vm, lam = logistic_params(C, p)
    if t < lam:
        return 0.0
    return Vm * x * (1 - x / Xm)

# Mechanistic model helpers

def hill_dec(C, k0, IC50, gamma):
    return k0 * IC50 ** gamma / (C ** gamma + IC50 ** gamma)

def hill_inc(C, k0, EC50, gamma):
    return k0 * C ** gamma / (C ** gamma + EC50 ** gamma)

def mechanistic_odes(t, y, C, params):
    Xl, X, S = y
    ka = hill_dec(C, params['k0_a'], params['IC50_a'], params['gamma_a'])
    kg = hill_dec(C, params['k0_g'], params['IC50_g'], params['gamma_g'])
    kd = params['k0_d'] + hill_inc(C, params['k_star_d'], params['EC50_d'], params['gamma_d'])
    dXl = -ka * Xl - kd * Xl
    dX = ka * Xl + (kg / (params['ki'] + X)) * S * X - kd * X
    dS = -params['Y'] * (kg / (params['ki'] + X)) * S * X
    return [dXl, dX, dS]

def simulate_logistic(times, C, p, x0=0.01):
    sol = solve_ivp(logistic_ode, (times.min(), times.max()), [x0], t_eval=times, args=(C, p))
    return sol.y[0]

def simulate_mech(times, C, p, y0=(0.01, 0.0, 1.0)):
    sol = solve_ivp(mechanistic_odes, (times.min(), times.max()), list(y0), t_eval=times, args=(C, p))
    Xl, X, _ = sol.y
    Xd = sum(y0[:2]) - (Xl + X)
    OD = (Xl + X) + p['alpha'] * Xd
    return OD

def residual_logistic(theta, data):
    p = {
        'Xm0': theta[0], 'Vm0': theta[1], 'lam0': theta[2],
        'Kx': theta[3], 'Cmx': theta[4], 'ax': theta[5],
        'Kv': theta[6], 'Cmv': theta[7], 'av': theta[8],
        'Kl': theta[9], 'Cml': theta[10], 'al': theta[11]
    }
    res = []
    for C, group in data.groupby('concentration'):
        times = group['time'].values
        od_pred = simulate_logistic(times, C, p)
        res.append(od_pred - group['od'].values)
    return np.concatenate(res)

def residual_mech(theta, data):
    p = {
        'k0_a': theta[0], 'IC50_a': theta[1], 'gamma_a': theta[2],
        'k0_g': theta[3], 'IC50_g': theta[4], 'gamma_g': theta[5],
        'k0_d': theta[6], 'k_star_d': theta[7], 'EC50_d': theta[8], 'gamma_d': theta[9],
        'ki': theta[10], 'Y': theta[11], 'alpha': theta[12]
    }
    res = []
    for C, group in data.groupby('concentration'):
        times = group['time'].values
        od_pred = simulate_mech(times, C, p)
        res.append(od_pred - group['od'].values)
    return np.concatenate(res)

def fit_model(data_path, model='logistic'):
    data = pd.read_csv(data_path)
    if model == 'logistic':
        theta0 = np.array([1.0, 0.8, 2.0, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0])
        result = least_squares(residual_logistic, theta0, args=(data,))
        params = result.x
    else:
        theta0 = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.01, 0.5, 1.0, 2.0, 0.5, 0.1, 0.3])
        result = least_squares(residual_mech, theta0, args=(data,))
        params = result.x
    print('Fitted parameters:', params)
    return data, params

def plot_fit(data, params, model='logistic'):
    plt.figure(figsize=(6, 4))
    for C, group in data.groupby('concentration'):
        times = np.linspace(group['time'].min(), group['time'].max(), 100)
        if model == 'logistic':
            p = {
                'Xm0': params[0], 'Vm0': params[1], 'lam0': params[2],
                'Kx': params[3], 'Cmx': params[4], 'ax': params[5],
                'Kv': params[6], 'Cmv': params[7], 'av': params[8],
                'Kl': params[9], 'Cml': params[10], 'al': params[11]
            }
            od_pred = simulate_logistic(times, C, p)
        else:
            p = {
                'k0_a': params[0], 'IC50_a': params[1], 'gamma_a': params[2],
                'k0_g': params[3], 'IC50_g': params[4], 'gamma_g': params[5],
                'k0_d': params[6], 'k_star_d': params[7], 'EC50_d': params[8], 'gamma_d': params[9],
                'ki': params[10], 'Y': params[11], 'alpha': params[12]
            }
            od_pred = simulate_mech(times, C, p)
        plt.plot(times, od_pred, label=f'C={C} fit')
        plt.scatter(group['time'], group['od'], color='k')
    plt.xlabel('Time (h)')
    plt.ylabel('OD')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fit DDAC growth data')
    parser.add_argument('--data', default='data/example_data.csv')
    parser.add_argument('--model', choices=['logistic', 'mechanistic'], default='logistic')
    args = parser.parse_args()
    data, params = fit_model(args.data, args.model)
    plot_fit(data, params, args.model)
