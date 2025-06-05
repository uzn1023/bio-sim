import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Logistic model with concentration dependent parameters

def weibull_inhibition(C, K, Cm, a):
    return 1 - K * (1 - np.exp(-np.log(2) * (C / Cm) ** a))

def logistic_params(C, p):
    Xm = p['Xm0'] * weibull_inhibition(C, p['Kx'], p['Cmx'], p['ax'])
    Vm = p['Vm0'] * weibull_inhibition(C, p['Kv'], p['Cmv'], p['av'])
    lam = p['lam0'] * weibull_inhibition(C, p['Kl'], p['Cml'], p['al'])
    return Xm, Vm, lam

def logistic_model(t, x, C, p):
    Xm, Vm, lam = logistic_params(C, p)
    if t < lam:
        return 0.0
    return Vm * x * (1 - x / Xm)

# Mechanistic model
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

def mechanistic_observables(y, params, X0):
    Xl, X, _ = y
    Xd = X0 - (Xl + X)
    OD = (Xl + X) + params['alpha'] * Xd
    CFU = params['beta'] * 10**10 * (Xl + X)
    return OD, CFU

def run_simulation():
    t_span = (0, 48)
    t_eval = np.linspace(0, 48, 200)
    # Example parameters (not from the paper)
    logistic_p = {
        'Xm0': 1.0, 'Vm0': 0.8, 'lam0': 2.0,
        'Kx': 0.9, 'Cmx': 1.0, 'ax': 1.0,
        'Kv': 0.9, 'Cmv': 1.0, 'av': 1.0,
        'Kl': 0.9, 'Cml': 1.0, 'al': 1.0
    }
    C = 0.5  # example concentration
    sol_log = solve_ivp(logistic_model, t_span, [0.01], t_eval=t_eval, args=(C, logistic_p))

    mech_params = {
        'k0_a': 1.0, 'IC50_a': 1.0, 'gamma_a': 2.0,
        'k0_g': 1.0, 'IC50_g': 1.0, 'gamma_g': 2.0,
        'k0_d': 0.01, 'k_star_d': 0.5, 'EC50_d': 1.0, 'gamma_d': 2.0,
        'ki': 0.5, 'Y': 0.1, 'alpha': 0.3, 'beta': 1.0
    }
    y0 = [0.01, 0.0, 1.0]
    sol_mech = solve_ivp(mechanistic_odes, t_span, y0, t_eval=t_eval, args=(C, mech_params))

    OD_mech, CFU_mech = mechanistic_observables(sol_mech.y, mech_params, sum(y0[:2]))

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(sol_log.t, sol_log.y[0], label='Logistic OD')
    plt.plot(sol_mech.t, OD_mech, label='Mechanistic OD')
    plt.xlabel('Time (h)')
    plt.ylabel('Optical Density')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(sol_mech.t, CFU_mech, label='Mechanistic CFUs')
    plt.xlabel('Time (h)')
    plt.ylabel('CFU (a.u.)')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_simulation()
