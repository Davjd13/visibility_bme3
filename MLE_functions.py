import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import zeta
from scipy.special import factorial
from scipy.stats import poisson


def freqTable(G):
    if type(G) == nx.classes.graph.Graph:
        degree_dict = dict(G.degree())
        degree_list = list(degree_dict.values())
    else:
        degree_list = G
    degree_list.sort()
    unique_deg = []

    table = {}
    for n in degree_list:
        if n in table:
            table[n] += 1
        else:
            table[n] = 1

    for n in degree_list:
        if n not in unique_deg:
            unique_deg += [n]
    return np.array(degree_list), np.array(unique_deg), table


def degree_list(G):
    if type(G) == nx.classes.graph.Graph:
        degree_dict = dict(G.degree())
        degree_list = list(degree_dict.values())
    else:
        degree_list = G
    degree_list.sort()
    return np.array(degree_list)


def empirical(X_list):
    N, f = np.unique(X_list, return_counts=True)
    cumul = np.cumsum(f[::-1])[::-1]
    p = f / X_list.size
    P = cumul / X_list.size
    return N, P, p


def CCDF(result, X, N, P):
    k_min = result[0]
    Input = np.unique(X)
    distribution = result[1]
    params = result[2][0]
    C_index = np.where(N == k_min)[0]
    C = P[C_index]
    try:
        inf = np.arange(np.amax(Input)) #np.arange(1000)
    except ValueError:
        inf = np.arange(1000)
    if distribution == 'Powerlaw':
        y = C * zeta(params[0], Input) / zeta(params[0], k_min)

    if distribution == 'Exponential':
        y = C * np.exp((-1 / params[0]) * (Input - k_min))

    if distribution == 'Weibull':
        sum1 = np.array(
            [np.sum((((j + inf) / params[0]) ** (params[1] - 1)) * np.exp(-(((j + inf) / params[0]) ** params[1]))) for
             j in Input])
        inf_sum = np.sum(
            (((inf + k_min) / params[0]) ** (params[1] - 1)) * np.exp(-1 * ((inf + k_min) / params[0]) ** params[1]))
        y = C * sum1 / inf_sum

    if distribution == 'Lognormal':
        sum1 = np.array(
            [np.sum((1.0 / (j + inf)) * np.exp(-((np.log(j + inf) - params[0]) ** 2) / (2 * (params[1] ** 2)))) for j in
             Input])
        inf_sum = np.sum(
            (1.0 / (inf + k_min)) * np.exp(-((np.log(inf + k_min) - params[0]) ** 2) / (2 * params[1] ** 2)))
        y = C * sum1 / (inf_sum)

    if distribution == 'Poisson':
        y = 1 - C * poisson.cdf(Input, params[0])

    if distribution == 'Trunc_pl':
        inf_sum = np.sum((inf + k_min) ** (-1 * params[1]) * np.exp(-1 * inf / params[0]))
        z = np.array([np.sum((inf + i) ** (-1 * params[1]) * np.exp(-1 * inf / params[0])) for i in Input])
        y = C * (np.exp(-(Input - k_min) / params[0])) * z / inf_sum

    if distribution == 'Normal':
        norm_n = np.sum(np.exp(-((inf - params[0]) ** 2) / (2 * params[1] ** 2)))
        sum1 = np.array([np.sum(np.exp(-((j + inf - params[0]) ** 2) / (2 * params[1] ** 2))) for j in Input])
        y = C * sum1 / norm_n

    return y


def PDF(result, X, N, p):
    k_min = result[0]
    distribution = result[1]
    Input = np.unique(X)
    params = result[2][0]
    C_index = np.where(N == k_min)[0]
    C = p[C_index]
    try:
        inf = np.arange(np.amax(Input))
    except ValueError:
        inf = np.arange(1000)
    if distribution == 'Powerlaw':
        y = (C / zeta(params[0], k_min)) * Input ** -params[0]

    if distribution == 'Exponential':
        y = C * ((1 - np.exp(-1 / params[0])) / np.exp(-k_min / params[0])) * np.exp(-Input / params[0])

    if distribution == 'Weibull':
        inf_sum = np.sum(
            (((inf + k_min) / params[0]) ** (params[1] - 1)) * np.exp(-1 * ((inf + k_min) / params[0]) ** params[1]))
        y = C * ((Input / params[0]) ** (params[1] - 1)) * (np.exp((-(Input / params[0]) ** params[1]))) / inf_sum

    if distribution == 'Lognormal':  # Not done
        inf_sum = np.sum(
            (1.0 / (inf + k_min)) * np.exp(-((np.log(inf + k_min) - params[0]) ** 2) / (2 * params[1] ** 2)))
        y = C * ((1 / Input) * np.exp(-((np.log(Input) - params[0]) ** 2) / (2 * params[1] ** 2))) / inf_sum

    if distribution == 'Poisson':  # Not done
        y = 1 - C * poisson.pdf(Input, params[0])

    if distribution == 'Trunc_pl':
        y = C * ((np.exp(-k_min / params[0])) / zeta(params[0], k_min)) * (Input ** (-params[0])) * np.exp(
            -Input / params[0])

    if distribution == 'Normal':
        norm_n = np.sum(np.exp(-((inf - params[0]) ** 2) / (2 * params[1] ** 2)))
        y = C * (np.exp(-((Input - params[0]) ** 2) / (2 * params[1] ** 2))) / norm_n

    return y


def CDF(result, X, N, P):
    k_min = result[0]
    Input = X
    distribution = result[1]
    params = result[2]
    C_index = np.where(N == k_min)[0]
    C = P[C_index]
    try:
        inf = np.arange(np.amax(Input))
    except ValueError:
        inf = np.arange(1000)
    if distribution == 'Powerlaw':
        y = C * zeta(params[0], Input) / zeta(params[0], k_min)

    if distribution == 'Exponential':
        y = C * np.exp((-1 / params[0]) * (Input - k_min))

    if distribution == 'Weibull':
        sum1 = np.array(
            [np.sum((((j + inf) / params[0]) ** (params[1] - 1)) * np.exp(-(((j + inf) / params[0]) ** params[1]))) for
             j in Input])
        inf_sum = np.sum(
            (((inf + k_min) / params[0]) ** (params[1] - 1)) * np.exp(-1 * ((inf + k_min) / params[0]) ** params[1]))
        y = C * sum1 / inf_sum

    if distribution == 'Lognormal':
        sum1 = np.array(
            [np.sum((1.0 / (j + inf)) * np.exp(-((np.log(j + inf) - params[0]) ** 2) / (2 * (params[1] ** 2)))) for j in
             Input])
        inf_sum = np.sum(
            (1.0 / (inf + k_min)) * np.exp(-((np.log(inf + k_min) - params[0]) ** 2) / (2 * params[1] ** 2)))
        y = C * sum1 / (inf_sum)

    if distribution == 'Poisson':
        y = 1 - C * poisson.cdf(Input, params[0])

    if distribution == 'Trunc_pl':
        inf_sum = np.sum((inf + k_min) ** (-1 * params[1]) * np.exp(-1 * inf / params[0]))
        z = np.array([np.sum((inf + i) ** (-1 * params[1]) * np.exp(-1 * inf / params[0])) for i in Input])
        y = C * (np.exp(-(Input - k_min) / params[0])) * z / inf_sum

    if distribution == 'Normal':
        norm_n = np.sum(np.exp(-((inf - params[0]) ** 2) / (2 * params[1] ** 2)))
        sum1 = np.array([np.sum(np.exp(-((j + inf - params[0]) ** 2) / (2 * params[1] ** 2))) for j in Input])
        y = C * sum1 / norm_n

    return 1 - y


def AIC(LnL: float, N: int, params: int = 1):
    if N < 4:
        AIC = -2 * LnL + 2 * params
    else:
        AIC = -2 * LnL + 2 * params + ((2 * params * (params + 1))) / (N - params - 1)
    return AIC


def BIC(LnL: float, N: int, params: int = 1):
    return params * np.log(N) - 2 * LnL


def powerlaw(params: np.ndarray, x: np.ndarray, sum_log, k_min, delta: float = 0):
    NegLnL = x.size * np.log(zeta(params[0], k_min)) + params[0] * (sum_log)
    return NegLnL


def exp_dist(params: np.ndarray, x: np.ndarray, k_min, delta: float = 0):
    NegLnL = -1 * x.size * (np.log(1 - np.exp(-1 / params[0]))) + (1 / params[0]) * (x.sum() - x.size * k_min)
    return NegLnL


def weibull(params, x: np.ndarray, inf, sum_log, k_min, delta: float = 0):
    inf_sum = np.sum(
        (((inf + k_min) / params[0]) ** (params[1] - 1)) * np.exp(-1 * ((inf + k_min) / params[0]) ** params[1]))
    if inf_sum == 0:
        return float('inf')
        pass
    else:
        LnL = -x.size * np.log(inf_sum) - x.size * (params[1] - 1) * np.log(params[0]) \
              + (params[1] - 1) * sum_log - np.sum((x / params[0]) ** params[1])
        NegLnL = -1 * LnL
        return NegLnL


def normal(params, x, inf):
    norm_n = np.sum(np.exp(-((inf - params[0]) ** 2) / (2 * params[1] ** 2)))
    NegLnL = x.size * np.log(norm_n) + np.sum(((x - params[0]) ** 2) / (2 * params[1] ** 2))
    return NegLnL


def stretched_exp(params, x, inf, k_min):
    norm_s = np.sum(np.exp(-((k_min + inf) / params[0]) ** params[1]))
    NegLnL = -1 * (-x.size * np.log(norm_s) - np.sum((x / params[0]) ** params[1]))
    return NegLnL


def trunc_powerlaw(params, x: np.ndarray, inf, delta: float, k_min):
    inf_sum = np.sum((inf + k_min) ** (-1 * params[1]) * np.exp(-1 * inf / params[0]))
    LnL = x.size * np.log(1 - delta) + x.size * k_min / params[0] - x.size * np.log(inf_sum) \
          - (params[1] * np.log(x) + x / params[0]).sum()
    NegLnL = -1 * LnL
    return NegLnL


def logn(params, x, inf, sum_log, k_min):
    inf_sum = np.sum((1.0 / (inf + k_min)) * np.exp(-((np.log(inf + k_min) - params[0]) ** 2) / (2 * params[1] ** 2)))
    NegLnL = -1 * (- x.size * np.log(inf_sum) - sum_log - np.sum(((np.log(x) - params[0]) ** 2) / (2 * params[1] ** 2)))
    return NegLnL


def poisson_dist(lam, x: np.ndarray, delta: float, k_min):
    m = np.arange(k_min)
    LnL = x.size * np.log(1 - delta) - np.log(1 - np.exp(-1 * lam) * np.sum((lam ** m) / factorial(m))) \
          - x.size * lam + np.log(lam) * x.sum() - np.sum(np.log(factorial(x)))
    NegLnL = -1 * LnL
    return NegLnL


def poisson_large_k(lam, x: np.ndarray):
    d1 = poisson.pmf(x, lam)
    d1 = d1[np.nonzero(d1)]
    NegLnL = -1 * np.sum(np.log(d1))
    return NegLnL


def MLE(X: np.ndarray, k_min, vt: int = 3, IC: str = 'AIC'):
    votes = [100, 10, 100, 10, 100]
    Results = {}
    Results['Powerlaw'] = {}
    Results['Exponential'] = {}
    Results['Weibull'] = {}
    Results['Normal'] = {}
    Results['Trunc_PL'] = {}
    Results['Lognormal'] = {}
    Results['Poisson'] = {}
    stop = False
    while stop == False:  # np.std(votes[-vt:]) >= 0.1: # while the last X votes have not been the same
        # where X is vt.
        x = X[X >= k_min]  # only include degree values over kmin
        delta = (X[X < k_min].size / X.size)  # fraction below kmin
        k_mean = x.mean()  # mean degree for initial parameter guesses

        try:
            inf = np.arange(np.amax(x) + 1000)  # list of numbers for infinite sums required below
        except ValueError:  # raised if x is empty.
            inf = 1000
        sum_log = np.sum(np.log(x))

        opt_pl = minimize(powerlaw, (2), (x, sum_log, delta, k_min), method='SLSQP', bounds=[(0.5, 4)])
        Results['Powerlaw'][k_min] = [opt_pl['x'], -1 * opt_pl['fun']]

        opt_exp = minimize(exp_dist, (k_mean), (x, delta, k_min), method='SLSQP', bounds=((0.5, k_mean + 20),))
        Results['Exponential'][k_min] = [opt_exp['x'], -1 * opt_exp['fun']]

        opt_wb = minimize(weibull, (k_mean, 1), (x, inf, sum_log, delta, k_min), method='SLSQP',
                          bounds=((0.05, None), (0.05, 4),))
        Results['Weibull'][k_min] = [opt_wb['x'], - 1 * opt_wb['fun']]

        opt_normal = minimize(normal, (k_mean, np.std(x)), (x, inf), method='SLSQP',
                              bounds=[(0., k_mean + 10), (0.1, None)])
        Results['Normal'][k_min] = [opt_normal['x'], -1 * opt_normal['fun']]

        opt_tpl = minimize(trunc_powerlaw, (k_mean, 1), (x, inf, delta, k_min), method='SLSQP',
                           bounds=((0.5, k_mean + 20), (0.5, 4),))
        Results['Trunc_PL'][k_min] = [opt_tpl['x'], -1 * opt_tpl['fun']]
        try:  # prevents valueerror when value goes out of bounds given in function
            opt_logn = minimize(logn, (np.log(k_mean), np.log(x).std()), (x, inf, sum_log, k_min), method='TNC',
                                bounds=[(0., np.log(k_mean) + 10), (0.01, np.log(x.std()) + 10)])
            Results['Lognormal'][k_min] = [opt_logn['x'], -1 * opt_logn['fun']]
        except ValueError:
            Results['Lognormal'][k_min] = [[0, 0], 10000]
        try:
            poisson_max = np.amax(x)
        except ValueError:
            poisson_max = 1
        if poisson_max > 170:  # different method used when k_max is large, due to infinity from factorial
            opt_p = minimize(poisson_large_k, x.mean(), (x), method='SLSQP')
        else:
            opt_p = minimize(poisson_dist, x.mean(), (x, delta, k_min), method='SLSQP', bounds=((0.5, None),))
        Results['Poisson'][k_min] = [opt_p['x'], -1 * opt_p['fun']]
        Distributions = list(Results.keys())

        #   x0 = [k_min*2, x.mean(), x.max()]
        #   opt_cmp = compound_poisson(x, x0)
        #   Results['Compound Poisson'][k_min] = [opt_cmp['x'], -1*opt_p['fun']]

        AICs = []
        BICs = []

        for i in Results.keys():
            if i == 'Lognormal':
                if Results[i][k_min][0][1] == 0:
                    AICs.append(float("inf"))
                    BICs.append(float("inf"))
            if AIC(Results[i][k_min][1], x.size, len(Results[i][k_min][0])) == float("-inf"):
                AICs.append(float("inf"))
            else:
                AICs.append(AIC(Results[i][k_min][1], x.size, len(Results[i][k_min][0])))
            if BIC(Results[i][k_min][1], x.size, len(Results[i][k_min][0])) == float("-inf"):
                BICs.append(float("inf"))
            else:
                BICs.append(BIC(Results[i][k_min][1], x.size, len(Results[i][k_min][0])))
        weights = []
        weight_total = 0
        for i in AICs:
            weight_total += np.exp(-1 * (i - np.min(AICs)) / 2)

        for i in AICs:
            weights += [np.exp(-1 * (i - np.min(AICs)) / 2) / weight_total]

        if IC == 'AIC':
            votes.append(np.argmax(weights).astype(np.int32))

        if IC == 'BIC':
            votes.append(np.argmin(BICs).astype(np.int32))
        # if we only want to fit at a specific k_min, break the loop and return the first result
        if vt == 1:
            Delta = (X[X < k_min]).size / X.size
            Final_dist = [k_min, Distributions[np.argmax(weights)], Results[Distributions[np.argmax(weights)]][k_min],
                          Delta]
            return Final_dist
        if vt > 1:
            if np.std(votes[-vt:]) <= 0.1:
                stop = True
        k_min += 1

    Delta = (X[X < (k_min - vt)]).size / X.size
    Final_dist = [k_min - vt, Distributions[np.argmax(weights)], Results[Distributions[np.argmax(weights)]][k_min - vt],
                  Delta]
    if len(weights) > 0:
        Final_dist.append(weights)
    return Final_dist


def opt_single_dist(X, result, k_min):
    # k_min = result[0]
    x = X[X >= k_min]
    delta = (X[X < k_min].size / X.size)
    k_mean = x.mean()
    try:
        inf = np.arange(np.amax(x))  # Creates a sequence of numbers for infinite sums
    except ValueError:  # raised if x is empty.
        inf = 1000
    sum_log = np.sum(np.log(x))
    if result[1] == 'Powerlaw':
        opt = minimize(powerlaw, (2), (x, sum_log, delta, k_min), method='SLSQP', bounds=[(0.5, 4)])

    if result[1] == 'Exponential':
        opt = minimize(exp_dist, (k_mean), (x, delta, k_min), method='SLSQP', bounds=((0.5, k_mean + 20),))

    if result[1] == 'Weibull':
        opt = minimize(weibull, (k_mean, 1), (x, inf, sum_log, delta, k_min), method='SLSQP',
                       bounds=((0.05, None), (0.05, 4),))

    if result[1] == 'Normal':
        opt = minimize(normal, (k_mean, np.std(x)), (x, inf), method='SLSQP', bounds=[(0., k_mean + 10), (0.1, None)])

    if result[1] == 'Stretched_Exp':
        opt = minimize(stretched_exp, (k_mean, 1), (x, inf, k_min), method='SLSQP', bounds=[(0.5, None), (0.05, 4)])

    if result[1] == 'Trunc_PL':
        opt = minimize(trunc_powerlaw, (k_mean, 1), (x, inf, delta, k_min), method='SLSQP',
                       bounds=((0.5, k_mean + 20), (0.5, 4),))

    if result[1] == 'Lognormal':
        try:  # prevents valueerror when value goes out of bounds given in function
            opt = minimize(logn, (np.log(k_mean), np.log(x).std()), (x, inf, sum_log, k_min),
                           method='TNC', bounds=[(0., np.log(k_mean) + 10), (0.01, np.log(x.std()) + 10)])
        except ValueError:
            return [0, 0]
    if result[1] == 'Poisson':
        try:
            poisson_max = np.amax(x)
        except ValueError:
            poisson_max = 1
        if poisson_max > 170:  # different method used when k_max is large, due to infinity from factorial
            opt = minimize(poisson_large_k, x.mean(), (x), method='SLSQP')
        else:
            opt = minimize(poisson_dist, x.mean(), (x, delta, k_min), method='SLSQP', bounds=((0.5, None),))
    return opt['x']


def plotting(N, Input, fit, result, emp, dist, Name='', save=False, saveloc=''):
    plt.step(N, emp, '.', ms=10, color='k', label='Actual')
    plt.plot(Input, fit, label=result[1], linewidth=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$k$', color='k', fontsize=50)  # Increased font size to 16
    plt.ylim(ymin=0.5 * np.min(emp))
    if dist == 'PDF':
        plt.ylabel(r'$p_k$', fontsize=50)  # Increased font size to 16
    if dist == 'CCDF':
        plt.ylabel(r'$P_k$', fontsize=50)  # Increased font size to 16
    plt.title(Name + ' ' + ' ' + r'$k_{\rm min} = $' + str(result[0]), fontsize=50)
    plt.legend(fontsize=30)  # Increased font size of legend labels to 14
    if save == True:
        if saveloc == None:
            raise ValueError
        else:
            plt.savefig(saveloc + 'MLE_plot_' + dist, dpi=300)
    plt.xticks(fontsize=50)  # Increased font size to 16 for x-axis ticks
    plt.yticks(fontsize=50)  # Increased font size to 16 for y-axis ticks
    plt.tight_layout()  # Add tight layout
    plt.show()


def bootstrap(G_list, result):
    params1 = []
    params2 = []
    while len(params1) < 1000:
        sample = np.array(np.random.choice(G_list, len(G_list), replace=True))
        opt = opt_single_dist(sample, result, result[0])
        if len(opt) == 1:  # if it is a one parameter distribution
            if np.isnan(opt[0]) == False:  # we only count non-nan parameter values
                params1.append(opt[0])

        if len(opt) == 2:
            if np.isnan(opt[0]) == False & np.isnan(opt[1]) == False:
                params1.append(opt[0])
                params2.append(opt[1])

    if len(params2) == 0:  # graph has one distribution
        parameters = [params1]
    else:
        parameters = [params1, params2]

    return parameters


def summary_stats(Name, result, params):
    means = []
    devs = []
    perc1 = []
    perc2 = []
    for i in params:
        means.append(np.round(np.mean(i), 2))
        devs.append(np.round(np.std(i), 2))
        perc1.append(np.round(np.percentile(i, 2.5), 2))
        perc2.append(np.round(np.percentile(i, 97.5), 2))

    row = [Name, result[1], result[0], result[2][0][0], means[0], devs[0], perc1[0], perc2[0]]
    if len(result[2][0]) == 2:
        row.extend([result[2][0][1], means[1], devs[1], perc1[1], perc2[1]])
    return row


def fit(Name, G, k_min, vt=None, plot_type='auto', save=False, saveloc=None, IC='AIC'):
    X = degree_list(G)  # Get the degree list
    if len(X) < 2000:
        if vt == None:
            vt = 2  # for small datasets we need fewer consecutive votes to
            # determine the distribution
        if plot_type == 'auto':
            plot_type = 'ccdf'  # for small datasets we do not want to display the PDF
    else:
        if vt == None:
            vt = 3
        if plot_type == 'auto':
            plot_type = 'both'  # for larger datasets we display both PDF and
            # CCDF

    result = MLE(X, k_min, vt, IC)  # Perform the MLE using kmin=1 as default
    N, P, p = empirical(X)  # Get the unique degree list, empirical CCDF and
    # PDF values

    Input = np.arange(result[0], np.amax(X) + 1)  # generate complete list of integers
    # kmin to kmax

    # if plot_type == 'both' or plot_type == 'pdf':
    #     pdf = PDF(result, Input, N, P)
    #     plotting(N, Input, pdf, result, p, 'PDF', Name, save, saveloc)
    # if plot_type == 'both' or plot_type == 'ccdf':
    #     ccdf = CCDF(result, Input, N, P)
    #     plotting(N, Input, ccdf, result, P, 'CCDF', Name, save, saveloc)

    # print('kmin', result[0], result[1], 'parameters', np.round(result[2][0], 3))

    return result
