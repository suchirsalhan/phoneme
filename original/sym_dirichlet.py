
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sys
sys.path.append('/Users/fermin/CODE/PYTHON3/') # just to add to my searchpath
from entropies import *

def plot_ranks(counts, alpha=None, loglog=True, title=None, types="order",\
               error=95,reduced=False,ylim=None):
    """
    Plot predicted and observed probability ranks using estimated order statistics.

    Parameters:
    - counts: array-like, frequency counts of observed items
    - loglog: bool, whether to use log-log scale for the plot
    - title: str or None, optional plot title
    - types: str, "order" or other (for reversed rank axis)
    """
    V = len(counts)
    ps = np.sort(np.array(counts) / np.sum(counts))

    if alpha is None:
        #entropy_val = Entropy(counts, method="CWJ")
        #alpha, _, _ = estimate_alpha_bma_order_entropy_constrained(ps, entropy_val)
        alpha, _ = estimate_alpha_entropy(counts,len(counts), n_boot=0)

    ranks = np.arange(1, V + 1)
    if reduced:
        ranks2 = np.unique(np.round(np.exp(np.linspace(0,np.log(V),reduced)))).astype(int)
        marker1 = marker2 = ""
    else:
        ranks2 = ranks
        marker1="."
        marker2="o"
    means, std_errors = zip(*[
        numerical_beta_order_stat_moments(i, V, alpha, (V - 1) * alpha)
        for i in ranks2
    ])
    
    if reduced:
        means2=np.interp(ranks,ranks2,means)
        means = np.array(means)/np.sum(means2)
        std_errors = np.array(std_errors)/np.sum(means2)
    else:
        means = np.array(means)/np.sum(means)
        std_errors = np.array(std_errors)/np.sum(means)


    if type(error) is int:
        n = -sp.stats.norm.ppf((1-error/100)/2)
    elif type(error) is float:
        n = -sp.stats.norm.ppf((1-error)/2)
    else:
        n = 1

    if types != "order":
        ranks = V + 1 - ranks
        ranks2 = V + 1 - ranks2

    # Plot predicted means with confidence intervals
    plt.plot(ranks2, means, color="green", marker=marker1, label="Predicted Mean")
    plt.fill_between(ranks2,
                     np.clip(means - n * std_errors, 0, 1),
                     np.clip(means + n * std_errors, 0, 1),
                     alpha=0.2, color="green")

    plt.fill_between(ranks2,
                     np.clip(means - std_errors, 0, 1),
                     np.clip(means + std_errors, 0, 1),
                     alpha=0.4, color="green")


    # Plot observed probabilities
    plt.plot(ranks, np.sort(ps), color="red", label="Observed", marker=marker2)

    # Labels and scales
    if types == "order":
        plt.xlabel(r"$i$-th Order Statistic")
        plt.ylabel(r"$\mathbb{P}_{(i)}$")
    else:
        plt.xlabel(r"Rank ($r$)")
        plt.ylabel(r"$P(r)$")

    if loglog:
        plt.xscale("log")
        plt.yscale("log")

    if title:
        plt.title(title)
    
    if not ylim is None:
        plt.ylim(ylim)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def beta_order_stat_pdf(x, i, n, a, b,ll=False):
    """
    PDF of the i-th order statistic from a Beta(a, b) distribution.

    Parameters:
        x (float): evaluation point
        i (int): order statistic index (1-based)
        n (int): number of total samples
        a (float): alpha parameter of Beta
        b (float): beta parameter of Beta

    Returns:
        float: value of the PDF at x
    """
    coeff = sp.special.loggamma(n)-sp.special.loggamma(i)-sp.special.loggamma(n-i+1)+np.log(n)
    beta_pdf = sp.stats.beta.logpdf(x, a, b)
    beta_cdf = sp.stats.beta.logcdf(x, a, b)
    beta_sf = sp.stats.beta.logsf(x, a, b)
    res = coeff + beta_pdf + (i-1)*beta_cdf + (n-i)*beta_sf
    if not ll:
        res = np.exp(res)
    return res
    


def numerical_beta_order_stat_moments(i, n, a, b):
    """
    Numerically compute the mean and standard deviation of the i-th order statistic
    from n samples of a Beta(a, b) distribution.

    Parameters:
        i (int): order statistic index (1-based)
        n (int): total number of samples
        a (float): alpha parameter of Beta
        b (float): beta parameter of Beta

    Returns:
        mean (float), std_dev (float)
    """
    def integrand_mean(x):
        return x * beta_order_stat_pdf(x, i, n, a, b)

    def integrand_sq(x):
        return x**2 * beta_order_stat_pdf(x, i, n, a, b)

    mean, _ = quad(integrand_mean, 0, 1, epsabs=1e-10)
    mean_sq, _ = quad(integrand_sq, 0, 1, epsabs=1e-10)
    std_dev = np.sqrt(mean_sq - mean**2)

    return mean, std_dev

def dirichlet_entropy(alpha, V):
    """Entropy of a symmetric Dirichlet(alpha,...,alpha) distribution."""
    return sp.special.psi(V * alpha + 1) - sp.special.psi(alpha + 1)

def estimate_alpha_entropy(counts,k=None,n_boot=500,method="CWJ",x0=1):
    """
    Estimate symmetric Dirichlet alpha via its Entropy.

    Parameters:
        counts (list or array): counts from a Dirichlet sample
        k (int): True dimensionality of the Dirichlet distribution (
                 (k >= len(counts))

    Returns:
        alpha (float): Estimate of alpha
    """
    if k is None or  k < len(counts):
        k = len(counts)
    h = Entropy(counts,method=method)
    opt = _h_estimate(h,k,x0=x0)
    if opt.converged:
        if n_boot:
            N = np.sum(counts)
            ps = np.array(counts)/N
            samp = sp.stats.multinomial.rvs(N,ps,size=n_boot)
            hsamp = np.array([_h_estimate(Entropy(samp[i,:],method=method),k,x0=x0).root \
                     for i in range(n_boot)])
            se = sp.stats.sem(hsamp**2)
        else:
            se=-1.
        return opt.root**2,se
    else:
        return np.nan,np.nan
    
def _h_estimate(h,k,method=None,x0=1,x1=0):
    eq = lambda a: dirichlet_entropy(a**2,k)-h
    fprime1 = lambda a: 2 * a * (k * sp.special.polygamma(1, k * a**2 + 1) - \
                                 sp.special.polygamma(1, a**2 + 1))

    opt = sp.optimize.root_scalar(eq,x0=x0,x1=x1,fprime=fprime1)
    return opt
    
    

