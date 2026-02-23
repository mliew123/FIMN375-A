import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def price_floorlet(df, maturity, strike, implied_vol, delta=0.25, notional=100):
    """
    Price a single floorlet using Black's model.

    The function extracts the relevant inputs (forward rate, discount factor,
    forward volatility, and swap rate strike) from the provided dataframe
    and computes the present value of the floorlet.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the interest rate curve and volatility data.
        Must include the columns:
            - "tenor"
            - "swap rates"
            - "discounts"
            - "forwards"
            - "fwd vols"

    maturity : float
        Payment date of the floorlet (in years).

    delta : float
        Accrual fraction (e.g., 0.25 for quarterly).

    notional : float, optional
        Notional amount of the floorlet. Default is 100.

    Returns
    -------
    float
        Present value of the floorlet under Black's model.

    Notes
    -----
    The pricing formula used is:

        Floorlet = N * delta * Z(T) *
                    [ K * N(-d2) - F * N(-d1) ]

    where:
        d1 = [ln(F/K) + 0.5 * sigma^2 * tau] / (sigma * sqrt(tau))
        d2 = d1 - sigma * sqrt(tau)

    This corresponds to a put option on the forward rate under the
    forward measure.
    """
    tau = maturity - delta
    # Discount factor to payment date T
    Z = df.loc[df["tenor"] == maturity, "discounts"].values[0]
    # Forward rate for [T-0.25, T]
    F = df.loc[df["tenor"] == maturity, "forwards"].values[0]
    # Forward vol
    # Black d1, d2
    d1 = (np.log(F / strike) + 0.5 * implied_vol**2 * tau) / (implied_vol * np.sqrt(tau))
    d2 = d1 - implied_vol * np.sqrt(tau)
    
    # Black floorlet formula (put)
    price = notional * delta * Z * (
        strike * norm.cdf(-d2) - F * norm.cdf(-d1)
    )
    return price

def price_floor(df, maturity, strike, implied_vol):
    expirations = np.arange(2, maturity*4+1) / 4
    floor_price = 0
    for exp in expirations:
        floor_price += price_floorlet(df, exp, strike, implied_vol)

    return floor_price

def rates_to_disc_factors(tenor, discount_rates, freq=4, continous = False):
    discount_factors = []
    for i in range(len(discount_rates)):
        if continous:
            discount_factors.append(np.e ** (-discount_rates[i] * tenor[i]))
        else:
            discount_factors.append((1/(1 + discount_rates[i]/freq))**(freq * tenor[i]))

    return discount_factors

def compute_forwards(rate_data):
    """
    Compute forward rates from discount factors.

    Assumes:
        rate_data has columns:
            - "tenor"
            - "discounts"
    Returns:
        numpy array of forward rates
    """

    df = rate_data.copy().sort_values("tenor").reset_index(drop=True)

    tenors = df["tenor"].values
    discounts = df["discounts"].values

    forwards = np.zeros(len(tenors))
    forwards[0] = np.nan  # no forward for first point

    for i in range(1, len(tenors)):
        delta = tenors[i] - tenors[i-1]
        forwards[i] = (discounts[i-1] / discounts[i] - 1) / delta

    return forwards



def price_caplet(df, maturity, strike, implied_vol, delta=0.25, notional=100):
    """
    Price a single caplet using Black's model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing term structure data. Must include:
            - "tenor" (payment dates in years)
            - "discounts"
            - "forwards"

    maturity : float
        Payment date of the caplet (T).

    strike : float
        Strike rate (constant across caplets in a cap).

    implied_vol : float
        Black implied volatility for this caplet.

    delta : float, optional
        Accrual fraction (default 0.25 for quarterly).

    notional : float, optional
        Notional amount (default 100).

    Returns
    -------
    float
        Present value of the caplet.

    Notes
    -----
    The payoff is:

        N * delta * max(L - K, 0)

    paid at time T, where the rate L is fixed at T - delta.

    Black formula:

        Caplet = N * delta * Z(T) *
                  [ F * N(d1) - K * N(d2) ]

    where:
        tau = T - delta
        d1 = [ln(F/K) + 0.5 σ² tau] / (σ √tau)
        d2 = d1 - σ √tau
    """

    tau = maturity-delta
    # Discount factor to payment date T
    Z = df.loc[df["tenor"] == maturity, "discounts"].values[0]
    # Forward rate for [T-0.25, T]
    F = df.loc[df["tenor"] == maturity, "forwards"].values[0]

    # Black d1, d2
    d1 = (np.log(F / strike) + 0.5 * implied_vol**2 * tau) / (implied_vol * np.sqrt(tau))
    d2 = d1 - implied_vol * np.sqrt(tau)
    
    # Black caplet formula (call)
    price = notional * delta * Z * (
        F * norm.cdf(d1) - strike * norm.cdf(d2)
    )
    return price

def price_cap(df, maturity, strike, implied_vol):
    """
    Price an interest rate cap as the sum of its caplets.

    Parameters
    ----------
    df : pandas.DataFrame
        Term structure data containing:
            - "tenor"
            - "discounts"
            - "forwards"

    maturity : float
        Final payment date of the cap (e.g., 3 for a 3Y cap).

    strike : float
        Strike rate of the cap (constant across all caplets).

    implied_vol : float
        Implied volatility used for pricing (flat cap vol or per-caplet vol).

    delta : float, optional
        Accrual fraction (default 0.25).

    notional : float, optional
        Notional amount (default 100).

    Returns
    -------
    float
        Present value of the cap.

    Notes
    -----
    A cap is a portfolio of caplets:

        Cap = Σ Caplet(T_i)

    where payment dates T_i range from delta to maturity
    (excluding already-fixed initial period).

    Payment dates are used as the indexing variable.
    """

    expirations = np.arange(2, maturity*4+1) / 4
    cap_price = 0
    for exp in expirations:
        cap_price += price_caplet(df, exp, strike, implied_vol)

    return cap_price


import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def find_caplet_implied_vol(df, maturity, strike, market_price,
                            delta=0.25, notional=100):
    """
    Compute Black implied volatility for a single caplet.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
            - "tenor"
            - "discounts"
            - "forwards"

    maturity : float
        Payment date T.

    strike : float
        Strike rate.

    market_price : float
        Observed caplet market price.

    delta : float
        Accrual fraction.

    notional : float
        Notional amount.

    vol_lower, vol_upper : float
        Search interval for volatility.

    Returns
    -------
    float
        Implied Black volatility.
    """

    def error(vol):
        return price_caplet(df, maturity, strike, vol, delta, notional) - market_price

    return brentq(error, 1e-6, 5.0)


def find_forward_swap_rate(rate_infos, t, t_forward, maturity, freq=4):
    """
    Compute forward swap rate starting at t_forward
    and ending at maturity.

    Parameters
    ----------
    rate_infos : DataFrame
        Must contain:
            - "tenor"
            - "discounts"

    t : float
        Valuation time (usually 0).

    t_forward : float
        Swap start time (option expiry).

    maturity : float
        Swap end time.

    freq : int
        Payment frequency (4=quarterly, 2=semiannual).

    Returns
    -------
    float
        Forward swap rate.
    """

    df = rate_infos.copy()

    Z_fwd = df.loc[np.isclose(df["tenor"], t_forward), "discounts"].values[0]
    Z_T = df.loc[np.isclose(df["tenor"], maturity), "discounts"].values[0]

    delta = 1 / freq
    payment_dates = np.arange(t_forward + delta, maturity + 1e-10, delta)

    annuity = 0.0
    for T_i in payment_dates:
        Z_i = df.loc[np.isclose(df["tenor"], T_i), "discounts"].values[0]
        annuity += Z_i

    forward_swap_rate = freq * (Z_fwd - Z_T) / annuity

    return forward_swap_rate


def compute_swap_annuity(rate_infos, t_forward, maturity, freq=4):
    df = rate_infos.copy()
    
    delta = 1 / freq
    payment_dates = np.arange(t_forward + delta, maturity + 1e-10, delta)
    
    annuity = 0.0
    for T_i in payment_dates:
        Z_i = df.loc[np.isclose(df["tenor"], T_i), "discounts"].values[0]
        annuity += delta * Z_i
    
    return annuity


def black_swaption_price(F, K, sigma, T, annuity, notional=1.0, payer=True):
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if payer:
        price = notional * annuity * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = notional * annuity * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return price
    
