import numpy as np
import pandas as pd
from scipy.stats import norm



def price_vanilla_bond(maturity, face_value, coupon_rate, discount_curve, frequency=1):
    """Price a bond by discounting its cashflows using a discount curve.

    Parameters
    ----------
    maturity : float
        Time to maturity in years (e.g. 3 for a 3-year bond).
    face_value : float
        Par / face value of the bond.
    coupon_rate : float
        Annualized coupon rate (e.g. 0.06 for 6%).
    discount_curve : pd.DataFrame
        Must contain columns 'ttm' (time to maturity in years)
        and 'discount' (discount factor for that tenor).

    Returns
    -------
    float
        The present value (price) of the bond.
    """
    period = 1 / frequency
    coupon = face_value * coupon_rate * period

    # coupon dates at every period interval up to maturity
    coupon_times = np.arange(period, maturity + period / 2, period)

    # build cashflows: coupon at each date, plus face value at maturity
    cashflows = np.full_like(coupon_times, coupon)
    cashflows[-1] += face_value

    # look up discount factors by matching ttm
    discount_factors = np.array([
        discount_curve.loc[discount_curve['ttm'] == t, 'discount'].values[0]
        for t in coupon_times
    ])

    price = np.sum(cashflows * discount_factors)
    return price

def call_option_bond(expiration, strike, vol, forward_price, discount_curve):
    """Price a call option on a bond using Black's model.

    Parameters
    ----------
    expiration : float
        Time to expiration in years.
    strike : float
        Strike price of the option.
    vol : float
        Volatility of the forward bond price (annualized).
    forward_price : float
        Forward price of the underlying bond.
    discount_curve : pd.DataFrame or float
        Either a DataFrame with columns 'ttm' and 'discount',
        or a scalar discount factor.

    Returns
    -------
    float
        The present value of the call option.
    """

    if isinstance(discount_curve, (int, float)):
        discount_factor = discount_curve
    else:
        discount_factor = discount_curve.loc[discount_curve['ttm'] == expiration, 'discount'].values[0]

    d1 = np.log(forward_price/strike) + (vol**2 / 2) * expiration
    d1 /= vol * np.sqrt(expiration)

    d2 = d1 - vol * np.sqrt(expiration)

    option_price = forward_price * norm.cdf(d1) - strike * norm.cdf(d2)
    option_price *= discount_factor
    return option_price


def put_option_bond(expiration, strike, vol, forward_price, discount_curve):
    """Price a put option on a bond using Black's model.

    Parameters
    ----------
    expiration : float
        Time to expiration in years.
    strike : float
        Strike price of the option.
    vol : float
        Volatility of the forward bond price (annualized).
    forward_price : float
        Forward price of the underlying bond.
    discount_curve : pd.DataFrame or float
        Either a DataFrame with columns 'ttm' and 'discount',
        or a scalar discount factor.

    Returns
    -------
    float
        The present value of the put option.
    """
    if isinstance(discount_curve, (int, float)):
        discount_factor = discount_curve
    else:
        discount_factor = discount_curve.loc[discount_curve['ttm'] == expiration, 'discount'].values[0]

    d1 = np.log(forward_price/strike) + (vol**2 / 2) * expiration
    d1 /= vol * np.sqrt(expiration)

    d2 = d1 - vol * np.sqrt(expiration)

    option_price = strike * norm.cdf(-d2) - forward_price * norm.cdf(-d1)
    option_price *= discount_factor
    return option_price


def price_callable_bond(maturity, face_value, coupon_rate, discount_curve,
                        expiration, strike, vol, forward_price, frequency=1):
    """Price a callable bond as: vanilla bond - call option.

    Parameters
    ----------
    maturity : float
        Time to maturity in years.
    face_value : float
        Par / face value of the bond.
    coupon_rate : float
        Annualized coupon rate.
    discount_curve : pd.DataFrame
        Must contain columns 'ttm' and 'discount'.
    expiration : float
        Time to expiration of the embedded call option in years.
    strike : float
        Strike price of the embedded call option.
    vol : float
        Volatility of the forward bond price (annualized).
    forward_price : float
        Forward price of the underlying bond at option expiration.
    frequency : int
        Coupon frequency per year (default 1 = annual).

    Returns
    -------
    float
        The price of the callable bond.
    """
    vanilla = price_vanilla_bond(maturity, face_value, coupon_rate, discount_curve, frequency)
    call = call_option_bond(expiration, strike, vol, forward_price, discount_curve)
    return vanilla - call


def value_option_chain(option_data, futures_price, option_type='C'):
    """Value all options at the nearest expiration using Black's model.

    Parameters
    ----------
    option_data : pd.DataFrame
        Option chain with columns: 'days to expiration', 'option type',
        'strike price', 'implied vol', 'finance rate', 'price'.
    futures_price : float
        The underlying futures price.
    option_type : str
        'C' for calls, 'P' for puts (default 'C').

    Returns
    -------
    pd.DataFrame
        DataFrame with strike, market price, model price, and difference.
    """
    # filter to earliest expiration and option type
    nearest_exp = option_data['days to expiration'].min()
    chain = option_data[
        (option_data['days to expiration'] == nearest_exp) &
        (option_data['option type'] == option_type)
    ].copy()

    # compute shared values
    tau = nearest_exp / 365
    rate = 100 - futures_price

    model_prices = []
    for _, row in chain.iterrows():
        strike = row['strike price']
        implied_vol_rate = row['implied vol']
        finance_rate = row['finance rate']

        vol_price = implied_vol_rate * (rate / futures_price)
        discount_factor = 1 / (1 + finance_rate * tau)
        if option_type == "C":
            model_price = call_option_bond(
                expiration=tau,
                strike=strike,
                vol=vol_price,
                forward_price=futures_price,
                discount_curve=discount_factor
            )
        else:
            model_price = put_option_bond(
                expiration=tau,
                strike=strike,
                vol=vol_price,
                forward_price=futures_price,
                discount_curve=discount_factor
            )

        model_prices.append(model_price)

    chain['model price'] = model_prices
    chain['difference'] = chain['model price'] - chain['price']

    return chain[['strike price', 'price', 'model price', 'difference']]
