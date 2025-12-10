import math

# ---------- Normal distribution helpers ----------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


# =================================================
# FX OPTIONS (Garman–Kohlhagen + Binomial)
# =================================================

def garman_kohlhagen(option_type: str, S: float, K: float,
                     rd: float, rf: float, sigma: float, T: float) -> float:
    """
    Garman–Kohlhagen formula for FX European options.
    rd = domestic rate, rf = foreign rate
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    if T <= 0.0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    if sigma <= 0.0:
        forward = S * math.exp((rd - rf) * T)
        disc = math.exp(-rd * T)
        if option_type == "call":
            return disc * max(forward - K, 0.0)
        else:
            return disc * max(K - forward, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (rd - rf + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_d = math.exp(-rd * T)
    disc_f = math.exp(-rf * T)

    if option_type == "call":
        price = S * disc_f * norm_cdf(d1) - K * disc_d * norm_cdf(d2)
    else:
        price = K * disc_d * norm_cdf(-d2) - S * disc_f * norm_cdf(-d1)

    return float(price)


def gk_greeks(option_type: str, S: float, K: float,
              rd: float, rf: float, sigma: float, T: float):
    """
    Greeks for Garman–Kohlhagen FX options.
    Returns dict: delta, gamma, vega, theta, rho_domestic, rho_foreign
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    if T <= 0.0 or sigma <= 0.0:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho_domestic": 0.0,
            "rho_foreign": 0.0,
        }

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (rd - rf + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_d = math.exp(-rd * T)
    disc_f = math.exp(-rf * T)
    pdf_d1 = norm_pdf(d1)

    if option_type == "call":
        delta = disc_f * norm_cdf(d1)
    else:
        delta = -disc_f * norm_cdf(-d1)

    gamma = disc_f * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_f * pdf_d1 * sqrtT

    if option_type == "call":
        theta = (
            -S * disc_f * pdf_d1 * sigma / (2.0 * sqrtT)
            - rd * K * disc_d * norm_cdf(d2)
            + rf * S * disc_f * norm_cdf(d1)
        )
        rho_dom = T * K * disc_d * norm_cdf(d2)
        rho_for = -T * S * disc_f * norm_cdf(d1)
    else:
        theta = (
            -S * disc_f * pdf_d1 * sigma / (2.0 * sqrtT)
            + rd * K * disc_d * norm_cdf(-d2)
            - rf * S * disc_f * norm_cdf(-d1)
        )
        rho_dom = -T * K * disc_d * norm_cdf(-d2)
        rho_for = T * S * disc_f * norm_cdf(-d1)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho_domestic": float(rho_dom),
        "rho_foreign": float(rho_for),
    }


def fx_binomial(option_type: str, exercise_type: str,
                S: float, K: float, rd: float, rf: float,
                sigma: float, T: float, N: int) -> float:
    """
    CRR binomial tree for FX options (European or American).
    """
    option_type = option_type.lower()
    exercise_type = exercise_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if exercise_type not in ("european", "american"):
        raise ValueError("exercise_type must be 'european' or 'american'.")
    if N < 1:
        raise ValueError("N must be >= 1.")

    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-rd * dt)
    p = (math.exp((rd - rf) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError("Risk-neutral probability out of (0,1).")

    prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
    if option_type == "call":
        vals = [max(price - K, 0.0) for price in prices]
    else:
        vals = [max(K - price, 0.0) for price in prices]

    for i in range(N - 1, -1, -1):
        new_vals = []
        new_prices = [S * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
        for j in range(i + 1):
            cont = disc * (p * vals[j + 1] + (1 - p) * vals[j])
            if exercise_type == "american":
                intr = max(
                    new_prices[j] - K, 0.0
                ) if option_type == "call" else max(K - new_prices[j], 0.0)
                new_vals.append(max(cont, intr))
            else:
                new_vals.append(cont)
        vals = new_vals

    return float(vals[0])


# =================================================
# STOCK OPTIONS (Black–Scholes + Binomial)
# =================================================

def black_scholes(option_type: str, S: float, K: float,
                  r: float, sigma: float, T: float, q: float = 0.0) -> float:
    """
    Black–Scholes for equity options with continuous dividend yield q.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    if T <= 0.0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type == "call":
        price = S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    else:
        price = K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)

    return float(price)


def bs_greeks(option_type: str, S: float, K: float,
              r: float, sigma: float, T: float, q: float = 0.0):
    """
    Greeks for Black–Scholes equity option with dividend yield q.
    Returns dict: delta, gamma, vega, theta, rho
    """
    option_type = option_type.lower()
    if T <= 0.0 or sigma <= 0.0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf_d1 = norm_pdf(d1)

    if option_type == "call":
        delta = disc_q * norm_cdf(d1)
    else:
        delta = disc_q * (norm_cdf(d1) - 1.0)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT

    if option_type == "call":
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2.0 * sqrtT)
            - r * K * disc_r * norm_cdf(d2)
            + q * S * disc_q * norm_cdf(d1)
        )
        rho = K * T * disc_r * norm_cdf(d2)
    else:
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2.0 * sqrtT)
            + r * K * disc_r * norm_cdf(-d2)
            - q * S * disc_q * norm_cdf(-d1)
        )
        rho = -K * T * disc_r * norm_cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def equity_binomial(option_type: str, exercise_type: str,
                    S: float, K: float, r: float,
                    sigma: float, T: float, N: int, q: float = 0.0) -> float:
    """
    CRR binomial tree for equity options with dividend yield q.
    """
    option_type = option_type.lower()
    exercise_type = exercise_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if exercise_type not in ("european", "american"):
        raise ValueError("exercise_type must be 'european' or 'american'.")
    if N < 1:
        raise ValueError("N must be >= 1.")

    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError("Risk-neutral probability out of (0,1).")

    prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]
    if option_type == "call":
        vals = [max(price - K, 0.0) for price in prices]
    else:
        vals = [max(K - price, 0.0) for price in prices]

    for i in range(N - 1, -1, -1):
        new_vals = []
        new_prices = [S * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
        for j in range(i + 1):
            cont = disc * (p * vals[j + 1] + (1 - p) * vals[j])
            if exercise_type == "american":
                intr = max(
                    new_prices[j] - K, 0.0
                ) if option_type == "call" else max(K - new_prices[j], 0.0)
                new_vals.append(max(cont, intr))
            else:
                new_vals.append(cont)
        vals = new_vals

    return float(vals[0])
