import os
import math
import time
import json
import re
import requests
from html import unescape
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

_price_cache = {}
_chain_cache = {}
_PRICE_TTL = 60
_CHAIN_TTL = 600

def cache_get(cache, key, ttl):
    rec = cache.get(key)
    if not rec:
        return None
    ts, val = rec
    if time.time() - ts > ttl:
        cache.pop(key, None)
        return None
    return val

def cache_set(cache, key, val):
    cache[key] = (time.time(), val)

def get_ohlc_for_symbol(symbol: str, period: str = "1mo"):
    key = f"ohlc::{symbol}::{period}"
    cached = cache_get(_price_cache, key, _PRICE_TTL)
    if cached:
        return cached
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d", auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        dates = [idx.strftime("%Y-%m-%d") for idx in df.index]
        out = {
            "symbol": symbol,
            "period": period,
            "dates": dates,
            "open": df["Open"].round(6).tolist(),
            "high": df["High"].round(6).tolist(),
            "low": df["Low"].round(6).tolist(),
            "close": df["Close"].round(6).tolist(),
        }
        cache_set(_price_cache, key, out)
        return out
    except Exception as e:
        print("ohlc fetch err", e)
        return None

SAMPLE_UNIVERSE = [
    "AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOGL", "META", "SPY", "QQQ", "EURUSD=X", "USDINR=X"
]

def get_movers(universe: List[str] = SAMPLE_UNIVERSE):
    key = f"movers::{','.join(universe)}"
    cached = cache_get(_price_cache, key, 30)
    if cached:
        return cached
    rows = []
    try:
        data = yf.download(tickers=" ".join(universe), period="5d", interval="1d", group_by="ticker", threads=False, progress=False)
        for s in universe:
            try:
                if len(universe) == 1:
                    df = data
                else:
                    df = data[s]
                last = df["Close"].iloc[-1]
                prev = df["Close"].iloc[-2]
                vol = df["Volume"].iloc[-1] if "Volume" in df.columns else 0
                pct = ((last - prev) / prev) * 100 if prev != 0 else 0.0
                rows.append({"symbol": s, "last": float(last), "pct_change": float(pct), "volume": int(vol)})
            except Exception:
                pass
    except Exception:
        for s in universe:
            try:
                t = yf.Ticker(s)
                hist = t.history(period="5d")
                if hist.shape[0] >= 2:
                    last = hist["Close"].iloc[-1]
                    prev = hist["Close"].iloc[-2]
                    vol = hist["Volume"].iloc[-1] if "Volume" in hist.columns else 0
                    pct = ((last - prev) / prev) * 100 if prev != 0 else 0.0
                    rows.append({"symbol": s, "last": float(last), "pct_change": float(pct), "volume": int(vol)})
            except Exception:
                pass
    dfm = pd.DataFrame(rows)
    if dfm.empty:
        result = {"gainers": [], "losers": [], "active": []}
        cache_set(_price_cache, key, result)
        return result
    gainers = dfm.sort_values("pct_change", ascending=False).head(10).to_dict("records")
    losers = dfm.sort_values("pct_change", ascending=True).head(10).to_dict("records")
    active = dfm.sort_values("volume", ascending=False).head(10).to_dict("records")
    result = {"gainers": gainers, "losers": losers, "active": active}
    cache_set(_price_cache, key, result)
    return result

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_scholes_price(S, K, r, sigma, T, option_type="call", q=0.0):
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(0.0, S - K)
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        price = S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)
    return price

def bs_greeks(S, K, r, sigma, T, q=0.0):
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta_call = math.exp(-q * T) * norm_cdf(d1)
    delta_put = -math.exp(-q * T) * norm_cdf(-d1)
    gamma = math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T)
    theta_call = (-S * sigma * math.exp(-q * T) * norm_pdf(d1) / (2 * math.sqrt(T))
                  - r * K * math.exp(-r * T) * norm_cdf(d2)
                  + q * S * math.exp(-q * T) * norm_cdf(d1))
    theta_put = (-S * sigma * math.exp(-q * T) * norm_pdf(d1) / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm_cdf(-d2)
                 - q * S * math.exp(-q * T) * norm_cdf(-d1))
    rho_call = K * T * math.exp(-r * T) * norm_cdf(d2)
    rho_put = -K * T * math.exp(-r * T) * norm_cdf(-d2)
    return {
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "vega": vega / 100.0,
        "theta_call": theta_call / 365.0,
        "theta_put": theta_put / 365.0,
        "rho_call": rho_call / 100.0,
        "rho_put": rho_put / 100.0
    }

def garman_kohlhagen_price(S, K, rd, rf, sigma, T, option_type="call"):
    return black_scholes_price(S, K, rd, sigma, T, option_type=option_type, q=rf)

def gk_greeks(S, K, rd, rf, sigma, T):
    g = bs_greeks(S, K, rd, sigma, T, q=rf)
    return {
        "delta": g["delta_call"],
        "gamma": g["gamma"],
        "vega": g["vega"],
        "theta": g["theta_call"],
        "rho_domestic": g["rho_call"],
        "rho_foreign": -g["delta_call"] * T
    }

def binomial_tree_price(S, K, r, sigma, T, steps=50, option_type="call", american=False, q=0.0):
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    a = math.exp((r - q) * dt)
    p = (a - d) / (u - d)
    if p < 0 or p > 1:
        p = max(0.0, min(1.0, 0.5))
    prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
    if option_type == "call":
        values = [max(0.0, p0 - K) for p0 in prices]
    else:
        values = [max(0.0, K - p0) for p0 in prices]
    disc = math.exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        new_values = []
        for j in range(i + 1):
            cont = disc * (p * values[j] + (1 - p) * values[j + 1])
            if american:
                spot = S * (u ** j) * (d ** (i - j))
                exercise = max(0.0, spot - K) if option_type == "call" else max(0.0, K - spot)
                newv = max(cont, exercise)
            else:
                newv = cont
            new_values.append(newv)
        values = new_values
    return values[0]

def _recursive_find(obj, key_name: str):
    if isinstance(obj, dict):
        if key_name in obj:
            return obj[key_name]
        for v in obj.values():
            res = _recursive_find(v, key_name)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = _recursive_find(item, key_name)
            if res is not None:
                return res
    return None

def get_yahoo_option_chain(symbol: str) -> Tuple[List[dict], List[dict]]:
    key = f"chain::{symbol}"
    cached = cache_get(_chain_cache, key, _CHAIN_TTL)
    if cached:
        return cached
    calls = []
    puts = []
    try:
        t = yf.Ticker(symbol)
        expiries = list(t.options or [])
        if expiries:
            expiry = expiries[0]
            chain = t.option_chain(expiry)
            for r in chain.calls.to_dict("records"):
                calls.append({
                    "expirationDate": expiry,
                    "strike": r.get("strike"),
                    "last": r.get("lastPrice"),
                    "bid": r.get("bid"),
                    "ask": r.get("ask"),
                    "volume": r.get("volume"),
                    "openInterest": r.get("openInterest"),
                    "impliedVolatility": r.get("impliedVolatility"),
                    "lastPrice": r.get("lastPrice"),
                })
            for r in chain.puts.to_dict("records"):
                puts.append({
                    "expirationDate": expiry,
                    "strike": r.get("strike"),
                    "last": r.get("lastPrice"),
                    "bid": r.get("bid"),
                    "ask": r.get("ask"),
                    "volume": r.get("volume"),
                    "openInterest": r.get("openInterest"),
                    "impliedVolatility": r.get("impliedVolatility"),
                    "lastPrice": r.get("lastPrice"),
                })
            cache_set(_chain_cache, key, (calls, puts))
            return calls, puts
    except Exception:
        pass
    url = f"https://finance.yahoo.com/quote/{symbol}/options?p={symbol}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; options-scraper/1.0; +https://example.com)"}
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code != 200:
            cache_set(_chain_cache, key, ([], []))
            return [], []
        text = resp.text
        m = re.search(r"root\.App\.main\s*=\s*({.*?})\s*;\n", text, re.S)
        data = None
        if m:
            try:
                payload = unescape(m.group(1))
                data = json.loads(payload)
            except Exception:
                data = None
        if data is None:
            idx = text.find("optionChain")
            if idx != -1:
                start = text.rfind("{", 0, idx)
                end = text.find("};", idx)
                if start != -1 and end != -1:
                    blob = text[start:end + 1]
                    try:
                        data = json.loads(blob)
                    except Exception:
                        data = None
        if data:
            options_obj = _recursive_find(data, "optionChain") or _recursive_find(data, "options") or _recursive_find(data, "optionContracts")
            if options_obj is None:
                stores = _recursive_find(data, "stores") or data
                options_obj = _recursive_find(stores, "OptionChainStore") or _recursive_find(stores, "OptionContractsStore") or _recursive_find(stores, "OptionStore")
            results = []
            if isinstance(options_obj, dict):
                results = options_obj.get("result") or options_obj.get("options") or []
            elif isinstance(options_obj, list):
                results = options_obj
            for block in results:
                exp = block.get("expirationDate") or block.get("expirationDates")
                ops = block.get("options") or []
                if isinstance(ops, dict):
                    ops = [ops]
                for opgrp in ops:
                    for r in (opgrp.get("calls") or []):
                        calls.append({
                            "expirationDate": exp,
                            "strike": r.get("strike") or r.get("strikePrice"),
                            "last": r.get("lastPrice") or r.get("last"),
                            "bid": r.get("bid"),
                            "ask": r.get("ask"),
                            "volume": r.get("volume"),
                            "openInterest": r.get("openInterest"),
                            "impliedVolatility": r.get("impliedVolatility"),
                            "lastPrice": r.get("lastPrice") or r.get("last"),
                        })
                    for r in (opgrp.get("puts") or []):
                        puts.append({
                            "expirationDate": exp,
                            "strike": r.get("strike") or r.get("strikePrice"),
                            "last": r.get("lastPrice") or r.get("last"),
                            "bid": r.get("bid"),
                            "ask": r.get("ask"),
                            "volume": r.get("volume"),
                            "openInterest": r.get("openInterest"),
                            "impliedVolatility": r.get("impliedVolatility"),
                            "lastPrice": r.get("lastPrice") or r.get("last"),
                        })
        else:
            soup = BeautifulSoup(text, "html.parser")
            tables = soup.find_all("table")
            for table in tables:
                ths = [th.get_text(strip=True).lower() for th in table.find_all("th")]
                trs = table.find_all("tr")[1:]
                for tr in trs:
                    tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if not tds:
                        continue
                    try:
                        strike = float(tds[0].replace(",", ""))
                    except Exception:
                        strike = None
                    last = tds[1] if len(tds) > 1 else None
                    bid = tds[2] if len(tds) > 2 else None
                    ask = tds[3] if len(tds) > 3 else None
                    vol = tds[4] if len(tds) > 4 else None
                    oi = tds[5] if len(tds) > 5 else None
                    header_text = " ".join(ths)
                    if "call" in header_text:
                        calls.append({"expirationDate": None, "strike": strike, "last": last, "bid": bid, "ask": ask, "volume": vol, "openInterest": oi})
                    else:
                        puts.append({"expirationDate": None, "strike": strike, "last": last, "bid": bid, "ask": ask, "volume": vol, "openInterest": oi})
        calls_sorted = sorted([c for c in calls if c.get("strike") is not None], key=lambda x: float(x["strike"])) + [c for c in calls if c.get("strike") is None]
        puts_sorted = sorted([p for p in puts if p.get("strike") is not None], key=lambda x: float(x["strike"])) + [p for p in puts if p.get("strike") is None]
        cache_set(_chain_cache, key, (calls_sorted, puts_sorted))
        return calls_sorted, puts_sorted
    except Exception as e:
        print("yahoo chain err:", e)
        cache_set(_chain_cache, key, ([], []))
        return [], []

@app.route("/", methods=["GET", "POST"])
def index():
    chart_form = {"symbol": "AAPL", "period": "1mo"}
    fx_form = {"option_type": "call", "model": "gk", "symbol": "EURUSD=X", "use_live_price": "yes",
               "S": "", "K": "", "rd": 0.05, "rf": 0.02, "sigma": 0.15, "T": 1.0, "steps": 200}
    eq_form = {"option_type": "call", "model": "bs", "symbol": "AAPL", "use_live_price": "yes",
               "S": "", "K": "", "r": 0.05, "q": 0.0, "sigma": 0.2, "T": 1.0, "steps": 200}
    fx_chain_form = {"root": "", "contract": "", "apikey": ""}
    eq_chain_form = {"symbol": "", "expiry": "", "expiries": []}

    active_tab = "chart"
    chart_data = None
    chart_error = None
    movers = get_movers()

    fx_result = None
    fx_chart = None
    fx_error = None

    eq_result = None
    eq_chart = None
    eq_error = None

    fx_chain_calls, fx_chain_puts = [], []
    fx_chain_error = None

    eq_chain_calls, eq_chain_puts = [], []
    eq_chain_error = None

    if request.method == "POST":
        tab_type = request.form.get("tab_type", "chart")
        if tab_type == "chart":
            active_tab = "chart"
            symbol = request.form.get("chart_symbol", chart_form["symbol"]).strip()
            period = request.form.get("chart_period", chart_form["period"]).strip()
            chart_form["symbol"] = symbol
            chart_form["period"] = period
            if not symbol:
                chart_error = "Enter a symbol"
            else:
                cdata = get_ohlc_for_symbol(symbol, period)
                if not cdata:
                    chart_error = f"Could not fetch OHLC for {symbol}"
                else:
                    chart_data = cdata
        elif tab_type == "fx":
            active_tab = "fx"
            fx_form["option_type"] = request.form.get("fx_option_type", fx_form["option_type"])
            fx_form["model"] = request.form.get("fx_model", fx_form["model"])
            fx_form["symbol"] = request.form.get("fx_symbol", fx_form["symbol"]).strip()
            fx_form["use_live_price"] = request.form.get("fx_use_live_price")
            fx_form["S"] = request.form.get("fx_S", fx_form["S"])
            fx_form["K"] = request.form.get("fx_K", fx_form["K"])
            fx_form["rd"] = float(request.form.get("fx_rd", fx_form["rd"]) or 0.0)
            fx_form["rf"] = float(request.form.get("fx_rf", fx_form["rf"]) or 0.0)
            fx_form["sigma"] = float(request.form.get("fx_sigma", fx_form["sigma"]) or 0.0)
            fx_form["T"] = float(request.form.get("fx_T", fx_form["T"]) or 0.0)
            fx_form["steps"] = int(request.form.get("fx_steps", fx_form["steps"]) or 200)
            S_used = None
            if fx_form["use_live_price"]:
                try:
                    t = yf.Ticker(fx_form["symbol"])
                    hist = t.history(period="5d")
                    if hist is not None and not hist.empty:
                        S_used = float(hist["Close"].iloc[-1])
                except Exception:
                    S_used = None
            if (not S_used) and fx_form["S"]:
                try:
                    S_used = float(fx_form["S"])
                except Exception:
                    S_used = None
            if S_used is None:
                fx_error = "Could not determine spot rate S — provide S or enable live price with a valid symbol."
            else:
                model = fx_form["model"]
                optype = fx_form["option_type"]
                K = float(fx_form["K"]) if fx_form["K"] else S_used
                rd = float(fx_form["rd"])
                rf = float(fx_form["rf"])
                sigma = float(fx_form["sigma"])
                T = float(fx_form["T"])
                steps = int(fx_form["steps"])
                price = None
                greeks = {}
                try:
                    if model == "gk":
                        price = garman_kohlhagen_price(S_used, K, rd, rf, sigma, T, option_type=optype)
                        greeks = gk_greeks(S_used, K, rd, rf, sigma, T)
                    elif model == "binomial_eur":
                        price = binomial_tree_price(S_used, K, rd, sigma, T, steps=steps, option_type=optype, american=False, q=rf)
                        greeks = gk_greeks(S_used, K, rd, rf, sigma, T)
                    elif model == "binomial_amer":
                        price = binomial_tree_price(S_used, K, rd, sigma, T, steps=steps, option_type=optype, american=True, q=rf)
                        greeks = gk_greeks(S_used, K, rd, rf, sigma, T)
                    fx_result = {
                        "model_label": {"gk": "Garman–Kohlhagen", "binomial_eur": "Binomial (European)", "binomial_amer": "Binomial (American)"}[model],
                        "price": float(price),
                        "S_used": float(S_used),
                        "greeks": greeks
                    }
                    s_range = np.linspace(S_used * 0.8, S_used * 1.2, 30).tolist()
                    price_arr = []
                    delta_arr = []
                    gamma_arr = []
                    for s in s_range:
                        if model == "gk":
                            p = garman_kohlhagen_price(s, K, rd, rf, sigma, T, option_type=optype)
                        else:
                            p = binomial_tree_price(s, K, rd, sigma, T, steps=steps, option_type=optype, american=(model=="binomial_amer"), q=rf)
                        price_arr.append(float(p))
                        h = max(1e-4, s * 1e-4)
                        if model == "gk":
                            p_plus = garman_kohlhagen_price(s + h, K, rd, rf, sigma, T, option_type=optype)
                            p_minus = garman_kohlhagen_price(s - h, K, rd, rf, sigma, T, option_type=optype)
                        else:
                            p_plus = binomial_tree_price(s + h, K, rd, sigma, T, steps=steps, option_type=optype, american=(model=="binomial_amer"), q=rf)
                            p_minus = binomial_tree_price(s - h, K, rd, sigma, T, steps=steps, option_type=optype, american=(model=="binomial_amer"), q=rf)
                        delta = (p_plus - p_minus) / (2 * h)
                        gamma = (p_plus - 2 * p + p_minus) / (h * h)
                        delta_arr.append(float(delta))
                        gamma_arr.append(float(gamma))
                    fx_chart = {"S": [round(float(v),6) for v in s_range], "price": price_arr, "delta": delta_arr, "gamma": gamma_arr}
                except Exception as e:
                    fx_error = f"Pricing error: {e}"
        elif tab_type == "equity":
            active_tab = "equity"
            eq_form["option_type"] = request.form.get("eq_option_type", eq_form["option_type"])
            eq_form["model"] = request.form.get("eq_model", eq_form["model"])
            eq_form["symbol"] = request.form.get("eq_symbol", eq_form["symbol"]).strip()
            eq_form["use_live_price"] = request.form.get("eq_use_live_price")
            eq_form["S"] = request.form.get("eq_S", eq_form["S"])
            eq_form["K"] = request.form.get("eq_K", eq_form["K"])
            eq_form["r"] = float(request.form.get("eq_r", eq_form["r"]) or 0.0)
            eq_form["q"] = float(request.form.get("eq_q", eq_form["q"]) or 0.0)
            eq_form["sigma"] = float(request.form.get("eq_sigma", eq_form["sigma"]) or 0.0)
            eq_form["T"] = float(request.form.get("eq_T", eq_form["T"]) or 0.0)
            eq_form["steps"] = int(request.form.get("eq_steps", eq_form["steps"]) or 200)
            S_used = None
            if eq_form["use_live_price"]:
                try:
                    t = yf.Ticker(eq_form["symbol"])
                    hist = t.history(period="5d")
                    if hist is not None and not hist.empty:
                        S_used = float(hist["Close"].iloc[-1])
                except Exception:
                    S_used = None
            if (not S_used) and eq_form["S"]:
                try:
                    S_used = float(eq_form["S"])
                except Exception:
                    S_used = None
            if S_used is None:
                eq_error = "Could not determine stock price S — provide S or enable live price with a valid symbol."
            else:
                model = eq_form["model"]
                optype = eq_form["option_type"]
                K = float(eq_form["K"]) if eq_form["K"] else S_used
                r = float(eq_form["r"])
                q = float(eq_form["q"])
                sigma = float(eq_form["sigma"])
                T = float(eq_form["T"])
                steps = int(eq_form["steps"])
                try:
                    if model == "bs":
                        price = black_scholes_price(S_used, K, r, sigma, T, option_type=optype, q=q)
                        g = bs_greeks(S_used, K, r, sigma, T, q=q)
                        greeks = {
                            "delta": g["delta_call"] if optype == "call" else g["delta_put"],
                            "gamma": g["gamma"],
                            "vega": g["vega"],
                            "theta": g["theta_call"] if optype == "call" else g["theta_put"],
                            "rho": g["rho_call"] if optype == "call" else g["rho_put"]
                        }
                    elif model == "binomial_eur":
                        price = binomial_tree_price(S_used, K, r, sigma, T, steps=steps, option_type=optype, american=False, q=q)
                        g = bs_greeks(S_used, K, r, sigma, T, q=q)
                        greeks = {"delta": g["delta_call"], "gamma": g["gamma"], "vega": g["vega"], "theta": g["theta_call"], "rho": g["rho_call"]}
                    elif model == "binomial_amer":
                        price = binomial_tree_price(S_used, K, r, sigma, T, steps=steps, option_type=optype, american=True, q=q)
                        g = bs_greeks(S_used, K, r, sigma, T, q=q)
                        greeks = {"delta": g["delta_call"], "gamma": g["gamma"], "vega": g["vega"], "theta": g["theta_call"], "rho": g["rho_call"]}
                    eq_result = {"model_label": {"bs":"Black–Scholes","binomial_eur":"Binomial (European)","binomial_amer":"Binomial (American)"}[model],
                                 "price": float(price), "S_used": float(S_used), "greeks": greeks}
                    s_range = np.linspace(S_used * 0.8, S_used * 1.2, 30).tolist()
                    price_arr = []
                    delta_arr = []
                    gamma_arr = []
                    for s in s_range:
                        if model == "bs":
                            p = black_scholes_price(s, K, r, sigma, T, option_type=optype, q=q)
                        else:
                            p = binomial_tree_price(s, K, r, sigma, T, steps=steps, option_type=optype, american=(model=="binomial_amer"), q=q)
                        price_arr.append(float(p))
                        h = max(1e-4, s * 1e-4)
                        if model == "bs":
                            p_plus = black_scholes_price(s + h, K, r, sigma, T, option_type=optype, q=q)
                            p_minus = black_scholes_price(s - h, K, r, sigma, T, option_type=optype, q=q)
                        else:
                            p_plus = binomial_tree_price(s + h, K, r, sigma, T, steps=steps, option_type=optype, american=(model=="binomial_amer"), q=q)
                            p_minus = binomial_tree_price(s - h, K, r, sigma, T, steps=steps, option_type=optype, american=(model=="binomial_amer"), q=q)
                        delta = (p_plus - p_minus) / (2 * h)
                        gamma = (p_plus - 2 * p + p_minus) / (h * h)
                        delta_arr.append(float(delta))
                        gamma_arr.append(float(gamma))
                    eq_chart = {"S": [round(float(v),6) for v in s_range], "price": price_arr, "delta": delta_arr, "gamma": gamma_arr}
                except Exception as e:
                    eq_error = f"Pricing error: {e}"
        elif tab_type == "fx_chain":
            active_tab = "fx_chain"
            fx_chain_form["root"] = request.form.get("fx_chain_root", "").strip()
            fx_chain_form["contract"] = request.form.get("fx_chain_contract", "").strip()
            symbol_for_yahoo = fx_chain_form["contract"] or fx_chain_form["root"]
            if not symbol_for_yahoo:
                fx_chain_error = "Enter contract or root to fetch from Yahoo."
            else:
                try:
                    calls, puts = get_yahoo_option_chain(symbol_for_yahoo)
                    fx_chain_calls, fx_chain_puts = calls, puts
                    if not calls and not puts:
                        fx_chain_error = "No options found on Yahoo for that symbol."
                except Exception as e:
                    fx_chain_error = str(e)
        elif tab_type == "equity_chain":
            active_tab = "equity_chain"
            eq_chain_form["symbol"] = request.form.get("eq_chain_symbol", "").strip()
            eq_chain_form["expiry"] = request.form.get("eq_chain_expiry", "").strip()
            try:
                if eq_chain_form["symbol"]:
                    t = yf.Ticker(eq_chain_form["symbol"])
                    expiries = list(t.options or [])
                    eq_chain_form["expiries"] = expiries
                    if eq_chain_form["expiry"] and eq_chain_form["expiry"] in expiries:
                        chain = t.option_chain(eq_chain_form["expiry"])
                        calls = chain.calls.to_dict("records")
                        puts = chain.puts.to_dict("records")
                        eq_chain_calls = calls
                        eq_chain_puts = puts
                    else:
                        if expiries:
                            chain = t.option_chain(expiries[0])
                            eq_chain_calls = chain.calls.to_dict("records")
                            eq_chain_puts = chain.puts.to_dict("records")
            except Exception as e:
                eq_chain_error = f"Error fetching option chain: {e}"

    return render_template("index.html",
        active_tab=active_tab,
        chart_form=chart_form, chart_data=chart_data, chart_error=chart_error,
        movers=movers,
        fx_form=fx_form, fx_result=fx_result, fx_chart=fx_chart, fx_error=fx_error,
        eq_form=eq_form, eq_result=eq_result, eq_chart=eq_chart, eq_error=eq_error,
        fx_chain_form=fx_chain_form, fx_chain_calls=fx_chain_calls, fx_chain_puts=fx_chain_puts, fx_chain_error=fx_chain_error,
        eq_chain_form=eq_chain_form, eq_chain_calls=eq_chain_calls, eq_chain_puts=eq_chain_puts, eq_chain_error=eq_chain_error
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

