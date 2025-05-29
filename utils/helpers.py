# 📁 utils/helpers.py
"""
Funkcje pomocnicze używane w całej aplikacji
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional


def format_number(value: float, precision: int = 2) -> str:
    """Formatuje liczbę z odpowiednią precyzją"""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:.{precision}f}"


def format_percentage(value: float, precision: int = 2) -> str:
    """Formatuje procent"""
    return f"{value:+.{precision}f}%"


def validate_timeframe(timeframe: str) -> bool:
    """Sprawdza czy timeframe jest poprawny"""
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    return timeframe in valid_timeframes


def calculate_price_change(current: float, previous: float) -> tuple:
    """Oblicza zmianę ceny (absolutną i procentową)"""
    if previous == 0:
        return 0.0, 0.0

    absolute_change = current - previous
    percentage_change = (absolute_change / previous) * 100

    return absolute_change, percentage_change


def generate_test_ohlcv(periods: int = 200, base_price: float = 45000.0) -> pd.DataFrame:
    """Generuje testowe dane OHLCV"""
    np.random.seed(42)

    # Random walk dla cen
    returns = np.random.randn(periods) * 0.015
    prices = base_price * np.exp(np.cumsum(returns))

    data = []
    for i in range(periods):
        open_price = prices[i - 1] if i > 0 else base_price
        close_price = prices[i]

        # Realistyczne high/low
        high_mult = 1 + abs(np.random.randn()) * 0.008
        low_mult = 1 - abs(np.random.randn()) * 0.008

        high_price = max(open_price, close_price) * high_mult
        low_price = min(open_price, close_price) * low_mult
        volume = np.random.uniform(50, 500)

        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

    dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
    return pd.DataFrame(data, index=dates)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Bezpieczne dzielenie z domyślną wartością"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default