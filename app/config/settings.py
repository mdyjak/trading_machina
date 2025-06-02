# 📁 app/config/settings.py
"""
Konfiguracja aplikacji - centralne ustawienia
"""

from dataclasses import dataclass
from typing import Dict, List
import os


@dataclass
class ExchangeConfig:
    """Konfiguracja giełdy"""
    name: str
    id: str
    config: Dict


@dataclass
class AppSettings:
    """Główne ustawienia aplikacji"""
    # GUI
    window_title: str = "HUSTLER 3.0"
    window_size: tuple = (1800, 1000)
    theme: str = "dark"

    # Dane
    default_symbol: str = "BTC/USDT"
    default_exchange: str = "binance"
    default_timeframe: str = "5m"
    default_candles: int = 200
    refresh_interval: int = 5

    # Wskaźniki
    indicators_enabled: bool = True
    max_indicators: int = 10

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True


# Konfiguracje giełd
EXCHANGES = {
    'Binance': ExchangeConfig(
        name='Binance',
        id='binance',
        config={'sandbox': False, 'enableRateLimit': True}
    ),
    'Bybit': ExchangeConfig(
        name='Bybit',
        id='bybit',
        config={'sandbox': False, 'enableRateLimit': True}
    ),
    'OKX': ExchangeConfig(
        name='OKX',
        id='okx',
        config={'sandbox': False, 'enableRateLimit': True}
    ),
    'Kraken': ExchangeConfig(
        name='Kraken',
        id='kraken',
        config={'enableRateLimit': True}
    ),
    'Coinbase': ExchangeConfig(
        name='Coinbase',
        id='coinbasepro',
        config={'enableRateLimit': True}
    ),
    'KuCoin': ExchangeConfig(
        name='KuCoin',
        id='kucoin',
        config={'sandbox': False, 'enableRateLimit': True}
    )
}

# Timeframes
TIMEFRAMES = {
    'M1': '1m', 'M5': '5m', 'M15': '15m', 'M30': '30m',
    'H1': '1h', 'H4': '4h', 'D1': '1d', 'W1': '1w'
}

# Popularne symbole
POPULAR_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT',
    'ADA/USDT', 'SOL/USDT', 'DOT/USDT', 'DOGE/USDT',
    'AVAX/USDT', 'MATIC/USDT', 'LTC/USDT', 'LINK/USDT'
]

# Kolory dla dark theme
COLORS = {
    'bg_primary': '#1a1a1a',
    'bg_secondary': '#0d1117',
    'text_primary': '#F1B95D',
    'text_secondary': '#cccccc',
    'accent_green': '#00ff88',
    'accent_red': '#ff4444',
    'accent_blue': '#4A90E2',
    'accent_pink': '#E24A90',
    'accent_gold': '#FFD700',
    'grid_color': '#404040'
}
