# 📁 app/data/data_fetcher.py
"""
Data Fetcher - pobieranie i zarządzanie danymi rynkowymi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging
from .exchange_manager import ExchangeManager
from ..config.settings import AppSettings

logger = logging.getLogger(__name__)


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


def calculate_price_change(current: float, previous: float) -> tuple:
    """Oblicza zmianę ceny (absolutną i procentową)"""
    if previous == 0:
        return 0.0, 0.0

    absolute_change = current - previous
    percentage_change = (absolute_change / previous) * 100

    return absolute_change, percentage_change


class DataFetcher:
    """
    Klasa do pobierania i zarządzania danymi rynkowymi
    """

    def __init__(self, exchange_manager: ExchangeManager):
        self.exchange_manager = exchange_manager
        self.current_data = pd.DataFrame()
        self.last_update = None
        self.market_stats = {}

    def fetch_market_data(self, symbol: str, timeframe: str,
                          limit: int = 200) -> Optional[pd.DataFrame]:
        """
        Pobiera dane rynkowe dla symbolu

        Args:
            symbol: Symbol (np. 'BTC/USDT')
            timeframe: Timeframe ('5m', '1h', etc.)
            limit: Liczba świec

        Returns:
            DataFrame z danymi OHLCV
        """
        try:
            # Pobierz dane z giełdy
            if self.exchange_manager.is_connected():
                ohlcv_data = self.exchange_manager.fetch_ohlcv(symbol, timeframe, limit)

                if ohlcv_data:
                    df = self._convert_ohlcv_to_dataframe(ohlcv_data)
                    self._update_market_stats(df, symbol)
                    self.current_data = df
                    self.last_update = datetime.now()
                    return df
                else:
                    logger.warning(f"No data received for {symbol}, using test data")
                    return self._generate_fallback_data(limit, timeframe)
            else:
                logger.info("Exchange not connected, using test data")
                return self._generate_fallback_data(limit, timeframe)

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return self._generate_fallback_data(limit, timeframe)

    def _convert_ohlcv_to_dataframe(self, ohlcv_data: list) -> pd.DataFrame:
        """Konwertuje dane OHLCV na DataFrame"""
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        return df

    def _generate_fallback_data(self, limit: int, timeframe: str) -> pd.DataFrame:
        """Generuje dane testowe jako fallback"""
        logger.info(f"Generating {limit} test candles for {timeframe}")
        return generate_test_ohlcv(limit)

    def _update_market_stats(self, df: pd.DataFrame, symbol: str):
        """Aktualizuje statystyki rynkowe"""
        if df.empty:
            return

        try:
            current_price = df['close'].iloc[-1]

            # Oblicz zmianę ceny
            if len(df) > 1:
                prev_price = df['close'].iloc[-2]
                price_change, price_change_pct = calculate_price_change(current_price, prev_price)
            else:
                price_change, price_change_pct = 0.0, 0.0

            # Pobierz volume 24h z tickera
            volume_24h = 0.0
            if self.exchange_manager.is_connected():
                ticker = self.exchange_manager.fetch_ticker(symbol)
                if ticker:
                    volume_24h = ticker.get('quoteVolume', 0) or ticker.get('baseVolume', 0)

            if volume_24h == 0:
                volume_24h = df['volume'].sum()  # Fallback

            # Zapisz statystyki
            self.market_stats = {
                'symbol': symbol,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_percent': price_change_pct,
                'volume_24h': volume_24h,
                'high_24h': df['high'].max(),
                'low_24h': df['low'].min(),
                'last_update': datetime.now()
            }

            logger.debug(f"Market stats updated for {symbol}: {current_price:.2f}")

        except Exception as e:
            logger.error(f"Error updating market stats: {e}")

    def get_market_stats(self) -> dict:
        """Zwraca aktualne statystyki rynkowe"""
        return self.market_stats.copy()

    def get_current_data(self) -> pd.DataFrame:
        """Zwraca aktualne dane"""
        return self.current_data.copy() if not self.current_data.empty else pd.DataFrame()

    def is_data_fresh(self, max_age_minutes: int = 5) -> bool:
        """Sprawdza czy dane są świeże"""
        if not self.last_update:
            return False

        age = datetime.now() - self.last_update
        return age.total_seconds() / 60 < max_age_minutes