# 📁 app/data/exchange_manager.py
"""
Manager giełd - obsługa połączeń z różnymi exchange'ami
"""

import ccxt
import logging
from typing import Dict, Optional, List
from ..config.settings import EXCHANGES

logger = logging.getLogger(__name__)


class ExchangeManager:
    """
    Manager do obsługi połączeń z giełdami kryptowalut
    """

    def __init__(self):
        self.current_exchange = None
        self.exchange_id = None
        self.available_markets = {}

    def connect_to_exchange(self, exchange_name: str) -> bool:
        """
        Łączy się z wybraną giełdą

        Args:
            exchange_name: Nazwa giełdy (np. 'Binance')

        Returns:
            True jeśli połączenie udane
        """
        try:
            if exchange_name not in EXCHANGES:
                logger.error(f"Unknown exchange: {exchange_name}")
                return False

            exchange_config = EXCHANGES[exchange_name]
            exchange_class = getattr(ccxt, exchange_config.id)

            self.current_exchange = exchange_class(exchange_config.config)
            self.exchange_id = exchange_config.id

            # Załaduj rynki
            self.current_exchange.load_markets()
            self.available_markets = self.current_exchange.markets

            logger.info(f"Connected to {exchange_name} ({len(self.available_markets)} markets)")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {exchange_name}: {e}")
            self.current_exchange = None
            self.exchange_id = None
            return False

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[List]:
        """
        Pobiera dane OHLCV

        Args:
            symbol: Symbol trading pair (np. 'BTC/USDT')
            timeframe: Timeframe ('5m', '1h', etc.)
            limit: Liczba świec

        Returns:
            Lista danych OHLCV lub None
        """
        if not self.current_exchange:
            logger.error("No exchange connected")
            return None

        try:
            # Specjalne przypadki dla niektórych giełd
            if self.exchange_id == 'kraken' and symbol == 'BTC/USDT':
                symbol = 'BTC/USD'

            ohlcv = self.current_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                logger.warning(f"No OHLCV data for {symbol}")
                return None

            logger.debug(f"Fetched {len(ohlcv)} candles for {symbol}")
            return ohlcv

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return None

    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Pobiera ticker (24h stats)"""
        if not self.current_exchange:
            return None

        try:
            ticker = self.current_exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

    def get_available_symbols(self, base_currency: str = None) -> List[str]:
        """
        Zwraca dostępne symbole na giełdzie

        Args:
            base_currency: Filtruj po walucie bazowej (np. 'BTC')

        Returns:
            Lista symboli
        """
        if not self.available_markets:
            return []

        symbols = list(self.available_markets.keys())

        if base_currency:
            symbols = [s for s in symbols if s.startswith(base_currency)]

        return sorted(symbols)

    def is_connected(self) -> bool:
        """Sprawdza czy połączenie jest aktywne"""
        return self.current_exchange is not None
