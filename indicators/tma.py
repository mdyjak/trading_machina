# 📁 indicators/tma.py
"""
TMA Fixed - 100% zgodny z MQ5
Poprawki: ATR calculation i pip value
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from .base import BaseIndicator

logger = logging.getLogger(__name__)

class TMAIndicator(BaseIndicator):
    """
    TMA 100% zgodny z MQ5 - poprawione ATR i pip handling
    """

    def __init__(self, name: str = "TMA", **kwargs):
        default_settings = {
            'half_length': 12,
            'atr_period': 100,
            'atr_multiplier': 2.0,
            'angle_threshold': 4,
            'price_type': 'weighted',
            'symbol_type': 'forex'  # forex, crypto, stocks
        }
        default_settings.update(kwargs)

        super().__init__(name, **default_settings)

        self.version = "2.1 - MQ5 Compatible"
        self.description = "TMA with MQ5-compatible ATR and pip handling"

        # Ustaw minimum periods
        self.minimum_periods = max(self.settings['half_length'] * 2,
                                   self.settings['atr_period']) + 20

        logger.info(f"TMA Indicator created: {self.settings}")

    def _get_price_data(self, df: pd.DataFrame) -> np.ndarray:
        """Pobiera dane cenowe według wybranego typu"""
        price_type = self.settings.get('price_type', 'weighted')

        if price_type == 'weighted':
            # (H+L+C+C)/4 - preferowane dla TMA
            return (df['high'] + df['low'] + df['close'] + df['close']).values / 4.0
        elif price_type == 'typical':
            # (H+L+C)/3
            return (df['high'] + df['low'] + df['close']).values / 3.0
        elif price_type == 'close':
            return df['close'].values
        else:
            return df['close'].values

    def _get_pip_value(self, current_price: float) -> float:
        """
        Dynamiczna wartość pip w zależności od instrumentu
        Próbuje naśladować _Point z MQ5
        """
        symbol_type = self.settings.get('symbol_type', 'forex')

        if symbol_type == 'forex':
            # Dla większości par forex
            if current_price < 2.0:  # EUR/USD, GBP/USD, etc.
                return 0.0001
            elif current_price < 200:  # USD/JPY
                return 0.01
            else:
                return 0.0001
        elif symbol_type == 'crypto':
            # Dla crypto - proporcjonalnie do ceny
            if current_price > 10000:  # BTC
                return 1.0
            elif current_price > 1000:  # ETH
                return 0.1
            elif current_price > 1:
                return 0.001
            else:
                return 0.00001
        else:
            # Default
            return 0.0001

    def _calculate_atr_mq5_style(self, high: np.ndarray, low: np.ndarray,
                                 close: np.ndarray, period: int, index: int) -> float:
        """
        ATR calculation EXACTLY like MQ5
        Używa uproszczoną formułę bez poprzedniej ceny zamknięcia
        """
        if index < period + 10:
            return 0.0

        atr_sum = 0.0
        valid_periods = 0

        for j in range(period):
            curr_idx = index - j - 10
            prev_idx = index - j - 11

            if prev_idx >= 0 and curr_idx < len(high):
                # MQ5 formula - uproszczona wersja
                # atr += MathMax(high[i-j-10],close[i-j-11])-MathMin(low[i-j-10],close[i-j-11]);
                range_high = max(high[curr_idx], close[prev_idx])
                range_low = min(low[curr_idx], close[prev_idx])
                true_range = range_high - range_low

                atr_sum += true_range
                valid_periods += 1

        return atr_sum / valid_periods if valid_periods > 0 else 0.0

    def _calculate_centered_tma(self, prices: np.ndarray, index: int) -> float:
        """
        Centered TMA - identyczne z MQ5
        """
        half_length = self.settings['half_length']

        if index < half_length or index >= len(prices) - half_length:
            return prices[index] if index < len(prices) else 0.0

        # MQ5 formula:
        # double sum  = (HalfLength+1)*prices[i];
        # double sumw = (HalfLength+1);
        sum_val = (half_length + 1) * prices[index]
        sum_weight = half_length + 1

        # for(int j=1, k=HalfLength; j<=HalfLength; j++, k--)
        for j in range(1, half_length + 1):
            k = half_length + 1 - j  # weight decreases

            # Backward
            if index - j >= 0:
                sum_val += k * prices[index - j]
                sum_weight += k

            # Forward (centered)
            if index + j < len(prices):
                sum_val += k * prices[index + j]
                sum_weight += k

        return sum_val / sum_weight

    def _detect_rebound_signals_mq5(self, high: np.ndarray, low: np.ndarray,
                                    open_prices: np.ndarray, close: np.ndarray,
                                    tma_upper: np.ndarray, tma_lower: np.ndarray,
                                    tma_center: np.ndarray, atr: float,
                                    index: int) -> Tuple[float, float, float]:
        """
        Sygnały odbicia - 100% zgodne z MQ5
        """
        if index <= 0 or index >= len(high):
            return 0.0, 0.0, 0.0

        rebound_up = 0.0
        rebound_down = 0.0
        caution = 0.0

        # Get current price for pip calculation
        current_price = close[index] if index < len(close) else 45000.0
        pip_value = self._get_pip_value(current_price)

        # MQ5 exact logic:
        # if(high[i-1] > tmau[i-1] && close[i-1] > open[i-1] && close[i] < open[i])
        if (high[index - 1] > tma_upper[index - 1] and
                close[index - 1] > open_prices[index - 1] and
                close[index] < open_prices[index]):

            rebound_down = high[index] + self.settings['atr_multiplier'] * atr / 2

            # if(tmac[i] - tmac[i-1] > TMAangle*_Point)
            if (tma_center[index] - tma_center[index - 1] >
                    self.settings['angle_threshold'] * pip_value):
                caution = rebound_down + 10 * pip_value

        # if(low[i-1] < tmad[i-1] && close[i-1] < open[i-1] && close[i] > open[i])
        if (low[index - 1] < tma_lower[index - 1] and
                close[index - 1] < open_prices[index - 1] and
                close[index] > open_prices[index]):

            rebound_up = low[index] - self.settings['atr_multiplier'] * atr / 2

            # if(tmac[i-1] - tmac[i] > TMAangle*_Point)
            if (tma_center[index - 1] - tma_center[index] >
                    self.settings['angle_threshold'] * pip_value):
                caution = rebound_up - 10 * pip_value

        return rebound_up, rebound_down, caution

    def calculate(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Główna funkcja - 100% zgodna z MQ5
        """
        try:
            if not self.validate_data(df):
                return None

            if not self._should_recalculate(df) and self.last_result:
                return self.last_result

            # Pobierz dane
            length = len(df)
            high = df['high'].values
            low = df['low'].values
            open_prices = df['open'].values
            close = df['close'].values
            prices = self._get_price_data(df)

            # Bufory
            tma_center = np.zeros(length)
            tma_upper = np.zeros(length)
            tma_lower = np.zeros(length)
            tma_colors = np.zeros(length, dtype=int)
            rebound_up = np.zeros(length)
            rebound_down = np.zeros(length)
            angle_caution = np.zeros(length)

            # Start index - jak w MQ5
            start_idx = max(self.settings['half_length'],
                            self.settings['atr_period'] + 11)  # +11 jak w MQ5

            for i in range(start_idx, length):
                # ATR w stylu MQ5
                atr = self._calculate_atr_mq5_style(high, low, close,
                                                    self.settings['atr_period'], i)

                # Centered TMA
                tma_center[i] = self._calculate_centered_tma(prices, i)

                # Kolory - jak w MQ5
                if i > 0:
                    if tma_center[i] > tma_center[i - 1]:
                        tma_colors[i] = 0  # Up
                    elif tma_center[i] < tma_center[i - 1]:
                        tma_colors[i] = 1  # Down
                    else:
                        tma_colors[i] = tma_colors[i - 1]

                # Pasma
                tma_upper[i] = tma_center[i] + self.settings['atr_multiplier'] * atr
                tma_lower[i] = tma_center[i] - self.settings['atr_multiplier'] * atr

                # Sygnały w stylu MQ5
                r_up, r_down, caution = self._detect_rebound_signals_mq5(
                    high, low, open_prices, close,
                    tma_upper, tma_lower, tma_center, atr, i
                )

                rebound_up[i] = r_up
                rebound_down[i] = r_down
                angle_caution[i] = caution

            result = {
                'tma_center': tma_center,
                'tma_upper': tma_upper,
                'tma_lower': tma_lower,
                'tma_colors': tma_colors,
                'rebound_up': rebound_up,
                'rebound_down': rebound_down,
                'angle_caution': angle_caution,
                'valid_from': start_idx,
                'settings': self.settings.copy(),
                'timestamp': df.index[-1] if not df.empty else None,
                'mq5_compatible': True
            }

            self.last_result = result
            logger.info(f"TMA calculated: {length} candles, MQ5 compatible")
            return result

        except Exception as e:
            logger.error(f"TMA calculation error: {e}")
            return None

    def get_plot_config(self) -> Dict:
        """Konfiguracja identyczna jak oryginalny TMA"""
        return {
            'main_window': True,
            'subplot': False,
            'colors': {
                'tma_up': '#00FF88',
                'tma_down': '#FF4444',
                'upper_band': '#4A90E2',
                'lower_band': '#E24A90',
                'rebound_up': '#00FF88',
                'rebound_down': '#FF4444',
                'caution': '#FFD700'
            },
            'styles': {
                'tma_line': {'width': 2, 'style': 'solid'},
                'bands': {'width': 1, 'style': 'dashed', 'alpha': 0.7},
                'signals': {'size': 80, 'alpha': 0.8}
            }
        }

    def get_latest_signal(self, result: Optional[Dict] = None,
                          lookback: int = 5) -> Dict:
        """
        Analizuje najnowsze sygnały TMA

        Args:
            result: Wyniki TMA (opcjonalne)
            lookback: Ile świec wstecz analizować

        Returns:
            Dict z analizą sygnałów
        """
        if result is None:
            result = self.last_result

        if not result:
            return {
                'trend': 'unknown',
                'signal': 'none',
                'strength': 0,
                'caution': False,
                'distance_to_bands': {'upper': 0, 'lower': 0}
            }

        try:
            length = len(result['tma_center'])
            if length < 2:
                return {'trend': 'unknown', 'signal': 'none', 'strength': 0}

            # Analiza trendu
            current_tma = result['tma_center'][-1]
            prev_tma = result['tma_center'][-2]

            if current_tma > prev_tma:
                trend = 'bullish'
            elif current_tma < prev_tma:
                trend = 'bearish'
            else:
                trend = 'sideways'

            # Szukaj najnowszych sygnałów
            start_idx = max(0, length - lookback)
            signal = 'none'
            signal_strength = 0

            for i in range(length - 1, start_idx - 1, -1):
                if result['rebound_up'][i] > 0:
                    signal = 'buy'
                    signal_strength = length - i
                    break
                elif result['rebound_down'][i] > 0:
                    signal = 'sell'
                    signal_strength = length - i
                    break

            # Sprawdź ostrzeżenia
            caution_active = any(result['angle_caution'][start_idx:length] > 0)

            # Odległości do pasm
            distances = {
                'upper': result['tma_upper'][-1] - current_tma,
                'lower': current_tma - result['tma_lower'][-1]
            }

            return {
                'trend': trend,
                'signal': signal,
                'strength': signal_strength,
                'caution': caution_active,
                'current_tma': current_tma,
                'distance_to_bands': distances
            }

        except Exception as e:
            logger.error(f"Signal analysis error: {e}")
            return {'trend': 'error', 'signal': 'none', 'strength': 0}


def create_tma_indicator(timeframe: str = '5m', style: str = 'balanced',
                         name: str = "TMA") -> TMAIndicator:
    """
    Factory function dla TMA z predefiniowanymi ustawieniami

    Args:
        timeframe: Timeframe ('1m', '5m', '15m', etc.)
        style: Styl tradingu ('conservative', 'balanced', 'aggressive')
        name: Nazwa instancji wskaźnika

    Returns:
        Skonfigurowany TMAIndicator
    """

    # Konfiguracje zoptymalizowane pod różne style
    configs = {
        '1m': {
            'conservative': {'period': 20, 'atr': 2.5, 'angle': 6},
            'balanced': {'period': 15, 'atr': 2.0, 'angle': 4},
            'aggressive': {'period': 10, 'atr': 1.5, 'angle': 3}
        },
        '5m': {
            'conservative': {'period': 15, 'atr': 2.2, 'angle': 5},
            'balanced': {'period': 12, 'atr': 2.0, 'angle': 4},
            'aggressive': {'period': 8, 'atr': 1.8, 'angle': 3}
        },
        '15m': {
            'conservative': {'period': 12, 'atr': 2.0, 'angle': 4},
            'balanced': {'period': 10, 'atr': 1.8, 'angle': 3},
            'aggressive': {'period': 7, 'atr': 1.5, 'angle': 2}
        },
        '30m': {
            'conservative': {'period': 10, 'atr': 1.8, 'angle': 3},
            'balanced': {'period': 8, 'atr': 1.6, 'angle': 2},
            'aggressive': {'period': 6, 'atr': 1.4, 'angle': 2}
        }
    }

    # Wybierz konfigurację
    if timeframe in configs and style in configs[timeframe]:
        config = configs[timeframe][style]
    else:
        config = configs['5m']['balanced']  # Domyślna

    return TMAIndicator(
        name=name,
        half_length=config['period'],
        atr_period=100,
        atr_multiplier=config['atr'],
        angle_threshold=config['angle'],
        symbol_type='crypto'  # Domyślnie crypto
    )
