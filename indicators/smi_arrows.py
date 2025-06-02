# 📁 indicators/smi_arrows.py
"""
SMI Arrows Indicator - Professional Implementation
Stochastic Momentum Index z sygnałami strzałek (kupno/sprzedaż) według Williama Blau

Zgodny z MT5 SMI-Arrows - analiza momentum i identyfikacja punktów zwrotnych
Inspirowany: https://tradingfinder.com/products/indicators/mt5/stochastic-momentum-with-arrows-free-download/
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from .base import BaseIndicator

logger = logging.getLogger(__name__)


class SMIArrowsIndicator(BaseIndicator):
    """
    SMI Arrows - Professional Trading Signals

    Funkcje:
    - SMI calculation (Stochastic Momentum Index) według Williama Blau
    - Overbought/Oversold detection (-40/+40)
    - Buy/Sell arrows na crossoverach i ekstremalnych poziomach
    - Divergence detection
    - Triple smoothing (EMA) dla precyzyjnych sygnałów
    - Kompatybilność z MT5 logic
    """

    def __init__(self, name: str = "SMI_Arrows", **kwargs):
        default_settings = {
            'smi_period': 10,  # Okres SMI (jak %K w Stochastic)
            'first_smoothing': 3,  # Pierwsza EMA
            'second_smoothing': 3,  # Druga EMA (double smoothing)
            'signal_smoothing': 3,  # Signal line smoothing
            'overbought_level': 40,  # Poziom wykupienia
            'oversold_level': -40,  # Poziom wyprzedania
            'extreme_level': 60,  # Poziom ekstremalny
            'arrow_sensitivity': 'medium',  # low, medium, high
            'use_divergence': True,  # Wykrywanie dywergencji
            'min_bars_between_signals': 3,  # Min. odstęp między sygnałami
            'price_type': 'close'  # close, typical, weighted
        }
        default_settings.update(kwargs)

        super().__init__(name, **default_settings)

        self.version = "1.0 - Professional SMI Arrows (William Blau)"
        self.description = "SMI with professional buy/sell signals and divergence detection"

        # Ustaw minimum periods - SMI potrzebuje więcej danych przez triple smoothing
        self.minimum_periods = max(
            self.settings['smi_period'] * 2 +
            self.settings['first_smoothing'] +
            self.settings['second_smoothing'] +
            self.settings['signal_smoothing'],
            60
        )

        # Sensitivity settings
        self._setup_sensitivity()

        logger.info(f"SMI Arrows Indicator created: {self.settings}")

    def _setup_sensitivity(self):
        """Konfiguruje czułość sygnałów"""
        sensitivity = self.settings.get('arrow_sensitivity', 'medium')

        if sensitivity == 'low':
            self.signal_threshold = 0.8
            self.divergence_threshold = 0.9
            self.trend_strength_min = 0.7
        elif sensitivity == 'high':
            self.signal_threshold = 0.2
            self.divergence_threshold = 0.3
            self.trend_strength_min = 0.2
        else:  # medium
            self.signal_threshold = 0.5
            self.divergence_threshold = 0.6
            self.trend_strength_min = 0.4

    def _get_price_data(self, df: pd.DataFrame) -> np.ndarray:
        """Pobiera dane cenowe według wybranego typu"""
        price_type = self.settings.get('price_type', 'close')

        if price_type == 'typical':
            # Typical Price (H+L+C)/3
            return (df['high'] + df['low'] + df['close']).values / 3.0
        elif price_type == 'weighted':
            # Weighted Price (H+L+C+C)/4
            return (df['high'] + df['low'] + df['close'] + df['close']).values / 4.0
        elif price_type == 'close':
            return df['close'].values
        else:
            return df['close'].values

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Oblicza EMA (Exponential Moving Average)
        Używane do triple smoothing w SMI
        """
        length = len(data)
        ema = np.zeros(length)

        if length < period:
            return ema

        # Smoothing factor
        alpha = 2.0 / (period + 1.0)

        # Pierwszy punkt - SMA
        ema[period - 1] = np.mean(data[:period])

        # Kolejne punkty - EMA
        for i in range(period, length):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _calculate_smi_william_blau(self, high: np.ndarray, low: np.ndarray,
                                    close: np.ndarray, period: int,
                                    smooth1: int, smooth2: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oblicza SMI według oryginalnej formuły Williama Blau

        SMI Formula (William Blau):
        1. M = (HighMAX + LowMIN) / 2  (midpoint)
        2. D = Close - M  (distance from midpoint)
        3. Ds = EMA(EMA(D))  (double smoothed distance)
        4. Dhl = EMA(EMA(HighMAX - LowMIN))  (double smoothed range)
        5. SMI = 100 * (Ds / Dhl)
        """
        length = len(close)
        smi_values = np.zeros(length)

        # Bufory dla obliczeń
        distance_from_midpoint = np.zeros(length)
        high_low_range = np.zeros(length)

        for i in range(period - 1, length):
            # 1. Znajdź highest high i lowest low w okresie
            window_start = max(0, i - period + 1)
            window_high = high[window_start:i + 1]
            window_low = low[window_start:i + 1]

            high_max = np.max(window_high)
            low_min = np.min(window_low)

            # 2. Midpoint (M)
            midpoint = (high_max + low_min) / 2.0

            # 3. Distance from midpoint (D)
            distance_from_midpoint[i] = close[i] - midpoint

            # 4. High-Low range
            high_low_range[i] = high_max - low_min

        # 5. Double smoothing (EMA of EMA)
        # First smoothing
        distance_smooth1 = self._calculate_ema(distance_from_midpoint, smooth1)
        range_smooth1 = self._calculate_ema(high_low_range, smooth1)

        # Second smoothing
        distance_smooth2 = self._calculate_ema(distance_smooth1, smooth2)
        range_smooth2 = self._calculate_ema(range_smooth1, smooth2)

        # 6. Calculate SMI
        for i in range(length):
            if range_smooth2[i] != 0:
                smi_values[i] = 100.0 * (distance_smooth2[i] / range_smooth2[i])
            else:
                smi_values[i] = 0.0

        # 7. Signal line (additional EMA smoothing)
        signal_line = self._calculate_ema(smi_values, self.settings['signal_smoothing'])

        return smi_values, signal_line

    def _detect_smi_signals(self, smi: np.ndarray, signal_line: np.ndarray,
                            prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wykrywa sygnały kupna i sprzedaży na podstawie SMI

        Sygnały bazują na:
        1. Crossover SMI vs Signal Line w strefach ekstremalnych
        2. Wyjście z overbought/oversold
        3. Zero line cross z momentum
        """
        length = len(smi)
        buy_signals = np.zeros(length)
        sell_signals = np.zeros(length)

        overbought = self.settings['overbought_level']
        oversold = self.settings['oversold_level']
        min_bars = self.settings['min_bars_between_signals']

        last_signal_bar = -min_bars - 1

        for i in range(3, length):  # Start from 3 for lookback
            current_bar = i

            # Sprawdź czy minął wystarczający czas od ostatniego sygnału
            if current_bar - last_signal_bar < min_bars:
                continue

            # === BULLISH SIGNALS ===

            # Signal 1: SMI crosses above signal line from oversold
            if (smi[i - 1] <= signal_line[i - 1] and
                    smi[i] > signal_line[i] and
                    smi[i - 1] < oversold and smi[i] >= oversold):

                if self._validate_buy_signal(smi, signal_line, prices, i):
                    buy_signals[i] = prices[i]
                    last_signal_bar = current_bar
                    continue

            # Signal 2: SMI exits oversold with momentum
            if (smi[i - 2] < oversold and smi[i - 1] < oversold and
                    smi[i] > oversold and smi[i] > smi[i - 1]):

                if self._validate_buy_signal(smi, signal_line, prices, i):
                    buy_signals[i] = prices[i]
                    last_signal_bar = current_bar
                    continue

            # === BEARISH SIGNALS ===

            # Signal 1: SMI crosses below signal line from overbought
            if (smi[i - 1] >= signal_line[i - 1] and
                    smi[i] < signal_line[i] and
                    smi[i - 1] > overbought and smi[i] <= overbought):

                if self._validate_sell_signal(smi, signal_line, prices, i):
                    sell_signals[i] = prices[i]
                    last_signal_bar = current_bar
                    continue

            # Signal 2: SMI exits overbought with momentum
            if (smi[i - 2] > overbought and smi[i - 1] > overbought and
                    smi[i] < overbought and smi[i] < smi[i - 1]):

                if self._validate_sell_signal(smi, signal_line, prices, i):
                    sell_signals[i] = prices[i]
                    last_signal_bar = current_bar

        return buy_signals, sell_signals

    def _validate_buy_signal(self, smi: np.ndarray, signal_line: np.ndarray,
                             prices: np.ndarray, index: int) -> bool:
        """Waliduje sygnał kupna - dodatkowe filtry"""
        try:
            # 1. Sprawdź momentum SMI
            if index < 2:
                return True

            smi_momentum = smi[index] - smi[index - 2]
            if smi_momentum <= 0:
                return False

            # 2. Sprawdź czy nie jesteśmy w bardzo silnym downtrend
            if index < 5:
                return True

            recent_prices = prices[index - 4:index + 1]
            price_trend = np.polyfit(range(5), recent_prices, 1)[0]

            # Jeśli trend cenowy jest bardzo negatywny, bądź ostrożny
            if price_trend < -0.7 * np.std(recent_prices):
                return False

            # 3. SMI powinno mieć przestrzeń do wzrostu
            if smi[index] > self.settings['overbought_level'] * 0.8:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating buy signal: {e}")
            return True

    def _validate_sell_signal(self, smi: np.ndarray, signal_line: np.ndarray,
                              prices: np.ndarray, index: int) -> bool:
        """Waliduje sygnał sprzedaży - dodatkowe filtry"""
        try:
            # 1. Sprawdź momentum SMI
            if index < 2:
                return True

            smi_momentum = smi[index] - smi[index - 2]
            if smi_momentum >= 0:
                return False

            # 2. Sprawdź czy nie jesteśmy w bardzo silnym uptrend
            if index < 5:
                return True

            recent_prices = prices[index - 4:index + 1]
            price_trend = np.polyfit(range(5), recent_prices, 1)[0]

            # Jeśli trend cenowy jest bardzo pozytywny, bądź ostrożny
            if price_trend > 0.7 * np.std(recent_prices):
                return False

            # 3. SMI powinno mieć przestrzeń do spadku
            if smi[index] < self.settings['oversold_level'] * 0.8:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating sell signal: {e}")
            return True

    def _detect_divergences(self, smi: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wykrywa dywergencje między SMI a ceną

        Returns:
            Tuple[bullish_divergence, bearish_divergence]
        """
        if not self.settings.get('use_divergence', True):
            return np.zeros(len(smi)), np.zeros(len(smi))

        length = len(smi)
        bullish_div = np.zeros(length)
        bearish_div = np.zeros(length)

        # Minimalny odstęp dla analizy dywergencji
        min_bars = 12

        for i in range(min_bars, length - 5):
            # Bullish divergence: cena robi niższe dołki, SMI wyższe dołki
            if self._check_bullish_divergence(prices, smi, i, min_bars):
                bullish_div[i] = prices[i]

            # Bearish divergence: cena robi wyższe szczyty, SMI niższe szczyty
            if self._check_bearish_divergence(prices, smi, i, min_bars):
                bearish_div[i] = prices[i]

        return bullish_div, bearish_div

    def _check_bullish_divergence(self, prices: np.ndarray, smi: np.ndarray,
                                  current_idx: int, lookback: int) -> bool:
        """Sprawdza bullish divergence"""
        try:
            # Znajdź lokalne minima w cenach i SMI
            price_window = prices[current_idx - lookback:current_idx + 1]
            smi_window = smi[current_idx - lookback:current_idx + 1]

            # Znajdź ostatnie dwa lokalne minima
            price_mins = []
            smi_mins = []

            for i in range(3, len(price_window) - 3):
                if (price_window[i] < price_window[i - 1] and
                        price_window[i] < price_window[i + 1] and
                        price_window[i] < price_window[i - 2] and
                        price_window[i] < price_window[i + 2] and
                        price_window[i] < price_window[i - 3] and
                        price_window[i] < price_window[i + 3]):
                    price_mins.append((i, price_window[i]))
                    smi_mins.append((i, smi_window[i]))

            if len(price_mins) < 2:
                return False

            # Porównaj ostatnie dwa minima
            last_price_min = price_mins[-1][1]
            prev_price_min = price_mins[-2][1]
            last_smi_min = smi_mins[-1][1]
            prev_smi_min = smi_mins[-2][1]

            # Bullish divergence: cena niżej, SMI wyżej
            if (last_price_min < prev_price_min and last_smi_min > prev_smi_min):
                # Dodatkowo: SMI powinno być w oversold lub blisko
                if last_smi_min < self.settings['oversold_level'] * 1.5:
                    return True

            return False

        except Exception:
            return False

    def _check_bearish_divergence(self, prices: np.ndarray, smi: np.ndarray,
                                  current_idx: int, lookback: int) -> bool:
        """Sprawdza bearish divergence"""
        try:
            # Znajdź lokalne maxima w cenach i SMI
            price_window = prices[current_idx - lookback:current_idx + 1]
            smi_window = smi[current_idx - lookback:current_idx + 1]

            # Znajdź ostatnie dwa lokalne maxima
            price_maxs = []
            smi_maxs = []

            for i in range(3, len(price_window) - 3):
                if (price_window[i] > price_window[i - 1] and
                        price_window[i] > price_window[i + 1] and
                        price_window[i] > price_window[i - 2] and
                        price_window[i] > price_window[i + 2] and
                        price_window[i] > price_window[i - 3] and
                        price_window[i] > price_window[i + 3]):
                    price_maxs.append((i, price_window[i]))
                    smi_maxs.append((i, smi_window[i]))

            if len(price_maxs) < 2:
                return False

            # Porównaj ostatnie dwa maxima
            last_price_max = price_maxs[-1][1]
            prev_price_max = price_maxs[-2][1]
            last_smi_max = smi_maxs[-1][1]
            prev_smi_max = smi_maxs[-2][1]

            # Bearish divergence: cena wyżej, SMI niżej
            if (last_price_max > prev_price_max and last_smi_max < prev_smi_max):
                # Dodatkowo: SMI powinno być w overbought lub blisko
                if last_smi_max > self.settings['overbought_level'] * 0.7:
                    return True

            return False

        except Exception:
            return False

    def _calculate_trend_strength(self, smi: np.ndarray, signal_line: np.ndarray,
                                  index: int) -> float:
        """Oblicza siłę trendu na podstawie SMI"""
        if index < 10:
            return 0.5

        try:
            # Analiza ostatnich 10 wartości SMI
            recent_smi = smi[index - 9:index + 1]
            recent_signal = signal_line[index - 9:index + 1]

            # 1. Sprawdź konsystencję kierunku
            smi_above_signal = np.sum(recent_smi > recent_signal)
            direction_consistency = max(smi_above_signal, 10 - smi_above_signal) / 10.0

            # 2. Sprawdź volatility/stabilność
            smi_std = np.std(recent_smi)
            volatility_factor = min(1.0, smi_std / 30.0)  # Normalizacja

            # 3. Sprawdź momentum
            smi_momentum = recent_smi[-1] - recent_smi[0]
            momentum_factor = min(1.0, abs(smi_momentum) / 50.0)

            # Kombinacja czynników
            trend_strength = (direction_consistency * 0.5 +
                              momentum_factor * 0.3 +
                              (1 - volatility_factor) * 0.2)

            return max(0.0, min(1.0, trend_strength))

        except Exception:
            return 0.5

    def calculate(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Główna funkcja obliczeniowa SMI Arrows
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
            close = df['close'].values
            prices = self._get_price_data(df)

            # Settings
            period = self.settings['smi_period']
            smooth1 = self.settings['first_smoothing']
            smooth2 = self.settings['second_smoothing']

            # Oblicz SMI według formuły Williama Blau
            smi_values, signal_line = self._calculate_smi_william_blau(
                high, low, close, period, smooth1, smooth2
            )

            # Wykryj sygnały kupna/sprzedaży
            buy_signals, sell_signals = self._detect_smi_signals(
                smi_values, signal_line, prices
            )

            # Wykryj dywergencje
            bullish_div, bearish_div = self._detect_divergences(smi_values, prices)

            # Oblicz poziomy SMI
            overbought_line = np.full(length, self.settings['overbought_level'])
            oversold_line = np.full(length, self.settings['oversold_level'])
            zero_line = np.zeros(length)

            # Oblicz siłę trendu dla każdego punktu
            trend_strength = np.array([
                self._calculate_trend_strength(smi_values, signal_line, i)
                for i in range(length)
            ])

            # Przygotuj wynik
            result = {
                'smi': smi_values,
                'signal_line': signal_line,
                'buy_arrows': buy_signals,
                'sell_arrows': sell_signals,
                'bullish_divergence': bullish_div,
                'bearish_divergence': bearish_div,
                'overbought_line': overbought_line,
                'oversold_line': oversold_line,
                'zero_line': zero_line,
                'trend_strength': trend_strength,
                'valid_from': max(period + smooth1 + smooth2, 20),
                'settings': self.settings.copy(),
                'timestamp': df.index[-1] if not df.empty else None,
                'levels': {
                    'overbought': self.settings['overbought_level'],
                    'oversold': self.settings['oversold_level'],
                    'extreme': self.settings['extreme_level']
                },
                'william_blau_compatible': True
            }

            self.last_result = result
            logger.info(f"SMI Arrows calculated: {length} candles, "
                        f"{np.sum(buy_signals > 0)} buy signals, "
                        f"{np.sum(sell_signals > 0)} sell signals")

            return result

        except Exception as e:
            logger.error(f"SMI Arrows calculation error: {e}")
            return None

    def get_plot_config(self) -> Dict:
        """Konfiguracja wyświetlania SMI Arrows"""
        return {
            'main_window': False,
            'subplot': True,
            'subplot_height_ratio': 0.35,
            'colors': {
                'smi_line': '#FFD700',  # Gold
                'signal_line': '#87CEEB',  # Sky Blue
                'buy_arrow': '#00FF88',
                'sell_arrow': '#FF4444',
                'bullish_divergence': '#4A90E2',
                'bearish_divergence': '#E24A90',
                'overbought_line': '#FF6B6B',
                'oversold_line': '#4ECDC4',
                'zero_line': '#999999'
            },
            'styles': {
                'smi_line': {'width': 2.5, 'style': 'solid'},
                'signal_line': {'width': 1.5, 'style': 'dashed', 'alpha': 0.8},
                'level_lines': {'width': 1, 'style': 'dashed', 'alpha': 0.7},
                'arrows': {'size': 120, 'alpha': 0.9},
                'divergence': {'size': 80, 'alpha': 0.8}
            },
            'levels': {
                'overbought': self.settings['overbought_level'],
                'oversold': self.settings['oversold_level'],
                'zero': 0
            }
        }

    def get_latest_signal(self, result: Optional[Dict] = None, lookback: int = 10) -> Dict:
        """
        Analizuje najnowsze sygnały SMI Arrows

        Args:
            result: Wyniki SMI (opcjonalne)
            lookback: Ile świec wstecz analizować

        Returns:
            Dict z analizą sygnałów
        """
        if result is None:
            result = self.last_result

        if not result:
            return {
                'signal': 'none',
                'strength': 0,
                'smi_value': 0,
                'signal_value': 0,
                'trend': 'neutral',
                'divergence': False
            }

        try:
            length = len(result['smi'])
            if length < 2:
                return {'signal': 'none', 'strength': 0}

            # Sprawdź najnowsze sygnały
            start_idx = max(0, length - lookback)
            signal = 'none'
            signal_strength = 0
            signal_age = 0

            # Szukaj ostatnich sygnałów
            for i in range(length - 1, start_idx - 1, -1):
                if result['buy_arrows'][i] > 0:
                    signal = 'buy'
                    signal_age = length - 1 - i
                    signal_strength = result['trend_strength'][i]
                    break
                elif result['sell_arrows'][i] > 0:
                    signal = 'sell'
                    signal_age = length - 1 - i
                    signal_strength = result['trend_strength'][i]
                    break

            # Analiza aktualnego stanu SMI
            current_smi = result['smi'][-1]
            current_signal = result['signal_line'][-1]
            prev_smi = result['smi'][-2] if length > 1 else current_smi

            # Określ trend
            if current_smi > self.settings['overbought_level']:
                trend = 'overbought'
            elif current_smi < self.settings['oversold_level']:
                trend = 'oversold'
            elif current_smi > current_signal:
                trend = 'bullish'
            elif current_smi < current_signal:
                trend = 'bearish'
            else:
                trend = 'neutral'

            # Sprawdź dywergencje
            divergence_detected = (
                    result['bullish_divergence'][start_idx:].any() or
                    result['bearish_divergence'][start_idx:].any()
            )

            return {
                'signal': signal,
                'strength': signal_strength,
                'signal_age': signal_age,
                'smi_value': current_smi,
                'signal_value': current_signal,
                'smi_change': current_smi - prev_smi,
                'trend': trend,
                'divergence': divergence_detected,
                'crossover_status': 'above' if current_smi > current_signal else 'below',
                'overbought': current_smi > self.settings['overbought_level'],
                'oversold': current_smi < self.settings['oversold_level']
            }

        except Exception as e:
            logger.error(f"SMI signal analysis error: {e}")
            return {'signal': 'error', 'strength': 0}


def create_smi_arrows_indicator(timeframe: str = '5m', sensitivity: str = 'medium',
                                name: str = "SMI_Arrows") -> SMIArrowsIndicator:
    """
    Factory function dla SMI Arrows z predefiniowanymi ustawieniami

    Args:
        timeframe: Timeframe ('1m', '5m', '15m', etc.)
        sensitivity: Czułość sygnałów ('low', 'medium', 'high')
        name: Nazwa instancji wskaźnika

    Returns:
        Skonfigurowany SMIArrowsIndicator
    """

    # Konfiguracje zoptymalizowane pod różne timeframes
    configs = {
        '1m': {
            'smi_period': 14,
            'first_smoothing': 2,
            'second_smoothing': 2,
            'signal_smoothing': 2,
            'overbought_level': 50,
            'oversold_level': -50,
            'min_bars_between_signals': 2
        },
        '5m': {
            'smi_period': 10,
            'first_smoothing': 3,
            'second_smoothing': 3,
            'signal_smoothing': 3,
            'overbought_level': 40,
            'oversold_level': -40,
            'min_bars_between_signals': 3
        },
        '15m': {
            'smi_period': 10,
            'first_smoothing': 3,
            'second_smoothing': 3,
            'signal_smoothing': 3,
            'overbought_level': 40,
            'oversold_level': -40,
            'min_bars_between_signals': 2
        },
        '30m': {
            'smi_period': 8,
            'first_smoothing': 3,
            'second_smoothing': 3,
            'signal_smoothing': 2,
            'overbought_level': 35,
            'oversold_level': -35,
            'min_bars_between_signals': 2
        },
        '1h': {
            'smi_period': 8,
            'first_smoothing': 2,
            'second_smoothing': 2,
            'signal_smoothing': 2,
            'overbought_level': 35,
            'oversold_level': -35,
            'min_bars_between_signals': 1
        }
    }

    # Wybierz konfigurację
    if timeframe in configs:
        config = configs[timeframe]
    else:
        config = configs['5m']  # Domyślna

    return SMIArrowsIndicator(
        name=name,
        smi_period=config['smi_period'],
        first_smoothing=config['first_smoothing'],
        second_smoothing=config['second_smoothing'],
        signal_smoothing=config['signal_smoothing'],
        overbought_level=config['overbought_level'],
        oversold_level=config['oversold_level'],
        extreme_level=60,
        arrow_sensitivity=sensitivity,
        use_divergence=True,
        min_bars_between_signals=config['min_bars_between_signals'],
        price_type='close'
    )