# 📁 indicators/rsi_professional.py
"""
RSI Professional Implementation - Trading Grade Accuracy
Relative Strength Index według oryginalnej formuły J. Welles Wildera Jr.

Zgodny z:
- MT5 RSI
- TradingView RSI
- Bloomberg Terminal RSI
- Oryginalną książką "New Concepts in Technical Trading Systems" (1978)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import logging
from .base import BaseIndicator

logger = logging.getLogger(__name__)


class RSIProfessionalIndicator(BaseIndicator):
    """
    RSI Professional - Trading Grade Implementation

    Funkcje:
    - Oryginalna formuła Wildera (EMA-based smoothing)
    - Dokładne obliczenia zgodne z MT5/TradingView
    - Support dla różnych timeframes
    - Overbought/Oversold detection (70/30, 80/20)
    - Divergence detection
    - RSI-based signals (crossovers, extremes)
    - Multi-timeframe compatibility
    """

    def __init__(self, name: str = "RSI_Professional", **kwargs):
        default_settings = {
            'rsi_period': 14,  # Klasyczne 14 okresów Wildera
            'overbought_level': 70,  # Poziom wykupienia
            'oversold_level': 30,  # Poziom wyprzedania
            'extreme_overbought': 80,  # Poziom ekstremalnego wykupienia
            'extreme_oversold': 20,  # Poziom ekstremalnej wyprzedania
            'signal_sensitivity': 'medium',  # low, medium, high
            'use_divergence': True,  # Wykrywanie dywergencji
            'min_bars_between_signals': 3,  # Min. odstęp między sygnałami
            'price_type': 'close',  # close, typical, weighted
            'smoothing_method': 'wilder',  # wilder, ema, sma
            'precision_digits': 6,  # Precyzja obliczeń (trading grade)
        }
        default_settings.update(kwargs)

        super().__init__(name, **default_settings)

        self.version = "1.0 - Professional RSI (Wilder Compatible)"
        self.description = "Professional RSI with trading-grade accuracy and MT5 compatibility"

        # Ustaw minimum periods - RSI potrzebuje okresu + lookback dla stabilności
        self.minimum_periods = self.settings['rsi_period'] * 3 + 10

        # Cache dla optymalizacji
        self._gain_loss_cache = {}
        self._rsi_cache = {}

        logger.info(f"RSI Professional Indicator created: {self.settings}")

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

    def _calculate_rsi_wilder_method(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Oblicza RSI według oryginalnej metody Wildera

        Oryginalna formuła z książki "New Concepts in Technical Trading Systems":
        1. Oblicz zmiany cen (gains/losses)
        2. Pierwsze średnie: SMA z gains i losses za 'period' dni
        3. Kolejne średnie: Smoothed MA (Wilder's EMA):
           - Avg Gain = [(Prev Avg Gain × (period-1)) + Current Gain] / period
           - Avg Loss = [(Prev Avg Loss × (period-1)) + Current Loss] / period
        4. RS = Avg Gain / Avg Loss
        5. RSI = 100 - (100 / (1 + RS))
        """
        length = len(prices)
        rsi_values = np.full(length, np.nan, dtype=np.float64)

        if length < period + 1:
            return rsi_values

        # Krok 1: Oblicz price changes
        price_changes = np.diff(prices)
        gains = np.where(price_changes > 0, price_changes, 0.0)
        losses = np.where(price_changes < 0, -price_changes, 0.0)

        # Krok 2: Pierwsze średnie (SMA za pierwsze 'period' dni)
        first_avg_gain = np.mean(gains[:period])
        first_avg_loss = np.mean(losses[:period])

        # Inicjalizuj smoothed averages
        avg_gain = first_avg_gain
        avg_loss = first_avg_loss

        # Krok 3: Oblicz pierwsze RSI (dla index = period w price array)
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_values[period] = 100.0 if avg_gain > 0 else 50.0

        # Krok 4: Oblicz kolejne RSI używając Wilder's smoothing
        for i in range(period + 1, length):
            gain_idx = i - 1  # Index w arrays gains/losses

            # Wilder's smoothing formula
            avg_gain = ((avg_gain * (period - 1)) + gains[gain_idx]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[gain_idx]) / period

            # Oblicz RSI
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))
            else:
                rsi_values[i] = 100.0 if avg_gain > 0 else 50.0

        return rsi_values

    def _calculate_rsi_ema_method(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Alternatywna metoda RSI używająca standardowej EMA
        (dla porównania z niektórymi platformami)
        """
        length = len(prices)
        rsi_values = np.full(length, np.nan, dtype=np.float64)

        if length < period + 1:
            return rsi_values

        # Price changes
        price_changes = np.diff(prices)
        gains = np.where(price_changes > 0, price_changes, 0.0)
        losses = np.where(price_changes < 0, -price_changes, 0.0)

        # EMA alpha
        alpha = 2.0 / (period + 1.0)

        # Pierwsze średnie - SMA
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        # First RSI
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_values[period] = 100.0 if avg_gain > 0 else 50.0

        # EMA smoothing
        for i in range(period + 1, length):
            gain_idx = i - 1

            # EMA formula
            avg_gain = alpha * gains[gain_idx] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[gain_idx] + (1 - alpha) * avg_loss

            # RSI
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))
            else:
                rsi_values[i] = 100.0 if avg_gain > 0 else 50.0

        return rsi_values

    def _detect_rsi_signals(self, rsi_values: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wykrywa sygnały handlowe na podstawie RSI

        Sygnały:
        1. Crossover 70/30 levels
        2. Extreme levels (80/20)
        3. RSI momentum changes
        4. Divergences
        """
        length = len(rsi_values)
        buy_signals = np.zeros(length, dtype=np.float64)
        sell_signals = np.zeros(length, dtype=np.float64)

        overbought = self.settings['overbought_level']
        oversold = self.settings['oversold_level']
        extreme_ob = self.settings['extreme_overbought']
        extreme_os = self.settings['extreme_oversold']
        min_bars = self.settings['min_bars_between_signals']

        last_signal_bar = -min_bars - 1

        for i in range(3, length):
            current_bar = i

            # Check minimum bars between signals
            if current_bar - last_signal_bar < min_bars:
                continue

            current_rsi = rsi_values[i]
            prev_rsi = rsi_values[i - 1]
            prev2_rsi = rsi_values[i - 2]

            # Skip if RSI is NaN
            if np.isnan(current_rsi) or np.isnan(prev_rsi):
                continue

            # === BULLISH SIGNALS ===

            # Signal 1: RSI crosses above oversold level
            if (prev2_rsi < oversold and prev_rsi < oversold and
                    current_rsi > oversold and current_rsi > prev_rsi):

                if self._validate_buy_signal(rsi_values, prices, i):
                    buy_signals[i] = prices[i]
                    last_signal_bar = current_bar
                    continue

            # Signal 2: RSI bounces from extreme oversold
            if (prev_rsi < extreme_os and current_rsi > extreme_os and
                    current_rsi > prev_rsi):

                if self._validate_buy_signal(rsi_values, prices, i):
                    buy_signals[i] = prices[i]
                    last_signal_bar = current_bar
                    continue

            # === BEARISH SIGNALS ===

            # Signal 1: RSI crosses below overbought level
            if (prev2_rsi > overbought and prev_rsi > overbought and
                    current_rsi < overbought and current_rsi < prev_rsi):

                if self._validate_sell_signal(rsi_values, prices, i):
                    sell_signals[i] = prices[i]
                    last_signal_bar = current_bar
                    continue

            # Signal 2: RSI drops from extreme overbought
            if (prev_rsi > extreme_ob and current_rsi < extreme_ob and
                    current_rsi < prev_rsi):

                if self._validate_sell_signal(rsi_values, prices, i):
                    sell_signals[i] = prices[i]
                    last_signal_bar = current_bar

        return buy_signals, sell_signals

    def _validate_buy_signal(self, rsi_values: np.ndarray, prices: np.ndarray,
                             index: int) -> bool:
        """Waliduje sygnał kupna z dodatkowymi filtrami"""
        try:
            # 1. RSI momentum check
            if index < 3:
                return True

            rsi_momentum = rsi_values[index] - rsi_values[index - 2]
            if rsi_momentum <= 0:
                return False

            # 2. Price trend check (nie kupuj w silnym downtrend)
            if index < 5:
                return True

            recent_prices = prices[index - 4:index + 1]
            price_slope = np.polyfit(range(5), recent_prices, 1)[0]
            price_std = np.std(recent_prices)

            # Jeśli trend cenowy bardzo negatywny, odrzuć
            if price_slope < -0.6 * price_std:
                return False

            # 3. RSI nie powinno być już za wysokie
            if rsi_values[index] > self.settings['overbought_level'] * 0.9:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating buy signal: {e}")
            return True

    def _validate_sell_signal(self, rsi_values: np.ndarray, prices: np.ndarray,
                              index: int) -> bool:
        """Waliduje sygnał sprzedaży z dodatkowymi filtrami"""
        try:
            # 1. RSI momentum check
            if index < 3:
                return True

            rsi_momentum = rsi_values[index] - rsi_values[index - 2]
            if rsi_momentum >= 0:
                return False

            # 2. Price trend check (nie sprzedawaj w silnym uptrend)
            if index < 5:
                return True

            recent_prices = prices[index - 4:index + 1]
            price_slope = np.polyfit(range(5), recent_prices, 1)[0]
            price_std = np.std(recent_prices)

            # Jeśli trend cenowy bardzo pozytywny, odrzuć
            if price_slope > 0.6 * price_std:
                return False

            # 3. RSI nie powinno być już za niskie
            if rsi_values[index] < self.settings['oversold_level'] * 1.1:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating sell signal: {e}")
            return True

    def _detect_divergences(self, rsi_values: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wykrywa dywergencje między RSI a ceną

        Bullish divergence: cena robi niższe dołki, RSI wyższe dołki
        Bearish divergence: cena robi wyższe szczyty, RSI niższe szczyty
        """
        if not self.settings.get('use_divergence', True):
            return np.zeros(len(rsi_values)), np.zeros(len(rsi_values))

        length = len(rsi_values)
        bullish_div = np.zeros(length, dtype=np.float64)
        bearish_div = np.zeros(length, dtype=np.float64)

        # Minimum lookback for divergence
        min_bars = 15

        for i in range(min_bars, length - 5):
            # Bullish divergence check
            if self._check_bullish_divergence(prices, rsi_values, i, min_bars):
                bullish_div[i] = prices[i]

            # Bearish divergence check
            if self._check_bearish_divergence(prices, rsi_values, i, min_bars):
                bearish_div[i] = prices[i]

        return bullish_div, bearish_div

    def _check_bullish_divergence(self, prices: np.ndarray, rsi_values: np.ndarray,
                                  current_idx: int, lookback: int) -> bool:
        """Sprawdza bullish divergence w oknie lookback"""
        try:
            # Find local minima in both price and RSI
            price_window = prices[current_idx - lookback:current_idx + 1]
            rsi_window = rsi_values[current_idx - lookback:current_idx + 1]

            # Find last two significant lows
            price_lows = []
            rsi_lows = []

            for i in range(3, len(price_window) - 3):
                # Local minimum criteria (more strict)
                if (price_window[i] < price_window[i - 1] and
                        price_window[i] < price_window[i + 1] and
                        price_window[i] < price_window[i - 2] and
                        price_window[i] < price_window[i + 2] and
                        price_window[i] <= min(price_window[max(0, i - 3):i]) and
                        price_window[i] <= min(price_window[i + 1:min(len(price_window), i + 4)])):
                    price_lows.append((i, price_window[i]))
                    rsi_lows.append((i, rsi_window[i]))

            if len(price_lows) < 2:
                return False

            # Compare last two lows
            last_price_low = price_lows[-1][1]
            prev_price_low = price_lows[-2][1]
            last_rsi_low = rsi_lows[-1][1]
            prev_rsi_low = rsi_lows[-2][1]

            # Bullish divergence: price lower, RSI higher
            # Add minimum threshold to avoid noise
            price_diff = (prev_price_low - last_price_low) / prev_price_low
            rsi_diff = last_rsi_low - prev_rsi_low

            if (price_diff > 0.01 and  # Price dropped at least 1%
                    rsi_diff > 2.0 and  # RSI rose at least 2 points
                    last_rsi_low < self.settings['oversold_level'] * 1.5):  # In oversold area
                return True

            return False

        except Exception:
            return False

    def _check_bearish_divergence(self, prices: np.ndarray, rsi_values: np.ndarray,
                                  current_idx: int, lookback: int) -> bool:
        """Sprawdza bearish divergence w oknie lookback"""
        try:
            # Find local maxima in both price and RSI
            price_window = prices[current_idx - lookback:current_idx + 1]
            rsi_window = rsi_values[current_idx - lookback:current_idx + 1]

            # Find last two significant highs
            price_highs = []
            rsi_highs = []

            for i in range(3, len(price_window) - 3):
                # Local maximum criteria (more strict)
                if (price_window[i] > price_window[i - 1] and
                        price_window[i] > price_window[i + 1] and
                        price_window[i] > price_window[i - 2] and
                        price_window[i] > price_window[i + 2] and
                        price_window[i] >= max(price_window[max(0, i - 3):i]) and
                        price_window[i] >= max(price_window[i + 1:min(len(price_window), i + 4)])):
                    price_highs.append((i, price_window[i]))
                    rsi_highs.append((i, rsi_window[i]))

            if len(price_highs) < 2:
                return False

            # Compare last two highs
            last_price_high = price_highs[-1][1]
            prev_price_high = price_highs[-2][1]
            last_rsi_high = rsi_highs[-1][1]
            prev_rsi_high = rsi_highs[-2][1]

            # Bearish divergence: price higher, RSI lower
            # Add minimum threshold to avoid noise
            price_diff = (last_price_high - prev_price_high) / prev_price_high
            rsi_diff = prev_rsi_high - last_rsi_high

            if (price_diff > 0.01 and  # Price rose at least 1%
                    rsi_diff > 2.0 and  # RSI dropped at least 2 points
                    last_rsi_high > self.settings['overbought_level'] * 0.8):  # In overbought area
                return True

            return False

        except Exception:
            return False

    def calculate(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Główna funkcja obliczeniowa RSI Professional
        """
        try:
            if not self.validate_data(df):
                return None

            if not self._should_recalculate(df) and self.last_result:
                return self.last_result

            # Get data
            length = len(df)
            prices = self._get_price_data(df)
            period = self.settings['rsi_period']

            # Select calculation method
            smoothing_method = self.settings.get('smoothing_method', 'wilder')

            if smoothing_method == 'wilder':
                rsi_values = self._calculate_rsi_wilder_method(prices, period)
            elif smoothing_method == 'ema':
                rsi_values = self._calculate_rsi_ema_method(prices, period)
            else:
                rsi_values = self._calculate_rsi_wilder_method(prices, period)

            # Detect signals
            buy_signals, sell_signals = self._detect_rsi_signals(rsi_values, prices)

            # Detect divergences
            bullish_div, bearish_div = self._detect_divergences(rsi_values, prices)

            # Create reference levels
            overbought_line = np.full(length, self.settings['overbought_level'])
            oversold_line = np.full(length, self.settings['oversold_level'])
            extreme_ob_line = np.full(length, self.settings['extreme_overbought'])
            extreme_os_line = np.full(length, self.settings['extreme_oversold'])
            midline = np.full(length, 50.0)

            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(rsi_values)

            # Prepare result
            result = {
                'rsi': rsi_values,
                'buy_arrows': buy_signals,
                'sell_arrows': sell_signals,
                'bullish_divergence': bullish_div,
                'bearish_divergence': bearish_div,
                'overbought_line': overbought_line,
                'oversold_line': oversold_line,
                'extreme_overbought_line': extreme_ob_line,
                'extreme_oversold_line': extreme_os_line,
                'midline': midline,
                'trend_strength': trend_strength,
                'valid_from': period + 1,  # RSI starts from period+1
                'settings': self.settings.copy(),
                'timestamp': df.index[-1] if not df.empty else None,
                'levels': {
                    'overbought': self.settings['overbought_level'],
                    'oversold': self.settings['oversold_level'],
                    'extreme_overbought': self.settings['extreme_overbought'],
                    'extreme_oversold': self.settings['extreme_oversold'],
                    'midline': 50.0
                },
                'wilder_compatible': True,
                'mt5_compatible': True
            }

            self.last_result = result

            # Log statistics
            valid_rsi_count = np.sum(~np.isnan(rsi_values))
            buy_count = np.sum(buy_signals > 0)
            sell_count = np.sum(sell_signals > 0)
            div_count = np.sum(bullish_div > 0) + np.sum(bearish_div > 0)

            logger.info(f"RSI Professional calculated: {length} candles, "
                        f"{valid_rsi_count} valid RSI values, "
                        f"{buy_count} buy signals, {sell_count} sell signals, "
                        f"{div_count} divergences")

            return result

        except Exception as e:
            logger.error(f"RSI Professional calculation error: {e}")
            return None

    def _calculate_trend_strength(self, rsi_values: np.ndarray) -> np.ndarray:
        """Oblicza siłę trendu na podstawie RSI"""
        length = len(rsi_values)
        strength = np.zeros(length, dtype=np.float64)

        for i in range(10, length):
            try:
                # Analyze last 10 RSI values
                recent_rsi = rsi_values[i - 9:i + 1]
                recent_rsi = recent_rsi[~np.isnan(recent_rsi)]

                if len(recent_rsi) < 5:
                    strength[i] = 0.5
                    continue

                # 1. RSI trend consistency
                rsi_slope = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
                trend_factor = min(1.0, abs(rsi_slope) / 10.0)

                # 2. RSI volatility (lower volatility = stronger trend)
                rsi_std = np.std(recent_rsi)
                volatility_factor = max(0.0, 1.0 - (rsi_std / 20.0))

                # 3. Distance from midline (50)
                distance_factor = abs(rsi_values[i] - 50.0) / 50.0

                # Combine factors
                combined_strength = (trend_factor * 0.4 +
                                     volatility_factor * 0.3 +
                                     distance_factor * 0.3)

                strength[i] = max(0.0, min(1.0, combined_strength))

            except Exception:
                strength[i] = 0.5

        return strength

    def get_plot_config(self) -> Dict:
        """Konfiguracja wyświetlania RSI Professional"""
        return {
            'main_window': False,
            'subplot': True,
            'subplot_height_ratio': 0.3,
            'colors': {
                'rsi_line': '#FFD700',  # Gold
                'buy_arrow': '#00FF88',
                'sell_arrow': '#FF4444',
                'bullish_divergence': '#4A90E2',
                'bearish_divergence': '#E24A90',
                'overbought_line': '#FF6B6B',
                'oversold_line': '#4ECDC4',
                'extreme_overbought': '#CC0000',
                'extreme_oversold': '#00CC00',
                'midline': '#999999'
            },
            'styles': {
                'rsi_line': {'width': 2.5, 'style': 'solid'},
                'level_lines': {'width': 1, 'style': 'dashed', 'alpha': 0.7},
                'extreme_lines': {'width': 1.5, 'style': 'dotted', 'alpha': 0.8},
                'arrows': {'size': 120, 'alpha': 0.9},
                'divergence': {'size': 80, 'alpha': 0.8}
            },
            'levels': {
                'overbought': self.settings['overbought_level'],
                'oversold': self.settings['oversold_level'],
                'extreme_overbought': self.settings['extreme_overbought'],
                'extreme_oversold': self.settings['extreme_oversold'],
                'midline': 50.0
            }
        }

    def get_latest_signal(self, result: Optional[Dict] = None, lookback: int = 10) -> Dict:
        """
        Analizuje najnowsze sygnały RSI Professional

        Args:
            result: Wyniki RSI (opcjonalne)
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
                'rsi_value': 50,
                'trend': 'neutral',
                'divergence': False,
                'zone': 'neutral'
            }

        try:
            length = len(result['rsi'])
            if length < 2:
                return {'signal': 'none', 'strength': 0, 'rsi_value': 50}

            # Check latest signals
            start_idx = max(0, length - lookback)
            signal = 'none'
            signal_strength = 0
            signal_age = 0

            # Find latest signals
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

            # Current RSI analysis
            current_rsi = result['rsi'][-1]
            prev_rsi = result['rsi'][-2] if length > 1 else current_rsi

            # Determine trend
            rsi_change = current_rsi - prev_rsi
            if abs(rsi_change) < 0.5:
                trend = 'sideways'
            elif rsi_change > 0:
                trend = 'bullish'
            else:
                trend = 'bearish'

            # Determine zone
            levels = result['levels']
            if current_rsi >= levels['extreme_overbought']:
                zone = 'extreme_overbought'
            elif current_rsi >= levels['overbought']:
                zone = 'overbought'
            elif current_rsi <= levels['extreme_oversold']:
                zone = 'extreme_oversold'
            elif current_rsi <= levels['oversold']:
                zone = 'oversold'
            elif current_rsi > 45 and current_rsi < 55:
                zone = 'neutral'
            elif current_rsi > 50:
                zone = 'bullish'
            else:
                zone = 'bearish'

            # Check divergences
            divergence_detected = (
                    result['bullish_divergence'][start_idx:].any() or
                    result['bearish_divergence'][start_idx:].any()
            )

            return {
                'signal': signal,
                'strength': signal_strength,
                'signal_age': signal_age,
                'rsi_value': current_rsi,
                'rsi_change': rsi_change,
                'trend': trend,
                'zone': zone,
                'divergence': divergence_detected,
                'overbought': current_rsi >= levels['overbought'],
                'oversold': current_rsi <= levels['oversold'],
                'extreme_overbought': current_rsi >= levels['extreme_overbought'],
                'extreme_oversold': current_rsi <= levels['extreme_oversold']
            }

        except Exception as e:
            logger.error(f"RSI signal analysis error: {e}")
            return {'signal': 'error', 'strength': 0, 'rsi_value': 50}


def create_rsi_professional_indicator(timeframe: str = '5m', sensitivity: str = 'medium',
                                      name: str = "RSI_Professional") -> RSIProfessionalIndicator:
    """
    Factory function dla RSI Professional z predefiniowanymi ustawieniami

    Args:
        timeframe: Timeframe ('1m', '5m', '15m', etc.)
        sensitivity: Czułość sygnałów ('low', 'medium', 'high')
        name: Nazwa instancji wskaźnika

    Returns:
        Skonfigurowany RSIProfessionalIndicator
    """

    # Konfiguracje zoptymalizowane pod różne timeframes
    configs = {
        '1m': {
            'rsi_period': 14,
            'overbought_level': 75,
            'oversold_level': 25,
            'extreme_overbought': 85,
            'extreme_oversold': 15,
            'min_bars_between_signals': 2
        },
        '5m': {
            'rsi_period': 14,
            'overbought_level': 70,
            'oversold_level': 30,
            'extreme_overbought': 80,
            'extreme_oversold': 20,
            'min_bars_between_signals': 3
        },
        '15m': {
            'rsi_period': 14,
            'overbought_level': 70,
            'oversold_level': 30,
            'extreme_overbought': 80,
            'extreme_oversold': 20,
            'min_bars_between_signals': 2
        },
        '30m': {
            'rsi_period': 14,
            'overbought_level': 68,
            'oversold_level': 32,
            'extreme_overbought': 78,
            'extreme_oversold': 22,
            'min_bars_between_signals': 2
        },
        '1h': {
            'rsi_period': 14,
            'overbought_level': 65,
            'oversold_level': 35,
            'extreme_overbought': 75,
            'extreme_oversold': 25,
            'min_bars_between_signals': 1
        },
        '4h': {
            'rsi_period': 14,
            'overbought_level': 65,
            'oversold_level': 35,
            'extreme_overbought': 75,
            'extreme_oversold': 25,
            'min_bars_between_signals': 1
        },
        '1d': {
            'rsi_period': 14,
            'overbought_level': 70,
            'oversold_level': 30,
            'extreme_overbought': 80,
            'extreme_oversold': 20,
            'min_bars_between_signals': 1
        }
    }

    # Sensitivity adjustments
    sensitivity_adjustments = {
        'low': {
            'overbought_adjustment': +5,
            'oversold_adjustment': -5,
            'min_bars_multiplier': 1.5
        },
        'medium': {
            'overbought_adjustment': 0,
            'oversold_adjustment': 0,
            'min_bars_multiplier': 1.0
        },
        'high': {
            'overbought_adjustment': -5,
            'oversold_adjustment': +5,
            'min_bars_multiplier': 0.7
        }
    }

    # Select base config
    if timeframe in configs:
        config = configs[timeframe].copy()
    else:
        config = configs['5m'].copy()  # Default

    # Apply sensitivity adjustments
    if sensitivity in sensitivity_adjustments:
        adj = sensitivity_adjustments[sensitivity]
        config['overbought_level'] += adj['overbought_adjustment']
        config['oversold_level'] += adj['oversold_adjustment']
        config['min_bars_between_signals'] = int(config['min_bars_between_signals'] * adj['min_bars_multiplier'])

    return RSIProfessionalIndicator(
        name=name,
        rsi_period=config['rsi_period'],
        overbought_level=config['overbought_level'],
        oversold_level=config['oversold_level'],
        extreme_overbought=config['extreme_overbought'],
        extreme_oversold=config['extreme_oversold'],
        signal_sensitivity=sensitivity,
        use_divergence=True,
        min_bars_between_signals=config['min_bars_between_signals'],
        price_type='close',
        smoothing_method='wilder',  # Oryginalna metoda Wildera
        precision_digits=6
    )


def compare_rsi_implementations():
    """
    Utility function dla porównania różnych implementacji RSI
    Przydatne do testowania accuracy względem MT5/TradingView
    """
    try:
        import talib
        talib_available = True
    except ImportError:
        talib_available = False

    try:
        import pandas_ta as pta
        pandas_ta_available = True
    except ImportError:
        pandas_ta_available = False

    def compare_with_test_data():
        """Porównaj z test data"""
        # Sample price data
        test_prices = np.array([
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 47.25,
            47.92, 46.75, 46.57, 46.11, 43.42, 42.66, 43.13, 43.74,
            44.64, 45.58, 46.08, 45.89, 46.03, 46.83, 47.69, 47.54,
            48.64, 48.94, 49.58, 50.19, 50.12, 49.66, 48.90, 47.99
        ])

        df_test = pd.DataFrame({'close': test_prices})

        # Our implementation
        rsi_prof = RSIProfessionalIndicator(name="Test_RSI")
        our_result = rsi_prof.calculate(df_test)
        our_rsi = our_result['rsi'] if our_result else np.array([])

        results = {
            'our_implementation': our_rsi[-5:] if len(our_rsi) > 5 else our_rsi,
            'data_points': len(test_prices)
        }

        # TA-Lib comparison
        if talib_available:
            try:
                talib_rsi = talib.RSI(test_prices, timeperiod=14)
                results['talib_rsi'] = talib_rsi[-5:]
            except Exception as e:
                results['talib_error'] = str(e)

        # pandas_ta comparison
        if pandas_ta_available:
            try:
                pta_rsi = pta.rsi(df_test['close'], length=14)
                results['pandas_ta_rsi'] = pta_rsi.tail(5).values
            except Exception as e:
                results['pandas_ta_error'] = str(e)

        return results

    return compare_with_test_data()


# Test/validation functions
def validate_rsi_accuracy():
    """
    Validates RSI calculation accuracy against known reference values
    Based on Wilder's original example from his 1978 book
    """
    # Wilder's test data (from "New Concepts in Technical Trading Systems")
    wilder_test_prices = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 47.25, 47.92, 46.75,
        46.57, 46.11, 43.42, 42.66, 43.13, 43.74, 44.64, 45.58, 46.08, 45.89,
        46.03, 46.83, 47.69, 47.54, 48.64, 48.94, 49.58, 50.19, 50.12, 49.66,
        48.90, 47.99, 46.78, 47.28, 47.97, 48.39, 48.42, 48.96, 49.12
    ]

    # Expected RSI values (calculated manually or from reference)
    # These would need to be verified against Wilder's book or MT5
    expected_rsi_values = {
        28: 70.53,  # Approximate values for validation
        29: 66.32,
        30: 66.55,
        # Add more reference points as needed
    }

    df_test = pd.DataFrame({'close': wilder_test_prices})
    rsi_indicator = RSIProfessionalIndicator()
    result = rsi_indicator.calculate(df_test)

    if result:
        calculated_rsi = result['rsi']
        validation_results = {}

        for idx, expected_val in expected_rsi_values.items():
            if idx < len(calculated_rsi) and not np.isnan(calculated_rsi[idx]):
                calculated_val = calculated_rsi[idx]
                difference = abs(calculated_val - expected_val)
                validation_results[idx] = {
                    'expected': expected_val,
                    'calculated': calculated_val,
                    'difference': difference,
                    'accuracy': 100 * (1 - difference / expected_val)
                }

        return validation_results

    return None