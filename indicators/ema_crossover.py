# 📁 indicators/ema_crossover.py
"""
EMA Crossover Signal Indicator - Professional Implementation
Exponential Moving Average Crossover with Buy/Sell signals

Based on MQ5 EMA Crossover logic - trend following signals
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import logging
from .base import BaseIndicator

logger = logging.getLogger(__name__)


class EMACrossoverIndicator(BaseIndicator):
    """
    EMA Crossover Signal Indicator - Professional Trading Signals

    Funkcje:
    - Dual EMA system (Fast EMA vs Slow EMA)
    - Buy/Sell signals on crossovers
    - Trend strength analysis
    - Signal filtering and confirmation
    - Multi-timeframe compatibility
    """

    def __init__(self, name: str = "EMA_Crossover", **kwargs):
        default_settings = {
            'fast_ema_period': 12,  # Szybka EMA
            'slow_ema_period': 26,  # Wolna EMA
            'signal_ema_period': 9,  # Signal EMA (dla potwierdzenia)
            'min_separation': 0.0005,  # Min. odległość między EMA (jako % ceny)
            'use_signal_line': True,  # Użyj trzeciej EMA jako filtr
            'trend_strength_bars': 5,  # Ile świec analizować dla siły trendu
            'crossover_confirmation': 2,  # Ile świec potwierdzenia crossover
            'price_type': 'close'  # close, typical, weighted
        }
        default_settings.update(kwargs)

        super().__init__(name, **default_settings)

        self.version = "1.0 - Professional EMA Crossover"
        self.description = "EMA Crossover with professional signal filtering and trend confirmation"

        # Ustaw minimum periods
        self.minimum_periods = max(
            self.settings['slow_ema_period'] * 3,
            self.settings['signal_ema_period'] * 2,
            50
        )

        logger.info(f"EMA Crossover Indicator created: {self.settings}")

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

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Oblicza EMA (Exponential Moving Average)
        Identyczna logika jak w MQ5
        """
        length = len(prices)
        ema = np.zeros(length)

        if length < period:
            return ema

        # Smoothing factor
        alpha = 2.0 / (period + 1.0)

        # Pierwszy punkt - SMA
        ema[period - 1] = np.mean(prices[:period])

        # Kolejne punkty - EMA
        for i in range(period, length):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _detect_crossover_signals(self, fast_ema: np.ndarray, slow_ema: np.ndarray,
                                  signal_ema: np.ndarray = None,
                                  prices: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wykrywa sygnały crossover z zaawansowanymi filtrami

        Returns:
            Tuple[buy_signals, sell_signals] - arrays z pozycjami sygnałów
        """

        #print(f"🔍 EMA DEBUG: Detecting crossovers, fast_ema sample: {fast_ema[-5:]}")
        #print(f"🔍 EMA DEBUG: slow_ema sample: {slow_ema[-5:]}")

        length = len(fast_ema)
        buy_signals = np.zeros(length)
        sell_signals = np.zeros(length)

        confirmation_bars = self.settings['crossover_confirmation']
        min_separation = self.settings['min_separation']
        use_signal_line = self.settings['use_signal_line']

        for i in range(confirmation_bars + 1, length):
            # Sprawdź podstawowy crossover
            fast_curr = fast_ema[i]
            fast_prev = fast_ema[i - 1]
            slow_curr = slow_ema[i]
            slow_prev = slow_ema[i - 1]

            # Bullish crossover: Fast EMA crosses above Slow EMA
            if (fast_prev <= slow_prev and fast_curr > slow_curr):
                if self._validate_bullish_crossover(fast_ema, slow_ema, signal_ema,
                                                    prices, i, min_separation,
                                                    confirmation_bars, use_signal_line):
                    buy_signals[i] = prices[i] if prices is not None else slow_curr

            # Bearish crossover: Fast EMA crosses below Slow EMA
            elif (fast_prev >= slow_prev and fast_curr < slow_curr):
                if self._validate_bearish_crossover(fast_ema, slow_ema, signal_ema,
                                                    prices, i, min_separation,
                                                    confirmation_bars, use_signal_line):
                    sell_signals[i] = prices[i] if prices is not None else slow_curr

        buy_count = np.sum(buy_signals > 0)
        sell_count = np.sum(sell_signals > 0)
        #print(f"🔍 EMA DEBUG: Generated {buy_count} buy, {sell_count} sell signals")

        return buy_signals, sell_signals

    def _validate_bullish_crossover(self, fast_ema: np.ndarray, slow_ema: np.ndarray,
                                    signal_ema: np.ndarray, prices: np.ndarray,
                                    index: int, min_separation: float,
                                    confirmation_bars: int, use_signal_line: bool) -> bool:
        """Waliduje sygnał bullish crossover"""
        try:
            current_price = prices[index] if prices is not None else fast_ema[index]

            # 1. Sprawdź minimalną separację
            if min_separation > 0:
                separation = abs(fast_ema[index] - slow_ema[index]) / current_price
                if separation < min_separation:
                    return False

            # 2. Potwierdzenie przez kilka świec
            if confirmation_bars > 0:
                confirmed = 0
                for j in range(1, min(confirmation_bars + 1, index)):
                    if fast_ema[index - j] > slow_ema[index - j]:
                        confirmed += 1

                if confirmed < confirmation_bars // 2:
                    return False

            # 3. Signal line filter (jeśli używany)
            if use_signal_line and signal_ema is not None:
                # Fast EMA powinna być także powyżej Signal EMA
                if fast_ema[index] <= signal_ema[index]:
                    return False

                # Sprawdź trend Signal EMA
                if index > 2:
                    signal_trend = signal_ema[index] - signal_ema[index - 2]
                    if signal_trend < 0:  # Signal EMA spada
                        return False

            # 4. Sprawdź momentum
            if self._check_bullish_momentum(fast_ema, slow_ema, index):
                return True

            return False

        except Exception as e:
            logger.error(f"Error validating bullish crossover: {e}")
            return False

    def _validate_bearish_crossover(self, fast_ema: np.ndarray, slow_ema: np.ndarray,
                                    signal_ema: np.ndarray, prices: np.ndarray,
                                    index: int, min_separation: float,
                                    confirmation_bars: int, use_signal_line: bool) -> bool:
        """Waliduje sygnał bearish crossover"""
        try:
            current_price = prices[index] if prices is not None else fast_ema[index]

            # 1. Sprawdź minimalną separację
            if min_separation > 0:
                separation = abs(fast_ema[index] - slow_ema[index]) / current_price
                if separation < min_separation:
                    return False

            # 2. Potwierdzenie przez kilka świec
            if confirmation_bars > 0:
                confirmed = 0
                for j in range(1, min(confirmation_bars + 1, index)):
                    if fast_ema[index - j] < slow_ema[index - j]:
                        confirmed += 1

                if confirmed < confirmation_bars // 2:
                    return False

            # 3. Signal line filter (jeśli używany)
            if use_signal_line and signal_ema is not None:
                # Fast EMA powinna być także poniżej Signal EMA
                if fast_ema[index] >= signal_ema[index]:
                    return False

                # Sprawdź trend Signal EMA
                if index > 2:
                    signal_trend = signal_ema[index] - signal_ema[index - 2]
                    if signal_trend > 0:  # Signal EMA rośnie
                        return False

            # 4. Sprawdź momentum
            if self._check_bearish_momentum(fast_ema, slow_ema, index):
                return True

            return False

        except Exception as e:
            logger.error(f"Error validating bearish crossover: {e}")
            return False

    def _check_bullish_momentum(self, fast_ema: np.ndarray, slow_ema: np.ndarray,
                                index: int) -> bool:
        """Sprawdza bullish momentum"""
        if index < 3:
            return True

        try:
            # Sprawdź czy Fast EMA przyspiesza w górę
            fast_slope_recent = fast_ema[index] - fast_ema[index - 1]
            fast_slope_prev = fast_ema[index - 1] - fast_ema[index - 2]

            # Sprawdź czy Slow EMA też idzie w górę (lub przynajmniej nie spada mocno)
            slow_slope = slow_ema[index] - slow_ema[index - 2]

            # Momentum jest pozytywny gdy Fast EMA przyspiesza i Slow nie spada mocno
            return (fast_slope_recent > fast_slope_prev and
                    slow_slope >= -abs(fast_slope_recent) * 0.5)

        except Exception:
            return True

    def _check_bearish_momentum(self, fast_ema: np.ndarray, slow_ema: np.ndarray,
                                index: int) -> bool:
        """Sprawdza bearish momentum"""
        if index < 3:
            return True

        try:
            # Sprawdź czy Fast EMA przyspiesza w dół
            fast_slope_recent = fast_ema[index] - fast_ema[index - 1]
            fast_slope_prev = fast_ema[index - 1] - fast_ema[index - 2]

            # Sprawdź czy Slow EMA też idzie w dół (lub przynajmniej nie rośnie mocno)
            slow_slope = slow_ema[index] - slow_ema[index - 2]

            # Momentum jest negatywny gdy Fast EMA przyspiesza w dół i Slow nie rośnie mocno
            return (fast_slope_recent < fast_slope_prev and
                    slow_slope <= abs(fast_slope_recent) * 0.5)

        except Exception:
            return True

    def _calculate_trend_strength(self, fast_ema: np.ndarray, slow_ema: np.ndarray,
                                  signal_ema: np.ndarray, index: int) -> float:
        """Oblicza siłę trendu na podstawie EMA"""
        if index < self.settings['trend_strength_bars']:
            return 0.5

        try:
            bars_to_check = self.settings['trend_strength_bars']

            # Sprawdź konsystencję kierunku
            fast_above_slow = 0
            slow_above_signal = 0

            for i in range(bars_to_check):
                idx = index - i
                if idx >= 0:
                    if fast_ema[idx] > slow_ema[idx]:
                        fast_above_slow += 1

                    if signal_ema is not None and slow_ema[idx] > signal_ema[idx]:
                        slow_above_signal += 1

            # Oblicz siłę trendu
            fast_slow_strength = fast_above_slow / bars_to_check

            if signal_ema is not None:
                slow_signal_strength = slow_above_signal / bars_to_check
                # Średnia z obu pomiarów
                trend_strength = (fast_slow_strength + slow_signal_strength) / 2.0
            else:
                trend_strength = fast_slow_strength

            # Sprawdź separację EMA (większa separacja = silniejszy trend)
            current_price = fast_ema[index]
            if current_price > 0:
                separation = abs(fast_ema[index] - slow_ema[index]) / current_price
                separation_factor = min(1.0, separation * 1000)  # Scale separation
                trend_strength = (trend_strength + separation_factor) / 2.0

            return max(0.0, min(1.0, trend_strength))

        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.5

    def calculate(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Główna funkcja obliczeniowa EMA Crossover
        """
        try:
            if not self.validate_data(df):
                return None

            if not self._should_recalculate(df) and self.last_result:
                return self.last_result

            # Pobierz dane
            length = len(df)
            prices = self._get_price_data(df)

            # Oblicz EMA
            fast_ema = self._calculate_ema(prices, self.settings['fast_ema_period'])
            slow_ema = self._calculate_ema(prices, self.settings['slow_ema_period'])

            signal_ema = None
            if self.settings['use_signal_line']:
                signal_ema = self._calculate_ema(prices, self.settings['signal_ema_period'])

            # Wykryj sygnały crossover
            buy_signals, sell_signals = self._detect_crossover_signals(
                fast_ema, slow_ema, signal_ema, prices
            )

            # Oblicz siłę trendu dla każdego punktu
            trend_strength = np.array([
                self._calculate_trend_strength(fast_ema, slow_ema, signal_ema, i)
                for i in range(length)
            ])

            # Określ aktualny trend
            trend_direction = np.zeros(length)
            for i in range(1, length):
                if fast_ema[i] > slow_ema[i]:
                    trend_direction[i] = 1  # Bullish
                elif fast_ema[i] < slow_ema[i]:
                    trend_direction[i] = -1  # Bearish
                else:
                    trend_direction[i] = trend_direction[i - 1] if i > 0 else 0  # Maintain previous

            # Wyznacz obszary trendu
            bullish_areas, bearish_areas = self._identify_trend_areas(
                fast_ema, slow_ema, length
            )

            # Przygotuj wynik
            result = {
                'fast_ema': fast_ema,
                'slow_ema': slow_ema,
                'signal_ema': signal_ema if signal_ema is not None else np.zeros(length),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'bullish_areas': bullish_areas,
                'bearish_areas': bearish_areas,
                'crossover_points': self._find_crossover_points(fast_ema, slow_ema),
                'valid_from': max(self.settings['fast_ema_period'],
                                  self.settings['slow_ema_period']),
                'settings': self.settings.copy(),
                'timestamp': df.index[-1] if not df.empty else None,
                'ema_compatible': True
            }

            self.last_result = result

            # Loguj statystyki
            buy_count = np.sum(buy_signals > 0)
            sell_count = np.sum(sell_signals > 0)
            logger.info(f"EMA Crossover calculated: {length} candles, "
                        f"{buy_count} buy signals, {sell_count} sell signals")

            return result

        except Exception as e:
            logger.error(f"EMA Crossover calculation error: {e}")
            return None

    def _identify_trend_areas(self, fast_ema: np.ndarray, slow_ema: np.ndarray,
                              length: int) -> Tuple[List[Tuple], List[Tuple]]:
        """Identyfikuje obszary trendu bullish/bearish"""
        bullish_areas = []
        bearish_areas = []

        current_trend = None
        trend_start = 0

        for i in range(1, length):
            if fast_ema[i] > slow_ema[i]:
                # Bullish
                if current_trend != 'bullish':
                    if current_trend == 'bearish':
                        bearish_areas.append((trend_start, i - 1))
                    current_trend = 'bullish'
                    trend_start = i

            elif fast_ema[i] < slow_ema[i]:
                # Bearish
                if current_trend != 'bearish':
                    if current_trend == 'bullish':
                        bullish_areas.append((trend_start, i - 1))
                    current_trend = 'bearish'
                    trend_start = i

        # Dodaj ostatni obszar
        if current_trend == 'bullish':
            bullish_areas.append((trend_start, length - 1))
        elif current_trend == 'bearish':
            bearish_areas.append((trend_start, length - 1))

        return bullish_areas, bearish_areas

    def _find_crossover_points(self, fast_ema: np.ndarray, slow_ema: np.ndarray) -> List[Dict]:
        """Znajduje wszystkie punkty crossover"""
        crossovers = []

        for i in range(1, len(fast_ema)):
            if fast_ema[i - 1] <= slow_ema[i - 1] and fast_ema[i] > slow_ema[i]:
                crossovers.append({
                    'index': i,
                    'type': 'bullish',
                    'fast_value': fast_ema[i],
                    'slow_value': slow_ema[i]
                })
            elif fast_ema[i - 1] >= slow_ema[i - 1] and fast_ema[i] < slow_ema[i]:
                crossovers.append({
                    'index': i,
                    'type': 'bearish',
                    'fast_value': fast_ema[i],
                    'slow_value': slow_ema[i]
                })

        return crossovers

    def get_plot_config(self) -> Dict:
        """Konfiguracja wyświetlania EMA Crossover"""
        return {
            'main_window': True,
            'subplot': False,
            'colors': {
                'fast_ema': '#FFD700',  # Gold
                'slow_ema': '#87CEEB',  # Sky Blue
                'signal_ema': '#DDA0DD',  # Plum
                'buy_signal': '#00FF88',  # Green
                'sell_signal': '#FF4444',  # Red
                'bullish_area': '#00FF88',  # Light Green
                'bearish_area': '#FF6B6B'  # Light Red
            },
            'styles': {
                'fast_ema': {'width': 2.5, 'style': 'solid', 'alpha': 0.9},
                'slow_ema': {'width': 2.5, 'style': 'solid', 'alpha': 0.9},
                'signal_ema': {'width': 2, 'style': 'dashed', 'alpha': 0.8},
                'signals': {'size': 120, 'alpha': 0.9},
                'trend_areas': {'alpha': 0.1}
            }
        }

    def get_latest_signal(self, result: Optional[Dict] = None, lookback: int = 10) -> Dict:
        """
        Analizuje najnowsze sygnały EMA Crossover

        Args:
            result: Wyniki EMA (opcjonalne)
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
                'trend': 'neutral',
                'fast_ema': 0,
                'slow_ema': 0,
                'signal_ema': 0
            }

        try:
            length = len(result['fast_ema'])
            if length < 2:
                return {'signal': 'none', 'strength': 0}

            # Sprawdź najnowsze sygnały
            start_idx = max(0, length - lookback)
            signal = 'none'
            signal_strength = 0
            signal_age = 0

            # Szukaj ostatnich sygnałów
            for i in range(length - 1, start_idx - 1, -1):
                if result['buy_signals'][i] > 0:
                    signal = 'buy'
                    signal_age = length - 1 - i
                    signal_strength = result['trend_strength'][i]
                    break
                elif result['sell_signals'][i] > 0:
                    signal = 'sell'
                    signal_age = length - 1 - i
                    signal_strength = result['trend_strength'][i]
                    break

            # Analiza aktualnego trendu
            current_trend_dir = result['trend_direction'][-1]
            if current_trend_dir > 0:
                trend = 'bullish'
            elif current_trend_dir < 0:
                trend = 'bearish'
            else:
                trend = 'neutral'

            # Aktualne wartości EMA
            fast_ema = result['fast_ema'][-1]
            slow_ema = result['slow_ema'][-1]
            signal_ema = result['signal_ema'][-1] if result['signal_ema'] is not None else 0

            # Sprawdź siłę trendu
            current_strength = result['trend_strength'][-1]

            return {
                'signal': signal,
                'strength': signal_strength,
                'signal_age': signal_age,
                'trend': trend,
                'trend_strength': current_strength,
                'fast_ema': fast_ema,
                'slow_ema': slow_ema,
                'signal_ema': signal_ema,
                'ema_separation': abs(fast_ema - slow_ema),
                'crossover_recent': signal_age <= 3
            }

        except Exception as e:
            logger.error(f"EMA signal analysis error: {e}")
            return {'signal': 'error', 'strength': 0}


def create_ema_crossover_indicator(timeframe: str = '5m', style: str = 'balanced',
                                   name: str = "EMA_Crossover") -> EMACrossoverIndicator:
    """
    Factory function dla EMA Crossover z predefiniowanymi ustawieniami

    Args:
        timeframe: Timeframe ('1m', '5m', '15m', etc.)
        style: Styl tradingu ('conservative', 'balanced', 'aggressive')
        name: Nazwa instancji wskaźnika

    Returns:
        Skonfigurowany EMACrossoverIndicator
    """

    # Konfiguracje zoptymalizowane pod różne style i timeframes
    configs = {
        '1m': {
            'conservative': {'fast': 21, 'slow': 55, 'signal': 13, 'confirm': 3},
            'balanced': {'fast': 12, 'slow': 26, 'signal': 9, 'confirm': 2},
            'aggressive': {'fast': 8, 'slow': 21, 'signal': 5, 'confirm': 1}
        },
        '5m': {
            'conservative': {'fast': 18, 'slow': 39, 'signal': 11, 'confirm': 2},
            'balanced': {'fast': 12, 'slow': 26, 'signal': 9, 'confirm': 2},
            'aggressive': {'fast': 9, 'slow': 21, 'signal': 7, 'confirm': 1}
        },
        '15m': {
            'conservative': {'fast': 15, 'slow': 35, 'signal': 10, 'confirm': 2},
            'balanced': {'fast': 12, 'slow': 26, 'signal': 9, 'confirm': 1},
            'aggressive': {'fast': 8, 'slow': 18, 'signal': 6, 'confirm': 1}
        },
        '30m': {
            'conservative': {'fast': 14, 'slow': 30, 'signal': 9, 'confirm': 1},
            'balanced': {'fast': 10, 'slow': 22, 'signal': 7, 'confirm': 1},
            'aggressive': {'fast': 7, 'slow': 15, 'signal': 5, 'confirm': 1}
        },
        '1h': {
            'conservative': {'fast': 12, 'slow': 26, 'signal': 9, 'confirm': 1},
            'balanced': {'fast': 9, 'slow': 21, 'signal': 7, 'confirm': 1},
            'aggressive': {'fast': 6, 'slow': 14, 'signal': 4, 'confirm': 1}
        }
    }

    # Wybierz konfigurację
    if timeframe in configs and style in configs[timeframe]:
        config = configs[timeframe][style]
    else:
        config = configs['5m']['balanced']  # Domyślna

    return EMACrossoverIndicator(
        name=name,
        fast_ema_period=config['fast'],
        slow_ema_period=config['slow'],
        signal_ema_period=config['signal'],
        min_separation=0.0005,
        use_signal_line=True,
        trend_strength_bars=5,
        crossover_confirmation=config['confirm'],
        price_type='close'
    )