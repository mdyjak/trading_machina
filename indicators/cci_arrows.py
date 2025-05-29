# 📁 indicators/cci_arrows.py
"""
CCI Arrows Indicator - Professional Implementation
Commodity Channel Index z sygnałami strzałek (kupno/sprzedaż)

Zgodny z MQ5 CCI-Arrows - analiza momentum i identyfikacja punktów zwrotnych
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from .base import BaseIndicator

logger = logging.getLogger(__name__)


class CCIArrowsIndicator(BaseIndicator):
    """
    CCI Arrows - Professional Trading Signals

    Funkcje:
    - CCI calculation (Commodity Channel Index)
    - Overbought/Oversold detection
    - Buy/Sell arrows on trend reversals
    - Divergence detection
    - Multi-timeframe analysis
    """

    def __init__(self, name: str = "CCI_Arrows", **kwargs):
        default_settings = {
            'cci_period': 14,  # Okres CCI
            'overbought_level': 100,  # Poziom wykupienia
            'oversold_level': -100,  # Poziom wyprzedania
            'extreme_level': 200,  # Poziom ekstremalny
            'arrow_sensitivity': 'medium',  # low, medium, high
            'use_divergence': True,  # Wykrywanie dywergencji
            'min_bars_between_signals': 3,  # Min. odstęp między sygnałami
            'price_type': 'typical'  # typical, weighted, close
        }
        default_settings.update(kwargs)

        super().__init__(name, **default_settings)

        self.version = "1.0 - Professional CCI Arrows"
        self.description = "CCI with professional buy/sell signals and divergence detection"

        # Ustaw minimum periods
        self.minimum_periods = max(self.settings['cci_period'] * 3, 50)

        # Sensitivity settings
        self._setup_sensitivity()

        logger.info(f"CCI Arrows Indicator created: {self.settings}")

    def _setup_sensitivity(self):
        """Konfiguruje czułość sygnałów"""
        sensitivity = self.settings.get('arrow_sensitivity', 'medium')

        if sensitivity == 'low':
            self.signal_threshold = 0.7
            self.divergence_threshold = 0.8
            self.trend_strength_min = 0.6
        elif sensitivity == 'high':
            self.signal_threshold = 0.3
            self.divergence_threshold = 0.4
            self.trend_strength_min = 0.3
        else:  # medium
            self.signal_threshold = 0.5
            self.divergence_threshold = 0.6
            self.trend_strength_min = 0.4

    def _get_price_data(self, df: pd.DataFrame) -> np.ndarray:
        """Pobiera dane cenowe według wybranego typu"""
        price_type = self.settings.get('price_type', 'typical')

        if price_type == 'typical':
            # Typical Price (H+L+C)/3
            return (df['high'] + df['low'] + df['close']).values / 3.0
        elif price_type == 'weighted':
            # Weighted Price (H+L+C+C)/4
            return (df['high'] + df['low'] + df['close'] + df['close']).values / 4.0
        elif price_type == 'close':
            return df['close'].values
        else:
            return (df['high'] + df['low'] + df['close']).values / 3.0

    def _calculate_cci(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Oblicza CCI (Commodity Channel Index)
        CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
        """
        length = len(prices)
        cci = np.zeros(length)

        for i in range(period - 1, length):
            # Pobierz dane dla okresu
            window_prices = prices[i - period + 1:i + 1]

            # Simple Moving Average
            sma = np.mean(window_prices)

            # Mean Deviation
            mean_deviation = np.mean(np.abs(window_prices - sma))

            # CCI calculation
            if mean_deviation != 0:
                cci[i] = (prices[i] - sma) / (0.015 * mean_deviation)
            else:
                cci[i] = 0

        return cci

    def _detect_cci_signals(self, cci: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wykrywa sygnały kupna i sprzedaży na podstawie CCI

        Returns:
            Tuple[buy_signals, sell_signals] - arrays z pozycjami sygnałów
        """
        length = len(cci)
        buy_signals = np.zeros(length)
        sell_signals = np.zeros(length)

        overbought = self.settings['overbought_level']
        oversold = self.settings['oversold_level']
        min_bars = self.settings['min_bars_between_signals']

        last_signal_bar = -min_bars - 1

        for i in range(2, length):
            current_bar = i

            # Sprawdź czy minął wystarczający czas od ostatniego sygnału
            if current_bar - last_signal_bar < min_bars:
                continue

            # Sygnał kupna: CCI wychodzi z obszaru wyprzedania w górę
            if (cci[i - 2] < oversold and
                    cci[i - 1] < oversold and
                    cci[i] > oversold and
                    cci[i] > cci[i - 1]):

                # Dodatkowa walidacja - trend ceny
                if self._validate_buy_signal(prices, cci, i):
                    buy_signals[i] = prices[i]
                    last_signal_bar = current_bar

            # Sygnał sprzedaży: CCI wychodzi z obszaru wykupienia w dół
            elif (cci[i - 2] > overbought and
                  cci[i - 1] > overbought and
                  cci[i] < overbought and
                  cci[i] < cci[i - 1]):

                # Dodatkowa walidacja - trend ceny
                if self._validate_sell_signal(prices, cci, i):
                    sell_signals[i] = prices[i]
                    last_signal_bar = current_bar

        return buy_signals, sell_signals

    def _validate_buy_signal(self, prices: np.ndarray, cci: np.ndarray, index: int) -> bool:
        """Waliduje sygnał kupna - dodatkowe filtry"""
        try:
            # Sprawdź czy cena nie jest w silnym downtrend
            if index < 5:
                return True

            # Analiza ostatnich 5 świec
            recent_prices = prices[index - 4:index + 1]
            price_trend = np.polyfit(range(5), recent_prices, 1)[0]

            # Jeśli trend cenowy jest bardzo negatywny, odrzuć sygnał
            if price_trend < -0.5 * np.std(recent_prices):
                return False

            # Sprawdź momentum CCI
            cci_momentum = cci[index] - cci[index - 2]
            if cci_momentum <= 0:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating buy signal: {e}")
            return True

    def _validate_sell_signal(self, prices: np.ndarray, cci: np.ndarray, index: int) -> bool:
        """Waliduje sygnał sprzedaży - dodatkowe filtry"""
        try:
            # Sprawdź czy cena nie jest w silnym uptrend
            if index < 5:
                return True

            # Analiza ostatnich 5 świec
            recent_prices = prices[index - 4:index + 1]
            price_trend = np.polyfit(range(5), recent_prices, 1)[0]

            # Jeśli trend cenowy jest bardzo pozytywny, odrzuć sygnał
            if price_trend > 0.5 * np.std(recent_prices):
                return False

            # Sprawdź momentum CCI
            cci_momentum = cci[index] - cci[index - 2]
            if cci_momentum >= 0:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating sell signal: {e}")
            return True

    def _detect_divergences(self, cci: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wykrywa dywergencje między CCI a ceną

        Returns:
            Tuple[bullish_divergence, bearish_divergence]
        """
        if not self.settings.get('use_divergence', True):
            return np.zeros(len(cci)), np.zeros(len(cci))

        length = len(cci)
        bullish_div = np.zeros(length)
        bearish_div = np.zeros(length)

        # Minimalny odstęp dla analizy dywergencji
        min_bars = 10

        for i in range(min_bars, length - 5):
            # Bullish divergence: cena robi niższe dołki, CCI wyższe dołki
            if self._check_bullish_divergence(prices, cci, i, min_bars):
                bullish_div[i] = prices[i]

            # Bearish divergence: cena robi wyższe szczyty, CCI niższe szczyty
            if self._check_bearish_divergence(prices, cci, i, min_bars):
                bearish_div[i] = prices[i]

        return bullish_div, bearish_div

    def _check_bullish_divergence(self, prices: np.ndarray, cci: np.ndarray,
                                  current_idx: int, lookback: int) -> bool:
        """Sprawdza bullish divergence"""
        try:
            # Znajdź lokalne minima w cenach i CCI
            price_window = prices[current_idx - lookback:current_idx + 1]
            cci_window = cci[current_idx - lookback:current_idx + 1]

            # Znajdź ostatnie dwa lokalne minima
            price_mins = []
            cci_mins = []

            for i in range(2, len(price_window) - 2):
                if (price_window[i] < price_window[i - 1] and
                        price_window[i] < price_window[i + 1] and
                        price_window[i] < price_window[i - 2] and
                        price_window[i] < price_window[i + 2]):
                    price_mins.append((i, price_window[i]))
                    cci_mins.append((i, cci_window[i]))

            if len(price_mins) < 2:
                return False

            # Porównaj ostatnie dwa minima
            last_price_min = price_mins[-1][1]
            prev_price_min = price_mins[-2][1]
            last_cci_min = cci_mins[-1][1]
            prev_cci_min = cci_mins[-2][1]

            # Bullish divergence: cena niżej, CCI wyżej
            if (last_price_min < prev_price_min and
                    last_cci_min > prev_cci_min):
                return True

            return False

        except Exception:
            return False

    def _check_bearish_divergence(self, prices: np.ndarray, cci: np.ndarray,
                                  current_idx: int, lookback: int) -> bool:
        """Sprawdza bearish divergence"""
        try:
            # Znajdź lokalne maxima w cenach i CCI
            price_window = prices[current_idx - lookback:current_idx + 1]
            cci_window = cci[current_idx - lookback:current_idx + 1]

            # Znajdź ostatnie dwa lokalne maxima
            price_maxs = []
            cci_maxs = []

            for i in range(2, len(price_window) - 2):
                if (price_window[i] > price_window[i - 1] and
                        price_window[i] > price_window[i + 1] and
                        price_window[i] > price_window[i - 2] and
                        price_window[i] > price_window[i + 2]):
                    price_maxs.append((i, price_window[i]))
                    cci_maxs.append((i, cci_window[i]))

            if len(price_maxs) < 2:
                return False

            # Porównaj ostatnie dwa maxima
            last_price_max = price_maxs[-1][1]
            prev_price_max = price_maxs[-2][1]
            last_cci_max = cci_maxs[-1][1]
            prev_cci_max = cci_maxs[-2][1]

            # Bearish divergence: cena wyżej, CCI niżej
            if (last_price_max > prev_price_max and
                    last_cci_max < prev_cci_max):
                return True

            return False

        except Exception:
            return False

    def _calculate_trend_strength(self, cci: np.ndarray, index: int) -> float:
        """Oblicza siłę trendu na podstawie CCI"""
        if index < 10:
            return 0.5

        try:
            # Analiza ostatnich 10 wartości CCI
            recent_cci = cci[index - 9:index + 1]

            # Sprawdź konsystencję kierunku
            positive_count = np.sum(recent_cci > 0)
            negative_count = np.sum(recent_cci < 0)

            # Oblicz średnią i odchylenie
            cci_mean = np.mean(recent_cci)
            cci_std = np.std(recent_cci)

            # Siła trendu na podstawie konsystencji i volatility
            consistency = max(positive_count, negative_count) / 10.0
            volatility_factor = min(1.0, cci_std / 50.0)  # Normalizacja volatility

            trend_strength = consistency * (1 - volatility_factor * 0.5)

            return max(0.0, min(1.0, trend_strength))

        except Exception:
            return 0.5

    def calculate(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Główna funkcja obliczeniowa CCI Arrows
        """
        try:
            if not self.validate_data(df):
                return None

            if not self._should_recalculate(df) and self.last_result:
                return self.last_result

            # Pobierz dane
            length = len(df)
            prices = self._get_price_data(df)
            period = self.settings['cci_period']

            # Oblicz CCI
            cci_values = self._calculate_cci(prices, period)

            # Wykryj sygnały kupna/sprzedaży
            buy_signals, sell_signals = self._detect_cci_signals(cci_values, prices)

            # Wykryj dywergencje
            bullish_div, bearish_div = self._detect_divergences(cci_values, prices)

            # Oblicz poziomy CCI
            overbought_line = np.full(length, self.settings['overbought_level'])
            oversold_line = np.full(length, self.settings['oversold_level'])
            zero_line = np.zeros(length)

            # Oblicz siłę trendu dla każdego punktu
            trend_strength = np.array([
                self._calculate_trend_strength(cci_values, i)
                for i in range(length)
            ])

            # Przygotuj wynik
            result = {
                'cci': cci_values,
                'buy_arrows': buy_signals,
                'sell_arrows': sell_signals,
                'bullish_divergence': bullish_div,
                'bearish_divergence': bearish_div,
                'overbought_line': overbought_line,
                'oversold_line': oversold_line,
                'zero_line': zero_line,
                'trend_strength': trend_strength,
                'valid_from': period,
                'settings': self.settings.copy(),
                'timestamp': df.index[-1] if not df.empty else None,
                'levels': {
                    'overbought': self.settings['overbought_level'],
                    'oversold': self.settings['oversold_level'],
                    'extreme': self.settings['extreme_level']
                }
            }

            self.last_result = result
            logger.info(f"CCI Arrows calculated: {length} candles, "
                        f"{np.sum(buy_signals > 0)} buy signals, "
                        f"{np.sum(sell_signals > 0)} sell signals")

            return result

        except Exception as e:
            logger.error(f"CCI Arrows calculation error: {e}")
            return None

    def get_plot_config(self) -> Dict:
        """Konfiguracja wyświetlania CCI Arrows"""
        return {
            'main_window': False,
            'subplot': True,
            'subplot_height_ratio': 0.3,
            'colors': {
                'cci_line': '#FFD700',
                'buy_arrow': '#00FF88',
                'sell_arrow': '#FF4444',
                'bullish_divergence': '#4A90E2',
                'bearish_divergence': '#E24A90',
                'overbought_line': '#FF6B6B',
                'oversold_line': '#4ECDC4',
                'zero_line': '#999999'
            },
            'styles': {
                'cci_line': {'width': 2, 'style': 'solid'},
                'level_lines': {'width': 1, 'style': 'dashed', 'alpha': 0.7},
                'arrows': {'size': 100, 'alpha': 0.9},
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
        Analizuje najnowsze sygnały CCI Arrows

        Args:
            result: Wyniki CCI (opcjonalne)
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
                'cci_value': 0,
                'trend': 'neutral',
                'divergence': False
            }

        try:
            length = len(result['cci'])
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

            # Analiza aktualnego CCI
            current_cci = result['cci'][-1]
            prev_cci = result['cci'][-2] if length > 1 else current_cci

            # Określ trend
            if current_cci > self.settings['overbought_level']:
                trend = 'overbought'
            elif current_cci < self.settings['oversold_level']:
                trend = 'oversold'
            elif current_cci > prev_cci:
                trend = 'bullish'
            elif current_cci < prev_cci:
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
                'cci_value': current_cci,
                'cci_change': current_cci - prev_cci,
                'trend': trend,
                'divergence': divergence_detected,
                'overbought': current_cci > self.settings['overbought_level'],
                'oversold': current_cci < self.settings['oversold_level']
            }

        except Exception as e:
            logger.error(f"CCI signal analysis error: {e}")
            return {'signal': 'error', 'strength': 0}


def create_cci_arrows_indicator(timeframe: str = '5m', sensitivity: str = 'medium',
                                name: str = "CCI_Arrows") -> CCIArrowsIndicator:
    """
    Factory function dla CCI Arrows z predefiniowanymi ustawieniami

    Args:
        timeframe: Timeframe ('1m', '5m', '15m', etc.)
        sensitivity: Czułość sygnałów ('low', 'medium', 'high')
        name: Nazwa instancji wskaźnika

    Returns:
        Skonfigurowany CCIArrowsIndicator
    """

    # Konfiguracje zoptymalizowane pod różne timeframes
    configs = {
        '1m': {
            'cci_period': 20,
            'overbought_level': 150,
            'oversold_level': -150,
            'min_bars_between_signals': 2
        },
        '5m': {
            'cci_period': 14,
            'overbought_level': 100,
            'oversold_level': -100,
            'min_bars_between_signals': 3
        },
        '15m': {
            'cci_period': 14,
            'overbought_level': 100,
            'oversold_level': -100,
            'min_bars_between_signals': 2
        },
        '30m': {
            'cci_period': 12,
            'overbought_level': 100,
            'oversold_level': -100,
            'min_bars_between_signals': 2
        },
        '1h': {
            'cci_period': 10,
            'overbought_level': 100,
            'oversold_level': -100,
            'min_bars_between_signals': 1
        }
    }

    # Wybierz konfigurację
    if timeframe in configs:
        config = configs[timeframe]
    else:
        config = configs['5m']  # Domyślna

    return CCIArrowsIndicator(
        name=name,
        cci_period=config['cci_period'],
        overbought_level=config['overbought_level'],
        oversold_level=config['oversold_level'],
        extreme_level=200,
        arrow_sensitivity=sensitivity,
        use_divergence=True,
        min_bars_between_signals=config['min_bars_between_signals'],
        price_type='typical'
    )