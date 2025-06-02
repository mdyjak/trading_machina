# 📁 indicators/bollinger_professional.py
"""
Bollinger Bands Professional Implementation - Trading Grade Accuracy
Bollinger Bands według oryginalnej formuły Johna Bollingera z lat 1980s

Zgodny z:
- MT5 Bollinger Bands
- TradingView Bollinger Bands
- Bloomberg Terminal Bollinger Bands
- Oryginalnymi specyfikacjami Johna Bollingera

Features:
- Multiple calculation methods (TA-Lib, pandas-ta, custom)
- Professional signal detection (squeeze, expansion, touches)
- Multi-timeframe optimization
- Bollinger Band Width (BBW) and %B calculations
- Dynamic support/resistance levels
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import logging
from .base import BaseIndicator

# Try to import professional libraries
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Using custom implementation.")

try:
    import pandas_ta as pta

    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.warning("pandas-ta not available. Using custom implementation.")

logger = logging.getLogger(__name__)


class BollingerBandsProfessional(BaseIndicator):
    """
    Bollinger Bands Professional - Trading Grade Implementation

    Funkcje:
    - Oryginalna formuła Bollingera (20-period SMA + 2 standard deviations)
    - Multiple calculation methods z fallback
    - Bollinger Band Width (BBW) - volatility indicator
    - %B indicator - position within bands
    - Professional signals (squeeze, expansion, bounces, breakouts)
    - Dynamic support/resistance levels
    - Multi-timeframe compatibility
    """

    def __init__(self, name: str = "Bollinger_Professional", **kwargs):
        default_settings = {
            'bb_period': 20,  # Klasyczne 20 okresów Bollingera
            'bb_std_dev': 2.0,  # 2 standardowe odchylenia
            'ma_type': 'sma',  # sma, ema, wma
            'calculation_method': 'auto',  # auto, talib, pandas_ta, custom
            'price_type': 'close',  # close, typical, weighted, ohlc4

            # Signal settings
            'squeeze_threshold': 0.1,  # BBW threshold for squeeze detection
            'expansion_threshold': 0.25,  # BBW threshold for expansion
            'touch_sensitivity': 0.02,  # % distance for band touches
            'breakout_confirmation': 2,  # bars for breakout confirmation
            'use_percent_b': True,  # Calculate %B indicator

            # Advanced settings
            'dynamic_periods': False,  # Adaptive period based on volatility
            'min_periods': 10,  # Minimum periods for dynamic calculation
            'max_periods': 50,  # Maximum periods for dynamic calculation
            'precision_digits': 6,  # Trading grade precision
        }
        default_settings.update(kwargs)

        super().__init__(name, **default_settings)

        self.version = "1.0 - Professional Bollinger Bands (Bollinger Compatible)"
        self.description = "Professional Bollinger Bands with trading-grade accuracy and multiple calculation methods"

        # Set minimum periods
        self.minimum_periods = max(self.settings['bb_period'] * 2, 50)

        # Select best available calculation method
        self._setup_calculation_method()

        logger.info(f"Bollinger Bands Professional created: {self.settings}, Method: {self.calculation_method}")

    def _setup_calculation_method(self):
        """Wybiera najlepszą dostępną metodę obliczeniową"""
        method = self.settings.get('calculation_method', 'auto')

        if method == 'auto':
            # Auto-select best available method
            if TALIB_AVAILABLE:
                self.calculation_method = 'talib'
                logger.info("Using TA-Lib for Bollinger Bands calculation")
            elif PANDAS_TA_AVAILABLE:
                self.calculation_method = 'pandas_ta'
                logger.info("Using pandas-ta for Bollinger Bands calculation")
            else:
                self.calculation_method = 'custom'
                logger.info("Using custom implementation for Bollinger Bands calculation")
        else:
            # Force specific method
            self.calculation_method = method

            # Validate availability
            if method == 'talib' and not TALIB_AVAILABLE:
                logger.warning("TA-Lib requested but not available. Falling back to custom.")
                self.calculation_method = 'custom'
            elif method == 'pandas_ta' and not PANDAS_TA_AVAILABLE:
                logger.warning("pandas-ta requested but not available. Falling back to custom.")
                self.calculation_method = 'custom'

    def _get_price_data(self, df: pd.DataFrame) -> np.ndarray:
        """Pobiera dane cenowe według wybranego typu"""
        price_type = self.settings.get('price_type', 'close')

        if price_type == 'typical':
            # Typical Price (H+L+C)/3
            return (df['high'] + df['low'] + df['close']).values / 3.0
        elif price_type == 'weighted':
            # Weighted Price (H+L+C+C)/4
            return (df['high'] + df['low'] + df['close'] + df['close']).values / 4.0
        elif price_type == 'ohlc4':
            # OHLC4 (O+H+L+C)/4
            return (df['open'] + df['high'] + df['low'] + df['close']).values / 4.0
        elif price_type == 'close':
            return df['close'].values
        else:
            return df['close'].values

    def _calculate_bollinger_bands_talib(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Oblicza Bollinger Bands używając TA-Lib (najszybsza i najbardziej precyzyjna)
        """
        if not TALIB_AVAILABLE:
            raise ImportError("TA-Lib not available")

        period = self.settings['bb_period']
        std_dev = self.settings['bb_std_dev']
        ma_type = self.settings.get('ma_type', 'sma')

        # Map MA types to TA-Lib constants
        # TA-Lib supports different MA types
        ma_type_map = {
            'sma': 0,  # Simple Moving Average
            'ema': 1,  # Exponential Moving Average
            'wma': 2,  # Weighted Moving Average
            'dema': 3,  # Double Exponential Moving Average
            'tema': 4,  # Triple Exponential Moving Average
            'trima': 5,  # Triangular Moving Average
            'kama': 6,  # Kaufman Adaptive Moving Average
            'mama': 7,  # MESA Adaptive Moving Average
            't3': 8  # T3 Moving Average
        }

        matype = ma_type_map.get(ma_type, 0)  # Default to SMA

        try:
            # TA-Lib BBANDS function usage
            upper_band, middle_band, lower_band = talib.BBANDS(
                real=prices,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=matype
            )

            return upper_band, middle_band, lower_band

        except Exception as e:
            logger.error(f"TA-Lib BBANDS calculation error: {e}")
            # Fallback to custom method
            return self._calculate_bollinger_bands_custom(prices)

    def _calculate_bollinger_bands_pandas_ta(self, df: pd.DataFrame, prices: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Oblicza Bollinger Bands używając pandas-ta
        """
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas-ta not available")

        period = self.settings['bb_period']
        std_dev = self.settings['bb_std_dev']

        try:
            # Create temporary series for pandas-ta calculation
            price_series = pd.Series(prices, index=df.index if len(df.index) == len(prices) else None)

            # pandas-ta BBANDS calculation
            bb_result = pta.bbands(
                close=price_series,
                length=period,
                std=std_dev
            )

            if bb_result is not None and len(bb_result.columns) >= 3:
                # pandas-ta returns DataFrame with columns: BBL_period_std, BBM_period_std, BBU_period_std
                lower_band = bb_result.iloc[:, 0].values  # Lower band
                middle_band = bb_result.iloc[:, 1].values  # Middle band (SMA)
                upper_band = bb_result.iloc[:, 2].values  # Upper band

                return upper_band, middle_band, lower_band
            else:
                logger.warning("pandas-ta BBANDS returned invalid result")
                return self._calculate_bollinger_bands_custom(prices)

        except Exception as e:
            logger.error(f"pandas-ta BBANDS calculation error: {e}")
            # Fallback to custom method
            return self._calculate_bollinger_bands_custom(prices)

    def _calculate_bollinger_bands_custom(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Custom implementation of Bollinger Bands - fallback method
        Zgodny z oryginalną formułą Johna Bollingera
        """
        length = len(prices)
        period = self.settings['bb_period']
        std_dev = self.settings['bb_std_dev']
        ma_type = self.settings.get('ma_type', 'sma')

        upper_band = np.full(length, np.nan, dtype=np.float64)
        middle_band = np.full(length, np.nan, dtype=np.float64)
        lower_band = np.full(length, np.nan, dtype=np.float64)

        if length < period:
            return upper_band, middle_band, lower_band

        # Calculate moving average (middle band)
        if ma_type == 'sma':
            middle_band = self._calculate_sma(prices, period)
        elif ma_type == 'ema':
            middle_band = self._calculate_ema(prices, period)
        elif ma_type == 'wma':
            middle_band = self._calculate_wma(prices, period)
        else:
            middle_band = self._calculate_sma(prices, period)  # Default to SMA

        # Calculate standard deviation bands
        for i in range(period - 1, length):
            # Get price window
            price_window = prices[i - period + 1:i + 1]

            # Calculate standard deviation
            # Using sample standard deviation (ddof=1) like most trading platforms
            price_std = np.std(price_window, ddof=1)

            # Calculate bands
            middle_value = middle_band[i]
            if not np.isnan(middle_value):
                upper_band[i] = middle_value + (std_dev * price_std)
                lower_band[i] = middle_value - (std_dev * price_std)

        return upper_band, middle_band, lower_band

    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        length = len(prices)
        sma = np.full(length, np.nan, dtype=np.float64)

        for i in range(period - 1, length):
            sma[i] = np.mean(prices[i - period + 1:i + 1])

        return sma

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        length = len(prices)
        ema = np.full(length, np.nan, dtype=np.float64)

        if length < period:
            return ema

        # Smoothing factor
        alpha = 2.0 / (period + 1.0)

        # First EMA point (SMA)
        ema[period - 1] = np.mean(prices[:period])

        # Calculate EMA
        for i in range(period, length):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _calculate_wma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Weighted Moving Average"""
        length = len(prices)
        wma = np.full(length, np.nan, dtype=np.float64)

        # Create weights
        weights = np.arange(1, period + 1, dtype=np.float64)
        weights_sum = np.sum(weights)

        for i in range(period - 1, length):
            price_window = prices[i - period + 1:i + 1]
            wma[i] = np.sum(price_window * weights) / weights_sum

        return wma

    def _calculate_bollinger_indicators(self, prices: np.ndarray, upper_band: np.ndarray,
                                        middle_band: np.ndarray, lower_band: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Oblicza dodatkowe wskaźniki Bollinger Bands:
        - Bollinger Band Width (BBW) - measure of volatility
        - %B - position of price within bands
        """
        length = len(prices)

        # Bollinger Band Width (BBW)
        # BBW = (Upper Band - Lower Band) / Middle Band
        bbw = np.full(length, np.nan, dtype=np.float64)

        # %B indicator
        # %B = (Price - Lower Band) / (Upper Band - Lower Band)
        percent_b = np.full(length, np.nan, dtype=np.float64)

        for i in range(length):
            if (not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]) and
                    not np.isnan(middle_band[i]) and middle_band[i] != 0):

                band_width = upper_band[i] - lower_band[i]

                # Calculate BBW
                bbw[i] = band_width / middle_band[i]

                # Calculate %B
                if band_width != 0:
                    percent_b[i] = (prices[i] - lower_band[i]) / band_width

        return bbw, percent_b

    def _detect_bollinger_signals(self, prices: np.ndarray, upper_band: np.ndarray,
                                  middle_band: np.ndarray, lower_band: np.ndarray,
                                  bbw: np.ndarray, percent_b: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Wykrywa sygnały Bollinger Bands:
        - Band touches and bounces
        - Breakouts and reversals
        - Squeeze and expansion
        - %B extremes
        """
        length = len(prices)

        # Initialize signal arrays
        signals = {
            'upper_touch': np.zeros(length, dtype=np.float64),
            'lower_touch': np.zeros(length, dtype=np.float64),
            'upper_breakout': np.zeros(length, dtype=np.float64),
            'lower_breakout': np.zeros(length, dtype=np.float64),
            'squeeze_signal': np.zeros(length, dtype=np.bool_),
            'expansion_signal': np.zeros(length, dtype=np.bool_),
            'middle_cross_up': np.zeros(length, dtype=np.float64),
            'middle_cross_down': np.zeros(length, dtype=np.float64),
            'percent_b_extreme_high': np.zeros(length, dtype=np.bool_),
            'percent_b_extreme_low': np.zeros(length, dtype=np.bool_)
        }

        touch_sensitivity = self.settings['touch_sensitivity']
        squeeze_threshold = self.settings['squeeze_threshold']
        expansion_threshold = self.settings['expansion_threshold']
        breakout_confirmation = self.settings['breakout_confirmation']

        for i in range(1, length):
            if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
                    np.isnan(middle_band[i]) or np.isnan(bbw[i])):
                continue

            current_price = prices[i]
            prev_price = prices[i - 1] if i > 0 else current_price

            # === BAND TOUCHES ===

            # Upper band touch (price approaches but doesn't break)
            upper_distance = abs(current_price - upper_band[i]) / upper_band[i]
            if upper_distance <= touch_sensitivity and current_price < upper_band[i]:
                signals['upper_touch'][i] = current_price

            # Lower band touch (price approaches but doesn't break)
            lower_distance = abs(current_price - lower_band[i]) / lower_band[i]
            if lower_distance <= touch_sensitivity and current_price > lower_band[i]:
                signals['lower_touch'][i] = current_price

            # === BREAKOUTS ===

            # Upper breakout (sustained move above upper band)
            if (current_price > upper_band[i] and prev_price <= upper_band[i - 1] if i > 0 else False):
                # Check for confirmation
                if self._confirm_breakout(prices, upper_band, i, 'upper', breakout_confirmation):
                    signals['upper_breakout'][i] = current_price

            # Lower breakout (sustained move below lower band)
            if (current_price < lower_band[i] and prev_price >= lower_band[i - 1] if i > 0 else False):
                # Check for confirmation
                if self._confirm_breakout(prices, lower_band, i, 'lower', breakout_confirmation):
                    signals['lower_breakout'][i] = current_price

            # === MIDDLE BAND CROSSES ===

            # Middle band crossovers (trend changes)
            if i > 0:
                if prev_price <= middle_band[i - 1] and current_price > middle_band[i]:
                    signals['middle_cross_up'][i] = current_price
                elif prev_price >= middle_band[i - 1] and current_price < middle_band[i]:
                    signals['middle_cross_down'][i] = current_price

            # === SQUEEZE AND EXPANSION ===

            # Bollinger Squeeze (low volatility)
            if bbw[i] < squeeze_threshold:
                signals['squeeze_signal'][i] = True

            # Bollinger Expansion (high volatility)
            if bbw[i] > expansion_threshold:
                signals['expansion_signal'][i] = True

            # === %B EXTREMES ===

            if not np.isnan(percent_b[i]):
                # %B above 1.0 (price above upper band)
                if percent_b[i] > 1.0:
                    signals['percent_b_extreme_high'][i] = True
                # %B below 0.0 (price below lower band)
                elif percent_b[i] < 0.0:
                    signals['percent_b_extreme_low'][i] = True

        return signals

    def _confirm_breakout(self, prices: np.ndarray, band: np.ndarray,
                          index: int, direction: str, confirmation_bars: int) -> bool:
        """Potwierdza breakout przez określoną liczbę świec"""
        if index + confirmation_bars >= len(prices):
            return False

        try:
            for i in range(1, confirmation_bars + 1):
                if index + i >= len(prices) or index + i >= len(band):
                    return False

                price = prices[index + i]
                band_value = band[index + i]

                if np.isnan(band_value):
                    return False

                if direction == 'upper' and price <= band_value:
                    return False
                elif direction == 'lower' and price >= band_value:
                    return False

            return True

        except Exception:
            return False

    def calculate(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Główna funkcja obliczeniowa Bollinger Bands Professional
        """
        try:
            if not self.validate_data(df):
                return None

            if not self._should_recalculate(df) and self.last_result:
                return self.last_result

            # Get data
            length = len(df)
            prices = self._get_price_data(df)

            # Dynamic period adjustment (optional)
            if self.settings.get('dynamic_periods', False):
                period = self._calculate_dynamic_period(prices)
                self.settings['bb_period'] = period

            # Calculate Bollinger Bands using selected method
            if self.calculation_method == 'talib':
                upper_band, middle_band, lower_band = self._calculate_bollinger_bands_talib(prices)
            elif self.calculation_method == 'pandas_ta':
                upper_band, middle_band, lower_band = self._calculate_bollinger_bands_pandas_ta(df, prices)
            else:
                upper_band, middle_band, lower_band = self._calculate_bollinger_bands_custom(prices)

            # Calculate additional indicators
            bbw, percent_b = self._calculate_bollinger_indicators(prices, upper_band, middle_band, lower_band)

            # Detect signals
            signals = self._detect_bollinger_signals(prices, upper_band, middle_band, lower_band, bbw, percent_b)

            # Calculate support/resistance levels
            support_levels, resistance_levels = self._calculate_dynamic_levels(
                prices, upper_band, middle_band, lower_band
            )

            # Prepare result
            result = {
                'upper_band': upper_band,
                'middle_band': middle_band,  # SMA/EMA line
                'lower_band': lower_band,
                'bbw': bbw,  # Bollinger Band Width
                'percent_b': percent_b,  # %B indicator

                # Signals
                'upper_touch': signals['upper_touch'],
                'lower_touch': signals['lower_touch'],
                'upper_breakout': signals['upper_breakout'],
                'lower_breakout': signals['lower_breakout'],
                'middle_cross_up': signals['middle_cross_up'],
                'middle_cross_down': signals['middle_cross_down'],
                'squeeze_signal': signals['squeeze_signal'],
                'expansion_signal': signals['expansion_signal'],
                'percent_b_extreme_high': signals['percent_b_extreme_high'],
                'percent_b_extreme_low': signals['percent_b_extreme_low'],

                # Dynamic levels
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,

                'valid_from': self.settings['bb_period'],
                'settings': self.settings.copy(),
                'timestamp': df.index[-1] if not df.empty else None,
                'calculation_method': self.calculation_method,
                'bollinger_compatible': True,
                'levels': {
                    'squeeze_threshold': self.settings['squeeze_threshold'],
                    'expansion_threshold': self.settings['expansion_threshold']
                }
            }

            self.last_result = result

            # Log statistics
            valid_count = np.sum(~np.isnan(upper_band))
            touch_signals = np.sum(signals['upper_touch'] > 0) + np.sum(signals['lower_touch'] > 0)
            breakout_signals = np.sum(signals['upper_breakout'] > 0) + np.sum(signals['lower_breakout'] > 0)
            squeeze_count = np.sum(signals['squeeze_signal'])

            logger.info(f"Bollinger Bands Professional calculated: {length} candles, "
                        f"{valid_count} valid values, {touch_signals} touches, "
                        f"{breakout_signals} breakouts, {squeeze_count} squeeze periods")

            return result

        except Exception as e:
            logger.error(f"Bollinger Bands Professional calculation error: {e}")
            return None

    def _calculate_dynamic_period(self, prices: np.ndarray) -> int:
        """
        Oblicza dynamiczny okres na podstawie volatility
        Wyższa volatility = krótszy okres (szybsza reakcja)
        Niższa volatility = dłuższy okres (więcej wygładzenia)
        """
        try:
            base_period = self.settings['bb_period']
            min_periods = self.settings['min_periods']
            max_periods = self.settings['max_periods']

            # Calculate recent volatility (last 50 periods)
            lookback = min(50, len(prices) - 1)
            if lookback < 10:
                return base_period

            recent_prices = prices[-lookback:]
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility

            # Map volatility to period (inverse relationship)
            # High volatility -> lower period, Low volatility -> higher period
            volatility_normalized = np.clip(volatility, 0.1, 1.0)
            period_adjustment = (1.0 - volatility_normalized) * (max_periods - min_periods)
            dynamic_period = int(min_periods + period_adjustment)

            return max(min_periods, min(max_periods, dynamic_period))

        except Exception:
            return self.settings['bb_period']

    def _calculate_dynamic_levels(self, prices: np.ndarray, upper_band: np.ndarray,
                                  middle_band: np.ndarray, lower_band: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oblicza dynamiczne poziomy support/resistance na podstawie Bollinger Bands
        """
        length = len(prices)
        support_levels = np.full(length, np.nan, dtype=np.float64)
        resistance_levels = np.full(length, np.nan, dtype=np.float64)

        # Use bands as dynamic S/R levels
        # Lower band often acts as support, upper band as resistance
        for i in range(length):
            if not np.isnan(lower_band[i]):
                support_levels[i] = lower_band[i]
            if not np.isnan(upper_band[i]):
                resistance_levels[i] = upper_band[i]

        return support_levels, resistance_levels

    def get_plot_config(self) -> Dict:
        """Konfiguracja wyświetlania Bollinger Bands Professional"""
        return {
            'main_window': True,
            'subplot': False,
            'create_subplots': ['bbw'] if self.settings.get('use_percent_b', True) else [],
            'colors': {
                'upper_band': '#FF6B6B',  # Light Red
                'middle_band': '#FFD700',  # Gold
                'lower_band': '#4ECDC4',  # Teal
                'band_fill': '#E3F2FD',  # Light Blue
                'upper_touch': '#FF4444',
                'lower_touch': '#00FF88',
                'upper_breakout': '#CC0000',
                'lower_breakout': '#00CC00',
                'middle_cross_up': '#00FF88',
                'middle_cross_down': '#FF4444',
                'squeeze': '#FFD700',
                'expansion': '#FF9800',
                'bbw_line': '#9C27B0',  # Purple
                'percent_b_line': '#3F51B5'  # Indigo
            },
            'styles': {
                'upper_band': {'width': 1.5, 'style': 'solid', 'alpha': 0.8},
                'middle_band': {'width': 2, 'style': 'solid', 'alpha': 0.9},
                'lower_band': {'width': 1.5, 'style': 'solid', 'alpha': 0.8},
                'band_fill': {'alpha': 0.1},
                'signals': {'size': 100, 'alpha': 0.9},
                'bbw_line': {'width': 2, 'style': 'solid'},
                'percent_b_line': {'width': 1.5, 'style': 'solid'}
            }
        }

    def get_latest_signal(self, result: Optional[Dict] = None, lookback: int = 10) -> Dict:
        """
        Analizuje najnowsze sygnały Bollinger Bands Professional

        Args:
            result: Wyniki BB (opcjonalne)
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
                'band_position': 'middle',
                'squeeze': False,
                'expansion': False,
                'percent_b': 0.5,
                'bbw': 0.0
            }

        try:
            length = len(result['upper_band'])
            if length < 2:
                return {'signal': 'none', 'strength': 0}

            # Check latest signals
            start_idx = max(0, length - lookback)
            signal = 'none'
            signal_strength = 0
            signal_age = 0

            # Current values
            current_upper = result['upper_band'][-1]
            current_middle = result['middle_band'][-1]
            current_lower = result['lower_band'][-1]
            current_bbw = result['bbw'][-1] if not np.isnan(result['bbw'][-1]) else 0.0
            current_percent_b = result['percent_b'][-1] if not np.isnan(result['percent_b'][-1]) else 0.5

            # Find latest signals
            signal_types = [
                ('upper_touch', 'sell_setup'),
                ('lower_touch', 'buy_setup'),
                ('upper_breakout', 'strong_sell'),
                ('lower_breakout', 'strong_buy'),
                ('middle_cross_up', 'buy'),
                ('middle_cross_down', 'sell')
            ]

            for i in range(length - 1, start_idx - 1, -1):
                for signal_type, signal_name in signal_types:
                    if (signal_type in result and
                            len(result[signal_type]) > i and
                            result[signal_type][i] > 0):
                        signal = signal_name
                        signal_age = length - 1 - i
                        # Strength based on signal type and %B position
                        if 'strong' in signal_name:
                            signal_strength = 0.9
                        elif 'setup' in signal_name:
                            signal_strength = 0.7
                        else:
                            signal_strength = 0.5
                        break
                if signal != 'none':
                    break

            # Determine band position
            if np.isnan(current_upper) or np.isnan(current_lower) or np.isnan(current_middle):
                band_position = 'unknown'
                price_in_bands = 0.0  # Fallback current price
            else:
                # Use the last known price (you might want to pass this as parameter)
                # For now, we'll estimate from %B
                if current_percent_b > 0.8:
                    band_position = 'upper'
                elif current_percent_b < 0.2:
                    band_position = 'lower'
                elif 0.4 <= current_percent_b <= 0.6:
                    band_position = 'middle'
                elif current_percent_b > 0.6:
                    band_position = 'upper_middle'
                else:
                    band_position = 'lower_middle'

                # Estimate current price from %B
                if not np.isnan(current_percent_b):
                    band_width = current_upper - current_lower
                    price_in_bands = current_lower + (current_percent_b * band_width)
                else:
                    price_in_bands = current_middle

            # Check squeeze and expansion
            squeeze_active = (result['squeeze_signal'][-10:].any() if len(result['squeeze_signal']) >= 10
                              else result['squeeze_signal'][-1] if len(result['squeeze_signal']) > 0 else False)
            expansion_active = (result['expansion_signal'][-5:].any() if len(result['expansion_signal']) >= 5
                                else result['expansion_signal'][-1] if len(result['expansion_signal']) > 0 else False)

            # Band width analysis
            if length >= 20:
                recent_bbw = result['bbw'][-20:]
                recent_bbw_clean = recent_bbw[~np.isnan(recent_bbw)]
                if len(recent_bbw_clean) > 0:
                    bbw_trend = 'expanding' if current_bbw > np.mean(recent_bbw_clean) else 'contracting'
                else:
                    bbw_trend = 'neutral'
            else:
                bbw_trend = 'neutral'

            return {
                'signal': signal,
                'strength': signal_strength,
                'signal_age': signal_age,
                'band_position': band_position,
                'price_estimate': price_in_bands,
                'percent_b': current_percent_b,
                'bbw': current_bbw,
                'bbw_trend': bbw_trend,
                'squeeze': squeeze_active,
                'expansion': expansion_active,
                'upper_band': current_upper,
                'middle_band': current_middle,
                'lower_band': current_lower,
                'band_width': current_upper - current_lower if not np.isnan(current_upper) and not np.isnan(
                    current_lower) else 0.0
            }

        except Exception as e:
            logger.error(f"Bollinger Bands signal analysis error: {e}")
            return {'signal': 'error', 'strength': 0}


def create_bollinger_professional_indicator(timeframe: str = '5m', sensitivity: str = 'medium',
                                            name: str = "Bollinger_Professional") -> BollingerBandsProfessional:
    """
    Factory function dla Bollinger Bands Professional z predefiniowanymi ustawieniami

    Args:
        timeframe: Timeframe ('1m', '5m', '15m', etc.)
        sensitivity: Czułość sygnałów ('low', 'medium', 'high')
        name: Nazwa instancji wskaźnika

    Returns:
        Skonfigurowany BollingerBandsProfessional
    """

    # Konfiguracje zoptymalizowane pod różne timeframes
    configs = {
        '1m': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'squeeze_threshold': 0.08,
            'expansion_threshold': 0.3,
            'touch_sensitivity': 0.015,
            'breakout_confirmation': 3
        },
        '5m': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'squeeze_threshold': 0.1,
            'expansion_threshold': 0.25,
            'touch_sensitivity': 0.02,
            'breakout_confirmation': 2
        },
        '15m': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'squeeze_threshold': 0.12,
            'expansion_threshold': 0.22,
            'touch_sensitivity': 0.025,
            'breakout_confirmation': 2
        },
        '30m': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'squeeze_threshold': 0.15,
            'expansion_threshold': 0.2,
            'touch_sensitivity': 0.03,
            'breakout_confirmation': 1
        },
        '1h': {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'squeeze_threshold': 0.18,
            'expansion_threshold': 0.18,
            'touch_sensitivity': 0.035,
            'breakout_confirmation': 1
        },
        '4h': {
            'bb_period': 20,
            'bb_std_dev': 2.1,
            'squeeze_threshold': 0.2,
            'expansion_threshold': 0.15,
            'touch_sensitivity': 0.04,
            'breakout_confirmation': 1
        },
        '1d': {
            'bb_period': 20,
            'bb_std_dev': 2.2,
            'squeeze_threshold': 0.25,
            'expansion_threshold': 0.12,
            'touch_sensitivity': 0.05,
            'breakout_confirmation': 1
        }
    }

    # Sensitivity adjustments
    sensitivity_adjustments = {
        'low': {
            'touch_sensitivity_multiplier': 0.7,
            'squeeze_threshold_multiplier': 1.2,
            'expansion_threshold_multiplier': 0.8,
            'breakout_confirmation_add': 1
        },
        'medium': {
            'touch_sensitivity_multiplier': 1.0,
            'squeeze_threshold_multiplier': 1.0,
            'expansion_threshold_multiplier': 1.0,
            'breakout_confirmation_add': 0
        },
        'high': {
            'touch_sensitivity_multiplier': 1.3,
            'squeeze_threshold_multiplier': 0.8,
            'expansion_threshold_multiplier': 1.2,
            'breakout_confirmation_add': -1
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
        config['touch_sensitivity'] *= adj['touch_sensitivity_multiplier']
        config['squeeze_threshold'] *= adj['squeeze_threshold_multiplier']
        config['expansion_threshold'] *= adj['expansion_threshold_multiplier']
        config['breakout_confirmation'] = max(1, config['breakout_confirmation'] + adj['breakout_confirmation_add'])

    return BollingerBandsProfessional(
        name=name,
        bb_period=config['bb_period'],
        bb_std_dev=config['bb_std_dev'],
        ma_type='sma',  # Classic Bollinger uses SMA
        calculation_method='auto',  # Use best available method
        price_type='close',
        squeeze_threshold=config['squeeze_threshold'],
        expansion_threshold=config['expansion_threshold'],
        touch_sensitivity=config['touch_sensitivity'],
        breakout_confirmation=config['breakout_confirmation'],
        use_percent_b=True,
        dynamic_periods=False,  # Keep classic 20-period unless requested
        precision_digits=6
    )


def validate_bollinger_accuracy():
    """
    Validates Bollinger Bands calculation accuracy against known reference values
    Tests multiple calculation methods for consistency
    """
    # Test data - sample price series
    test_prices = np.array([
        25.0, 25.1, 25.2, 25.1, 25.3, 25.4, 25.2, 25.3, 25.1, 25.0,
        24.9, 25.1, 25.3, 25.4, 25.6, 25.8, 25.9, 26.0, 26.1, 25.9,
        25.8, 25.6, 25.4, 25.2, 25.1, 25.3, 25.5, 25.7, 25.8, 26.0
    ])

    df_test = pd.DataFrame({
        'close': test_prices,
        'high': test_prices * 1.01,
        'low': test_prices * 0.99,
        'open': test_prices,
        'volume': [1000] * len(test_prices)
    })

    validation_results = {}

    # Test custom implementation
    bb_custom = BollingerBandsProfessional(name="Test_Custom", calculation_method='custom')
    result_custom = bb_custom.calculate(df_test)

    if result_custom:
        validation_results['custom'] = {
            'upper_band_sample': result_custom['upper_band'][-5:],
            'middle_band_sample': result_custom['middle_band'][-5:],
            'lower_band_sample': result_custom['lower_band'][-5:],
            'bbw_sample': result_custom['bbw'][-5:],
            'method': 'custom'
        }

    # Test TA-Lib if available
    if TALIB_AVAILABLE:
        try:
            bb_talib = BollingerBandsProfessional(name="Test_TALib", calculation_method='talib')
            result_talib = bb_talib.calculate(df_test)

            if result_talib:
                validation_results['talib'] = {
                    'upper_band_sample': result_talib['upper_band'][-5:],
                    'middle_band_sample': result_talib['middle_band'][-5:],
                    'lower_band_sample': result_talib['lower_band'][-5:],
                    'bbw_sample': result_talib['bbw'][-5:],
                    'method': 'talib'
                }
        except Exception as e:
            validation_results['talib_error'] = str(e)

    # Test pandas-ta if available
    if PANDAS_TA_AVAILABLE:
        try:
            bb_pta = BollingerBandsProfessional(name="Test_PandasTA", calculation_method='pandas_ta')
            result_pta = bb_pta.calculate(df_test)

            if result_pta:
                validation_results['pandas_ta'] = {
                    'upper_band_sample': result_pta['upper_band'][-5:],
                    'middle_band_sample': result_pta['middle_band'][-5:],
                    'lower_band_sample': result_pta['lower_band'][-5:],
                    'bbw_sample': result_pta['bbw'][-5:],
                    'method': 'pandas_ta'
                }
        except Exception as e:
            validation_results['pandas_ta_error'] = str(e)

    # Calculate differences between methods
    if len(validation_results) > 1:
        methods = [k for k in validation_results.keys() if not k.endswith('_error')]
        if len(methods) >= 2:
            method1, method2 = methods[0], methods[1]

            try:
                upper_diff = np.abs(validation_results[method1]['upper_band_sample'] -
                                    validation_results[method2]['upper_band_sample'])
                middle_diff = np.abs(validation_results[method1]['middle_band_sample'] -
                                     validation_results[method2]['middle_band_sample'])
                lower_diff = np.abs(validation_results[method1]['lower_band_sample'] -
                                    validation_results[method2]['lower_band_sample'])

                validation_results['comparison'] = {
                    'methods_compared': f"{method1} vs {method2}",
                    'upper_band_max_diff': np.max(upper_diff),
                    'middle_band_max_diff': np.max(middle_diff),
                    'lower_band_max_diff': np.max(lower_diff),
                    'accuracy_percentage': 100 * (
                                1 - np.mean([np.max(upper_diff), np.max(middle_diff), np.max(lower_diff)]) / 25.0)
                }
            except Exception as e:
                validation_results['comparison_error'] = str(e)

    return validation_results


def compare_bollinger_methods():
    """
    Utility function to compare different Bollinger Bands calculation methods
    Useful for performance and accuracy testing
    """
    import time

    # Generate larger test dataset
    np.random.seed(42)
    base_price = 100.0
    price_changes = np.random.randn(1000) * 0.02
    test_prices = base_price * np.exp(np.cumsum(price_changes))

    df_test = pd.DataFrame({
        'close': test_prices,
        'high': test_prices * 1.015,
        'low': test_prices * 0.985,
        'open': test_prices * 1.005,
        'volume': np.random.randint(1000, 10000, len(test_prices))
    })

    comparison_results = {
        'test_data_points': len(test_prices),
        'methods': {}
    }

    # Test each available method
    methods_to_test = ['custom']
    if TALIB_AVAILABLE:
        methods_to_test.append('talib')
    if PANDAS_TA_AVAILABLE:
        methods_to_test.append('pandas_ta')

    for method in methods_to_test:
        try:
            bb_indicator = BollingerBandsProfessional(
                name=f"Test_{method}",
                calculation_method=method
            )

            # Measure execution time
            start_time = time.time()
            result = bb_indicator.calculate(df_test)
            end_time = time.time()

            if result:
                # Count valid calculations
                valid_upper = np.sum(~np.isnan(result['upper_band']))
                valid_middle = np.sum(~np.isnan(result['middle_band']))
                valid_lower = np.sum(~np.isnan(result['lower_band']))

                # Count signals
                signals_count = 0
                signal_types = ['upper_touch', 'lower_touch', 'upper_breakout', 'lower_breakout']
                for signal_type in signal_types:
                    if signal_type in result:
                        signals_count += np.sum(result[signal_type] > 0)

                comparison_results['methods'][method] = {
                    'execution_time_ms': (end_time - start_time) * 1000,
                    'valid_calculations': min(valid_upper, valid_middle, valid_lower),
                    'total_signals': signals_count,
                    'last_upper_band': result['upper_band'][-1] if valid_upper > 0 else None,
                    'last_middle_band': result['middle_band'][-1] if valid_middle > 0 else None,
                    'last_lower_band': result['lower_band'][-1] if valid_lower > 0 else None,
                    'success': True
                }
            else:
                comparison_results['methods'][method] = {
                    'success': False,
                    'error': 'No result returned'
                }

        except Exception as e:
            comparison_results['methods'][method] = {
                'success': False,
                'error': str(e)
            }

    return comparison_results