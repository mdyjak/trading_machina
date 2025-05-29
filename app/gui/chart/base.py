# 📁 app/gui/chart/base.py
"""
Bazowe komponenty dla chart widget
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ChartDataManager:
    """
    Manager danych dla wykresu
    Odpowiada za przechowywanie i walidację danych
    """

    def __init__(self):
        self.current_df = pd.DataFrame()
        self.current_indicators = {}
        self.last_update = None

    def update_data(self, df: pd.DataFrame, indicator_results: Dict) -> bool:
        """Aktualizuje dane wykresu"""
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided")
                return False

            self.current_df = df.copy()
            self.current_indicators = indicator_results.copy()

            logger.debug(f"Chart data updated: {len(df)} candles, {len(indicator_results)} indicators")
            return True

        except Exception as e:
            logger.error(f"Error updating chart data: {e}")
            return False

    def get_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Zwraca aktualne dane"""
        return self.current_df.copy(), self.current_indicators.copy()

    def is_empty(self) -> bool:
        """Sprawdza czy dane są puste"""
        return self.current_df.empty

    def get_candle_info(self, x_index: int) -> Optional[Dict]:
        """Pobiera informacje o świecy"""
        if 0 <= x_index < len(self.current_df):
            candle = self.current_df.iloc[x_index]
            return {
                'index': x_index,
                'candle': candle,
                'datetime': candle.name if hasattr(candle, 'name') else None
            }
        return None


class ChartLayoutManager:
    """
    Manager layoutu wykresów
    Odpowiada za konfigurację subplotów
    """

    def __init__(self):
        self.subplot_config = {
            'price': {'height': 3, 'enabled': True},
            'volume': {'height': 1, 'enabled': True},
            'cci': {'height': 1, 'enabled': True}
        }

    def calculate_layout(self, show_volume: bool, show_cci: bool,
                         has_cci_data: bool) -> List[Dict]:
        """Oblicza dynamiczny layout subplotów"""
        active_subplots = []

        # Główny wykres cenowy - zawsze aktywny
        active_subplots.append({
            'type': 'price',
            'height': self.subplot_config['price']['height']
        })

        # Volume
        if show_volume and self.subplot_config['volume']['enabled']:
            active_subplots.append({
                'type': 'volume',
                'height': self.subplot_config['volume']['height']
            })

        # CCI
        if (show_cci and has_cci_data and
                self.subplot_config['cci']['enabled']):
            active_subplots.append({
                'type': 'cci',
                'height': self.subplot_config['cci']['height']
            })

        return active_subplots

    def update_config(self, subplot_type: str, **kwargs):
        """Aktualizuje konfigurację subplot"""
        if subplot_type in self.subplot_config:
            self.subplot_config[subplot_type].update(kwargs)


class ChartTooltipBuilder:
    """
    Builder dla tooltipów wykresu
    """

    def __init__(self):
        pass

    def build_tooltip(self, candle_info: Dict, indicators: Dict) -> List[str]:
        """Buduje tooltip dla świecy"""
        info_parts = []

        if not candle_info:
            return info_parts

        candle = candle_info['candle']
        x_index = candle_info['index']

        # Podstawowe OHLCV
        info_parts.extend(self._build_basic_info(candle))

        # TMA info
        tma_info = self._build_tma_info(x_index, indicators)
        if tma_info:
            info_parts.extend(tma_info)

        # CCI info
        cci_info = self._build_cci_info(x_index, indicators)
        if cci_info:
            info_parts.extend(cci_info)

        return info_parts

    def _build_basic_info(self, candle) -> List[str]:
        """Buduje podstawowe info OHLCV"""
        info = []

        # Format daty
        if hasattr(candle, 'name') and hasattr(candle.name, 'strftime'):
            time_str = candle.name.strftime('%H:%M %d/%m')
        else:
            time_str = 'N/A'

        info.extend([
            f"T: {time_str}",
            f"O: {candle['open']:.4f}",
            f"H: {candle['high']:.4f}",
            f"L: {candle['low']:.4f}",
            f"C: {candle['close']:.4f}",
            f"V: {candle['volume']:.0f}"
        ])

        return info

    def _build_tma_info(self, x_index: int, indicators: Dict) -> List[str]:
        """Buduje info TMA"""
        if 'TMA_Main' not in indicators:
            return []

        try:
            tma_data = indicators['TMA_Main']
            valid_from = tma_data.get('valid_from', 0)

            if x_index >= valid_from and x_index < len(tma_data['tma_center']):
                info = []
                info.append(f"TMA: {tma_data['tma_center'][x_index]:.4f}")
                info.append(f"Bands: {tma_data['tma_lower'][x_index]:.4f}-{tma_data['tma_upper'][x_index]:.4f}")

                # Sygnały
                if tma_data['rebound_up'][x_index] > 0:
                    info.append("🔵 TMA BUY")
                elif tma_data['rebound_down'][x_index] > 0:
                    info.append("🔴 TMA SELL")

                if tma_data['angle_caution'][x_index] > 0:
                    info.append("⚠️ TMA Caution")

                return info
        except Exception as e:
            logger.error(f"Error building TMA tooltip: {e}")

        return []

    def _build_cci_info(self, x_index: int, indicators: Dict) -> List[str]:
        """Buduje info CCI"""
        if 'CCI_Arrows_Main' not in indicators:
            return []

        try:
            cci_data = indicators['CCI_Arrows_Main']
            valid_from = cci_data.get('valid_from', 0)

            if x_index >= valid_from and x_index < len(cci_data['cci']):
                info = []
                cci_value = cci_data['cci'][x_index]
                info.append(f"CCI: {cci_value:.1f}")

                # Sygnały
                if cci_data['buy_arrows'][x_index] > 0:
                    info.append("🡱 CCI BUY")
                elif cci_data['sell_arrows'][x_index] > 0:
                    info.append("🡳 CCI SELL")

                # Dywergencje
                if cci_data['bullish_divergence'][x_index] > 0:
                    info.append("⚡ Bull Div")
                elif cci_data['bearish_divergence'][x_index] > 0:
                    info.append("⚡ Bear Div")

                return info
        except Exception as e:
            logger.error(f"Error building CCI tooltip: {e}")

        return []


class ChartFormatter:
    """
    Formatowanie osi i etykiet wykresu
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config

    def format_axes(self, axes_dict: Dict, df: pd.DataFrame, app_ref=None):
        """Formatuje wszystkie osie"""
        if df.empty:
            return

        # Format X axes
        self._format_x_axes(axes_dict, df, app_ref)

        # Format price axis
        if 'price' in axes_dict and axes_dict['price']:
            self._format_price_axis(axes_dict['price'], df)

        # Format volume axis
        if 'volume' in axes_dict and axes_dict['volume']:
            self._format_volume_axis(axes_dict['volume'], df)

        # Format CCI axis
        if 'cci' in axes_dict and axes_dict['cci']:
            self._format_cci_axis(axes_dict['cci'], df)

    def _format_x_axes(self, axes_dict: Dict, df: pd.DataFrame, app_ref=None):
        """Formatuje osie X dla wszystkich subplotów"""
        df_len = len(df)

        # Synchronizuj X dla wszystkich osi
        for ax in axes_dict.values():
            if ax:
                ax.set_xlim(-1, df_len)

        # Inteligentne etykiety czasowe tylko dla głównej osi
        if 'price' in axes_dict and axes_dict['price']:
            self._format_time_labels(axes_dict['price'], df, app_ref)

    def _format_time_labels(self, ax, df: pd.DataFrame, app_ref=None):
        """Formatuje etykiety czasowe"""
        df_len = len(df)
        if df_len == 0:
            return

        # Dynamiczna liczba etykiet
        max_labels = min(12, max(4, df_len // 20))
        step = max(1, df_len // max_labels)

        tick_positions = list(range(0, df_len, step))
        if tick_positions[-1] != df_len - 1:
            tick_positions.append(df_len - 1)

        tick_labels = []
        for pos in tick_positions:
            if pos < len(df):
                date = df.index[pos]
                if hasattr(date, 'strftime'):
                    # Format zależny od timeframe
                    timeframe = '5m'  # default
                    if app_ref and hasattr(app_ref, 'get_current_timeframe'):
                        timeframe = app_ref.get_current_timeframe()

                    if timeframe in ['1m', '5m', '15m', '30m']:
                        tick_labels.append(date.strftime('%H:%M'))
                    else:
                        tick_labels.append(date.strftime('%d/%m'))
                else:
                    tick_labels.append(f"T{pos}")
            else:
                tick_labels.append("")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=8)

    def _format_price_axis(self, ax, df: pd.DataFrame):
        """Formatuje oś cen"""
        # Auto-scale Y axis z marginesem
        price_data = df[['high', 'low']].values.flatten()
        price_min = np.nanmin(price_data)
        price_max = np.nanmax(price_data)
        margin = (price_max - price_min) * 0.03

        ax.set_ylim(price_min - margin, price_max + margin)
        ax.set_ylabel('Cena (USDT)', color=self.colors['text_primary'],
                      fontweight='bold', fontsize=10)

    def _format_volume_axis(self, ax, df: pd.DataFrame):
        """Formatuje oś volume"""
        import matplotlib.pyplot as plt

        ax.set_xlim(-1, len(df))

        # Format volume labels
        volume_max = df['volume'].max() if not df.empty else 1
        if volume_max > 1000000:
            volume_formatter = lambda x, p: f'{x / 1000000:.1f}M'
        elif volume_max > 1000:
            volume_formatter = lambda x, p: f'{x / 1000:.1f}K'
        else:
            volume_formatter = lambda x, p: f'{x:.0f}'

        ax.yaxis.set_major_formatter(plt.FuncFormatter(volume_formatter))
        ax.set_ylabel('Volume', color=self.colors['text_primary'],
                      fontweight='bold', fontsize=10)

    def _format_cci_axis(self, ax, df: pd.DataFrame):
        """Formatuje oś CCI"""
        ax.set_xlim(-1, len(df))
        ax.set_ylabel('CCI', color=self.colors['text_primary'],
                      fontweight='bold', fontsize=10)


class ChartTitleBuilder:
    """
    Builder dla tytułu wykresu
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config

    def build_title(self, ax, indicators: Dict, app_ref=None) -> str:
        """Buduje tytuł wykresu"""
        try:
            # Podstawowe info
            symbol = 'BTC/USDT'
            timeframe = '5m'
            exchange_name = "Unknown"

            if app_ref:
                if hasattr(app_ref, 'get_current_symbol'):
                    symbol = app_ref.get_current_symbol()
                if hasattr(app_ref, 'get_current_timeframe'):
                    timeframe = app_ref.get_current_timeframe()
                if hasattr(app_ref, 'exchange_manager') and app_ref.exchange_manager:
                    exchange_name = app_ref.exchange_manager.exchange_id or "Unknown"

            # Główny tytuł
            main_title = f'{symbol} - {timeframe.upper()} - {exchange_name.title()}'

            # Informacje o wskaźnikach
            indicators_info = []

            if 'TMA_Main' in indicators:
                tma_settings = indicators['TMA_Main'].get('settings', {})
                half_length = tma_settings.get('half_length', 12)
                indicators_info.append(f'TMA({half_length})')

            if 'CCI_Arrows_Main' in indicators:
                cci_settings = indicators['CCI_Arrows_Main'].get('settings', {})
                cci_period = cci_settings.get('cci_period', 14)
                indicators_info.append(f'CCI({cci_period})')

            if indicators_info:
                main_title += f' | {", ".join(indicators_info)}'

            # Ustaw tytuł
            if ax:
                ax.set_title(main_title, color=self.colors['text_primary'],
                             fontsize=12, fontweight='bold', pad=15)

            return main_title

        except Exception as e:
            logger.error(f"Error building chart title: {e}")
            return "Chart"