# 📁 app/gui/chart/plotters.py
"""
Komponenty do rysowania różnych elementów wykresu
"""

import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CandlestickPlotter:
    """
    Rysowanie świec OHLC
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config

    def plot(self, ax, df: pd.DataFrame, show_grid: bool = True):
        """Rysuje świece na osi"""
        if df.empty or ax is None:
            return

        # Styling
        ax.set_facecolor(self.colors['bg_secondary'])

        if show_grid:
            ax.grid(True, alpha=0.25, color=self.colors['grid_color'],
                    linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.15, color=self.colors['grid_color'],
                    linestyle=':', linewidth=0.3, axis='x')

        ax.tick_params(colors=self.colors['text_primary'], labelsize=9)

        # Rysuj świece
        for i, (date, row) in enumerate(df.iterrows()):
            self._draw_single_candle(ax, i, row)

    def _draw_single_candle(self, ax, i: int, row):
        """Rysuje pojedynczą świecę"""
        # Kolory
        if row['close'] >= row['open']:
            body_color = self.colors['accent_green']
            fill_color = self.colors['bg_secondary']
            edge_color = self.colors['accent_green']
            alpha = 0.9
        else:
            body_color = self.colors['accent_red']
            fill_color = self.colors['accent_red']
            edge_color = self.colors['accent_red']
            alpha = 0.9

        # Korpus świecy
        height = abs(row['close'] - row['open'])
        bottom = min(row['open'], row['close'])

        if height > 0:
            rect = Rectangle((i - 0.35, bottom), 0.7, height,
                             facecolor=fill_color if row['close'] >= row['open'] else body_color,
                             edgecolor=edge_color, linewidth=1.0, alpha=alpha)
            ax.add_patch(rect)
        else:
            # Doji
            ax.plot([i - 0.35, i + 0.35], [row['open'], row['open']],
                    color=edge_color, linewidth=1.8, alpha=alpha)

        # Cień świecy (wick)
        ax.plot([i, i], [row['low'], row['high']],
                color=edge_color, linewidth=1.2, alpha=alpha - 0.1)


class VolumePlotter:
    """
    Rysowanie słupków volume
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config

    def plot(self, ax, df: pd.DataFrame, show_grid: bool = True):
        """Rysuje volume na osi"""
        if df.empty or ax is None:
            return

        ax.set_facecolor(self.colors['bg_secondary'])

        if show_grid:
            ax.grid(True, alpha=0.2, color=self.colors['grid_color'])

        ax.tick_params(colors=self.colors['text_primary'], labelsize=8)

        # Rysuj volume bars
        for i, (date, row) in enumerate(df.iterrows()):
            color = self.colors['accent_green'] if row['close'] >= row['open'] else self.colors['accent_red']
            # Gradient effect
            alpha = 0.7 if row['volume'] > df['volume'].median() else 0.5
            ax.bar(i, row['volume'], color=color, alpha=alpha, width=0.8)


class TMAPlotter:
    """
    Rysowanie wskaźnika TMA
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config
        self.legend_items = []

    def plot(self, ax, tma_result: Dict, df_len: int) -> List[Tuple]:
        """Rysuje TMA na osi"""
        if ax is None:
            return []

        try:
            valid_from = tma_result.get('valid_from', 20)
            if valid_from >= df_len:
                return []

            x_data = list(range(valid_from, df_len))
            if not x_data:
                return []

            self.legend_items = []

            # Główna linia TMA
            self._plot_tma_line(ax, tma_result, x_data)

            # Pasma ATR
            self._plot_tma_bands(ax, tma_result, x_data)

            # Sygnały
            self._plot_tma_signals(ax, tma_result, x_data)

            return self.legend_items

        except Exception as e:
            logger.error(f"Error plotting TMA: {e}")
            return []

    def _plot_tma_line(self, ax, tma_result: Dict, x_data: List[int]):
        """Rysuje główną linię TMA z kolorami"""
        tma_center = tma_result['tma_center']
        tma_colors = tma_result['tma_colors']

        # Segmentacja kolorów
        segments = self._create_color_segments(x_data, tma_center, tma_colors)

        for segment in segments:
            if len(segment['x']) > 1:
                color = self.colors['accent_green'] if segment['color'] == 0 else self.colors['accent_red']
                line = ax.plot(segment['x'], segment['y'], color=color,
                               linewidth=3, alpha=0.95, zorder=3)[0]

                # Dodaj do legendy tylko pierwszy segment każdego koloru
                trend_name = "TMA Up" if segment['color'] == 0 else "TMA Down"
                if not any(item[1] == trend_name for item in self.legend_items):
                    self.legend_items.append((line, trend_name))

    def _plot_tma_bands(self, ax, tma_result: Dict, x_data: List[int]):
        """Rysuje pasma ATR"""
        upper_y = [tma_result['tma_upper'][i] for i in x_data if i < len(tma_result['tma_upper'])]
        lower_y = [tma_result['tma_lower'][i] for i in x_data if i < len(tma_result['tma_lower'])]

        if len(upper_y) == len(x_data) and len(lower_y) == len(x_data):
            upper_line = ax.plot(x_data, upper_y, color=self.colors['accent_blue'],
                                 linestyle='--', alpha=0.8, linewidth=2, zorder=2)[0]
            lower_line = ax.plot(x_data, lower_y, color=self.colors['accent_pink'],
                                 linestyle='--', alpha=0.8, linewidth=2, zorder=2)[0]

            self.legend_items.extend([
                (upper_line, 'Upper Band'),
                (lower_line, 'Lower Band')
            ])

            # Fill między pasmami
            ax.fill_between(x_data, upper_y, lower_y,
                            alpha=0.08, color='cyan', zorder=1)

    def _plot_tma_signals(self, ax, tma_result: Dict, x_data: List[int]):
        """Rysuje sygnały TMA"""
        buy_signals = []
        sell_signals = []
        caution_signals = []

        for i in x_data:
            if (i >= len(tma_result['rebound_up']) or
                    i >= len(tma_result['rebound_down']) or
                    i >= len(tma_result['angle_caution'])):
                continue

            if tma_result['rebound_up'][i] > 0:
                buy_signals.append((i, tma_result['rebound_up'][i]))

            if tma_result['rebound_down'][i] > 0:
                sell_signals.append((i, tma_result['rebound_down'][i]))

            if tma_result['angle_caution'][i] > 0:
                caution_signals.append((i, tma_result['angle_caution'][i]))

        # Rysuj sygnały grupowo
        if buy_signals:
            x_buy, y_buy = zip(*buy_signals)
            buy_scatter = ax.scatter(x_buy, y_buy, marker='^',
                                     color=self.colors['accent_green'], s=150,
                                     alpha=0.95, zorder=5, edgecolors='white',
                                     linewidth=1.5)
            self.legend_items.append((buy_scatter, 'TMA Buy'))

        if sell_signals:
            x_sell, y_sell = zip(*sell_signals)
            sell_scatter = ax.scatter(x_sell, y_sell, marker='v',
                                      color=self.colors['accent_red'], s=150,
                                      alpha=0.95, zorder=5, edgecolors='white',
                                      linewidth=1.5)
            self.legend_items.append((sell_scatter, 'TMA Sell'))

        if caution_signals:
            x_caution, y_caution = zip(*caution_signals)
            caution_scatter = ax.scatter(x_caution, y_caution, marker='*',
                                         color=self.colors['accent_gold'], s=200,
                                         alpha=0.95, zorder=6, edgecolors='black',
                                         linewidth=0.8)
            self.legend_items.append((caution_scatter, 'TMA Caution'))

    def _create_color_segments(self, x_data: List[int], tma_center, tma_colors) -> List[Dict]:
        """Tworzy segmenty kolorów dla smooth rendering"""
        segments = []
        current_segment = {'x': [], 'y': [], 'color': None}

        for i in x_data:
            if i >= len(tma_center) or i >= len(tma_colors):
                continue

            color = tma_colors[i]

            if current_segment['color'] is None:
                current_segment['color'] = color

            if color == current_segment['color']:
                current_segment['x'].append(i)
                current_segment['y'].append(tma_center[i])
            else:
                if current_segment['x']:
                    segments.append(current_segment.copy())
                current_segment = {'x': [i], 'y': [tma_center[i]], 'color': color}

        # Dodaj ostatni segment
        if current_segment['x']:
            segments.append(current_segment)

        return segments


class CCIPlotter:
    """
    Rysowanie wskaźnika CCI Arrows
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config
        self.legend_items = []

    def plot(self, ax, cci_result: Dict, df_len: int) -> List[Tuple]:
        """Rysuje CCI na osi"""
        if ax is None:
            return []

        try:
            valid_from = cci_result.get('valid_from', 14)
            if valid_from >= df_len:
                return []

            ax.set_facecolor(self.colors['bg_secondary'])
            ax.tick_params(colors=self.colors['text_primary'], labelsize=8)

            x_data = list(range(valid_from, df_len))
            if not x_data:
                return []

            self.legend_items = []

            # Główna linia CCI
            self._plot_cci_line(ax, cci_result, x_data)

            # Poziomy referencyjne
            self._plot_cci_levels(ax, cci_result)

            # Sygnały
            self._plot_cci_signals(ax, cci_result, x_data)

            # Formatowanie osi
            self._format_cci_axis(ax, cci_result, x_data, df_len)

            return self.legend_items

        except Exception as e:
            logger.error(f"Error plotting CCI: {e}")
            return []

    def _plot_cci_line(self, ax, cci_result: Dict, x_data: List[int]):
        """Rysuje główną linię CCI"""
        cci_values = cci_result['cci']
        cci_line = [cci_values[i] for i in x_data if i < len(cci_values)]

        if len(cci_line) == len(x_data):
            line = ax.plot(x_data, cci_line, color='#FFD700',
                           linewidth=2.5, alpha=0.9, zorder=3)[0]
            self.legend_items.append((line, 'CCI'))

    def _plot_cci_levels(self, ax, cci_result: Dict):
        """Rysuje poziomy referencyjne CCI"""
        levels = cci_result.get('levels', {})
        overbought = levels.get('overbought', 100)
        oversold = levels.get('oversold', -100)

        ob_line = ax.axhline(y=overbought, color='#FF6B6B', linestyle='--',
                             alpha=0.8, linewidth=1.5, zorder=2)
        os_line = ax.axhline(y=oversold, color='#4ECDC4', linestyle='--',
                             alpha=0.8, linewidth=1.5, zorder=2)
        zero_line = ax.axhline(y=0, color='#999999', linestyle='-',
                               alpha=0.6, linewidth=1)

        self.legend_items.extend([
            (ob_line, f'Overbought ({overbought})'),
            (os_line, f'Oversold ({oversold})')
        ])

    def _plot_cci_signals(self, ax, cci_result: Dict, x_data: List[int]):
        """Rysuje sygnały CCI"""
        cci_values = cci_result['cci']

        buy_signals = []
        sell_signals = []
        bull_div = []
        bear_div = []

        for i in x_data:
            if i >= len(cci_result['buy_arrows']):
                continue

            cci_value = cci_values[i] if i < len(cci_values) else 0

            if cci_result['buy_arrows'][i] > 0:
                buy_signals.append((i, cci_value))

            if cci_result['sell_arrows'][i] > 0:
                sell_signals.append((i, cci_value))

            if cci_result['bullish_divergence'][i] > 0:
                bull_div.append((i, cci_value))

            if cci_result['bearish_divergence'][i] > 0:
                bear_div.append((i, cci_value))

        # Rysuj sygnały grupowo
        if buy_signals:
            x_buy, y_buy = zip(*buy_signals)
            buy_scatter = ax.scatter(x_buy, y_buy, marker='^',
                                     color=self.colors['accent_green'], s=150,
                                     alpha=0.9, zorder=5, edgecolors='white',
                                     linewidth=1.5)
            self.legend_items.append((buy_scatter, 'CCI Buy'))

        if sell_signals:
            x_sell, y_sell = zip(*sell_signals)
            sell_scatter = ax.scatter(x_sell, y_sell, marker='v',
                                      color=self.colors['accent_red'], s=150,
                                      alpha=0.9, zorder=5, edgecolors='white',
                                      linewidth=1.5)
            self.legend_items.append((sell_scatter, 'CCI Sell'))

        if bull_div:
            x_bull, y_bull = zip(*bull_div)
            bull_scatter = ax.scatter(x_bull, y_bull, marker='o',
                                      color=self.colors['accent_blue'], s=100,
                                      alpha=0.8, zorder=4)
            self.legend_items.append((bull_scatter, 'Bullish Divergence'))

        if bear_div:
            x_bear, y_bear = zip(*bear_div)
            bear_scatter = ax.scatter(x_bear, y_bear, marker='o',
                                      color=self.colors['accent_pink'], s=100,
                                      alpha=0.8, zorder=4)
            self.legend_items.append((bear_scatter, 'Bearish Divergence'))

    def _format_cci_axis(self, ax, cci_result: Dict, x_data: List[int], df_len: int):
        """Formatuje oś CCI"""
        ax.set_xlim(-1, df_len)

        # Inteligentne skalowanie Y
        cci_values = cci_result['cci']
        cci_line = [cci_values[i] for i in x_data if i < len(cci_values)]

        if cci_line:
            levels = cci_result.get('levels', {})
            overbought = levels.get('overbought', 100)
            oversold = levels.get('oversold', -100)

            cci_min = min(cci_line)
            cci_max = max(cci_line)
            margin = max(50, (cci_max - cci_min) * 0.1)

            ax.set_ylim(min(cci_min - margin, oversold - 50),
                        max(cci_max + margin, overbought + 50))


class SignalPlotter:
    """
    Rysowanie różnych typów sygnałów handlowych
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config

    def plot_buy_signals(self, ax, x_positions: List[int], y_positions: List[float],
                         label: str = "Buy Signal") -> Optional[object]:
        """Rysuje sygnały kupna"""
        if not x_positions or not y_positions:
            return None

        scatter = ax.scatter(x_positions, y_positions, marker='^',
                             color=self.colors['accent_green'], s=150,
                             alpha=0.9, zorder=5, edgecolors='white',
                             linewidth=1.5, label=label)
        return scatter

    def plot_sell_signals(self, ax, x_positions: List[int], y_positions: List[float],
                          label: str = "Sell Signal") -> Optional[object]:
        """Rysuje sygnały sprzedaży"""
        if not x_positions or not y_positions:
            return None

        scatter = ax.scatter(x_positions, y_positions, marker='v',
                             color=self.colors['accent_red'], s=150,
                             alpha=0.9, zorder=5, edgecolors='white',
                             linewidth=1.5, label=label)
        return scatter

    def plot_warning_signals(self, ax, x_positions: List[int], y_positions: List[float],
                             label: str = "Warning") -> Optional[object]:
        """Rysuje sygnały ostrzeżenia"""
        if not x_positions or not y_positions:
            return None

        scatter = ax.scatter(x_positions, y_positions, marker='*',
                             color=self.colors['accent_gold'], s=200,
                             alpha=0.95, zorder=6, edgecolors='black',
                             linewidth=0.8, label=label)
        return scatter

    def plot_divergence_signals(self, ax, x_positions: List[int], y_positions: List[float],
                                divergence_type: str = "bullish") -> Optional[object]:
        """Rysuje sygnały dywergencji"""
        if not x_positions or not y_positions:
            return None

        color = self.colors['accent_blue'] if divergence_type == 'bullish' else self.colors['accent_pink']
        label = f"{divergence_type.title()} Divergence"

        scatter = ax.scatter(x_positions, y_positions, marker='o',
                             color=color, s=100, alpha=0.8, zorder=4, label=label)
        return scatter


class GridPlotter:
    """
    Rysowanie siatki wykresu
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config

    def plot_grid(self, ax, show_grid: bool = True, style: str = 'standard'):
        """Rysuje siatkę na osi"""
        if not show_grid or ax is None:
            return

        if style == 'enhanced':
            # Enhanced grid z różnymi poziomami alpha
            ax.grid(True, alpha=0.25, color=self.colors['grid_color'],
                    linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.15, color=self.colors['grid_color'],
                    linestyle=':', linewidth=0.3, axis='x')
        else:
            # Standard grid
            ax.grid(True, alpha=0.3, color=self.colors['grid_color'],
                    linestyle='-', linewidth=0.5)


class LevelPlotter:
    """
    Rysowanie poziomów horyzontalnych (support/resistance, overbought/oversold)
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config

    def plot_horizontal_level(self, ax, y_level: float, color: str,
                              linestyle: str = '--', alpha: float = 0.7,
                              linewidth: float = 1.5, label: str = None) -> object:
        """Rysuje poziom horyzontalny"""
        line = ax.axhline(y=y_level, color=color, linestyle=linestyle,
                          alpha=alpha, linewidth=linewidth, label=label)
        return line

    def plot_overbought_oversold(self, ax, overbought: float, oversold: float) -> Tuple[object, object]:
        """Rysuje poziomy OB/OS"""
        ob_line = self.plot_horizontal_level(
            ax, overbought, '#FF6B6B', label=f'Overbought ({overbought})'
        )
        os_line = self.plot_horizontal_level(
            ax, oversold, '#4ECDC4', label=f'Oversold ({oversold})'
        )
        return ob_line, os_line

    def plot_zero_line(self, ax) -> object:
        """Rysuje linię zera"""
        return self.plot_horizontal_level(
            ax, 0, '#999999', linestyle='-', alpha=0.6, linewidth=1
        )


class FillPlotter:
    """
    Rysowanie wypełnień (fill_between)
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config

    def fill_between_bands(self, ax, x_data: List[int], upper_data: List[float],
                           lower_data: List[float], alpha: float = 0.1,
                           color: str = 'cyan') -> object:
        """Wypełnia obszar między pasmami"""
        fill = ax.fill_between(x_data, upper_data, lower_data,
                               alpha=alpha, color=color, zorder=1)
        return fill

    def fill_volume_gradient(self, ax, x_data: List[int], volume_data: List[float],
                             colors: List[str], alphas: List[float]) -> List[object]:
        """Wypełnia volume z gradientem"""
        fills = []
        for i, (x, vol, color, alpha) in enumerate(zip(x_data, volume_data, colors, alphas)):
            fill = ax.bar(x, vol, color=color, alpha=alpha, width=0.8)
            fills.append(fill)
        return fills

class EMAPlotter:
    """
    Rysowanie wskaźnika EMA Crossover
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config
        self.legend_items = []

    def plot(self, ax, ema_result: Dict, df_len: int) -> List[Tuple]:
        """Rysuje EMA Crossover na osi"""
        if ax is None:
            return []

        try:
            valid_from = ema_result.get('valid_from', 26)
            if valid_from >= df_len:
                return []

            x_data = list(range(valid_from, df_len))
            if not x_data:
                return []

            self.legend_items = []

            # Linie EMA
            self._plot_ema_lines(ax, ema_result, x_data)

            # Obszary trendu
            self._plot_trend_areas(ax, ema_result, x_data)

            # Sygnały crossover
            self._plot_crossover_signals(ax, ema_result, x_data)

            return self.legend_items

        except Exception as e:
            logger.error(f"Error plotting EMA: {e}")
            return []

    def _plot_ema_lines(self, ax, ema_result: Dict, x_data: List[int]):
        """Rysuje linie EMA"""
        fast_ema = ema_result['fast_ema']
        slow_ema = ema_result['slow_ema']
        signal_ema = ema_result['signal_ema']

        # Fast EMA
        fast_y = [fast_ema[i] for i in x_data if i < len(fast_ema)]
        if len(fast_y) == len(x_data):
            fast_line = ax.plot(x_data, fast_y, color='#FFD700',
                                linewidth=2.5, alpha=0.9, zorder=3)[0]
            self.legend_items.append((fast_line, 'Fast EMA'))

        # Slow EMA
        slow_y = [slow_ema[i] for i in x_data if i < len(slow_ema)]
        if len(slow_y) == len(x_data):
            slow_line = ax.plot(x_data, slow_y, color='#87CEEB',
                                linewidth=2.5, alpha=0.9, zorder=3)[0]
            self.legend_items.append((slow_line, 'Slow EMA'))

        # Signal EMA (jeśli używana)
        if np.any(signal_ema > 0):
            signal_y = [signal_ema[i] for i in x_data if i < len(signal_ema)]
            if len(signal_y) == len(x_data):
                signal_line = ax.plot(x_data, signal_y, color='#DDA0DD',
                                      linestyle='--', linewidth=2, alpha=0.8, zorder=2)[0]
                self.legend_items.append((signal_line, 'Signal EMA'))

    def _plot_trend_areas(self, ax, ema_result: Dict, x_data: List[int]):
        """Rysuje obszary trendu"""
        bullish_areas = ema_result.get('bullish_areas', [])
        bearish_areas = ema_result.get('bearish_areas', [])

        # Bullish areas
        for start, end in bullish_areas:
            if start < len(x_data) and end < len(x_data):
                ax.axvspan(start, end, alpha=0.1, color='#00FF88', zorder=1)

        # Bearish areas
        for start, end in bearish_areas:
            if start < len(x_data) and end < len(x_data):
                ax.axvspan(start, end, alpha=0.1, color='#FF6B6B', zorder=1)

    def _plot_crossover_signals(self, ax, ema_result: Dict, x_data: List[int]):
        """Rysuje sygnały crossover"""
        buy_signals = []
        sell_signals = []

        for i in x_data:
            if i < len(ema_result['buy_signals']) and ema_result['buy_signals'][i] > 0:
                buy_signals.append((i, ema_result['buy_signals'][i]))

            if i < len(ema_result['sell_signals']) and ema_result['sell_signals'][i] > 0:
                sell_signals.append((i, ema_result['sell_signals'][i]))

        # Buy signals
        if buy_signals:
            x_buy, y_buy = zip(*buy_signals)
            buy_scatter = ax.scatter(x_buy, y_buy, marker='^',
                                     color='#00FF88', s=150,
                                     alpha=0.9, zorder=5, edgecolors='white',
                                     linewidth=1.5)
            self.legend_items.append((buy_scatter, 'EMA Buy'))

        # Sell signals
        if sell_signals:
            x_sell, y_sell = zip(*sell_signals)
            sell_scatter = ax.scatter(x_sell, y_sell, marker='v',
                                      color='#FF4444', s=150,
                                      alpha=0.9, zorder=5, edgecolors='white',
                                      linewidth=1.5)
            self.legend_items.append((sell_scatter, 'EMA Sell'))

class SMIPlotter:
    """
    Rysowanie wskaźnika SMI Arrows
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config
        self.legend_items = []

    def plot(self, ax, smi_result: Dict, df_len: int) -> List[Tuple]:
        """Rysuje SMI na osi"""
        if ax is None:
            return []

        try:
            valid_from = smi_result.get('valid_from', 20)
            if valid_from >= df_len:
                return []

            ax.set_facecolor(self.colors['bg_secondary'])
            ax.tick_params(colors=self.colors['text_primary'], labelsize=8)

            x_data = list(range(valid_from, df_len))
            if not x_data:
                return []

            self.legend_items = []

            # Główna linia SMI
            self._plot_smi_line(ax, smi_result, x_data)

            # Signal line
            self._plot_signal_line(ax, smi_result, x_data)

            # Poziomy referencyjne
            self._plot_smi_levels(ax, smi_result)

            # Sygnały
            self._plot_smi_signals(ax, smi_result, x_data)

            # Formatowanie osi
            self._format_smi_axis(ax, smi_result, x_data, df_len)

            return self.legend_items

        except Exception as e:
            logger.error(f"Error plotting SMI: {e}")
            return []

    def _plot_smi_line(self, ax, smi_result: Dict, x_data: List[int]):
        """Rysuje główną linię SMI"""
        smi_values = smi_result['smi']
        smi_line = [smi_values[i] for i in x_data if i < len(smi_values)]

        if len(smi_line) == len(x_data):
            line = ax.plot(x_data, smi_line, color='#FF6B35',  # Orange-Red
                           linewidth=2.8, alpha=0.95, zorder=3)[0]
            self.legend_items.append((line, 'SMI'))

    def _plot_signal_line(self, ax, smi_result: Dict, x_data: List[int]):
        """Rysuje signal line SMI"""
        signal_line = smi_result['signal_line']
        signal_data = [signal_line[i] for i in x_data if i < len(signal_line)]

        if len(signal_data) == len(x_data):
            line = ax.plot(x_data, signal_data, color='#9B59B6',  # Purple
                           linestyle='--', linewidth=2, alpha=0.8, zorder=2)[0]
            self.legend_items.append((line, 'SMI Signal'))

    def _plot_smi_levels(self, ax, smi_result: Dict):
        """Rysuje poziomy referencyjne SMI"""
        levels = smi_result.get('levels', {})
        overbought = levels.get('overbought', 40)
        oversold = levels.get('oversold', -40)

        ob_line = ax.axhline(y=overbought, color='#E67E22', linestyle=':',  # Dark Orange
                             alpha=0.8, linewidth=1.8, zorder=2)
        os_line = ax.axhline(y=oversold, color='#16A085', linestyle=':',  # Teal
                             alpha=0.8, linewidth=1.8, zorder=2)
        zero_line = ax.axhline(y=0, color='#7F8C8D', linestyle='-',  # Gray
                               alpha=0.5, linewidth=1.2)

        self.legend_items.extend([
            (ob_line, f'Overbought ({overbought})'),
            (os_line, f'Oversold ({oversold})')
        ])

    def _plot_smi_signals(self, ax, smi_result: Dict, x_data: List[int]):
        """Rysuje sygnały SMI"""
        smi_values = smi_result['smi']

        buy_signals = []
        sell_signals = []
        bull_div = []
        bear_div = []

        for i in x_data:
            if i >= len(smi_result['buy_arrows']):
                continue

            smi_value = smi_values[i] if i < len(smi_values) else 0

            if smi_result['buy_arrows'][i] > 0:
                buy_signals.append((i, smi_value))

            if smi_result['sell_arrows'][i] > 0:
                sell_signals.append((i, smi_value))

            if smi_result['bullish_divergence'][i] > 0:
                bull_div.append((i, smi_value))

            if smi_result['bearish_divergence'][i] > 0:
                bear_div.append((i, smi_value))

        # Rysuj sygnały grupowo z nowymi symbolami
        if buy_signals:
            x_buy, y_buy = zip(*buy_signals)
            buy_scatter = ax.scatter(x_buy, y_buy, marker='D',  # Diamond
                                     color='#27AE60', s=140,  # Emerald Green
                                     alpha=0.95, zorder=5, edgecolors='white',
                                     linewidth=1.8)
            self.legend_items.append((buy_scatter, 'SMI Buy'))

        if sell_signals:
            x_sell, y_sell = zip(*sell_signals)
            sell_scatter = ax.scatter(x_sell, y_sell, marker='s',  # Square
                                      color='#C0392B', s=140,  # Dark Red
                                      alpha=0.95, zorder=5, edgecolors='white',
                                      linewidth=1.8)
            self.legend_items.append((sell_scatter, 'SMI Sell'))

        if bull_div:
            x_bull, y_bull = zip(*bull_div)
            bull_scatter = ax.scatter(x_bull, y_bull, marker='P',  # Plus filled
                                      color='#3498DB', s=120,  # Blue
                                      alpha=0.85, zorder=4)
            self.legend_items.append((bull_scatter, 'Bullish Div'))

        if bear_div:
            x_bear, y_bear = zip(*bear_div)
            bear_scatter = ax.scatter(x_bear, y_bear, marker='X',  # X filled
                                      color='#E74C3C', s=120,  # Red
                                      alpha=0.85, zorder=4)
            self.legend_items.append((bear_scatter, 'Bearish Div'))

    def _format_smi_axis(self, ax, smi_result: Dict, x_data: List[int], df_len: int):
        """Formatuje oś SMI"""
        ax.set_xlim(-1, df_len)

        # Inteligentne skalowanie Y
        smi_values = smi_result['smi']
        smi_line = [smi_values[i] for i in x_data if i < len(smi_values)]

        if smi_line:
            levels = smi_result.get('levels', {})
            overbought = levels.get('overbought', 40)
            oversold = levels.get('oversold', -40)

            smi_min = min(smi_line)
            smi_max = max(smi_line)
            margin = max(15, (smi_max - smi_min) * 0.1)

            ax.set_ylim(min(smi_min - margin, oversold - 20),
                        max(smi_max + margin, overbought + 20))

class RSIPlotter:
    """
    Rysowanie wskaźnika RSI Professional
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config
        self.legend_items = []

    def plot(self, ax, rsi_result: Dict, df_len: int) -> List[Tuple]:
        """Rysuje RSI na osi"""
        if ax is None:
            return []

        try:
            valid_from = rsi_result.get('valid_from', 15)
            if valid_from >= df_len:
                return []

            ax.set_facecolor(self.colors['bg_secondary'])
            ax.tick_params(colors=self.colors['text_primary'], labelsize=8)

            x_data = list(range(valid_from, df_len))
            if not x_data:
                return []

            self.legend_items = []

            # Główna linia RSI
            self._plot_rsi_line(ax, rsi_result, x_data)

            # Poziomy referencyjne
            self._plot_rsi_levels(ax, rsi_result)

            # Sygnały
            self._plot_rsi_signals(ax, rsi_result, x_data)

            # Formatowanie osi
            self._format_rsi_axis(ax, rsi_result, x_data, df_len)

            return self.legend_items

        except Exception as e:
            logger.error(f"Error plotting RSI: {e}")
            return []

    def _plot_rsi_line(self, ax, rsi_result: Dict, x_data: List[int]):
        """Rysuje główną linię RSI"""
        rsi_values = rsi_result['rsi']
        rsi_line = [rsi_values[i] for i in x_data if i < len(rsi_values) and not np.isnan(rsi_values[i])]
        x_clean = [x_data[i] for i in range(len(x_data)) if i < len(rsi_values) and not np.isnan(rsi_values[x_data[i]])]

        if len(rsi_line) == len(x_clean) and len(rsi_line) > 0:
            line = ax.plot(x_clean, rsi_line, color='#FFD700',
                           linewidth=2.5, alpha=0.9, zorder=3)[0]
            self.legend_items.append((line, 'RSI'))

    def _plot_rsi_levels(self, ax, rsi_result: Dict):
        """Rysuje poziomy referencyjne RSI"""
        levels = rsi_result.get('levels', {})
        overbought = levels.get('overbought', 70)
        oversold = levels.get('oversold', 30)
        extreme_ob = levels.get('extreme_overbought', 80)
        extreme_os = levels.get('extreme_oversold', 20)

        # Standard levels
        ob_line = ax.axhline(y=overbought, color='#FF6B6B', linestyle='--',
                             alpha=0.8, linewidth=1.5, zorder=2)
        os_line = ax.axhline(y=oversold, color='#4ECDC4', linestyle='--',
                             alpha=0.8, linewidth=1.5, zorder=2)

        # Extreme levels
        ext_ob_line = ax.axhline(y=extreme_ob, color='#CC0000', linestyle=':',
                                 alpha=0.9, linewidth=1.8, zorder=2)
        ext_os_line = ax.axhline(y=extreme_os, color='#00CC00', linestyle=':',
                                 alpha=0.9, linewidth=1.8, zorder=2)

        # Midline
        mid_line = ax.axhline(y=50, color='#999999', linestyle='-',
                              alpha=0.6, linewidth=1)

        self.legend_items.extend([
            (ob_line, f'Overbought ({overbought})'),
            (os_line, f'Oversold ({oversold})'),
            (ext_ob_line, f'Extreme OB ({extreme_ob})'),
            (ext_os_line, f'Extreme OS ({extreme_os})')
        ])

    def _plot_rsi_signals(self, ax, rsi_result: Dict, x_data: List[int]):
        """Rysuje sygnały RSI"""
        rsi_values = rsi_result['rsi']

        buy_signals = []
        sell_signals = []
        bull_div = []
        bear_div = []

        for i in x_data:
            if i >= len(rsi_result['buy_arrows']):
                continue

            rsi_value = rsi_values[i] if i < len(rsi_values) and not np.isnan(rsi_values[i]) else None
            if rsi_value is None:
                continue

            if rsi_result['buy_arrows'][i] > 0:
                buy_signals.append((i, rsi_value))

            if rsi_result['sell_arrows'][i] > 0:
                sell_signals.append((i, rsi_value))

            if rsi_result['bullish_divergence'][i] > 0:
                bull_div.append((i, rsi_value))

            if rsi_result['bearish_divergence'][i] > 0:
                bear_div.append((i, rsi_value))

        # Rysuj sygnały grupowo
        if buy_signals:
            x_buy, y_buy = zip(*buy_signals)
            buy_scatter = ax.scatter(x_buy, y_buy, marker='^',
                                     color=self.colors['accent_green'], s=150,
                                     alpha=0.9, zorder=5, edgecolors='white',
                                     linewidth=1.5)
            self.legend_items.append((buy_scatter, 'RSI Buy'))

        if sell_signals:
            x_sell, y_sell = zip(*sell_signals)
            sell_scatter = ax.scatter(x_sell, y_sell, marker='v',
                                      color=self.colors['accent_red'], s=150,
                                      alpha=0.9, zorder=5, edgecolors='white',
                                      linewidth=1.5)
            self.legend_items.append((sell_scatter, 'RSI Sell'))

        if bull_div:
            x_bull, y_bull = zip(*bull_div)
            bull_scatter = ax.scatter(x_bull, y_bull, marker='o',
                                      color=self.colors['accent_blue'], s=100,
                                      alpha=0.8, zorder=4)
            self.legend_items.append((bull_scatter, 'Bullish Divergence'))

        if bear_div:
            x_bear, y_bear = zip(*bear_div)
            bear_scatter = ax.scatter(x_bear, y_bear, marker='o',
                                      color=self.colors['accent_pink'], s=100,
                                      alpha=0.8, zorder=4)
            self.legend_items.append((bear_scatter, 'Bearish Divergence'))

    def _format_rsi_axis(self, ax, rsi_result: Dict, x_data: List[int], df_len: int):
        """Formatuje oś RSI"""
        ax.set_xlim(-1, df_len)
        ax.set_ylim(0, 100)  # RSI zawsze 0-100
        ax.set_ylabel('RSI', color=self.colors['text_primary'])


class BollingerPlotter:
    """
    Rysowanie wskaźnika Bollinger Bands Professional
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config
        self.legend_items = []

    def plot(self, ax, bb_result: Dict, df_len: int) -> List[Tuple]:
        """Rysuje Bollinger Bands na osi"""
        if ax is None:
            return []

        try:
            valid_from = bb_result.get('valid_from', 20)
            if valid_from >= df_len:
                return []

            x_data = list(range(valid_from, df_len))
            if not x_data:
                return []

            self.legend_items = []

            # Pasma Bollinger
            self._plot_bollinger_bands(ax, bb_result, x_data)

            # Fill między pasmami
            self._plot_band_fill(ax, bb_result, x_data)

            # Sygnały
            self._plot_bollinger_signals(ax, bb_result, x_data)

            return self.legend_items

        except Exception as e:
            logger.error(f"Error plotting Bollinger Bands: {e}")
            return []

    def _plot_bollinger_bands(self, ax, bb_result: Dict, x_data: List[int]):
        """Rysuje pasma Bollinger"""
        upper_band = bb_result['upper_band']
        middle_band = bb_result['middle_band']
        lower_band = bb_result['lower_band']

        # Upper Band
        upper_y = [upper_band[i] for i in x_data if i < len(upper_band) and not np.isnan(upper_band[i])]
        upper_x = [x_data[i] for i in range(len(x_data)) if i < len(upper_band) and not np.isnan(upper_band[x_data[i]])]

        if len(upper_y) > 0:
            upper_line = ax.plot(upper_x, upper_y, color='#FF6B6B',
                                 linewidth=1.5, alpha=0.8, zorder=3)[0]
            self.legend_items.append((upper_line, 'Upper Band'))

        # Middle Band (SMA)
        middle_y = [middle_band[i] for i in x_data if i < len(middle_band) and not np.isnan(middle_band[i])]
        middle_x = [x_data[i] for i in range(len(x_data)) if i < len(middle_band) and not np.isnan(middle_band[x_data[i]])]

        if len(middle_y) > 0:
            middle_line = ax.plot(middle_x, middle_y, color='#FFD700',
                                  linewidth=2, alpha=0.9, zorder=3)[0]
            self.legend_items.append((middle_line, 'Middle Band (SMA)'))

        # Lower Band
        lower_y = [lower_band[i] for i in x_data if i < len(lower_band) and not np.isnan(lower_band[i])]
        lower_x = [x_data[i] for i in range(len(x_data)) if i < len(lower_band) and not np.isnan(lower_band[x_data[i]])]

        if len(lower_y) > 0:
            lower_line = ax.plot(lower_x, lower_y, color='#4ECDC4',
                                 linewidth=1.5, alpha=0.8, zorder=3)[0]
            self.legend_items.append((lower_line, 'Lower Band'))

    def _plot_band_fill(self, ax, bb_result: Dict, x_data: List[int]):
        """Wypełnia obszar między pasmami"""
        upper_band = bb_result['upper_band']
        lower_band = bb_result['lower_band']

        # Przygotuj dane do fill_between
        valid_indices = []
        upper_values = []
        lower_values = []

        for i in x_data:
            if (i < len(upper_band) and i < len(lower_band) and
                    not np.isnan(upper_band[i]) and not np.isnan(lower_band[i])):
                valid_indices.append(i)
                upper_values.append(upper_band[i])
                lower_values.append(lower_band[i])

        if len(valid_indices) > 0:
            ax.fill_between(valid_indices, upper_values, lower_values,
                            alpha=0.1, color='#E3F2FD', zorder=1)

    def _plot_bollinger_signals(self, ax, bb_result: Dict, x_data: List[int]):
        """Rysuje sygnały Bollinger Bands"""
        # Touch signals
        upper_touch_signals = []
        lower_touch_signals = []
        upper_breakout_signals = []
        lower_breakout_signals = []

        for i in x_data:
            # Upper touches (sell setup)
            if (i < len(bb_result['upper_touch']) and
                bb_result['upper_touch'][i] > 0):
                upper_touch_signals.append((i, bb_result['upper_touch'][i]))

            # Lower touches (buy setup)
            if (i < len(bb_result['lower_touch']) and
                bb_result['lower_touch'][i] > 0):
                lower_touch_signals.append((i, bb_result['lower_touch'][i]))

            # Upper breakouts (strong sell)
            if (i < len(bb_result['upper_breakout']) and
                bb_result['upper_breakout'][i] > 0):
                upper_breakout_signals.append((i, bb_result['upper_breakout'][i]))

            # Lower breakouts (strong buy)
            if (i < len(bb_result['lower_breakout']) and
                bb_result['lower_breakout'][i] > 0):
                lower_breakout_signals.append((i, bb_result['lower_breakout'][i]))

        # Plot signals
        if upper_touch_signals:
            x_touch, y_touch = zip(*upper_touch_signals)
            touch_scatter = ax.scatter(x_touch, y_touch, marker='o',
                                       color='#FF9800', s=80,
                                       alpha=0.8, zorder=5, edgecolors='white')
            self.legend_items.append((touch_scatter, 'Upper Touch'))

        if lower_touch_signals:
            x_touch, y_touch = zip(*lower_touch_signals)
            touch_scatter = ax.scatter(x_touch, y_touch, marker='o',
                                       color='#4CAF50', s=80,
                                       alpha=0.8, zorder=5, edgecolors='white')
            self.legend_items.append((touch_scatter, 'Lower Touch'))

        if upper_breakout_signals:
            x_break, y_break = zip(*upper_breakout_signals)
            break_scatter = ax.scatter(x_break, y_break, marker='v',
                                       color='#F44336', s=120,
                                       alpha=0.9, zorder=6, edgecolors='white')
            self.legend_items.append((break_scatter, 'Upper Breakout'))

        if lower_breakout_signals:
            x_break, y_break = zip(*lower_breakout_signals)
            break_scatter = ax.scatter(x_break, y_break, marker='^',
                                       color='#2196F3', s=120,
                                       alpha=0.9, zorder=6, edgecolors='white')
            self.legend_items.append((break_scatter, 'Lower Breakout'))