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