# 📁 app/gui/chart/plotters.py
"""
Komponenty do rysowania różnych elementów wykresu
COMPLETE FIXED VERSION - Full Working EMA Crossover Integration
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


class EMACrossoverPlotter:
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

            # Rysuj obszary trendu (tło)
            self._plot_trend_areas(ax, ema_result, df_len)

            # Główne linie EMA
            self._plot_ema_lines(ax, ema_result, x_data)

            # Sygnały crossover
            self._plot_crossover_signals(ax, ema_result, x_data)

            # Dodatkowe punkty crossover
            self._plot_crossover_points(ax, ema_result)

            return self.legend_items

        except Exception as e:
            logger.error(f"Error plotting EMA Crossover: {e}")
            return []

    def _plot_trend_areas(self, ax, ema_result: Dict, df_len: int):
        """Rysuje obszary trendu jako tło"""
        try:
            # Bullish areas (green background)
            bullish_areas = ema_result.get('bullish_areas', [])
            for start, end in bullish_areas:
                if end - start > 3:  # Only show significant areas
                    ax.axvspan(start, end, alpha=0.08, color=self.colors['accent_green'],
                               zorder=0)

            # Bearish areas (red background)
            bearish_areas = ema_result.get('bearish_areas', [])
            for start, end in bearish_areas:
                if end - start > 3:  # Only show significant areas
                    ax.axvspan(start, end, alpha=0.08, color=self.colors['accent_red'],
                               zorder=0)

        except Exception as e:
            logger.error(f"Error plotting trend areas: {e}")

    def _plot_ema_lines(self, ax, ema_result: Dict, x_data: List[int]):
        """Rysuje główne linie EMA"""
        try:
            # Fast EMA (złota linia)
            fast_ema = ema_result['fast_ema']
            fast_y = [fast_ema[i] for i in x_data if i < len(fast_ema)]

            if len(fast_y) == len(x_data):
                fast_line = ax.plot(x_data, fast_y, color='#FFD700',
                                    linewidth=2.5, alpha=0.9, zorder=4)[0]
                self.legend_items.append((fast_line, f"Fast EMA({ema_result['settings']['fast_ema_period']})"))

            # Slow EMA (niebieska linia)
            slow_ema = ema_result['slow_ema']
            slow_y = [slow_ema[i] for i in x_data if i < len(slow_ema)]

            if len(slow_y) == len(x_data):
                slow_line = ax.plot(x_data, slow_y, color='#87CEEB',
                                    linewidth=2.5, alpha=0.9, zorder=4)[0]
                self.legend_items.append((slow_line, f"Slow EMA({ema_result['settings']['slow_ema_period']})"))

            # Signal EMA (jeśli używana)
            if ema_result.get('settings', {}).get('use_signal_line', False):
                signal_ema = ema_result.get('signal_ema')
                if signal_ema is not None:
                    signal_y = [signal_ema[i] for i in x_data if i < len(signal_ema)]

                    if len(signal_y) == len(x_data):
                        signal_line = ax.plot(x_data, signal_y, color='#DDA0DD',
                                              linewidth=2, linestyle='--', alpha=0.8, zorder=3)[0]
                        self.legend_items.append((signal_line, f"Signal EMA({ema_result['settings']['signal_ema_period']})"))

        except Exception as e:
            logger.error(f"Error plotting EMA lines: {e}")

    def _plot_crossover_signals(self, ax, ema_result: Dict, x_data: List[int]):
        """Rysuje sygnały crossover"""
        try:
            buy_signals = ema_result['buy_signals']
            sell_signals = ema_result['sell_signals']

            buy_points = []
            sell_points = []

            for i in x_data:
                if i < len(buy_signals) and buy_signals[i] > 0:
                    buy_points.append((i, buy_signals[i]))

                if i < len(sell_signals) and sell_signals[i] > 0:
                    sell_points.append((i, sell_signals[i]))

            # Rysuj sygnały kupna
            if buy_points:
                buy_x, buy_y = zip(*buy_points)
                buy_scatter = ax.scatter(buy_x, buy_y, marker='^',
                                         color=self.colors['accent_green'], s=150,
                                         alpha=0.9, zorder=6, edgecolors='white',
                                         linewidth=2)
                self.legend_items.append((buy_scatter, 'EMA Buy Signal'))

            # Rysuj sygnały sprzedaży
            if sell_points:
                sell_x, sell_y = zip(*sell_points)
                sell_scatter = ax.scatter(sell_x, sell_y, marker='v',
                                          color=self.colors['accent_red'], s=150,
                                          alpha=0.9, zorder=6, edgecolors='white',
                                          linewidth=2)
                self.legend_items.append((sell_scatter, 'EMA Sell Signal'))

        except Exception as e:
            logger.error(f"Error plotting crossover signals: {e}")

    def _plot_crossover_points(self, ax, ema_result: Dict):
        """Rysuje punkty crossover (dodatkowe oznaczenia)"""
        try:
            crossover_points = ema_result.get('crossover_points', [])

            bullish_crosses = []
            bearish_crosses = []

            for cross in crossover_points:
                if cross['type'] == 'bullish':
                    bullish_crosses.append((cross['index'], cross['fast_value']))
                elif cross['type'] == 'bearish':
                    bearish_crosses.append((cross['index'], cross['slow_value']))

            # Małe kropki na crossoverach (subtelne oznaczenie)
            if bullish_crosses:
                bull_x, bull_y = zip(*bullish_crosses)
                ax.scatter(bull_x, bull_y, marker='o', color='#00FF88',
                           s=30, alpha=0.6, zorder=5)

            if bearish_crosses:
                bear_x, bear_y = zip(*bearish_crosses)
                ax.scatter(bear_x, bear_y, marker='o', color='#FF4444',
                           s=30, alpha=0.6, zorder=5)

        except Exception as e:
            logger.error(f"Error plotting crossover points: {e}")

    def get_crossover_analysis(self, ema_result: Dict, current_index: int) -> str:
        """Zwraca analizę tekstową crossover dla tooltipa"""
        try:
            if current_index >= len(ema_result['fast_ema']):
                return ""

            fast_ema = ema_result['fast_ema'][current_index]
            slow_ema = ema_result['slow_ema'][current_index]
            trend_dir = ema_result['trend_direction'][current_index]
            trend_strength = ema_result['trend_strength'][current_index]

            analysis_parts = [
                f"Fast EMA: {fast_ema:.4f}",
                f"Slow EMA: {slow_ema:.4f}",
                f"Separation: {abs(fast_ema - slow_ema):.4f}"
            ]

            # Trend analysis
            if trend_dir > 0:
                analysis_parts.append("📈 Bullish Trend")
            elif trend_dir < 0:
                analysis_parts.append("📉 Bearish Trend")
            else:
                analysis_parts.append("➡️ Neutral")

            # Strength
            strength_pct = trend_strength * 100
            analysis_parts.append(f"Strength: {strength_pct:.1f}%")

            # Recent signals
            if ema_result['buy_signals'][current_index] > 0:
                analysis_parts.append("🟢 BUY SIGNAL")
            elif ema_result['sell_signals'][current_index] > 0:
                analysis_parts.append("🔴 SELL SIGNAL")

            return " | ".join(analysis_parts)

        except Exception as e:
            logger.error(f"Error in crossover analysis: {e}")
            return "EMA Analysis Error"


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


class MACDStyleEMAPlotter:
    """
    EMA Crossover w stylu MACD - z histogramem różnicy
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config
        self.legend_items = []

    def plot(self, ax_main, ax_macd, ema_result: Dict, df_len: int) -> List[Tuple]:
        """
        Rysuje EMA na głównym wykresie + MACD-style histogram na subplot

        Args:
            ax_main: Główna oś (dla linii EMA)
            ax_macd: Oś MACD (dla histogramu różnicy)
            ema_result: Wyniki EMA
            df_len: Długość danych
        """
        if ax_main is None:
            return []

        try:
            valid_from = ema_result.get('valid_from', 26)
            if valid_from >= df_len:
                return []

            x_data = list(range(valid_from, df_len))
            if not x_data:
                return []

            self.legend_items = []

            # Na głównym wykresie - tylko linie EMA
            self._plot_ema_lines_main(ax_main, ema_result, x_data)

            # Na MACD subplot - histogram różnicy
            if ax_macd is not None:
                self._plot_macd_style_histogram(ax_macd, ema_result, x_data)

            return self.legend_items

        except Exception as e:
            logger.error(f"Error plotting MACD-style EMA: {e}")
            return []

    def _plot_ema_lines_main(self, ax, ema_result: Dict, x_data: List[int]):
        """Rysuje linie EMA na głównym wykresie"""
        try:
            # Fast EMA
            fast_ema = ema_result['fast_ema']
            fast_y = [fast_ema[i] for i in x_data if i < len(fast_ema)]

            if len(fast_y) == len(x_data):
                fast_line = ax.plot(x_data, fast_y, color='#FFD700',
                                    linewidth=2, alpha=0.8, zorder=3)[0]
                self.legend_items.append((fast_line, "Fast EMA"))

            # Slow EMA
            slow_ema = ema_result['slow_ema']
            slow_y = [slow_ema[i] for i in x_data if i < len(slow_ema)]

            if len(slow_y) == len(x_data):
                slow_line = ax.plot(x_data, slow_y, color='#87CEEB',
                                    linewidth=2, alpha=0.8, zorder=3)[0]
                self.legend_items.append((slow_line, "Slow EMA"))

        except Exception as e:
            logger.error(f"Error plotting EMA lines on main: {e}")

    def _plot_macd_style_histogram(self, ax, ema_result: Dict, x_data: List[int]):
        """Rysuje histogram różnicy EMA w stylu MACD"""
        try:
            ax.set_facecolor(self.colors['bg_secondary'])
            ax.tick_params(colors=self.colors['text_primary'], labelsize=8)

            fast_ema = ema_result['fast_ema']
            slow_ema = ema_result['slow_ema']

            # Oblicz różnicę (Fast - Slow)
            differences = []
            colors = []

            for i in x_data:
                if i < len(fast_ema) and i < len(slow_ema):
                    diff = fast_ema[i] - slow_ema[i]
                    differences.append(diff)

                    # Kolor bazowany na kierunku
                    if diff > 0:
                        colors.append(self.colors['accent_green'])
                    else:
                        colors.append(self.colors['accent_red'])

            # Rysuj histogram
            if differences:
                bars = ax.bar(x_data, differences, color=colors, alpha=0.7, width=0.8)

            # Zero line
            ax.axhline(y=0, color=self.colors['text_primary'],
                       linestyle='-', alpha=0.5, linewidth=1)

            # Sygnały na histogramie
            self._plot_histogram_signals(ax, ema_result, x_data)

            ax.set_ylabel('EMA Difference', color=self.colors['text_primary'],
                          fontweight='bold', fontsize=10)
            ax.set_xlim(-1, len(x_data) + len(x_data))

        except Exception as e:
            logger.error(f"Error plotting MACD histogram: {e}")

    def _plot_histogram_signals(self, ax, ema_result: Dict, x_data: List[int]):
        """Dodaje sygnały na histogram"""
        try:
            buy_signals = ema_result['buy_signals']
            sell_signals = ema_result['sell_signals']
            fast_ema = ema_result['fast_ema']
            slow_ema = ema_result['slow_ema']

            for i in x_data:
                if i < len(buy_signals) and buy_signals[i] > 0:
                    if i < len(fast_ema) and i < len(slow_ema):
                        diff = fast_ema[i] - slow_ema[i]
                        ax.scatter(i, diff, marker='^', color='white',
                                   s=100, alpha=0.9, zorder=5, edgecolors='green')

                if i < len(sell_signals) and sell_signals[i] > 0:
                    if i < len(fast_ema) and i < len(slow_ema):
                        diff = fast_ema[i] - slow_ema[i]
                        ax.scatter(i, diff, marker='v', color='white',
                                   s=100, alpha=0.9, zorder=5, edgecolors='red')

        except Exception as e:
            logger.error(f"Error plotting histogram signals: {e}")