# 📁 app/gui/simple_chart_widget.py
"""
Simple Chart Widget - fallback version bez modularnych komponentów
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from ..config.settings import COLORS

logger = logging.getLogger(__name__)


class SimpleChartWidget:
    """
    Prosty Chart Widget - wszystko w jednym miejscu
    Do użycia jako fallback jeśli modularny nie działa
    """

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.widget = None

        # Matplotlib
        self.fig = None
        self.canvas = None
        self.ax_price = None
        self.ax_volume = None
        self.ax_cci = None

        # Dane
        self.current_df = pd.DataFrame()
        self.current_indicators = {}

        # Ustawienia wyświetlania
        self.show_volume = True
        self.show_grid = True
        self.show_indicators = True
        self.show_cci = True

        self._create_widget()
        self._setup_chart()

        logger.info("SimpleChartWidget initialized")

    def _create_widget(self):
        """Tworzy prosty widget wykresu"""
        self.widget = ttk.Frame(self.parent)

        # Prosty toolbar
        self._create_simple_toolbar()

        # Kontener na wykres
        self.chart_container = ttk.Frame(self.widget)
        self.chart_container.pack(fill=tk.BOTH, expand=True)

    def _create_simple_toolbar(self):
        """Tworzy prosty toolbar"""
        toolbar = ttk.Frame(self.widget)
        toolbar.pack(fill=tk.X, pady=(0, 5))

        # Zoom controls
        ttk.Button(toolbar, text="🔍+", command=self.zoom_in, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔍-", command=self.zoom_out, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🏠", command=self.reset_zoom, width=4).pack(side=tk.LEFT, padx=2)

        # View options
        volume_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Volume", variable=volume_var,
                        command=lambda: self._toggle_volume(volume_var.get())).pack(side=tk.LEFT, padx=10)

        cci_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="CCI", variable=cci_var,
                        command=lambda: self._toggle_cci(cci_var.get())).pack(side=tk.LEFT, padx=5)

        grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Grid", variable=grid_var,
                        command=lambda: self._toggle_grid(grid_var.get())).pack(side=tk.LEFT, padx=5)

        # Export
        ttk.Button(toolbar, text="💾", command=self.save_chart, width=4).pack(side=tk.LEFT, padx=10)
        ttk.Button(toolbar, text="📊", command=self.export_data, width=4).pack(side=tk.LEFT, padx=2)

    def _setup_chart(self):
        """Konfiguruje prosty wykres matplotlib"""
        plt.style.use('dark_background')

        self.fig = Figure(figsize=(16, 10), facecolor=COLORS['bg_primary'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Event handlers
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_click)

        logger.info("Simple chart setup completed")

    def update_chart(self, df: pd.DataFrame, indicator_results: Dict):
        """Aktualizuje wykres z nowymi danymi"""
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided to chart")
                return

            self.current_df = df.copy()
            self.current_indicators = indicator_results.copy()

            self._plot_chart()

            logger.debug(f"Simple chart updated with {len(df)} candles")

        except Exception as e:
            logger.error(f"Error updating simple chart: {e}")

    def _plot_chart(self):
        """Rysuje prosty wykres"""
        try:
            # Wyczyść wykres
            self.fig.clear()

            # Konfiguruj subploty
            subplots_count = 1
            height_ratios = [3]

            if self.show_volume:
                subplots_count += 1
                height_ratios.append(1)

            if self.show_cci and 'CCI_Arrows_Main' in self.current_indicators:
                subplots_count += 1
                height_ratios.append(1)

            # Stwórz grid
            gs = self.fig.add_gridspec(subplots_count, 1, height_ratios=height_ratios, hspace=0.1)

            # Główny wykres cen
            self.ax_price = self.fig.add_subplot(gs[0])

            subplot_idx = 1

            # Volume subplot
            if self.show_volume:
                self.ax_volume = self.fig.add_subplot(gs[subplot_idx], sharex=self.ax_price)
                subplot_idx += 1
            else:
                self.ax_volume = None

            # CCI subplot
            if self.show_cci and 'CCI_Arrows_Main' in self.current_indicators:
                self.ax_cci = self.fig.add_subplot(gs[subplot_idx], sharex=self.ax_price)
            else:
                self.ax_cci = None

            # Rysuj komponenty
            self._plot_candlesticks()

            if self.show_volume and self.ax_volume is not None:
                self._plot_volume()

            if self.show_indicators:
                self._plot_tma()

            if self.show_cci and self.ax_cci is not None:
                self._plot_cci_arrows()

            self._format_axes()
            self._add_title()

            # Odśwież canvas
            self.canvas.draw()

        except Exception as e:
            logger.error(f"Error plotting simple chart: {e}")

    def _plot_candlesticks(self):
        """Rysuje świece"""
        if self.current_df.empty:
            return

        self.ax_price.set_facecolor(COLORS['bg_secondary'])

        if self.show_grid:
            self.ax_price.grid(True, alpha=0.3, color=COLORS['grid_color'])

        self.ax_price.tick_params(colors=COLORS['text_primary'])

        # Rysuj świece
        df = self.current_df
        for i, (date, row) in enumerate(df.iterrows()):
            # Kolory świec
            if row['close'] >= row['open']:
                body_color = COLORS['accent_green']
                fill_color = COLORS['bg_secondary']
                edge_color = COLORS['accent_green']
            else:
                body_color = COLORS['accent_red']
                fill_color = COLORS['accent_red']
                edge_color = COLORS['accent_red']

            # Korpus świecy
            height = abs(row['close'] - row['open'])
            bottom = min(row['open'], row['close'])

            if height > 0:
                rect = Rectangle((i - 0.4, bottom), 0.8, height,
                                 facecolor=fill_color if row['close'] >= row['open'] else body_color,
                                 edgecolor=edge_color, linewidth=1.2, alpha=0.8)
                self.ax_price.add_patch(rect)
            else:
                self.ax_price.plot([i - 0.4, i + 0.4], [row['open'], row['open']],
                                   color=edge_color, linewidth=1.5, alpha=0.8)

            # Cień świecy
            self.ax_price.plot([i, i], [row['low'], row['high']],
                               color=edge_color, linewidth=1, alpha=0.7)

    def _plot_volume(self):
        """Rysuje volume"""
        if self.current_df.empty or self.ax_volume is None:
            return

        self.ax_volume.set_facecolor(COLORS['bg_secondary'])

        if self.show_grid:
            self.ax_volume.grid(True, alpha=0.3, color=COLORS['grid_color'])

        self.ax_volume.tick_params(colors=COLORS['text_primary'])

        # Rysuj volume bars
        df = self.current_df
        for i, (date, row) in enumerate(df.iterrows()):
            color = COLORS['accent_green'] if row['close'] >= row['open'] else COLORS['accent_red']
            self.ax_volume.bar(i, row['volume'], color=color, alpha=0.6, width=0.8)

        self.ax_volume.set_ylabel('Volume', color=COLORS['text_primary'])
        self.ax_volume.set_xlim(-1, len(df))

    def _plot_tma(self):
        """Rysuje TMA"""
        if 'TMA_Main' not in self.current_indicators:
            return

        try:
            tma_result = self.current_indicators['TMA_Main']
            valid_from = tma_result.get('valid_from', 20)
            df_len = len(self.current_df)

            if valid_from >= df_len:
                return

            x_data = list(range(valid_from, df_len))
            if not x_data:
                return

            # Główna linia TMA
            tma_center = tma_result['tma_center']
            tma_colors = tma_result['tma_colors']

            for i in x_data:
                if i >= len(tma_center) or i >= len(tma_colors):
                    continue

                if i > valid_from:
                    color = COLORS['accent_green'] if tma_colors[i] == 0 else COLORS['accent_red']
                    self.ax_price.plot([i - 1, i], [tma_center[i - 1], tma_center[i]],
                                       color=color, linewidth=2.5, alpha=0.9)

            # Pasma ATR
            upper_y = [tma_result['tma_upper'][i] for i in x_data if i < len(tma_result['tma_upper'])]
            lower_y = [tma_result['tma_lower'][i] for i in x_data if i < len(tma_result['tma_lower'])]

            if len(upper_y) == len(x_data) and len(lower_y) == len(x_data):
                self.ax_price.plot(x_data, upper_y, color=COLORS['accent_blue'],
                                   linestyle='--', alpha=0.7, linewidth=1.5)
                self.ax_price.plot(x_data, lower_y, color=COLORS['accent_pink'],
                                   linestyle='--', alpha=0.7, linewidth=1.5)

            # Sygnały
            for i in x_data:
                if i >= len(tma_result['rebound_up']) or i >= len(tma_result['rebound_down']):
                    continue

                if tma_result['rebound_up'][i] > 0:
                    self.ax_price.scatter(i, tma_result['rebound_up'][i],
                                          marker='^', color=COLORS['accent_green'],
                                          s=120, alpha=0.9, edgecolors='white')

                if tma_result['rebound_down'][i] > 0:
                    self.ax_price.scatter(i, tma_result['rebound_down'][i],
                                          marker='v', color=COLORS['accent_red'],
                                          s=120, alpha=0.9, edgecolors='white')

        except Exception as e:
            logger.error(f"Error plotting TMA: {e}")

    def _plot_cci_arrows(self):
        """Rysuje CCI Arrows"""
        if 'CCI_Arrows_Main' not in self.current_indicators or self.ax_cci is None:
            return

        try:
            cci_result = self.current_indicators['CCI_Arrows_Main']
            valid_from = cci_result.get('valid_from', 14)
            df_len = len(self.current_df)

            if valid_from >= df_len:
                return

            self.ax_cci.set_facecolor(COLORS['bg_secondary'])

            if self.show_grid:
                self.ax_cci.grid(True, alpha=0.3, color=COLORS['grid_color'])

            self.ax_cci.tick_params(colors=COLORS['text_primary'])

            x_data = list(range(valid_from, df_len))
            if not x_data:
                return

            # Główna linia CCI
            cci_values = cci_result['cci']
            cci_line = [cci_values[i] for i in x_data if i < len(cci_values)]

            if len(cci_line) == len(x_data):
                self.ax_cci.plot(x_data, cci_line, color='#FFD700', linewidth=2,
                                 alpha=0.9)

            # Poziomy referencyjne
            overbought = cci_result['levels']['overbought']
            oversold = cci_result['levels']['oversold']

            self.ax_cci.axhline(y=overbought, color='#FF6B6B', linestyle='--', alpha=0.7)
            self.ax_cci.axhline(y=oversold, color='#4ECDC4', linestyle='--', alpha=0.7)
            self.ax_cci.axhline(y=0, color='#999999', linestyle='-', alpha=0.5)

            # Sygnały
            for i in x_data:
                if i >= len(cci_result['buy_arrows']):
                    continue

                cci_value = cci_values[i] if i < len(cci_values) else 0

                if cci_result['buy_arrows'][i] > 0:
                    self.ax_cci.scatter(i, cci_value, marker='^',
                                        color=COLORS['accent_green'], s=120,
                                        alpha=0.9, edgecolors='white')

                if cci_result['sell_arrows'][i] > 0:
                    self.ax_cci.scatter(i, cci_value, marker='v',
                                        color=COLORS['accent_red'], s=120,
                                        alpha=0.9, edgecolors='white')

            self.ax_cci.set_ylabel('CCI', color=COLORS['text_primary'])
            self.ax_cci.set_xlim(-1, df_len)

        except Exception as e:
            logger.error(f"Error plotting CCI arrows: {e}")

    def _format_axes(self):
        """Formatuje osie wykresu"""
        if self.current_df.empty:
            return

        df = self.current_df

        # Formatowanie osi X
        self.ax_price.set_xlim(-1, len(df))

        if len(df) > 0:
            # Etykiety osi X
            step = max(1, len(df) // 10)
            tick_positions = list(range(0, len(df), step))

            tick_labels = []
            for pos in tick_positions:
                if pos < len(df):
                    date = df.index[pos]
                    if hasattr(date, 'strftime'):
                        tick_labels.append(date.strftime('%H:%M'))
                    else:
                        tick_labels.append(f"T{pos}")
                else:
                    tick_labels.append("")

            self.ax_price.set_xticks(tick_positions)
            self.ax_price.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=8)

            # Auto-scale Y axis
            price_data = df[['high', 'low']].values.flatten()
            price_min = np.nanmin(price_data)
            price_max = np.nanmax(price_data)
            margin = (price_max - price_min) * 0.05

            self.ax_price.set_ylim(price_min - margin, price_max + margin)

        self.ax_price.set_ylabel('Cena (USDT)', color=COLORS['text_primary'])

        # Volume axis formatting
        if self.ax_volume is not None:
            self.ax_volume.set_xlim(-1, len(df))

        # CCI axis formatting
        if self.ax_cci is not None:
            self.ax_cci.set_xlim(-1, len(df))

    def _add_title(self):
        """Dodaje tytuł wykresu"""
        try:
            symbol = self.app.get_current_symbol() if hasattr(self.app, 'get_current_symbol') else 'BTC/USDT'
            timeframe = self.app.get_current_timeframe() if hasattr(self.app, 'get_current_timeframe') else '5m'

            title = f'{symbol} - {timeframe.upper()}'

            if 'TMA_Main' in self.current_indicators:
                title += ' | TMA'
            if 'CCI_Arrows_Main' in self.current_indicators:
                title += ' | CCI'

            self.ax_price.set_title(title, color=COLORS['text_primary'],
                                    fontsize=14, fontweight='bold', pad=20)

        except Exception as e:
            logger.error(f"Error adding title: {e}")

    # Event handlers
    def _on_scroll(self, event):
        """Obsługa scroll wheel"""
        if event.inaxes is None:
            return

        try:
            scale = 1.1 if event.button == 'up' else 0.9

            xlim = event.inaxes.get_xlim()
            ylim = event.inaxes.get_ylim()

            xdata = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
            ydata = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2

            x_left = xdata - (xdata - xlim[0]) * scale
            x_right = xdata + (xlim[1] - xdata) * scale
            y_bottom = ydata - (ydata - ylim[0]) * scale
            y_top = ydata + (ylim[1] - ydata) * scale

            event.inaxes.set_xlim(x_left, x_right)
            event.inaxes.set_ylim(y_bottom, y_top)
            self.canvas.draw()

        except Exception as e:
            logger.error(f"Error in scroll handler: {e}")

    def _on_click(self, event):
        """Obsługa kliknięcia"""
        if event.inaxes is None or event.xdata is None:
            return

        try:
            x_index = int(round(event.xdata))

            if 0 <= x_index < len(self.current_df):
                candle = self.current_df.iloc[x_index]

                info_parts = [
                    f"C: {candle['close']:.4f}",
                    f"V: {candle['volume']:.0f}"
                ]

                info_text = " | ".join(info_parts)
                if hasattr(self.app, 'main_window'):
                    self.app.main_window.show_status(info_text)

        except Exception as e:
            logger.error(f"Error in click handler: {e}")

    # Control methods
    def zoom_in(self):
        """Przybliża wykres"""
        try:
            for ax in [self.ax_price, self.ax_volume, self.ax_cci]:
                if ax:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()

                    x_center = (xlim[0] + xlim[1]) / 2
                    y_center = (ylim[0] + ylim[1]) / 2
                    x_range = (xlim[1] - xlim[0]) * 0.4
                    y_range = (ylim[1] - ylim[0]) * 0.4

                    ax.set_xlim(x_center - x_range, x_center + x_range)
                    ax.set_ylim(y_center - y_range, y_center + y_range)

            self.canvas.draw()
        except Exception as e:
            logger.error(f"Error zooming in: {e}")

    def zoom_out(self):
        """Oddala wykres"""
        try:
            for ax in [self.ax_price, self.ax_volume, self.ax_cci]:
                if ax:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()

                    x_center = (xlim[0] + xlim[1]) / 2
                    y_center = (ylim[0] + ylim[1]) / 2
                    x_range = (xlim[1] - xlim[0]) * 0.6
                    y_range = (ylim[1] - ylim[0]) * 0.6

                    ax.set_xlim(x_center - x_range, x_center + x_range)
                    ax.set_ylim(y_center - y_range, y_center + y_range)

            self.canvas.draw()
        except Exception as e:
            logger.error(f"Error zooming out: {e}")

    def reset_zoom(self):
        """Resetuje zoom"""
        try:
            if not self.current_df.empty:
                df = self.current_df

                for ax in [self.ax_price, self.ax_volume, self.ax_cci]:
                    if ax:
                        ax.set_xlim(-1, len(df))

                if self.ax_price:
                    price_data = df[['high', 'low']].values.flatten()
                    price_min = np.nanmin(price_data)
                    price_max = np.nanmax(price_data)
                    margin = (price_max - price_min) * 0.05
                    self.ax_price.set_ylim(price_min - margin, price_max + margin)

                self.canvas.draw()
        except Exception as e:
            logger.error(f"Error resetting zoom: {e}")

    def _toggle_volume(self, show: bool):
        """Przełącza volume"""
        self.show_volume = show
        self._plot_chart()

    def _toggle_cci(self, show: bool):
        """Przełącza CCI"""
        self.show_cci = show
        self._plot_chart()

    def _toggle_grid(self, show: bool):
        """Przełącza grid"""
        self.show_grid = show
        self._plot_chart()

    def save_chart(self):
        """Zapisuje wykres"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Zapisz wykres"
            )

            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight',
                                 facecolor=COLORS['bg_primary'])

                if hasattr(self.app, 'main_window'):
                    self.app.main_window.show_status(f"Wykres zapisany: {filename}")

        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            messagebox.showerror("Błąd", f"Nie można zapisać wykresu: {e}")

    def export_data(self):
        """Eksportuje dane"""
        try:
            if self.current_df.empty:
                messagebox.showwarning("Uwaga", "Brak danych do eksportu")
                return

            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Eksportuj dane"
            )

            if filename:
                self.current_df.to_csv(filename)

                if hasattr(self.app, 'main_window'):
                    self.app.main_window.show_status(f"Dane wyeksportowane: {filename}")

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            messagebox.showerror("Błąd", f"Nie można wyeksportować danych: {e}")

    def get_widget(self):
        """Zwraca widget"""
        return self.widget


# Fallback aliases
ModularChartWidget = SimpleChartWidget
ChartWidget = SimpleChartWidget
EnhancedChartWidget = SimpleChartWidget