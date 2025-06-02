# 📁 app/gui/chart_widget.py
"""
Modular Chart Widget - główny kontroler używający komponentów
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

from ..config.settings import COLORS
from .collapsible_panel import CollapsibleLegend
from .chart.base import ChartDataManager, ChartLayoutManager, ChartTooltipBuilder, ChartFormatter, ChartTitleBuilder
from .chart.legends import ChartLegendManager
from .chart.plotters import CandlestickPlotter, VolumePlotter, TMAPlotter, CCIPlotter, RSIPlotter, BollingerPlotter, GridPlotter
from .chart.events import ChartEventHandler, ChartZoomHandler, ChartExportHandler, ChartInteractionHandler

logger = logging.getLogger(__name__)


class ModularChartWidget:
    """
    Modular Chart Widget - główny kontroler
    Deleguje funkcjonalność do wyspecjalizowanych komponentów
    """

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.widget = None

        # === CORE COMPONENTS ===
        self.data_manager = ChartDataManager()
        self.layout_manager = ChartLayoutManager()
        self.tooltip_builder = ChartTooltipBuilder()
        self.formatter = ChartFormatter(COLORS)
        self.title_builder = ChartTitleBuilder(COLORS)

        # === PLOTTERS ===
        self.candlestick_plotter = CandlestickPlotter(COLORS)
        self.volume_plotter = VolumePlotter(COLORS)
        self.tma_plotter = TMAPlotter(COLORS)
        self.cci_plotter = CCIPlotter(COLORS)
        self.grid_plotter = GridPlotter(COLORS)
        self.rsi_plotter = RSIPlotter(COLORS)
        self.bollinger_plotter = BollingerPlotter(COLORS)

        # === EVENT HANDLERS ===
        self.event_handler = ChartEventHandler(
            self.data_manager,
            self.tooltip_builder,
            self._show_status
        )
        self.zoom_handler = ChartZoomHandler()
        self.export_handler = ChartExportHandler(COLORS)
        self.interaction_handler = ChartInteractionHandler()

        # === MATPLOTLIB COMPONENTS ===
        self.fig = None
        self.canvas = None
        self.axes = {'price': None, 'volume': None, 'cci': None, 'rsi': None}

        # === DISPLAY STATE ===
        self.display_config = {
            'show_volume': True,
            'show_grid': True,
            'show_indicators': True,
            'show_cci': True,
            'show_legend': True,
            'show_tma': True,
            'show_cci_arrows': True,
            'show_rsi': True,
            'show_bollinger': True
        }

        # === LEGENDS ===
        self.legends = {}

        self._create_widget()
        self._setup_chart()

        logger.info("ModularChartWidget initialized")

        # Enhanced legend manager
        self.legend_manager = ChartLegendManager(COLORS)

    def _create_widget(self):
        """Tworzy główny widget z toolbar"""
        self.widget = ttk.Frame(self.parent)

        # Enhanced toolbar
        self._create_toolbar()

        # Chart container
        self.chart_container = ttk.Frame(self.widget)
        self.chart_container.pack(fill=tk.BOTH, expand=True)

    def _create_toolbar(self):
        """Tworzy toolbar (deleguje do istniejącego kodu)"""
        toolbar_frame = ttk.Frame(self.widget)
        toolbar_frame.pack(fill=tk.X, pady=(0, 5))

        # === ZOOM SECTION ===
        zoom_frame = ttk.LabelFrame(toolbar_frame, text="🔍 Zoom", padding=2)
        zoom_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(zoom_frame, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_frame, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_frame, text="🏠", command=self.reset_zoom, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_frame, text="🔍", command=self.auto_fit, width=3).pack(side=tk.LEFT, padx=1)

        # === VIEW SECTION ===
        view_frame = ttk.LabelFrame(toolbar_frame, text="👁️ Widok", padding=2)
        view_frame.pack(side=tk.LEFT, padx=5)

        self.volume_var = tk.BooleanVar(value=True)
        self.cci_var = tk.BooleanVar(value=True)
        self.grid_var = tk.BooleanVar(value=True)
        self.legend_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(view_frame, text="Vol", variable=self.volume_var,
                        command=self._on_volume_toggle).pack(side=tk.LEFT)
        ttk.Checkbutton(view_frame, text="CCI", variable=self.cci_var,
                        command=self._on_cci_toggle).pack(side=tk.LEFT)
        ttk.Checkbutton(view_frame, text="Grid", variable=self.grid_var,
                        command=self._on_grid_toggle).pack(side=tk.LEFT)
        ttk.Checkbutton(view_frame, text="Legend", variable=self.legend_var,
                        command=self._on_legend_toggle).pack(side=tk.LEFT)

        # === INDICATORS SECTION ===
        indicators_frame = ttk.LabelFrame(toolbar_frame, text="📊 Wskaźniki", padding=2)
        indicators_frame.pack(side=tk.LEFT, padx=5)

        self.tma_var = tk.BooleanVar(value=True)
        self.cci_arrows_var = tk.BooleanVar(value=True)
        self.rsi_var = tk.BooleanVar(value=True)
        self.bollinger_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(indicators_frame, text="TMA", variable=self.tma_var,
                        command=self._on_tma_toggle).pack(side=tk.LEFT)
        ttk.Checkbutton(indicators_frame, text="CCI", variable=self.cci_arrows_var,
                        command=self._on_cci_arrows_toggle).pack(side=tk.LEFT)
        ttk.Checkbutton(indicators_frame, text="RSI", variable=self.rsi_var,
                        command=self._on_rsi_toggle).pack(side=tk.LEFT)
        ttk.Checkbutton(indicators_frame, text="BB", variable=self.bollinger_var,
                        command=self._on_bollinger_toggle).pack(side=tk.LEFT)

        # === EXPORT SECTION ===
        export_frame = ttk.LabelFrame(toolbar_frame, text="💾 Eksport", padding=2)
        export_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(export_frame, text="PNG", command=self.save_chart, width=4).pack(side=tk.LEFT, padx=1)
        ttk.Button(export_frame, text="CSV", command=self.export_data, width=4).pack(side=tk.LEFT, padx=1)

    def _setup_chart(self):
        """Konfiguruje matplotlib z event handlerami"""
        plt.style.use('dark_background')

        self.fig = Figure(figsize=(18, 12), dpi=100, facecolor=COLORS['bg_primary'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind events
        self.canvas.mpl_connect('scroll_event', lambda e: self.event_handler.on_scroll(e, self.canvas))
        self.canvas.mpl_connect('button_press_event', self.event_handler.on_click)
        self.canvas.mpl_connect('key_press_event', lambda e: self.event_handler.on_key_press(e, self))
        self.canvas.mpl_connect('motion_notify_event', lambda e: self.event_handler.on_mouse_move(e))

        logger.info("Modular chart setup completed")

    # === PUBLIC INTERFACE ===
    def update_chart(self, df: pd.DataFrame, indicator_results: Dict):
        """Główna funkcja aktualizacji wykresu"""
        try:
            if not self.data_manager.update_data(df, indicator_results):
                return

            self._plot_chart()
            logger.debug(f"Modular chart updated: {len(df)} candles, {len(indicator_results)} indicators")

        except Exception as e:
            logger.error(f"Error updating modular chart: {e}")

    def _plot_chart(self):
        """Główna funkcja rysowania - koordinuje wszystkie plotters"""
        try:
            # Clear and setup
            self.fig.clear()
            self.legends.clear()

            df, indicators = self.data_manager.get_data()
            if df.empty:
                return

            # Calculate layout
            active_subplots = self.layout_manager.calculate_layout(
                self.display_config['show_volume'],
                self.display_config['show_cci'],
                'CCI_Arrows_Main' in indicators,
                self.display_config['show_rsi'],
                'RSI_Professional_Main' in indicators
            )

            if not active_subplots:
                return

            # Create subplots
            self._create_subplots(active_subplots)

            # Plot components
            self._plot_price_chart(df, indicators)
            self._plot_volume_chart(df)
            self._plot_cci_chart(indicators, len(df))
            self._plot_rsi_chart(indicators, len(df))
            self._plot_bollinger_chart(df, indicators)

            # Format and finalize
            self._finalize_chart(df, indicators)

            self.canvas.draw()

        except Exception as e:
            logger.error(f"Error in modular chart plotting: {e}")

    def _create_subplots(self, active_subplots: List[Dict]):
        """Tworzy subploty na podstawie layoutu"""
        height_ratios = [subplot['height'] for subplot in active_subplots]
        gs = self.fig.add_gridspec(len(active_subplots), 1,
                                   height_ratios=height_ratios, hspace=0.05)

        # Reset axes
        self.axes = {'price': None, 'volume': None, 'cci': None, 'rsi': None}

        # Create axes
        for i, subplot_info in enumerate(active_subplots):
            subplot_type = subplot_info['type']

            if subplot_type == 'price':
                self.axes['price'] = self.fig.add_subplot(gs[i])
            elif subplot_type == 'volume':
                self.axes['volume'] = self.fig.add_subplot(gs[i], sharex=self.axes['price'])
            elif subplot_type == 'cci':
                self.axes['cci'] = self.fig.add_subplot(gs[i], sharex=self.axes['price'])
            elif subplot_type == 'rsi':
                self.axes['rsi'] = self.fig.add_subplot(gs[i], sharex=self.axes['price'])

    def _plot_price_chart(self, df: pd.DataFrame, indicators: Dict):
        """Rysuje wykres cenowy z wskaźnikami"""
        if self.axes['price'] is None:
            return

        # Candlesticks
        self.candlestick_plotter.plot(self.axes['price'], df, self.display_config['show_grid'])

        # TMA indicator
        if (self.display_config['show_indicators'] and
                self.display_config['show_tma'] and
                'TMA_Main' in indicators):
            legend_items = self.tma_plotter.plot(self.axes['price'], indicators['TMA_Main'], len(df))
            if legend_items and self.display_config['show_legend']:
                self._setup_legend('tma', self.axes['price'], legend_items)

    def _plot_volume_chart(self, df: pd.DataFrame):
        """Rysuje wykres volume"""
        if self.axes['volume'] is None or not self.display_config['show_volume']:
            return

        self.volume_plotter.plot(self.axes['volume'], df, self.display_config['show_grid'])

    def _plot_cci_chart(self, indicators: Dict, df_len: int):
        """Rysuje wykres CCI"""
        if (self.axes['cci'] is None or
                not self.display_config['show_cci'] or
                'CCI_Arrows_Main' not in indicators):
            return

        # Grid first
        self.grid_plotter.plot_grid(self.axes['cci'], self.display_config['show_grid'])

        # CCI indicator
        if self.display_config['show_cci_arrows']:
            legend_items = self.cci_plotter.plot(self.axes['cci'], indicators['CCI_Arrows_Main'], df_len)
            if legend_items and self.display_config['show_legend']:
                self._setup_legend('cci', self.axes['cci'], legend_items)

    def _plot_rsi_chart(self, indicators: Dict, df_len: int):
        """Rysuje wykres RSI"""
        if (self.axes['rsi'] is None or
                not self.display_config['show_rsi'] or
                'RSI_Professional_Main' not in indicators):
            return

        # Grid first
        self.grid_plotter.plot_grid(self.axes['rsi'], self.display_config['show_grid'])

        # RSI indicator
        legend_items = self.rsi_plotter.plot(self.axes['rsi'], indicators['RSI_Professional_Main'], df_len)
        if legend_items and self.display_config['show_legend']:
            self._setup_legend('rsi', self.axes['rsi'], legend_items)

    def _plot_bollinger_chart(self, df: pd.DataFrame, indicators: Dict):
        """Rysuje Bollinger Bands na wykresie cenowym"""
        if (self.axes['price'] is None or
                not self.display_config['show_bollinger'] or
                'Bollinger_Professional_Main' not in indicators):
            return

        # Bollinger indicator (na main price chart)
        legend_items = self.bollinger_plotter.plot(self.axes['price'], indicators['Bollinger_Professional_Main'],
                                                   len(df))
        if legend_items and self.display_config['show_legend']:
            # Dodaj do price legend (nie twórz osobnej)
            if 'price' in self.legends:
                for handle, label in legend_items:
                    self.legends['price'].add_item(handle, label)

    def _finalize_chart(self, df: pd.DataFrame, indicators: Dict):
        """Finalizuje wykres - formatowanie i tytuł"""
        # Format axes
        self.formatter.format_axes(self.axes, df, self.app)

        # Add title
        self.title_builder.build_title(self.axes['price'], indicators, self.app)

        # Setup legends
        if self.display_config['show_legend']:
            self._setup_enhanced_legends(indicators)

    def _setup_legend(self, legend_key: str, ax, legend_items: List):
        """Konfiguruje legendę dla osi"""
        if legend_key not in self.legends:
            position = 'upper left' if legend_key == 'tma' else 'upper right'
            self.legends[legend_key] = CollapsibleLegend(ax, f"{legend_key.upper()} Legend", position)

        # Add items to legend
        for handle, label in legend_items:
            self.legends[legend_key].add_item(handle, label)

        # Update legend visibility
        if self.display_config['show_legend']:
            self.legends[legend_key]._update_legend()

    def _setup_enhanced_legends(self, indicators: Dict):
        """Konfiguruje enhanced legendy"""
        try:
            # Price chart legend
            if self.axes['price']:
                self.legend_manager.setup_price_legend(
                    self.axes['price'],
                    indicators,
                    self.display_config['show_legend']
                )

            # Volume legend
            if self.axes['volume'] and self.display_config['show_volume']:
                self.legend_manager.setup_volume_legend(
                    self.axes['volume'],
                    self.display_config['show_legend']
                )

            # CCI legend
            if self.axes['cci'] and self.display_config['show_cci']:
                self.legend_manager.setup_cci_legend(
                    self.axes['cci'],
                    indicators,
                    self.display_config['show_legend']
                )

            # RSI legend
            if self.axes.get('rsi') and self.display_config['show_rsi']:
                self.legend_manager.setup_rsi_legend(
                    self.axes['rsi'],
                    indicators,
                    self.display_config['show_legend']
                )

            # Bollinger legend (na price chart)
            if self.axes['price'] and self.display_config['show_bollinger']:
                self.legend_manager.setup_bollinger_legend(
                    self.axes['price'],
                    indicators,
                    self.display_config['show_legend']
                )

        except Exception as e:
            logger.error(f"Error setting up enhanced legends: {e}")

    # === ZOOM CONTROLS ===
    def zoom_in(self):
        """Zoom in używając zoom handler"""
        axes_list = [ax for ax in self.axes.values() if ax is not None]
        self.zoom_handler.zoom_in(axes_list)
        self.canvas.draw_idle()

    def zoom_out(self):
        """Zoom out używając zoom handler"""
        axes_list = [ax for ax in self.axes.values() if ax is not None]
        self.zoom_handler.zoom_out(axes_list)
        self.canvas.draw_idle()

    def reset_zoom(self):
        """Reset zoom używając zoom handler"""
        df, indicators = self.data_manager.get_data()
        if df.empty:
            return

        # Prepare data for reset
        price_data = df[['high', 'low']].values.flatten() if not df.empty else None
        cci_data = indicators.get('CCI_Arrows_Main', {}).get('cci', None)

        self.zoom_handler.reset_zoom(self.axes, len(df), price_data, cci_data)
        self.canvas.draw_idle()

    def auto_fit(self):
        """Auto-fit do najnowszych danych"""
        df, _ = self.data_manager.get_data()
        if df.empty:
            return

        axes_list = [ax for ax in self.axes.values() if ax is not None]
        self.zoom_handler.auto_fit(axes_list, len(df))
        self.canvas.draw_idle()

    # === DISPLAY TOGGLES ===
    def _on_volume_toggle(self):
        """Toggle volume display"""
        self.display_config['show_volume'] = self.volume_var.get()
        self.layout_manager.update_config('volume', enabled=self.display_config['show_volume'])
        self._plot_chart()

    def _on_cci_toggle(self):
        """Toggle CCI subplot"""
        self.display_config['show_cci'] = self.cci_var.get()
        self.layout_manager.update_config('cci', enabled=self.display_config['show_cci'])
        self._plot_chart()

    def _on_grid_toggle(self):
        """Toggle grid"""
        self.display_config['show_grid'] = self.grid_var.get()
        self._plot_chart()

    def _on_legend_toggle(self):
        """Toggle legends"""
        self.display_config['show_legend'] = self.legend_var.get()
        for legend in self.legends.values():
            if self.display_config['show_legend']:
                legend._update_legend()
            else:
                if legend.legend_obj:
                    legend.legend_obj.remove()
                    legend.legend_obj = None
        self.canvas.draw_idle()

    def _on_tma_toggle(self):
        """Toggle TMA indicator"""
        self.display_config['show_tma'] = self.tma_var.get()
        if hasattr(self.app, 'toggle_indicator'):
            self.app.toggle_indicator('TMA_Main', self.display_config['show_tma'])

    def _on_cci_arrows_toggle(self):
        """Toggle CCI Arrows indicator"""
        self.display_config['show_cci_arrows'] = self.cci_arrows_var.get()
        if hasattr(self.app, 'toggle_indicator'):
            self.app.toggle_indicator('CCI_Arrows_Main', self.display_config['show_cci_arrows'])

    def _on_rsi_toggle(self):
        """Toggle RSI indicator"""
        self.display_config['show_rsi'] = self.rsi_var.get()
        if hasattr(self.app, 'toggle_indicator'):
            self.app.toggle_indicator('RSI_Professional_Main', self.display_config['show_rsi'])

    def _on_bollinger_toggle(self):
        """Toggle Bollinger indicator"""
        self.display_config['show_bollinger'] = self.bollinger_var.get()
        if hasattr(self.app, 'toggle_indicator'):
            self.app.toggle_indicator('Bollinger_Professional_Main', self.display_config['show_bollinger'])

    # === EXPORT FUNCTIONS ===
    def save_chart(self):
        """Zapisuje wykres używając export handler"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ],
                title="Zapisz wykres"
            )

            if filename:
                success = self.export_handler.save_chart(self.fig, filename)
                if success:
                    self._show_status(f"Wykres zapisany: {filename}")
                else:
                    messagebox.showerror("Błąd", "Nie można zapisać wykresu")

        except Exception as e:
            logger.error(f"Error in save chart: {e}")
            messagebox.showerror("Błąd", f"Błąd zapisu: {e}")

    def export_data(self):
        """Eksportuje dane używając export handler"""
        try:
            df, indicators = self.data_manager.get_data()
            if df.empty:
                messagebox.showwarning("Uwaga", "Brak danych do eksportu")
                return

            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx"),
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ],
                title="Eksportuj dane"
            )

            if filename:
                success = self.export_handler.export_data(df, indicators, filename)
                if success:
                    self._show_status(f"Dane wyeksportowane: {filename}")
                else:
                    messagebox.showerror("Błąd", "Nie można wyeksportować danych")

        except Exception as e:
            logger.error(f"Error in export data: {e}")
            messagebox.showerror("Błąd", f"Błąd eksportu: {e}")

    # === HELPER METHODS ===
    def _show_status(self, message: str):
        """Pokazuje status (callback dla event handlerów)"""
        if hasattr(self.app, 'main_window') and hasattr(self.app.main_window, 'show_status'):
            self.app.main_window.show_status(message)

    def toggle_legend(self):
        """Public method dla toggle legend"""
        self.legend_var.set(not self.legend_var.get())
        self._on_legend_toggle()

        # Również toggle w legend manager
        self.legend_manager.toggle_all_legends()
        self.canvas.draw_idle()

    def toggle_grid(self):
        """Public method dla toggle grid"""
        self.grid_var.set(not self.grid_var.get())
        self._on_grid_toggle()

    def get_widget(self):
        """Zwraca główny widget"""
        return self.widget


# === BACKWARD COMPATIBILITY ===
# Aliasy dla kompatybilności wstecznej
ChartWidget = ModularChartWidget
EnhancedChartWidget = ModularChartWidget