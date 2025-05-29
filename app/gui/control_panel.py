# 📁 app/gui/control_panel.py
"""
Panel kontrolny - przeprojektowany z składanymi sekcjami
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, Callable
from ..config.settings import EXCHANGES, TIMEFRAMES, POPULAR_SYMBOLS
from .collapsible_panel import CollapsiblePanel, ExpandableSection, TabbedCollapsiblePanel

logger = logging.getLogger(__name__)


class ControlPanel:
    """
    Zaawansowany panel kontrolny z składanymi sekcjami
    """

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.widget = None

        # Zmienne GUI - podstawowe
        self.exchange_var = tk.StringVar(value='Binance')
        self.symbol_var = tk.StringVar(value='BTC/USDT')
        self.timeframe_var = tk.StringVar(value='M5')
        self.limit_var = tk.StringVar(value='200')
        self.interval_var = tk.StringVar(value='5')
        self.auto_refresh_var = tk.BooleanVar(value=True)

        # Zmienne GUI - widok
        self.volume_var = tk.BooleanVar(value=True)
        self.grid_var = tk.BooleanVar(value=True)
        self.indicators_var = tk.BooleanVar(value=True)

        # Zmienne GUI - TMA
        self.tma_enabled_var = tk.BooleanVar(value=True)
        self.tma_period_var = tk.StringVar(value='12')
        self.tma_atr_var = tk.StringVar(value='2.0')

        # Zmienne GUI - CCI
        self.cci_enabled_var = tk.BooleanVar(value=True)
        self.cci_period_var = tk.StringVar(value='14')
        self.cci_overbought_var = tk.StringVar(value='100')
        self.cci_oversold_var = tk.StringVar(value='-100')
        self.cci_sensitivity_var = tk.StringVar(value='medium')

        # Panele składane
        self.panels = {}

        self._create_widget()
        logger.info("Enhanced ControlPanel initialized")

    def _create_widget(self):
        """Tworzy główny widget z składanymi panelami"""
        # Główny frame
        self.widget = ttk.Frame(self.parent)

        # === SEKCJA 1: PODSTAWOWE USTAWIENIA ===
        basic_panel = CollapsiblePanel(
            self.widget,
            "🎯 Podstawowe ustawienia",
            expanded=True
        )
        basic_panel.get_frame().pack(fill=tk.X, padx=5, pady=2)
        self.panels['basic'] = basic_panel
        self._create_basic_controls(basic_panel.get_content_frame())

        # === SEKCJA 2: WIDOK I KONTROLKI ===
        view_panel = CollapsiblePanel(
            self.widget,
            "👁️ Widok i kontrolki",
            expanded=True
        )
        view_panel.get_frame().pack(fill=tk.X, padx=5, pady=2)
        self.panels['view'] = view_panel
        self._create_view_controls(view_panel.get_content_frame())

        # === SEKCJA 3: WSKAŹNIKI ===
        indicators_panel = CollapsiblePanel(
            self.widget,
            "📊 Wskaźniki techniczne",
            expanded=True,
            on_toggle=self._on_indicators_panel_toggle
        )
        indicators_panel.get_frame().pack(fill=tk.X, padx=5, pady=2)
        self.panels['indicators'] = indicators_panel
        self._create_indicators_controls(indicators_panel.get_content_frame())

    def _create_basic_controls(self, parent):
        """Tworzy podstawowe kontrolki"""
        # Rząd 1: Giełda, Symbol, Timeframe
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, pady=2)

        # Giełda
        ttk.Label(row1, text="Giełda:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        exchange_combo = ttk.Combobox(row1, textvariable=self.exchange_var,
                                      values=list(EXCHANGES.keys()), width=10, state='readonly')
        exchange_combo.pack(side=tk.LEFT, padx=5)
        exchange_combo.bind('<<ComboboxSelected>>', self._on_exchange_change)

        # Symbol
        ttk.Label(row1, text="Symbol:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(15, 5))
        symbol_combo = ttk.Combobox(row1, textvariable=self.symbol_var,
                                    values=POPULAR_SYMBOLS, width=12)
        symbol_combo.pack(side=tk.LEFT, padx=5)
        symbol_combo.bind('<<ComboboxSelected>>', self._on_symbol_change)
        symbol_combo.bind('<Return>', self._on_symbol_change)

        # Timeframe
        ttk.Label(row1, text="TF:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(15, 5))
        timeframe_combo = ttk.Combobox(row1, textvariable=self.timeframe_var,
                                       values=list(TIMEFRAMES.keys()), width=6, state='readonly')
        timeframe_combo.pack(side=tk.LEFT, padx=5)
        timeframe_combo.bind('<<ComboboxSelected>>', self._on_timeframe_change)

        # Rząd 2: Parametry i kontrolki
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, pady=5)

        # Limit świec
        ttk.Label(row2, text="Świece:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        limit_spin = ttk.Spinbox(row2, from_=50, to=1000, increment=50,
                                 textvariable=self.limit_var, width=6)
        limit_spin.pack(side=tk.LEFT, padx=5)

        # Interwał refresh
        ttk.Label(row2, text="Refresh:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(10, 5))
        interval_spin = ttk.Spinbox(row2, from_=1, to=60, increment=1,
                                    textvariable=self.interval_var, width=4)
        interval_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(row2, text="s", font=('Arial', 8)).pack(side=tk.LEFT)

        # Auto-refresh
        ttk.Checkbutton(row2, text="Auto", variable=self.auto_refresh_var,
                        command=self._on_auto_refresh_toggle).pack(side=tk.LEFT, padx=15)

        # Refresh button
        ttk.Button(row2, text="🔄", command=self._on_refresh, width=4).pack(side=tk.LEFT, padx=5)

    def _create_view_controls(self, parent):
        """Tworzy kontrolki widoku"""
        # Rząd 1: Opcje wyświetlania
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="Widok:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(row1, text="Volume", variable=self.volume_var,
                        command=self._on_volume_toggle).pack(side=tk.LEFT, padx=10)

        ttk.Checkbutton(row1, text="Grid", variable=self.grid_var,
                        command=self._on_grid_toggle).pack(side=tk.LEFT, padx=10)

        ttk.Checkbutton(row1, text="Wskaźniki", variable=self.indicators_var,
                        command=self._on_indicators_toggle).pack(side=tk.LEFT, padx=10)

        # Rząd 2: Kontrolki wykresu
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, pady=5)

        ttk.Label(row2, text="Wykres:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)

        ttk.Button(row2, text="🔍+", command=self._on_zoom_in, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="🔍-", command=self._on_zoom_out, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="🏠", command=self._on_reset_zoom, width=4).pack(side=tk.LEFT, padx=2)

        ttk.Button(row2, text="💾", command=self._on_save_chart, width=4).pack(side=tk.LEFT, padx=10)
        ttk.Button(row2, text="📊", command=self._on_export_data, width=4).pack(side=tk.LEFT, padx=2)

    def _create_indicators_controls(self, parent):
        """Tworzy kontrolki wskaźników z sekcjami"""
        # === TMA SEKCJA ===
        tma_section = ExpandableSection(parent, "📈 TMA (Triangular Moving Average)", True)
        self._create_tma_controls(tma_section.get_content_frame())

        # === CCI SEKCJA ===
        cci_section = ExpandableSection(parent, "📊 CCI Arrows (Commodity Channel Index)", True)
        self._create_cci_controls(cci_section.get_content_frame())

        # === PLACEHOLDER DLA KOLEJNYCH WSKAŹNIKÓW ===
        placeholder_section = ExpandableSection(parent, "⚡ Kolejne wskaźniki (wkrótce...)", False)
        placeholder_frame = placeholder_section.get_content_frame()
        ttk.Label(placeholder_frame,
                  text="🚧 Miejsce na RSI, MACD, Bollinger Bands, etc.",
                  font=('Arial', 8), foreground='gray').pack(pady=5)

    def _create_tma_controls(self, parent):
        """Tworzy kontrolki TMA"""
        # Włączenie TMA
        control_row = ttk.Frame(parent)
        control_row.pack(fill=tk.X, pady=2)

        ttk.Checkbutton(control_row, text="Włącz TMA", variable=self.tma_enabled_var,
                        command=self._on_tma_toggle).pack(side=tk.LEFT)

        # Parametry TMA
        params_row = ttk.Frame(parent)
        params_row.pack(fill=tk.X, pady=2)

        # Okres
        ttk.Label(params_row, text="Okres:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        period_spin = ttk.Spinbox(params_row, from_=5, to=30, textvariable=self.tma_period_var,
                                  width=4, command=self._on_tma_settings_change)
        period_spin.pack(side=tk.LEFT, padx=2)

        # ATR multiplier
        ttk.Label(params_row, text="ATR:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(10, 5))
        atr_spin = ttk.Spinbox(params_row, from_=1.0, to=3.0, increment=0.1,
                               textvariable=self.tma_atr_var, width=4,
                               command=self._on_tma_settings_change)
        atr_spin.pack(side=tk.LEFT, padx=2)

        # Presety TMA
        presets_row = ttk.Frame(parent)
        presets_row.pack(fill=tk.X, pady=2)

        ttk.Label(presets_row, text="Presety:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        ttk.Button(presets_row, text="Scalp",
                   command=lambda: self._load_tma_preset('scalping')).pack(side=tk.LEFT, padx=2)
        ttk.Button(presets_row, text="Swing",
                   command=lambda: self._load_tma_preset('swing')).pack(side=tk.LEFT, padx=2)
        ttk.Button(presets_row, text="Safe",
                   command=lambda: self._load_tma_preset('conservative')).pack(side=tk.LEFT, padx=2)

    def _create_cci_controls(self, parent):
        """Tworzy kontrolki CCI"""
        # Włączenie CCI
        control_row = ttk.Frame(parent)
        control_row.pack(fill=tk.X, pady=2)

        ttk.Checkbutton(control_row, text="Włącz CCI", variable=self.cci_enabled_var,
                        command=self._on_cci_toggle).pack(side=tk.LEFT)

        # Parametry CCI - rząd 1
        params_row1 = ttk.Frame(parent)
        params_row1.pack(fill=tk.X, pady=2)

        # Okres
        ttk.Label(params_row1, text="Okres:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        cci_period_spin = ttk.Spinbox(params_row1, from_=8, to=30, textvariable=self.cci_period_var,
                                      width=4, command=self._on_cci_settings_change)
        cci_period_spin.pack(side=tk.LEFT, padx=2)

        # Czułość
        ttk.Label(params_row1, text="Czułość:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(10, 5))
        sens_combo = ttk.Combobox(params_row1, textvariable=self.cci_sensitivity_var,
                                  values=['low', 'medium', 'high'], width=8, state='readonly')
        sens_combo.pack(side=tk.LEFT, padx=2)
        sens_combo.bind('<<ComboboxSelected>>', self._on_cci_settings_change)

        # Parametry CCI - rząd 2 (poziomy)
        params_row2 = ttk.Frame(parent)
        params_row2.pack(fill=tk.X, pady=2)

        # Overbought
        ttk.Label(params_row2, text="OB:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        ob_spin = ttk.Spinbox(params_row2, from_=50, to=200, increment=10,
                              textvariable=self.cci_overbought_var, width=4,
                              command=self._on_cci_settings_change)
        ob_spin.pack(side=tk.LEFT, padx=2)

        # Oversold
        ttk.Label(params_row2, text="OS:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(10, 5))
        os_spin = ttk.Spinbox(params_row2, from_=-200, to=-50, increment=10,
                              textvariable=self.cci_oversold_var, width=4,
                              command=self._on_cci_settings_change)
        os_spin.pack(side=tk.LEFT, padx=2)

        # Presety CCI
        presets_row = ttk.Frame(parent)
        presets_row.pack(fill=tk.X, pady=2)

        ttk.Label(presets_row, text="Presety:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        ttk.Button(presets_row, text="Fast",
                   command=lambda: self._load_cci_preset('fast')).pack(side=tk.LEFT, padx=2)
        ttk.Button(presets_row, text="Standard",
                   command=lambda: self._load_cci_preset('standard')).pack(side=tk.LEFT, padx=2)
        ttk.Button(presets_row, text="Smooth",
                   command=lambda: self._load_cci_preset('smooth')).pack(side=tk.LEFT, padx=2)

    # Event handlers - podstawowe
    def _on_exchange_change(self, event=None):
        """Obsługa zmiany giełdy"""
        exchange_name = self.exchange_var.get()
        success = self.app.change_exchange(exchange_name)
        if success:
            logger.info(f"Exchange changed to {exchange_name}")
        else:
            logger.error(f"Failed to change exchange to {exchange_name}")

    def _on_symbol_change(self, event=None):
        """Obsługa zmiany symbolu"""
        self.app.refresh_data()

    def _on_timeframe_change(self, event=None):
        """Obsługa zmiany timeframe"""
        timeframe = TIMEFRAMES.get(self.timeframe_var.get(), '5m')
        self.app.change_timeframe(timeframe)

    def _on_refresh(self):
        """Obsługa przycisku odśwież"""
        self.app.refresh_data()

    def _on_auto_refresh_toggle(self):
        """Obsługa auto-refresh"""
        if self.auto_refresh_var.get():
            self.app.start_auto_refresh()
        else:
            self.app.stop_auto_refresh()

    # Event handlers - widok
    def _on_volume_toggle(self):
        """Obsługa volume toggle"""
        if hasattr(self.app, 'main_window') and hasattr(self.app.main_window, 'chart_widget'):
            self.app.main_window.chart_widget._toggle_volume(self.volume_var.get())

    def _on_grid_toggle(self):
        """Obsługa grid toggle"""
        if hasattr(self.app, 'main_window') and hasattr(self.app.main_window, 'chart_widget'):
            self.app.main_window.chart_widget._toggle_grid(self.grid_var.get())

    def _on_indicators_toggle(self):
        """Obsługa indicators toggle"""
        if hasattr(self.app, 'main_window') and hasattr(self.app.main_window, 'chart_widget'):
            self.app.main_window.chart_widget._toggle_indicators(self.indicators_var.get())

    def _on_indicators_panel_toggle(self, expanded: bool):
        """Callback gdy panel wskaźników jest zwijany/rozwijany"""
        logger.info(f"Indicators panel {'expanded' if expanded else 'collapsed'}")

    # Event handlers - zoom i eksport
    def _on_zoom_in(self):
        """Zoom in"""
        if hasattr(self.app, 'main_window') and hasattr(self.app.main_window, 'chart_widget'):
            self.app.main_window.chart_widget.zoom_in()

    def _on_zoom_out(self):
        """Zoom out"""
        if hasattr(self.app, 'main_window') and hasattr(self.app.main_window, 'chart_widget'):
            self.app.main_window.chart_widget.zoom_out()

    def _on_reset_zoom(self):
        """Reset zoom"""
        if hasattr(self.app, 'main_window') and hasattr(self.app.main_window, 'chart_widget'):
            self.app.main_window.chart_widget.reset_zoom()

    def _on_save_chart(self):
        """Zapisz wykres"""
        if hasattr(self.app, 'main_window') and hasattr(self.app.main_window, 'chart_widget'):
            self.app.main_window.chart_widget.save_chart()

    def _on_export_data(self):
        """Eksportuj dane"""
        if hasattr(self.app, 'main_window') and hasattr(self.app.main_window, 'chart_widget'):
            self.app.main_window.chart_widget.export_data()

    # Event handlers - TMA
    def _on_tma_toggle(self):
        """Obsługa TMA toggle"""
        enabled = self.tma_enabled_var.get()
        self.app.toggle_indicator('TMA_Main', enabled)

    def _on_tma_settings_change(self):
        """Obsługa zmiany ustawień TMA"""
        try:
            new_period = int(self.tma_period_var.get())
            new_atr = float(self.tma_atr_var.get())

            self.app.update_indicator_settings('TMA_Main',
                                               half_length=new_period,
                                               atr_multiplier=new_atr)
        except ValueError as e:
            logger.error(f"Invalid TMA settings: {e}")

    def _load_tma_preset(self, preset_name: str):
        """Ładuje preset TMA"""
        presets = {
            'scalping': {'period': 8, 'atr': 1.8},
            'swing': {'period': 12, 'atr': 2.0},
            'conservative': {'period': 15, 'atr': 2.5}
        }

        if preset_name in presets:
            preset = presets[preset_name]
            self.tma_period_var.set(str(preset['period']))
            self.tma_atr_var.set(str(preset['atr']))
            self._on_tma_settings_change()

    # Event handlers - CCI
    def _on_cci_toggle(self):
        """Obsługa CCI toggle"""
        enabled = self.cci_enabled_var.get()
        self.app.toggle_indicator('CCI_Arrows_Main', enabled)

    def _on_cci_settings_change(self, event=None):
        """Obsługa zmiany ustawień CCI"""
        try:
            new_period = int(self.cci_period_var.get())
            new_overbought = int(self.cci_overbought_var.get())
            new_oversold = int(self.cci_oversold_var.get())
            new_sensitivity = self.cci_sensitivity_var.get()

            self.app.update_indicator_settings('CCI_Arrows_Main',
                                               cci_period=new_period,
                                               overbought_level=new_overbought,
                                               oversold_level=new_oversold,
                                               arrow_sensitivity=new_sensitivity)
        except ValueError as e:
            logger.error(f"Invalid CCI settings: {e}")

    def _load_cci_preset(self, preset_name: str):
        """Ładuje preset CCI"""
        presets = {
            'fast': {'period': 10, 'overbought': 80, 'oversold': -80, 'sensitivity': 'high'},
            'standard': {'period': 14, 'overbought': 100, 'oversold': -100, 'sensitivity': 'medium'},
            'smooth': {'period': 20, 'overbought': 120, 'oversold': -120, 'sensitivity': 'low'}
        }

        if preset_name in presets:
            preset = presets[preset_name]
            self.cci_period_var.set(str(preset['period']))
            self.cci_overbought_var.set(str(preset['overbought']))
            self.cci_oversold_var.set(str(preset['oversold']))
            self.cci_sensitivity_var.set(preset['sensitivity'])
            self._on_cci_settings_change()

    # Utility methods
    def collapse_all_panels(self):
        """Zwija wszystkie panele"""
        for panel in self.panels.values():
            panel.set_expanded(False)

    def expand_all_panels(self):
        """Rozwija wszystkie panele"""
        for panel in self.panels.values():
            panel.set_expanded(True)

    def get_panel(self, panel_name: str) -> CollapsiblePanel:
        """Zwraca panel po nazwie"""
        return self.panels.get(panel_name)

    # Getters
    def get_symbol(self) -> str:
        return self.symbol_var.get()

    def get_timeframe(self) -> str:
        return TIMEFRAMES.get(self.timeframe_var.get(), '5m')

    def get_candles_limit(self) -> int:
        try:
            return int(self.limit_var.get())
        except ValueError:
            return 200

    def get_widget(self):
        return self.widget


class IndicatorConfigPanel:
    """
    Dedykowany panel konfiguracji wskaźników
    Może być używany jako osobne okno lub zakładka
    """

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.widget = None

        # Słownik wszystkich wskaźników
        self.indicator_configs = {}

        self._create_widget()

    def _create_widget(self):
        """Tworzy zaawansowany panel konfiguracji"""
        self.widget = ttk.Frame(self.parent)

        # Nagłówek
        header = ttk.Label(self.widget, text="⚙️ Konfiguracja wskaźników",
                           font=('Arial', 12, 'bold'))
        header.pack(pady=10)

        # Notebook z zakładkami dla różnych kategorii
        notebook = ttk.Notebook(self.widget)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Zakładka: Trend Following
        trend_frame = ttk.Frame(notebook)
        notebook.add(trend_frame, text="📈 Trend Following")
        self._create_trend_indicators(trend_frame)

        # Zakładka: Oscillators
        osc_frame = ttk.Frame(notebook)
        notebook.add(osc_frame, text="🌊 Oscillators")
        self._create_oscillator_indicators(osc_frame)

        # Zakładka: Volume
        vol_frame = ttk.Frame(notebook)
        notebook.add(vol_frame, text="📊 Volume")
        self._create_volume_indicators(vol_frame)

        # Zakładka: Custom
        custom_frame = ttk.Frame(notebook)
        notebook.add(custom_frame, text="🔧 Custom")
        self._create_custom_indicators(custom_frame)

    def _create_trend_indicators(self, parent):
        """Tworzy sekcję wskaźników trendowych"""
        # TMA
        tma_panel = CollapsiblePanel(parent, "📈 TMA - Triangular Moving Average", True)
        tma_panel.get_frame().pack(fill=tk.X, padx=5, pady=2)

        # Placeholder dla kolejnych trend indicators
        ttk.Label(parent, text="🚧 Miejsce na: EMA, SMA, TEMA, etc.",
                  foreground='gray').pack(pady=20)

    def _create_oscillator_indicators(self, parent):
        """Tworzy sekcję oscylatorów"""
        # CCI
        cci_panel = CollapsiblePanel(parent, "📊 CCI - Commodity Channel Index", True)
        cci_panel.get_frame().pack(fill=tk.X, padx=5, pady=2)

        # Placeholder dla kolejnych oscillators
        ttk.Label(parent, text="🚧 Miejsce na: RSI, Stochastic, Williams %R, etc.",
                  foreground='gray').pack(pady=20)

    def _create_volume_indicators(self, parent):
        """Tworzy sekcję wskaźników volume"""
        ttk.Label(parent, text="🚧 Miejsce na: OBV, Volume Profile, A/D Line, etc.",
                  foreground='gray').pack(pady=20)

    def _create_custom_indicators(self, parent):
        """Tworzy sekcję custom indicators"""
        ttk.Label(parent, text="🚧 Miejsce na własne wskaźniki",
                  foreground='gray').pack(pady=20)


class QuickActionsPanel:
    """
    Panel szybkich akcji - często używane funkcje
    """

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.widget = None

        self._create_widget()

    def _create_widget(self):
        """Tworzy panel szybkich akcji"""
        self.widget = ttk.LabelFrame(self.parent, text="⚡ Szybkie akcje", padding=5)

        # Rząd 1: Trading actions
        row1 = ttk.Frame(self.widget)
        row1.pack(fill=tk.X, pady=2)

        ttk.Button(row1, text="🔄 Refresh All",
                   command=self._refresh_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="📸 Screenshot",
                   command=self._take_screenshot).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="🎯 Auto Setup",
                   command=self._auto_setup).pack(side=tk.LEFT, padx=2)

        # Rząd 2: Presets
        row2 = ttk.Frame(self.widget)
        row2.pack(fill=tk.X, pady=5)

        ttk.Label(row2, text="Presets:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="Scalping",
                   command=lambda: self._load_global_preset('scalping')).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Day Trading",
                   command=lambda: self._load_global_preset('daytrading')).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Swing",
                   command=lambda: self._load_global_preset('swing')).pack(side=tk.LEFT, padx=2)

    def _refresh_all(self):
        """Odświeża wszystkie dane"""
        if hasattr(self.app, 'refresh_data'):
            self.app.refresh_data()

    def _take_screenshot(self):
        """Robi screenshot wykresu"""
        if hasattr(self.app, 'main_window') and hasattr(self.app.main_window, 'chart_widget'):
            self.app.main_window.chart_widget.save_chart()

    def _auto_setup(self):
        """Automatyczne ustawienie dla aktualnego timeframe"""
        # Logika auto-setup bazująca na timeframe
        pass

    def _load_global_preset(self, preset_name: str):
        """Ładuje globalny preset dla wszystkich wskaźników"""
        # Logika ładowania presetów
        pass

    def get_widget(self):
        """Zwraca widget"""
        return self.widget