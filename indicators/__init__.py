# ============================================================
# COMPLETE EMA CROSSOVER INTEGRATION FIX
# ============================================================

# === 1. DODAJ DO indicators/__init__.py ===

# 📁 indicators/__init__.py
"""
Trading Indicators Package
Modułowe wskaźniki techniczne
"""

from .manager import IndicatorManager
from .tma import TMAIndicator, create_tma_indicator
from .cci_arrows import CCIArrowsIndicator, create_cci_arrows_indicator
from .ema_crossover import EMACrossoverIndicator, create_ema_crossover_indicator

__all__ = [
    'IndicatorManager',
    'TMAIndicator', 'create_tma_indicator',
    'CCIArrowsIndicator', 'create_cci_arrows_indicator',
    'EMACrossoverIndicator', 'create_ema_crossover_indicator'
]

# === 2. ZAKTUALIZUJ app/main.py ===

# W app/main.py - dodaj import na górze:
from indicators.ema_crossover import EMACrossoverIndicator, create_ema_crossover_indicator


# W metodzie _setup_indicators() zmień na:
def _setup_indicators(self):
    """Konfiguruje dostępne wskaźniki"""
    # Zarejestruj klasy wskaźników
    self.indicator_manager.register_indicator_class('TMA', TMAIndicator)
    self.indicator_manager.register_indicator_class('CCI_Arrows', CCIArrowsIndicator)
    self.indicator_manager.register_indicator_class('EMA_Crossover', EMACrossoverIndicator)

    # Dodaj domyślny TMA
    self.indicator_manager.add_indicator(
        name='TMA_Main',
        indicator_class_name='TMA',
        half_length=12,
        atr_period=100,
        atr_multiplier=2.0,
        angle_threshold=4,
        symbol_type='crypto'
    )

    # Dodaj domyślny CCI Arrows
    self.indicator_manager.add_indicator(
        name='CCI_Arrows_Main',
        indicator_class_name='CCI_Arrows',
        cci_period=14,
        overbought_level=100,
        oversold_level=-100,
        arrow_sensitivity='medium',
        use_divergence=True,
        min_bars_between_signals=3
    )

    # Dodaj domyślny EMA Crossover
    self.indicator_manager.add_indicator(
        name='EMA_Crossover_Main',
        indicator_class_name='EMA_Crossover',
        fast_ema_period=12,
        slow_ema_period=26,
        signal_ema_period=9,
        min_separation=0.0005,
        use_signal_line=True,
        trend_strength_bars=5,
        crossover_confirmation=2,
        price_type='close'
    )

    logger.info("Indicators setup completed - TMA + CCI Arrows + EMA Crossover")


# === 3. ZAKTUALIZUJ app/gui/chart_widget.py ===

# W app/gui/chart_widget.py - dodaj import:
from .chart.plotters import EMACrossoverPlotter

# W __init__ dodaj po innych plotters:
self.ema_plotter = EMACrossoverPlotter(COLORS)

# W display_config dodaj:
self.display_config = {
    'show_volume': True,
    'show_grid': True,
    'show_indicators': True,
    'show_cci': True,
    'show_legend': True,
    'show_tma': True,
    'show_cci_arrows': True,
    'show_ema_crossover': True
}

# W _create_toolbar() w sekcji indicators dodaj:
self.ema_var = tk.BooleanVar(value=True)
ttk.Checkbutton(indicators_frame, text="EMA", variable=self.ema_var,
                command=self._on_ema_toggle).pack(side=tk.LEFT)


# Dodaj event handler:
def _on_ema_toggle(self):
    """Toggle EMA Crossover indicator"""
    self.display_config['show_ema_crossover'] = self.ema_var.get()
    if hasattr(self.app, 'toggle_indicator'):
        self.app.toggle_indicator('EMA_Crossover_Main', self.display_config['show_ema_crossover'])


# W _plot_price_chart dodaj po TMA:
# EMA Crossover indicator
if (self.display_config['show_indicators'] and
        self.display_config['show_ema_crossover'] and
        'EMA_Crossover_Main' in indicators):
    legend_items = self.ema_plotter.plot(self.axes['price'], indicators['EMA_Crossover_Main'], len(df))
    if legend_items and self.display_config['show_legend']:
        self._setup_legend('ema', self.axes['price'], legend_items)

# === 4. ZAKTUALIZUJ app/gui/control_panel.py ===

# W app/gui/control_panel.py - dodaj zmienne w __init__:
# Zmienne GUI - EMA Crossover
self.ema_enabled_var = tk.BooleanVar(value=True)
self.ema_fast_var = tk.StringVar(value='12')
self.ema_slow_var = tk.StringVar(value='26')
self.ema_signal_var = tk.StringVar(value='9')

# W _create_indicators_controls dodaj po CCI sekcji:
# === EMA CROSSOVER SEKCJA ===
ema_section = ExpandableSection(parent, "📈 EMA Crossover (Exponential Moving Average)", True)
self._create_ema_controls(ema_section.get_content_frame())


# Dodaj nową metodę:
def _create_ema_controls(self, parent):
    """Tworzy kontrolki EMA Crossover"""
    # Włączenie EMA
    control_row = ttk.Frame(parent)
    control_row.pack(fill=tk.X, pady=2)

    ttk.Checkbutton(control_row, text="Włącz EMA Crossover", variable=self.ema_enabled_var,
                    command=self._on_ema_toggle).pack(side=tk.LEFT)

    # Parametry EMA - rząd 1
    params_row1 = ttk.Frame(parent)
    params_row1.pack(fill=tk.X, pady=2)

    # Fast EMA
    ttk.Label(params_row1, text="Fast:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
    fast_spin = ttk.Spinbox(params_row1, from_=5, to=50, textvariable=self.ema_fast_var,
                            width=4, command=self._on_ema_settings_change)
    fast_spin.pack(side=tk.LEFT, padx=2)

    # Slow EMA
    ttk.Label(params_row1, text="Slow:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(10, 5))
    slow_spin = ttk.Spinbox(params_row1, from_=10, to=100, textvariable=self.ema_slow_var,
                            width=4, command=self._on_ema_settings_change)
    slow_spin.pack(side=tk.LEFT, padx=2)

    # Signal EMA
    ttk.Label(params_row1, text="Signal:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(10, 5))
    signal_spin = ttk.Spinbox(params_row1, from_=3, to=30, textvariable=self.ema_signal_var,
                              width=4, command=self._on_ema_settings_change)
    signal_spin.pack(side=tk.LEFT, padx=2)

    # Presety EMA
    presets_row = ttk.Frame(parent)
    presets_row.pack(fill=tk.X, pady=2)

    ttk.Label(presets_row, text="Presety:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
    ttk.Button(presets_row, text="12/26/9",
               command=lambda: self._load_ema_preset('standard')).pack(side=tk.LEFT, padx=2)
    ttk.Button(presets_row, text="8/21/5",
               command=lambda: self._load_ema_preset('fast')).pack(side=tk.LEFT, padx=2)
    ttk.Button(presets_row, text="21/55/13",
               command=lambda: self._load_ema_preset('slow')).pack(side=tk.LEFT, padx=2)


# Dodaj event handlery:
def _on_ema_toggle(self):
    """Obsługa EMA toggle"""
    enabled = self.ema_enabled_var.get()
    self.app.toggle_indicator('EMA_Crossover_Main', enabled)


def _on_ema_settings_change(self):
    """Obsługa zmiany ustawień EMA"""
    try:
        fast_period = int(self.ema_fast_var.get())
        slow_period = int(self.ema_slow_var.get())
        signal_period = int(self.ema_signal_var.get())

        # Walidacja: Fast < Slow
        if fast_period >= slow_period:
            logger.warning("Fast EMA must be less than Slow EMA")
            return

        self.app.update_indicator_settings('EMA_Crossover_Main',
                                           fast_ema_period=fast_period,
                                           slow_ema_period=slow_period,
                                           signal_ema_period=signal_period)
    except ValueError as e:
        logger.error(f"Invalid EMA settings: {e}")


def _load_ema_preset(self, preset_name: str):
    """Ładuje preset EMA"""
    presets = {
        'standard': {'fast': 12, 'slow': 26, 'signal': 9},
        'fast': {'fast': 8, 'slow': 21, 'signal': 5},
        'slow': {'fast': 21, 'slow': 55, 'signal': 13}
    }

    if preset_name in presets:
        preset = presets[preset_name]
        self.ema_fast_var.set(str(preset['fast']))
        self.ema_slow_var.set(str(preset['slow']))
        self.ema_signal_var.set(str(preset['signal']))
        self._on_ema_settings_change()


# === 5. ZAKTUALIZUJ app/gui/info_panel.py ===

# W app/gui/info_panel.py - dodaj zmienną w __init__:
# EMA status
self.ema_status_var = tk.StringVar(value="---")

# W _create_indicators_row po CCI dodaj:
# Separator
ttk.Separator(indicators_row, orient='vertical').pack(side=tk.LEFT, padx=10, fill=tk.Y)

# EMA Status
ttk.Label(indicators_row, text="EMA:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
self.ema_status_label = ttk.Label(indicators_row, textvariable=self.ema_status_var,
                                  font=('Arial', 9, 'bold'))
self.ema_status_label.pack(side=tk.LEFT, padx=5)


# W _update_indicators_status dodaj:
def _update_indicators_status(self):
    """Aktualizuje status wszystkich wskaźników"""
    self._update_tma_status()
    self._update_cci_status()
    self._update_ema_status()


# Dodaj nową metodę:
def _update_ema_status(self):
    """Aktualizuje status EMA Crossover"""
    try:
        if not hasattr(self.app, 'get_indicator_manager'):
            self.ema_status_var.set("---")
            return

        indicator_manager = self.app.get_indicator_manager()
        ema_indicator = indicator_manager.get_indicator('EMA_Crossover_Main')

        if ema_indicator and ema_indicator.enabled:
            # Pobierz ostatnie wyniki
            results = indicator_manager.get_results('EMA_Crossover_Main')
            if results:
                signals = ema_indicator.get_latest_signal(results)

                signal = signals.get('signal', 'none')
                trend = signals.get('trend', 'neutral')
                strength = signals.get('trend_strength', 0)
                signal_age = signals.get('signal_age', 0)

                # Określ status i kolor
                if signal == 'buy' and signal_age <= 3:
                    status_text = "🡱 BUY"
                    color = COLORS['accent_green']
                elif signal == 'sell' and signal_age <= 3:
                    status_text = "🡳 SELL"
                    color = COLORS['accent_red']
                elif trend == 'bullish':
                    status_text = "↗ BULL"
                    color = COLORS['accent_green']
                elif trend == 'bearish':
                    status_text = "↘ BEAR"
                    color = COLORS['accent_red']
                else:
                    status_text = "→ FLAT"
                    color = 'gray'

                # Dodaj wskaźnik siły trendu
                if strength > 0.7:
                    status_text += " 💪"
                elif strength < 0.3:
                    status_text += " 🌊"

                self.ema_status_var.set(status_text)
                self.ema_status_label.configure(foreground=color)
            else:
                self.ema_status_var.set("CALC...")
                self.ema_status_label.configure(foreground='orange')
        else:
            self.ema_status_var.set("OFF")
            self.ema_status_label.configure(foreground='gray')

    except Exception as e:
        logger.error(f"Error updating EMA status: {e}")
        self.ema_status_var.set("---")


# === 6. ZAKTUALIZUJ app/gui/chart/base.py ===

# W app/gui/chart/base.py - w ChartTooltipBuilder dodaj metodę:
def _build_ema_info(self, x_index: int, indicators: Dict) -> List[str]:
    """Buduje info EMA Crossover"""
    if 'EMA_Crossover_Main' not in indicators:
        return []

    try:
        ema_data = indicators['EMA_Crossover_Main']
        valid_from = ema_data.get('valid_from', 0)

        if x_index >= valid_from and x_index < len(ema_data['fast_ema']):
            info = []
            fast_ema = ema_data['fast_ema'][x_index]
            slow_ema = ema_data['slow_ema'][x_index]

            info.append(f"Fast EMA: {fast_ema:.4f}")
            info.append(f"Slow EMA: {slow_ema:.4f}")
            info.append(f"Diff: {fast_ema - slow_ema:+.4f}")

            # Trend direction
            if fast_ema > slow_ema:
                info.append("📈 Bullish")
            else:
                info.append("📉 Bearish")

            # Sygnały
            if ema_data['buy_signals'][x_index] > 0:
                info.append("🟢 EMA BUY")
            elif ema_data['sell_signals'][x_index] > 0:
                info.append("🔴 EMA SELL")

            # Signal EMA (jeśli używana)
            if ema_data.get('settings', {}).get('use_signal_line', False):
                signal_ema = ema_data.get('signal_ema')
                if signal_ema is not None and x_index < len(signal_ema):
                    info.append(f"Signal: {signal_ema[x_index]:.4f}")

            return info
    except Exception as e:
        logger.error(f"Error building EMA tooltip: {e}")

    return []


# W build_tooltip po CCI info dodaj:
# EMA info
ema_info = self._build_ema_info(x_index, indicators)
if ema_info:
    info_parts.extend(ema_info)

# === 7. ZAKTUALIZUJ app/gui/chart/__init__.py ===

# W app/gui/chart/__init__.py - dodaj EMACrossoverPlotter do importów:
from .plotters import (
    CandlestickPlotter,
    VolumePlotter,
    TMAPlotter,
    CCIPlotter,
    EMACrossoverPlotter,
    MACDStyleEMAPlotter,
    SignalPlotter,
    GridPlotter,
    LevelPlotter,
    FillPlotter
)

# I do __all__:
__all__ = [
    # Base components
    'ChartDataManager',
    'ChartLayoutManager',
    'ChartTooltipBuilder',
    'ChartFormatter',
    'ChartTitleBuilder',

    # Plotters
    'CandlestickPlotter',
    'VolumePlotter',
    'TMAPlotter',
    'CCIPlotter',
    'EMACrossoverPlotter',
    'MACDStyleEMAPlotter',
    'SignalPlotter',
    'GridPlotter',
    'LevelPlotter',
    'FillPlotter',

    # Event handlers
    'ChartEventHandler',
    'ChartZoomHandler',
    'ChartExportHandler',
    'ChartInteractionHandler'
]

# === 8. ZAKTUALIZUJ run.py ===

# W run.py zmień opisy:
print("📈 Wskaźniki: TMA + CCI Arrows + EMA Crossover")
print("🎯 Strategia: Triple-Indicator Professional Trading")
print("=" * 60)
print("🔧 Funkcje:")
print("   • TMA z pasmami ATR - sygnały odbicia")
print("   • CCI Arrows - momentum i punkty zwrotne")
print("   • EMA Crossover - trend following signals")
print("   • Triple confirmation system")
print("   • Multi-timeframe analysis")
print("   • Auto-refresh danych w czasie rzeczywistym")
print("   • Eksport wykresów i danych")
print("   • Składane panele kontrolne")
print("   • Skróty klawiszowe i zaawansowane zoom")

# === 9. STWÓRZ test_integration.py ===

# 📁 test_integration.py
"""
Test integration po dodaniu EMA Crossover
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test czy wszystkie importy działają"""
    print("🧪 Testing imports...")

    try:
        # Test indicators
        from indicators import TMAIndicator, CCIArrowsIndicator, EMACrossoverIndicator
        print("✅ Indicators import OK")

        # Test manager
        from indicators.manager import IndicatorManager
        print("✅ IndicatorManager import OK")

        # Test plotters
        from app.gui.chart.plotters import EMACrossoverPlotter, TMAPlotter, CCIPlotter
        print("✅ Plotters import OK")

        # Test main app
        from app.main import TradingPlatform
        print("✅ TradingPlatform import OK")

        print("🎉 All imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_indicator_creation():
    """Test tworzenia wskaźników"""
    print("\n🧪 Testing indicator creation...")

    try:
        from indicators import EMACrossoverIndicator, TMAIndicator, CCIArrowsIndicator
        from indicators.manager import IndicatorManager

        # Test IndicatorManager
        manager = IndicatorManager()
        print("✅ IndicatorManager created")

        # Register indicators
        manager.register_indicator_class('TMA', TMAIndicator)
        manager.register_indicator_class('CCI_Arrows', CCIArrowsIndicator)
        manager.register_indicator_class('EMA_Crossover', EMACrossoverIndicator)
        print("✅ All indicator classes registered")

        # Add indicators
        success1 = manager.add_indicator('TMA_Test', 'TMA', half_length=12)
        success2 = manager.add_indicator('CCI_Test', 'CCI_Arrows', cci_period=14)
        success3 = manager.add_indicator('EMA_Test', 'EMA_Crossover', fast_ema_period=12, slow_ema_period=26)

        if success1 and success2 and success3:
            print("✅ All indicators added successfully")
            print(f"✅ Total indicators: {len(manager.indicators)}")
            return True
        else:
            print("❌ Failed to add some indicators")
            return False

    except Exception as e:
        print(f"❌ Indicator creation error: {e}")
        return False


def test_with_data():
    """Test ze rzeczywistymi danymi"""
    print("\n🧪 Testing with real data...")

    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        from indicators import EMACrossoverIndicator

        # Generate test data
        np.random.seed(42)
        periods = 100
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')

        base_price = 45000
        returns = np.random.randn(periods) * 0.01
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(periods) * 0.001),
            'high': prices * (1 + abs(np.random.randn(periods)) * 0.005),
            'low': prices * (1 - abs(np.random.randn(periods)) * 0.005),
            'close': prices,
            'volume': np.random.uniform(100, 1000, periods)
        }, index=dates)

        print(f"✅ Generated {len(df)} test candles")

        # Test EMA calculation
        ema_indicator = EMACrossoverIndicator(
            name="EMA_Test",
            fast_ema_period=12,
            slow_ema_period=26,
            signal_ema_period=9
        )

        result = ema_indicator.calculate(df)

        if result:
            buy_signals = np.sum(result['buy_signals'] > 0)
            sell_signals = np.sum(result['sell_signals'] > 0)
            print(f"✅ EMA calculation successful!")
            print(f"   • Buy signals: {buy_signals}")
            print(f"   • Sell signals: {sell_signals}")
            print(f"   • Crossovers: {len(result['crossover_points'])}")
            return True
        else:
            print("❌ EMA calculation failed")
            return False

    except Exception as e:
        print(f"❌ Data test error: {e}")
        return False


def run_integration_test():
    """Uruchom pełny test integracji"""
    print("🚀 EMA CROSSOVER INTEGRATION TEST")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Indicator Creation Test", test_indicator_creation),
        ("Data Processing Test", test_with_data)
    ]

    passed = 0
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")

    print(f"\n📊 RESULTS: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 ALL TESTS PASSED! EMA Crossover integration is ready!")
        print("\n✅ Ready to run: python run.py")
    else:
        print("❌ Some tests failed. Check the errors above.")

    return passed == len(tests)


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)

# === 10. QUICK SETUP CHECKLIST ===

print("""
🔧 EMA CROSSOVER INTEGRATION CHECKLIST
=====================================

□ 1. Stwórz indicators/ema_crossover.py (skopiuj z pierwszego artifact)
□ 2. Zastąp app/gui/chart/plotters.py (skopiuj fixed_plotters)
□ 3. Zaktualizuj indicators/__init__.py (dodaj EMA import)
□ 4. Zaktualizuj app/main.py (dodaj EMA setup)
□ 5. Zaktualizuj app/gui/chart_widget.py (dodaj EMA plotting)
□ 6. Zaktualizuj app/gui/control_panel.py (dodaj EMA controls)
□ 7. Zaktualizuj app/gui/info_panel.py (dodaj EMA status)
□ 8. Zaktualizuj app/gui/chart/base.py (dodaj EMA tooltip)
□ 9. Zaktualizuj app/gui/chart/__init__.py (dodaj EMA import)
□ 10. Zaktualizuj run.py (nowe opisy)

TESTING:
□ 1. Uruchom: python test_integration.py
□ 2. Jeśli OK, uruchom: python run.py
□ 3. Sprawdź GUI - powinien być checkbox "EMA" w toolbar
□ 4. Sprawdź control panel - sekcja EMA Crossover
□ 5. Sprawdź info panel - status EMA

EXPECTED RESULT:
✅ 3 wskaźniki: TMA + CCI + EMA
✅ Linie EMA na wykresie (złota + niebieska)
✅ Buy/Sell arrows
✅ Real-time status updates
✅ Parameter controls
✅ Triple confirmation system

🎯 TRADING COMBINATIONS:
• TMA Buy + EMA Bullish + CCI Oversold = STRONG LONG
• TMA Sell + EMA Bearish + CCI Overbought = STRONG SHORT
• All 3 aligned = HIGH CONFIDENCE TRADE
• Conflicting signals = WAIT FOR CLARITY

🚀 READY FOR PROFESSIONAL TRADING!
""")

# === TROUBLESHOOTING ===

print("""
🔧 TROUBLESHOOTING GUIDE
=======================

❌ ImportError: No module named 'ema_crossover'
   → Sprawdź czy indicators/ema_crossover.py istnieje
   → Sprawdź czy indicators/__init__.py ma import EMA

❌ AttributeError: 'EMACrossoverPlotter' object has no attribute...
   → Zastąp cały app/gui/chart/plotters.py fixed wersją
   → Sprawdź czy wszystkie metody są zdefiniowane

❌ KeyError: 'EMA_Crossover_Main'
   → Sprawdź czy app/main.py ma setup EMA w _setup_indicators()
   → Sprawdź czy IndicatorManager ma zarejestrowaną klasę

❌ GUI nie pokazuje EMA controls
   → Sprawdź czy control_panel.py ma _create_ema_controls()
   → Sprawdź czy jest wywoływana w _create_indicators_controls()

❌ Brak statusu EMA w info panel
   → Sprawdź czy info_panel.py ma _update_ema_status()
   → Sprawdź czy jest wywoływana w _update_indicators_status()

❌ Wykres nie pokazuje linii EMA
   → Sprawdź czy chart_widget.py ma ema_plotter
   → Sprawdź czy _plot_price_chart() ma kod EMA
   → Sprawdź czy checkbox EMA jest zaznaczony

💡 GENERAL DEBUG TIPS:
   • Sprawdź logi w logs/trading_platform.log
   • Uruchom python test_integration.py
   • Sprawdź czy wszystkie imports działają
   • Restartuj aplikację po zmianach
""")

# === FINAL SUCCESS MESSAGE ===

print("""
🎉 EMA CROSSOVER INTEGRATION COMPLETE!
=====================================

📊 YOUR PROFESSIONAL TRADING PLATFORM NOW HAS:

🔹 TMA (Triangular Moving Average)
   • Reversion signals
   • ATR bands
   • Trend analysis

🔹 CCI Arrows (Commodity Channel Index)  
   • Momentum signals
   • Overbought/Oversold
   • Divergence detection

🔹 EMA Crossover (Exponential Moving Average) ⭐ NEW!
   • Trend following
   • Crossover signals
   • Trend strength analysis
   • Signal filtering

🚀 TRIPLE CONFIRMATION SYSTEM READY!
📈 MT5-STYLE PROFESSIONAL INTERFACE!
⚡ REAL-TIME MULTI-INDICATOR ANALYSIS!

Ready to start professional trading! 🎯
""")