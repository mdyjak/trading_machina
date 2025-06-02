# 📁 app/gui/info_panel.py
"""
Panel informacyjny - statystyki rynkowe i wskaźniki
"""
import colorsys
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import logging
from typing import Dict

# Poprawiony import - używamy względnych importów
from ..config.settings import COLORS

logger = logging.getLogger(__name__)


def format_number(value: float, precision: int = 2) -> str:
    """Formatuje liczbę z odpowiednią precyzją"""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:.{precision}f}"


def format_percentage(value: float, precision: int = 2) -> str:
    """Formatuje procent"""
    return f"{value:+.{precision}f}%"


class InfoPanel:
    """
    Panel informacyjny z danymi rynkowymi i statusem wskaźników
    """

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.widget = None

        # Zmienne GUI
        self.price_var = tk.StringVar(value="0.00 USDT")
        self.change_var = tk.StringVar(value="0.00 (0.00%)")
        self.volume24h_var = tk.StringVar(value="0")
        self.last_update_var = tk.StringVar(value="Nigdy")
        self.status_var = tk.StringVar(value="Inicjalizacja...")
        self.stats_var = tk.StringVar(value="")

        # TMA status
        self.tma_status_var = tk.StringVar(value="---")

        # CCI status
        self.cci_status_var = tk.StringVar(value="---")
        self.cci_value_var = tk.StringVar(value="0")

        # EMA status
        self.ema_status_var = tk.StringVar(value="---")
        self.ema_values_var = tk.StringVar(value="12/26")

        # SMI status
        self.smi_status_var = tk.StringVar(value="---")
        self.smi_value_var = tk.StringVar(value="0")

        # RSI status
        self.rsi_status_var = tk.StringVar(value="---")
        self.rsi_value_var = tk.StringVar(value="50")

        # Bollinger status
        self.bollinger_status_var = tk.StringVar(value="---")
        self.bollinger_position_var = tk.StringVar(value="Middle")

        self._create_widget()
        logger.info("InfoPanel initialized")

    def _create_widget(self):
        """Tworzy widget panelu informacyjnego"""
        self.widget = ttk.LabelFrame(self.parent, text="Informacje rynkowe", padding=10)

        # Główny rząd z ceną
        self._create_price_row()

        # Wskaźniki
        self._create_indicators_row()

        # Status i statystyki
        self._create_status_row()

    def _create_price_row(self):
        """Tworzy rząd z informacjami o cenie"""
        price_row = ttk.Frame(self.widget)
        price_row.pack(fill=tk.X)

        # Lewa strona - cena
        left_info = ttk.Frame(price_row)
        left_info.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Cena
        ttk.Label(left_info, text="Cena:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT)
        price_label = ttk.Label(left_info, textvariable=self.price_var,
                                font=('Arial', 16, 'bold'), foreground='#B900CD')
        price_label.pack(side=tk.LEFT, padx=10)

        # Zmiana ceny
        self.change_label = ttk.Label(left_info, textvariable=self.change_var, font=('Arial', 12))
        self.change_label.pack(side=tk.LEFT, padx=10)

        # Prawa strona - dodatkowe info
        right_info = ttk.Frame(price_row)
        right_info.pack(side=tk.RIGHT)

        # Volume 24h
        ttk.Label(right_info, text="Vol 24h:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        ttk.Label(right_info, textvariable=self.volume24h_var,
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)

        # Ostatnia aktualizacja
        ttk.Label(right_info, text="Update:", font=('Arial', 9)).pack(side=tk.LEFT, padx=10)
        ttk.Label(right_info, textvariable=self.last_update_var, font=('Arial', 9)).pack(side=tk.LEFT)

    def _create_indicators_row(self):
        """Tworzy rząd ze wskaźnikami"""
        indicators_row = ttk.Frame(self.widget)
        indicators_row.pack(fill=tk.X, pady=(10, 0))

        # TMA Status
        ttk.Label(indicators_row, text="TMA:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.tma_status_label = ttk.Label(indicators_row, textvariable=self.tma_status_var,
                                          font=('Arial', 9, 'bold'))
        self.tma_status_label.pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(indicators_row, orient='vertical').pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # CCI Status
        ttk.Label(indicators_row, text="CCI:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.cci_status_label = ttk.Label(indicators_row, textvariable=self.cci_status_var,
                                          font=('Arial', 9, 'bold'))
        self.cci_status_label.pack(side=tk.LEFT, padx=5)

        # CCI Value
        ttk.Label(indicators_row, text="Val:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        self.cci_value_label = ttk.Label(indicators_row, textvariable=self.cci_value_var,
                                         font=('Arial', 8))
        self.cci_value_label.pack(side=tk.LEFT, padx=2)

        # EMA Status
        # Separator
        ttk.Separator(indicators_row, orient='vertical').pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # EMA Status
        ttk.Label(indicators_row, text="EMA:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.ema_status_label = ttk.Label(indicators_row, textvariable=self.ema_status_var,
                                          font=('Arial', 9, 'bold'))
        self.ema_status_label.pack(side=tk.LEFT, padx=5)

        # EMA Values
        ttk.Label(indicators_row, text="Val:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        self.ema_values_label = ttk.Label(indicators_row, textvariable=self.ema_values_var,
                                          font=('Arial', 8))
        self.ema_values_label.pack(side=tk.LEFT, padx=2)

        # Separator
        ttk.Separator(indicators_row, orient='vertical').pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # SMI Status
        ttk.Label(indicators_row, text="SMI:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.smi_status_label = ttk.Label(indicators_row, textvariable=self.smi_status_var,
                                          font=('Arial', 9, 'bold'))
        self.smi_status_label.pack(side=tk.LEFT, padx=5)

        # SMI Value
        ttk.Label(indicators_row, text="Val:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        self.smi_value_label = ttk.Label(indicators_row, textvariable=self.smi_value_var,
                                         font=('Arial', 8))
        self.smi_value_label.pack(side=tk.LEFT, padx=2)

        # Separator
        ttk.Separator(indicators_row, orient='vertical').pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # RSI Status
        ttk.Label(indicators_row, text="RSI:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.rsi_status_label = ttk.Label(indicators_row, textvariable=self.rsi_status_var,
                                          font=('Arial', 9, 'bold'))
        self.rsi_status_label.pack(side=tk.LEFT, padx=5)

        # RSI Value
        ttk.Label(indicators_row, text="Val:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        self.rsi_value_label = ttk.Label(indicators_row, textvariable=self.rsi_value_var,
                                         font=('Arial', 8))
        self.rsi_value_label.pack(side=tk.LEFT, padx=2)

        # Separator
        ttk.Separator(indicators_row, orient='vertical').pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # Bollinger Status
        ttk.Label(indicators_row, text="BB:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.bollinger_status_label = ttk.Label(indicators_row, textvariable=self.bollinger_status_var,
                                                font=('Arial', 9, 'bold'))
        self.bollinger_status_label.pack(side=tk.LEFT, padx=5)

        # Bollinger Position
        ttk.Label(indicators_row, text="Pos:", font=('Arial', 8)).pack(side=tk.LEFT, padx=5)
        self.bollinger_position_label = ttk.Label(indicators_row, textvariable=self.bollinger_position_var,
                                                  font=('Arial', 8))
        self.bollinger_position_label.pack(side=tk.LEFT, padx=2)

    def _create_status_row(self):
        """Tworzy rząd ze statusem"""
        status_row = ttk.Frame(self.widget)
        status_row.pack(fill=tk.X, pady=(5, 0))

        # Status
        ttk.Label(status_row, textvariable=self.status_var, font=('Arial', 9)).pack(side=tk.LEFT)

        # Statystyki
        ttk.Label(status_row, textvariable=self.stats_var, font=('Arial', 9)).pack(side=tk.RIGHT)

    def update_market_info(self, market_stats: Dict):
        """Aktualizuje informacje rynkowe"""
        try:
            if not market_stats:
                return

            # Cena
            current_price = market_stats.get('current_price', 0)
            self.price_var.set(f"{current_price:.2f} USDT")

            # Zmiana ceny
            price_change = market_stats.get('price_change', 0)
            price_change_pct = market_stats.get('price_change_percent', 0)
            change_text = f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            self.change_var.set(change_text)

            # Kolor zmiany
            if price_change > 0:
                self.change_label.configure(foreground=COLORS['accent_green'])
            elif price_change < 0:
                self.change_label.configure(foreground=COLORS['accent_red'])
            else:
                self.change_label.configure(foreground='gray')

            # Volume 24h
            volume_24h = market_stats.get('volume_24h', 0)
            self.volume24h_var.set(format_number(volume_24h))

            # Czas aktualizacji
            self.last_update_var.set(datetime.now().strftime('%H:%M:%S'))

            # Aktualizuj status wskaźników
            self._update_indicators_status()

            # Statystyki
            high_24h = market_stats.get('high_24h', 0)
            low_24h = market_stats.get('low_24h', 0)
            if high_24h > 0 and low_24h > 0:
                self.stats_var.set(f"H: {high_24h:.2f} | L: {low_24h:.2f}")

        except Exception as e:
            logger.error(f"Error updating market info: {e}")

    def _update_indicators_status(self):
        """Aktualizuje status wszystkich wskaźników"""
        self._update_tma_status()
        self._update_cci_status()
        self._update_ema_status()
        self._update_smi_status()
        self._update_rsi_status()
        self._update_bollinger_status()

    def _update_tma_status(self):
        """Aktualizuje status TMA"""
        try:
            if not hasattr(self.app, 'get_indicator_manager'):
                self.tma_status_var.set("---")
                return

            indicator_manager = self.app.get_indicator_manager()
            tma_indicator = indicator_manager.get_indicator('TMA_Main')

            if tma_indicator and tma_indicator.enabled:
                # Pobierz ostatnie wyniki
                results = indicator_manager.get_results('TMA_Main')
                if results:
                    signals = tma_indicator.get_latest_signal(results)

                    trend = signals.get('trend', 'unknown')
                    caution = signals.get('caution', False)

                    if trend == 'bullish':
                        status_text = "▲ UP"
                        color = COLORS['accent_green']
                    elif trend == 'bearish':
                        status_text = "▼ DOWN"
                        color = COLORS['accent_red']
                    elif trend == 'sideways':
                        status_text = "→ FLAT"
                        color = 'gray'
                    else:
                        status_text = "---"
                        color = 'gray'

                    if caution:
                        status_text += " ⚠️"

                    self.tma_status_var.set(status_text)
                    self.tma_status_label.configure(foreground=color)
                else:
                    self.tma_status_var.set("CALC...")
                    self.tma_status_label.configure(foreground='orange')
            else:
                self.tma_status_var.set("OFF")
                self.tma_status_label.configure(foreground='gray')

        except Exception as e:
            logger.error(f"Error updating TMA status: {e}")
            self.tma_status_var.set("---")

    def _update_cci_status(self):
        """Aktualizuje status CCI Arrows"""
        try:
            if not hasattr(self.app, 'get_indicator_manager'):
                self.cci_status_var.set("---")
                self.cci_value_var.set("0")
                return

            indicator_manager = self.app.get_indicator_manager()
            cci_indicator = indicator_manager.get_indicator('CCI_Arrows_Main')

            if cci_indicator and cci_indicator.enabled:
                # Pobierz ostatnie wyniki
                results = indicator_manager.get_results('CCI_Arrows_Main')
                if results:
                    signals = cci_indicator.get_latest_signal(results)

                    signal = signals.get('signal', 'none')
                    trend = signals.get('trend', 'neutral')
                    cci_value = signals.get('cci_value', 0)
                    signal_age = signals.get('signal_age', 0)

                    # Aktualizuj wartość CCI
                    self.cci_value_var.set(f"{cci_value:.0f}")

                    # Określ status i kolor
                    if signal == 'buy' and signal_age <= 5:
                        status_text = "🡱 BUY"
                        color = COLORS['accent_green']
                    elif signal == 'sell' and signal_age <= 5:
                        status_text = "🡳 SELL"
                        color = COLORS['accent_red']
                    elif trend == 'overbought':
                        status_text = "OB"
                        color = COLORS['accent_red']
                    elif trend == 'oversold':
                        status_text = "OS"
                        color = COLORS['accent_green']
                    elif trend == 'bullish':
                        status_text = "↗ UP"
                        color = COLORS['accent_green']
                    elif trend == 'bearish':
                        status_text = "↘ DOWN"
                        color = COLORS['accent_red']
                    else:
                        status_text = "NEUTRAL"
                        color = 'gray'

                    # Dodaj oznaczenie dywergencji
                    if signals.get('divergence', False):
                        status_text += " ⚡"

                    self.cci_status_var.set(status_text)
                    self.cci_status_label.configure(foreground=color)

                    # Kolor wartości CCI
                    if cci_value > 100:
                        self.cci_value_label.configure(foreground=COLORS['accent_red'])
                    elif cci_value < -100:
                        self.cci_value_label.configure(foreground=COLORS['accent_green'])
                    else:
                        self.cci_value_label.configure(foreground=COLORS['text_primary'])

                else:
                    self.cci_status_var.set("CALC...")
                    self.cci_value_var.set("0")
                    self.cci_status_label.configure(foreground='orange')
                    self.cci_value_label.configure(foreground='orange')
            else:
                self.cci_status_var.set("OFF")
                self.cci_value_var.set("0")
                self.cci_status_label.configure(foreground='gray')
                self.cci_value_label.configure(foreground='gray')

        except Exception as e:
            logger.error(f"Error updating CCI status: {e}")
            self.cci_status_var.set("---")
            self.cci_value_var.set("0")

    def _update_ema_status(self):
        """Aktualizuje status EMA Crossover"""
        try:
            if not hasattr(self.app, 'get_indicator_manager'):
                self.ema_status_var.set("---")
                self.ema_values_var.set("12/26")
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
                    signal_age = signals.get('signal_age', 0)
                    strength = signals.get('strength', 0)

                    # Określ status i kolor
                    if signal == 'buy' and signal_age <= 3:
                        status_text = "🡱 BUY"
                        color = COLORS['accent_green']
                    elif signal == 'sell' and signal_age <= 3:
                        status_text = "🡳 SELL"
                        color = COLORS['accent_red']
                    elif trend == 'bullish':
                        status_text = "↗ UP"
                        color = COLORS['accent_green']
                    elif trend == 'bearish':
                        status_text = "↘ DOWN"
                        color = COLORS['accent_red']
                    else:
                        status_text = "→ FLAT"
                        color = 'gray'

                    # Dodaj siłę trendu
                    if strength > 0.7:
                        status_text += " 💪"
                    elif strength < 0.3:
                        status_text += " 📉"

                    self.ema_status_var.set(status_text)
                    self.ema_status_label.configure(foreground=color)

                    # Wartości EMA
                    fast_ema = signals.get('fast_ema', 0)
                    slow_ema = signals.get('slow_ema', 0)
                    self.ema_values_var.set(f"{fast_ema:.1f}/{slow_ema:.1f}")
                    self.ema_values_label.configure(foreground=COLORS['text_primary'])

                else:
                    self.ema_status_var.set("CALC...")
                    self.ema_values_var.set("12/26")
                    self.ema_status_label.configure(foreground='orange')
                    self.ema_values_label.configure(foreground='orange')
            else:
                self.ema_status_var.set("OFF")
                self.ema_values_var.set("12/26")
                self.ema_status_label.configure(foreground='gray')
                self.ema_values_label.configure(foreground='gray')

        except Exception as e:
            logger.error(f"Error updating EMA status: {e}")
            self.ema_status_var.set("---")
            self.ema_values_var.set("12/26")

    def _update_smi_status(self):
        """Aktualizuje status SMI Arrows"""
        try:
            if not hasattr(self.app, 'get_indicator_manager'):
                self.smi_status_var.set("---")
                self.smi_value_var.set("0")
                return

            indicator_manager = self.app.get_indicator_manager()
            smi_indicator = indicator_manager.get_indicator('SMI_Arrows_Main')

            if smi_indicator and smi_indicator.enabled:
                # Pobierz ostatnie wyniki
                results = indicator_manager.get_results('SMI_Arrows_Main')
                if results:
                    signals = smi_indicator.get_latest_signal(results)

                    signal = signals.get('signal', 'none')
                    trend = signals.get('trend', 'neutral')
                    smi_value = signals.get('smi_value', 0)
                    signal_age = signals.get('signal_age', 0)

                    # Aktualizuj wartość SMI
                    self.smi_value_var.set(f"{smi_value:.1f}")

                    # Określ status i kolor
                    if signal == 'buy' and signal_age <= 5:
                        status_text = "🡱 BUY"
                        color = COLORS['accent_green']
                    elif signal == 'sell' and signal_age <= 5:
                        status_text = "🡳 SELL"
                        color = COLORS['accent_red']
                    elif trend == 'overbought':
                        status_text = "OB"
                        color = COLORS['accent_red']
                    elif trend == 'oversold':
                        status_text = "OS"
                        color = COLORS['accent_green']
                    elif trend == 'bullish':
                        status_text = "↗ UP"
                        color = COLORS['accent_green']
                    elif trend == 'bearish':
                        status_text = "↘ DOWN"
                        color = COLORS['accent_red']
                    else:
                        status_text = "NEUTRAL"
                        color = 'gray'

                    # Dodaj oznaczenie dywergencji
                    if signals.get('divergence', False):
                        status_text += " ⚡"

                    self.smi_status_var.set(status_text)
                    self.smi_status_label.configure(foreground=color)

                    # Kolor wartości SMI
                    if smi_value > 40:
                        self.smi_value_label.configure(foreground=COLORS['accent_red'])
                    elif smi_value < -40:
                        self.smi_value_label.configure(foreground=COLORS['accent_green'])
                    else:
                        self.smi_value_label.configure(foreground=COLORS['text_primary'])

                else:
                    self.smi_status_var.set("CALC...")
                    self.smi_value_var.set("0")
                    self.smi_status_label.configure(foreground='orange')
            else:
                self.smi_status_var.set("OFF")
                self.smi_value_var.set("0")
                self.smi_status_label.configure(foreground='gray')

        except Exception as e:
            logger.error(f"Error updating SMI status: {e}")
            self.smi_status_var.set("---")
            self.smi_value_var.set("0")

    def _update_rsi_status(self):
        """Aktualizuje status RSI Professional"""
        try:
            if not hasattr(self.app, 'get_indicator_manager'):
                self.rsi_status_var.set("---")
                self.rsi_value_var.set("50")
                return

            indicator_manager = self.app.get_indicator_manager()
            rsi_indicator = indicator_manager.get_indicator('RSI_Professional_Main')

            if rsi_indicator and rsi_indicator.enabled:
                # Pobierz ostatnie wyniki
                results = indicator_manager.get_results('RSI_Professional_Main')
                if results:
                    signals = rsi_indicator.get_latest_signal(results)

                    signal = signals.get('signal', 'none')
                    zone = signals.get('zone', 'neutral')
                    rsi_value = signals.get('rsi_value', 50)
                    signal_age = signals.get('signal_age', 0)

                    # Aktualizuj wartość RSI
                    self.rsi_value_var.set(f"{rsi_value:.0f}")

                    # Określ status i kolor
                    if signal == 'buy' and signal_age <= 5:
                        status_text = "🡱 BUY"
                        color = COLORS['accent_green']
                    elif signal == 'sell' and signal_age <= 5:
                        status_text = "🡳 SELL"
                        color = COLORS['accent_red']
                    elif zone == 'extreme_overbought':
                        status_text = "EXTR OB"
                        color = COLORS['accent_red']
                    elif zone == 'extreme_oversold':
                        status_text = "EXTR OS"
                        color = COLORS['accent_green']
                    elif zone == 'overbought':
                        status_text = "OB"
                        color = COLORS['accent_red']
                    elif zone == 'oversold':
                        status_text = "OS"
                        color = COLORS['accent_green']
                    elif zone == 'bullish':
                        status_text = "↗ UP"
                        color = COLORS['accent_green']
                    elif zone == 'bearish':
                        status_text = "↘ DOWN"
                        color = COLORS['accent_red']
                    else:
                        status_text = "NEUTRAL"
                        color = 'gray'

                    # Dodaj oznaczenie dywergencji
                    if signals.get('divergence', False):
                        status_text += " ⚡"

                    self.rsi_status_var.set(status_text)
                    self.rsi_status_label.configure(foreground=color)

                    # Kolor wartości RSI
                    if rsi_value >= 80:
                        self.rsi_value_label.configure(foreground=COLORS['accent_red'])
                    elif rsi_value <= 20:
                        self.rsi_value_label.configure(foreground=COLORS['accent_green'])
                    elif rsi_value >= 70:
                        self.rsi_value_label.configure(foreground='orange')
                    elif rsi_value <= 30:
                        self.rsi_value_label.configure(foreground='lightgreen')
                    else:
                        self.rsi_value_label.configure(foreground=COLORS['text_primary'])

                else:
                    self.rsi_status_var.set("CALC...")
                    self.rsi_value_var.set("50")
                    self.rsi_status_label.configure(foreground='orange')
                    self.rsi_value_label.configure(foreground='orange')
            else:
                self.rsi_status_var.set("OFF")
                self.rsi_value_var.set("50")
                self.rsi_status_label.configure(foreground='gray')
                self.rsi_value_label.configure(foreground='gray')

        except Exception as e:
            logger.error(f"Error updating RSI status: {e}")
            self.rsi_status_var.set("---")
            self.rsi_value_var.set("50")

    def _update_bollinger_status(self):
        """Aktualizuje status Bollinger Bands Professional"""
        try:
            if not hasattr(self.app, 'get_indicator_manager'):
                self.bollinger_status_var.set("---")
                self.bollinger_position_var.set("Middle")
                return

            indicator_manager = self.app.get_indicator_manager()
            bollinger_indicator = indicator_manager.get_indicator('Bollinger_Professional_Main')

            if bollinger_indicator and bollinger_indicator.enabled:
                # Pobierz ostatnie wyniki
                results = indicator_manager.get_results('Bollinger_Professional_Main')
                if results:
                    signals = bollinger_indicator.get_latest_signal(results)

                    signal = signals.get('signal', 'none')
                    band_position = signals.get('band_position', 'middle')
                    squeeze = signals.get('squeeze', False)
                    expansion = signals.get('expansion', False)
                    signal_age = signals.get('signal_age', 0)

                    # Aktualizuj pozycję w pasmach
                    position_text = band_position.replace('_', ' ').title()
                    self.bollinger_position_var.set(position_text)

                    # Określ status i kolor
                    if signal == 'strong_buy' and signal_age <= 3:
                        status_text = "🡱 STR BUY"
                        color = COLORS['accent_green']
                    elif signal == 'strong_sell' and signal_age <= 3:
                        status_text = "🡳 STR SELL"
                        color = COLORS['accent_red']
                    elif signal == 'buy_setup' and signal_age <= 5:
                        status_text = "🔵 BUY SETUP"
                        color = COLORS['accent_green']
                    elif signal == 'sell_setup' and signal_age <= 5:
                        status_text = "🔴 SELL SETUP"
                        color = COLORS['accent_red']
                    elif squeeze:
                        status_text = "🤏 SQUEEZE"
                        color = COLORS['accent_gold']
                    elif expansion:
                        status_text = "📈 EXPANSION"
                        color = COLORS['accent_blue']
                    elif band_position == 'upper':
                        status_text = "⬆️ UPPER"
                        color = COLORS['accent_red']
                    elif band_position == 'lower':
                        status_text = "⬇️ LOWER"
                        color = COLORS['accent_green']
                    else:
                        status_text = "➡️ MIDDLE"
                        color = 'gray'

                    self.bollinger_status_var.set(status_text)
                    self.bollinger_status_label.configure(foreground=color)

                    # Kolor pozycji
                    if 'upper' in band_position:
                        self.bollinger_position_label.configure(foreground=COLORS['accent_red'])
                    elif 'lower' in band_position:
                        self.bollinger_position_label.configure(foreground=COLORS['accent_green'])
                    else:
                        self.bollinger_position_label.configure(foreground=COLORS['text_primary'])

                else:
                    self.bollinger_status_var.set("CALC...")
                    self.bollinger_position_var.set("Middle")
                    self.bollinger_status_label.configure(foreground='orange')
                    self.bollinger_position_label.configure(foreground='orange')
            else:
                self.bollinger_status_var.set("OFF")
                self.bollinger_position_var.set("Middle")
                self.bollinger_status_label.configure(foreground='gray')
                self.bollinger_position_label.configure(foreground='gray')

        except Exception as e:
            logger.error(f"Error updating Bollinger status: {e}")
            self.bollinger_status_var.set("---")
            self.bollinger_position_var.set("Middle")

    def set_status(self, message: str):
        """Ustawia status"""
        self.status_var.set(message)

    def get_widget(self):
        return self.widget