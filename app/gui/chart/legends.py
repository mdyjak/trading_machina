# 📁 app/gui/chart/legends.py
"""
Legend management dla wykresów
"""

from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ChartLegendManager:
    """
    Manager legend dla wszystkich wykresów
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config
        self.legends = {}

    def setup_price_legend(self, ax, indicators: Dict, show_legend: bool = True):
        """Konfiguruje legendę dla wykresu cenowego"""
        if not show_legend:
            return

        # Import lokalny żeby uniknąć circular import
        from ..collapsible_panel import CollapsibleLegend

        legend_items = []

        # Podstawowe elementy cenowe
        legend_items.extend([
            (None, "OHLC Candles", self.colors['accent_green']),
        ])

        # TMA
        if 'TMA_Main' in indicators:
            legend_items.extend([
                (None, "TMA Center", self.colors['accent_green']),
                (None, "TMA Upper Band", self.colors['accent_blue']),
                (None, "TMA Lower Band", self.colors['accent_pink']),
                (None, "TMA Buy Signal", self.colors['accent_green']),
                (None, "TMA Sell Signal", self.colors['accent_red']),
                (None, "TMA Caution", self.colors['accent_gold'])
            ])

        # EMA Crossover
        if 'EMA_Crossover_Main' in indicators:
            settings = indicators['EMA_Crossover_Main'].get('settings', {})
            fast_period = settings.get('fast_ema_period', 12)
            slow_period = settings.get('slow_ema_period', 26)
            signal_period = settings.get('signal_ema_period', 9)

            legend_items.extend([
                (None, f"Fast EMA({fast_period})", '#FFD700'),
                (None, f"Slow EMA({slow_period})", '#87CEEB'),
                (None, f"Signal EMA({signal_period})", '#DDA0DD'),
                (None, "EMA Buy Signal", self.colors['accent_green']),
                (None, "EMA Sell Signal", self.colors['accent_red'])
            ])

        # Zawsze twórz legendę, nawet bez sygnałów
        if 'price' not in self.legends:
            self.legends['price'] = CollapsibleLegend(ax, "Price Chart", 'upper left')

        # Tylko jeśli są elementy
        if legend_items:
            self.legends['price'].clear_and_rebuild(legend_items)

    def setup_volume_legend(self, ax, show_legend: bool = True):
        """Konfiguruje legendę dla volume"""
        if not show_legend:
            return

        from ..collapsible_panel import CollapsibleLegend

        legend_items = [
            (None, "Volume Bars", self.colors['accent_green']),
            (None, "Up Volume", self.colors['accent_green']),
            (None, "Down Volume", self.colors['accent_red'])
        ]

        if 'volume' not in self.legends:
            self.legends['volume'] = CollapsibleLegend(ax, "Volume", 'upper right')

        if legend_items:
            self.legends['volume'].clear_and_rebuild(legend_items)

    def setup_cci_legend(self, ax, indicators: Dict, show_legend: bool = True):
        """Konfiguruje legendę dla CCI"""
        if not show_legend or 'CCI_Arrows_Main' not in indicators:
            return

        from ..collapsible_panel import CollapsibleLegend

        cci_settings = indicators['CCI_Arrows_Main'].get('settings', {})
        cci_period = cci_settings.get('cci_period', 14)
        overbought = cci_settings.get('overbought_level', 100)
        oversold = cci_settings.get('oversold_level', -100)

        legend_items = [
            (None, f"CCI({cci_period})", '#FFD700'),
            (None, f"Overbought ({overbought})", '#FF6B6B'),
            (None, f"Oversold ({oversold})", '#4ECDC4'),
            (None, "Zero Line", '#999999'),
            (None, "CCI Buy Signal", self.colors['accent_green']),
            (None, "CCI Sell Signal", self.colors['accent_red']),
            (None, "Bullish Div", self.colors['accent_blue']),
            (None, "Bearish Div", self.colors['accent_pink'])
        ]

        if 'cci' not in self.legends:
            self.legends['cci'] = CollapsibleLegend(ax, "CCI Oscillator", 'upper right')

        if legend_items:
            self.legends['cci'].clear_and_rebuild(legend_items)

    def setup_smi_legend(self, ax, indicators: Dict, show_legend: bool = True):
        """Konfiguruje legendę dla SMI"""
        if not show_legend or 'SMI_Arrows_Main' not in indicators:
            return

        from ..collapsible_panel import CollapsibleLegend

        smi_settings = indicators['SMI_Arrows_Main'].get('settings', {})
        smi_period = smi_settings.get('smi_period', 10)
        overbought = smi_settings.get('overbought_level', 40)
        oversold = smi_settings.get('oversold_level', -40)

        legend_items = [
            (None, f"SMI({smi_period})", '#FF6B35'),  # Orange-Red
            (None, "SMI Signal", '#9B59B6'),  # Purple
            (None, f"Overbought ({overbought})", '#E67E22'),  # Dark Orange
            (None, f"Oversold ({oversold})", '#16A085'),  # Teal
            (None, "Zero Line", '#7F8C8D'),  # Gray
            (None, "SMI Buy Signal", '#27AE60'),  # Emerald Green
            (None, "SMI Sell Signal", '#C0392B'),  # Dark Red
            (None, "Bullish Div", '#3498DB'),  # Blue
            (None, "Bearish Div", '#E74C3C')  # Red
        ]

        if 'smi' not in self.legends:
            self.legends['smi'] = CollapsibleLegend(ax, "SMI Oscillator", 'upper right')

        if legend_items:
            self.legends['smi'].clear_and_rebuild(legend_items)

    def setup_rsi_legend(self, ax, indicators: Dict, show_legend: bool = True):
        """Konfiguruje legendę dla RSI"""
        if not show_legend or 'RSI_Professional_Main' not in indicators:
            return

        from ..collapsible_panel import CollapsibleLegend

        rsi_settings = indicators['RSI_Professional_Main'].get('settings', {})
        rsi_period = rsi_settings.get('rsi_period', 14)
        overbought = rsi_settings.get('overbought_level', 70)
        oversold = rsi_settings.get('oversold_level', 30)

        legend_items = [
            (None, f"RSI({rsi_period})", '#FFD700'),
            (None, f"Overbought ({overbought})", '#FF6B6B'),
            (None, f"Oversold ({oversold})", '#4ECDC4'),
            (None, "Midline (50)", '#999999'),
            (None, "RSI Buy Signal", self.colors['accent_green']),
            (None, "RSI Sell Signal", self.colors['accent_red']),
            (None, "Bullish Div", self.colors['accent_blue']),
            (None, "Bearish Div", self.colors['accent_pink'])
        ]

        if 'rsi' not in self.legends:
            self.legends['rsi'] = CollapsibleLegend(ax, "RSI Professional", 'upper right')

        if legend_items:
            self.legends['rsi'].clear_and_rebuild(legend_items)

    def setup_bollinger_legend(self, ax, indicators: Dict, show_legend: bool = True):
        """Konfiguruje legendę dla Bollinger Bands"""
        if not show_legend or 'Bollinger_Professional_Main' not in indicators:
            return

        from ..collapsible_panel import CollapsibleLegend

        bb_settings = indicators['Bollinger_Professional_Main'].get('settings', {})
        bb_period = bb_settings.get('bb_period', 20)
        bb_std = bb_settings.get('bb_std_dev', 2.0)

        legend_items = [
            (None, f"Upper Band ({bb_period}, +{bb_std}σ)", '#FF6B6B'),
            (None, f"Middle Band (SMA {bb_period})", '#FFD700'),
            (None, f"Lower Band ({bb_period}, -{bb_std}σ)", '#4ECDC4'),
            (None, "Upper Touch", '#FF9800'),
            (None, "Lower Touch", '#4CAF50'),
            (None, "Upper Breakout", '#F44336'),
            (None, "Lower Breakout", '#2196F3'),
            (None, "Band Fill", '#E3F2FD')
        ]

        if 'bollinger' not in self.legends:
            self.legends['bollinger'] = CollapsibleLegend(ax, "Bollinger Bands", 'upper left')

        if legend_items:
            self.legends['bollinger'].clear_and_rebuild(legend_items)

    def toggle_all_legends(self):
        """Przełącza wszystkie legendy"""
        for legend in self.legends.values():
            legend.toggle()

    def get_legend(self, chart_type: str):
        """Zwraca legendę dla typu wykresu"""
        return self.legends.get(chart_type)