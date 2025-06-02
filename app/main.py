# 📁 app/main.py
"""
Główna aplikacja Trading Platform
"""

import tkinter as tk
import threading
import time
import logging
import sys
from pathlib import Path
from typing import Optional

from indicators import create_ema_crossover_indicator, EMACrossoverIndicator

# Dodaj ścieżkę do indicators i utils
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .gui.main_window import MainWindow
from .data.exchange_manager import ExchangeManager
from .data.data_fetcher import DataFetcher
from .config.settings import AppSettings
from indicators.manager import IndicatorManager
from indicators.tma import TMAIndicator, create_tma_indicator
from indicators.cci_arrows import CCIArrowsIndicator, create_cci_arrows_indicator
from indicators.smi_arrows import SMIArrowsIndicator, create_smi_arrows_indicator

logger = logging.getLogger(__name__)


class TradingPlatform:
    """
    Główna klasa aplikacji Trading Platform
    Koordynuje wszystkie komponenty
    """

    def __init__(self):
        # Podstawowa konfiguracja
        self.settings = AppSettings()

        # Komponenty aplikacji
        self.exchange_manager = ExchangeManager()
        self.data_fetcher = DataFetcher(self.exchange_manager)
        self.indicator_manager = IndicatorManager()

        # GUI
        self.root = tk.Tk()
        self.main_window = MainWindow(self.root, self)

        # Threading
        self.running = False
        self.update_thread = None

        # Inicjalizuj wskaźniki
        self._setup_indicators()

        # Połącz z domyślną giełdą
        self._connect_default_exchange()

        logger.info("TradingPlatform initialized")

    def _setup_indicators(self):
        """Konfiguruje dostępne wskaźniki"""
        # Zarejestruj klasy wskaźników
        self.indicator_manager.register_indicator_class('TMA', TMAIndicator)
        self.indicator_manager.register_indicator_class('CCI_Arrows', CCIArrowsIndicator)
        self.indicator_manager.register_indicator_class('EMA_Crossover', EMACrossoverIndicator)
        self.indicator_manager.register_indicator_class('SMI_Arrows', SMIArrowsIndicator)

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
            use_signal_line=True,
            crossover_confirmation=2,
            price_type='close'
        )

        # Dodaj domyślny SMI Arrows
        self.indicator_manager.add_indicator(
            name='SMI_Arrows_Main',
            indicator_class_name='SMI_Arrows',
            smi_period=10,
            first_smoothing=3,
            second_smoothing=3,
            signal_smoothing=3,
            overbought_level=40,
            oversold_level=-40,
            arrow_sensitivity='medium',
            use_divergence=True,
            min_bars_between_signals=3
        )

        logger.info("Indicators setup completed - TMA + CCI Arrows + EMA Crossover + SMI Arrows")  # ✅ ZMIEŃ

    def _connect_default_exchange(self):
        """Łączy z domyślną giełdą"""
        from .config.settings import EXCHANGES

        # Użyj pierwszej dostępnej giełdy z konfiguracji
        default_exchange = list(EXCHANGES.keys())[0] if EXCHANGES else 'Binance'

        success = self.exchange_manager.connect_to_exchange(default_exchange)
        if success:
            logger.info(f"Connected to default exchange: {default_exchange}")
        else:
            logger.warning(f"Failed to connect to default exchange: {default_exchange}")

    def refresh_data(self):
        """Odświeża dane rynkowe i wskaźniki"""
        try:
            # Pobierz aktualne ustawienia z GUI
            symbol = self.main_window.get_current_symbol()
            timeframe = self.main_window.get_current_timeframe()
            limit = self.main_window.get_candles_limit()

            # Pobierz dane
            df = self.data_fetcher.fetch_market_data(symbol, timeframe, limit)

            if df is not None and not df.empty:
                # Oblicz wskaźniki
                indicator_results = self.indicator_manager.calculate_all(df)

                # Aktualizuj GUI
                self.main_window.update_chart(df, indicator_results)
                self.main_window.update_market_info(self.data_fetcher.get_market_stats())

                logger.debug(f"Data refreshed: {len(df)} candles, {len(indicator_results)} indicators")
                return True
            else:
                logger.warning("No data received")
                return False

        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            return False

    def start_auto_refresh(self):
        """Uruchamia automatyczne odświeżanie"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._auto_refresh_loop, daemon=True)
            self.update_thread.start()
            logger.info("Auto-refresh started")

    def stop_auto_refresh(self):
        """Zatrzymuje automatyczne odświeżanie"""
        self.running = False
        logger.info("Auto-refresh stopped")

    def _auto_refresh_loop(self):
        """Pętla automatycznego odświeżania"""
        while self.running:
            try:
                self.refresh_data()
                time.sleep(self.settings.refresh_interval)
            except Exception as e:
                logger.error(f"Error in auto-refresh loop: {e}")
                time.sleep(10)  # Wait longer on error

    def change_exchange(self, exchange_name: str) -> bool:
        """Zmienia giełdę"""
        success = self.exchange_manager.connect_to_exchange(exchange_name)
        if success:
            self.refresh_data()
        return success

    def change_timeframe(self, timeframe: str):
        """Zmienia timeframe i dostosowuje wskaźniki"""
        # Aktualizuj ustawienia TMA dla nowego timeframe
        tma_indicator = self.indicator_manager.get_indicator('TMA_Main')
        if tma_indicator:
            # Stwórz nową konfigurację TMA dla timeframe
            new_tma = create_tma_indicator(timeframe, 'balanced', 'TMA_Main')

            # Usuń stary i dodaj nowy
            self.indicator_manager.remove_indicator('TMA_Main')
            self.indicator_manager.add_indicator(
                name='TMA_Main',
                indicator_class_name='TMA',
                **new_tma.get_settings()
            )

        # Aktualizuj ustawienia CCI Arrows dla nowego timeframe
        cci_indicator = self.indicator_manager.get_indicator('CCI_Arrows_Main')
        if cci_indicator:
            # Stwórz nową konfigurację CCI dla timeframe
            new_cci = create_cci_arrows_indicator(timeframe, 'medium', 'CCI_Arrows_Main')

            # Usuń stary i dodaj nowy
            self.indicator_manager.remove_indicator('CCI_Arrows_Main')
            self.indicator_manager.add_indicator(
                name='CCI_Arrows_Main',
                indicator_class_name='CCI_Arrows',
                **new_cci.get_settings()
            )

        # Aktualizuj ustawienia EMA Crossover dla nowego timeframe
        ema_indicator = self.indicator_manager.get_indicator('EMA_Crossover_Main')
        if ema_indicator:
            # Stwórz nową konfigurację EMA dla timeframe
            new_ema = create_ema_crossover_indicator(timeframe, 'balanced', 'EMA_Crossover_Main')

            # Usuń stary i dodaj nowy
            self.indicator_manager.remove_indicator('EMA_Crossover_Main')
            self.indicator_manager.add_indicator(
                name='EMA_Crossover_Main',
                indicator_class_name='EMA_Crossover',
                **new_ema.get_settings()
            )

        # Aktualizuj ustawienia SMI Arrows dla nowego timeframe
        smi_indicator = self.indicator_manager.get_indicator('SMI_Arrows_Main')
        if smi_indicator:
            # Stwórz nową konfigurację SMI dla timeframe
            new_smi = create_smi_arrows_indicator(timeframe, 'medium', 'SMI_Arrows_Main')

            # Usuń stary i dodaj nowy
            self.indicator_manager.remove_indicator('SMI_Arrows_Main')
            self.indicator_manager.add_indicator(
                name='SMI_Arrows_Main',
                indicator_class_name='SMI_Arrows',
                **new_smi.get_settings()
            )

        self.refresh_data()

    def toggle_indicator(self, indicator_name: str, enabled: bool):
        """Włącza/wyłącza wskaźnik"""
        if enabled:
            self.indicator_manager.enable_indicator(indicator_name)
        else:
            self.indicator_manager.disable_indicator(indicator_name)

        self.refresh_data()

    def update_indicator_settings(self, indicator_name: str, **kwargs):
        """Aktualizuje ustawienia wskaźnika"""
        success = self.indicator_manager.update_indicator_settings(indicator_name, **kwargs)
        if success:
            self.refresh_data()
        return success

    def get_indicator_manager(self) -> IndicatorManager:
        """Zwraca manager wskaźników"""
        return self.indicator_manager

    def get_exchange_manager(self) -> ExchangeManager:
        """Zwraca manager giełd"""
        return self.exchange_manager

    def get_current_symbol(self) -> str:
        """Zwraca aktualnie wybrany symbol"""
        return self.main_window.get_current_symbol()

    def get_current_timeframe(self) -> str:
        """Zwraca aktualnie wybrany timeframe"""
        return self.main_window.get_current_timeframe()

    def run(self):
        """Uruchamia aplikację"""
        try:
            # Pierwsze załadowanie danych
            self.refresh_data()

            # Uruchom auto-refresh
            self.start_auto_refresh()

            # Obsługa zamknięcia
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

            # Uruchom GUI
            self.root.mainloop()

        except Exception as e:
            logger.error(f"Error running application: {e}")
            raise

    def _on_closing(self):
        """Obsługa zamknięcia aplikacji"""
        try:
            self.stop_auto_refresh()
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=2)
            self.root.destroy()
            logger.info("Application closed")
        except Exception as e:
            logger.error(f"Error closing application: {e}")
            self.root.destroy()