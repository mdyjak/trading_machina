# 📁 indicators/base.py
"""
Bazowa klasa dla wszystkich wskaźników
Zapewnia spójny interfejs i funkcjonalność
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseIndicator(ABC):
    """
    Bazowa klasa dla wszystkich wskaźników technicznych

    Każdy wskaźnik musi implementować:
    - calculate() - główna logika obliczeniowa
    - get_plot_config() - konfiguracja wyświetlania
    - validate_data() - walidacja danych wejściowych
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.enabled = True
        self.settings = kwargs
        self.last_result = None
        self.minimum_periods = 50  # Domyślnie

        # Metadane
        self.version = "1.0"
        self.author = "Trading Platform"
        self.description = ""

        # Stan obliczeniowy
        self._cache = {}
        self._last_data_hash = None

        logger.info(f"Initialized indicator: {self.name}")

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Główna funkcja obliczeniowa wskaźnika

        Args:
            df: DataFrame z danymi OHLCV

        Returns:
            Dict z wynikami lub None w przypadku błędu
        """
        pass

    @abstractmethod
    def get_plot_config(self) -> Dict:
        """
        Zwraca konfigurację wyświetlania wskaźnika

        Returns:
            Dict z ustawieniami wykresu
        """
        pass

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Waliduje dane wejściowe

        Args:
            df: DataFrame do walidacji

        Returns:
            True jeśli dane są poprawne
        """
        if df is None or df.empty:
            logger.warning(f"{self.name}: Brak danych")
            return False

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"{self.name}: Brakuje kolumn: {missing_columns}")
            return False

        if len(df) < self.minimum_periods:
            logger.warning(f"{self.name}: Za mało danych: {len(df)} < {self.minimum_periods}")
            return False

        return True

    def update_settings(self, **kwargs):
        """Aktualizuje ustawienia wskaźnika"""
        old_settings = self.settings.copy()
        self.settings.update(kwargs)

        # Invalidate cache jeśli ustawienia się zmieniły
        if old_settings != self.settings:
            self._cache.clear()
            self._last_data_hash = None
            logger.info(f"{self.name}: Ustawienia zaktualizowane: {kwargs}")

    def get_settings(self) -> Dict:
        """Zwraca aktualne ustawienia"""
        return self.settings.copy()

    def enable(self):
        """Włącza wskaźnik"""
        self.enabled = True
        logger.info(f"{self.name}: Włączony")

    def disable(self):
        """Wyłącza wskaźnik"""
        self.enabled = False
        logger.info(f"{self.name}: Wyłączony")

    def get_info(self) -> Dict:
        """Zwraca informacje o wskaźniku"""
        return {
            'name': self.name,
            'version': self.version,
            'author': self.author,
            'description': self.description,
            'enabled': self.enabled,
            'settings': self.settings,
            'minimum_periods': self.minimum_periods
        }

    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Oblicza hash danych dla cache'owania"""
        try:
            # Użyj ostatnich kilku wierszy + ustawienia
            last_rows = df.tail(10).to_string()
            settings_str = str(sorted(self.settings.items()))
            return hash(last_rows + settings_str)
        except:
            return None

    def _should_recalculate(self, df: pd.DataFrame) -> bool:
        """Sprawdza czy trzeba przeliczyć wskaźnik"""
        if not self.enabled:
            return False

        current_hash = self._calculate_data_hash(df)
        if current_hash != self._last_data_hash:
            self._last_data_hash = current_hash
            return True

        return False