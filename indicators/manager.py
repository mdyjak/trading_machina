# 📁 indicators/manager.py
"""
Manager wskaźników - centralne zarządzanie wszystkimi wskaźnikami
"""

from typing import Dict, List, Optional, Type
import logging
from .base import BaseIndicator

logger = logging.getLogger(__name__)


class IndicatorManager:
    """
    Manager do zarządzania wskaźnikami technicznymi

    Funkcje:
    - Rejestracja wskaźników
    - Obliczanie wskaźników
    - Zarządzanie ustawieniami
    - Cache'owanie wyników
    """

    def __init__(self):
        self.indicators: Dict[str, BaseIndicator] = {}
        self.results: Dict[str, Dict] = {}
        self.indicator_classes: Dict[str, Type[BaseIndicator]] = {}

        logger.info("IndicatorManager initialized")

    def register_indicator_class(self, name: str, indicator_class: Type[BaseIndicator]):
        """Rejestruje klasę wskaźnika"""
        self.indicator_classes[name] = indicator_class
        logger.info(f"Registered indicator class: {name}")

    def add_indicator(self, name: str, indicator_class_name: str, **kwargs) -> bool:
        """
        Dodaje nowy wskaźnik

        Args:
            name: Nazwa instancji wskaźnika
            indicator_class_name: Nazwa klasy wskaźnika
            **kwargs: Parametry wskaźnika
        """
        try:
            if indicator_class_name not in self.indicator_classes:
                logger.error(f"Unknown indicator class: {indicator_class_name}")
                return False

            indicator_class = self.indicator_classes[indicator_class_name]
            indicator = indicator_class(name=name, **kwargs)

            self.indicators[name] = indicator
            logger.info(f"Added indicator: {name} ({indicator_class_name})")
            return True

        except Exception as e:
            logger.error(f"Error adding indicator {name}: {e}")
            return False

    def remove_indicator(self, name: str) -> bool:
        """Usuwa wskaźnik"""
        if name in self.indicators:
            del self.indicators[name]
            if name in self.results:
                del self.results[name]
            logger.info(f"Removed indicator: {name}")
            return True
        return False

    def calculate_all(self, df) -> Dict[str, Dict]:
        """
        Oblicza wszystkie aktywne wskaźniki

        Args:
            df: DataFrame z danymi OHLCV

        Returns:
            Dict z wynikami wszystkich wskaźników
        """
        results = {}

        for name, indicator in self.indicators.items():
            if not indicator.enabled:
                continue

            try:
                result = indicator.calculate(df)
                if result:
                    results[name] = result
                    self.results[name] = result

            except Exception as e:
                logger.error(f"Error calculating {name}: {e}")

        logger.debug(f"Calculated {len(results)} indicators")
        return results

    def get_indicator(self, name: str) -> Optional[BaseIndicator]:
        """Zwraca wskaźnik po nazwie"""
        return self.indicators.get(name)

    def get_results(self, name: str = None) -> Dict:
        """Zwraca wyniki wskaźników"""
        if name:
            return self.results.get(name, {})
        return self.results.copy()

    def update_indicator_settings(self, name: str, **kwargs) -> bool:
        """Aktualizuje ustawienia wskaźnika"""
        if name not in self.indicators:
            return False

        self.indicators[name].update_settings(**kwargs)
        return True

    def enable_indicator(self, name: str) -> bool:
        """Włącza wskaźnik"""
        if name in self.indicators:
            self.indicators[name].enable()
            return True
        return False

    def disable_indicator(self, name: str) -> bool:
        """Wyłącza wskaźnik"""
        if name in self.indicators:
            self.indicators[name].disable()
            return True
        return False

    def get_all_indicators_info(self) -> Dict[str, Dict]:
        """Zwraca informacje o wszystkich wskaźnikach"""
        return {name: indicator.get_info()
                for name, indicator in self.indicators.items()}

    def clear_cache(self):
        """Czyści cache wszystkich wskaźników"""
        for indicator in self.indicators.values():
            indicator._cache.clear()
            indicator._last_data_hash = None
        logger.info("Cleared all indicators cache")
