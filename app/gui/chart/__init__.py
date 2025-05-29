# 📁 app/gui/chart/__init__.py
"""
Chart Components Package

Modularny system wykresów:
- Base components (data, layout, tooltip, formatter)
- Plotters (candlestick, volume, indicators)
- Event handlers (zoom, export, interaction)
- Main widget (controller)
"""

from .base import (
    ChartDataManager,
    ChartLayoutManager,
    ChartTooltipBuilder,
    ChartFormatter,
    ChartTitleBuilder
)

from .plotters import (
    CandlestickPlotter,
    VolumePlotter,
    TMAPlotter,
    CCIPlotter,
    SignalPlotter,
    GridPlotter,
    LevelPlotter,
    FillPlotter
)

from .events import (
    ChartEventHandler,
    ChartZoomHandler,
    ChartExportHandler,
    ChartInteractionHandler
)

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