# 📁 app/gui/__init__.py
"""
Enhanced GUI Components Package
"""

from .main_window import EnhancedMainWindow, MainWindow
from .control_panel import ControlPanel, IndicatorConfigPanel, QuickActionsPanel
from .info_panel import InfoPanel
from .chart_widget import EnhancedChartWidget, ChartWidget
from .collapsible_panel import (
    CollapsiblePanel,
    TabbedCollapsiblePanel,
    CollapsibleLegend,
    ExpandableSection
)

__all__ = [
    # Main components
    'EnhancedMainWindow', 'MainWindow',
    'ControlPanel', 'IndicatorConfigPanel', 'QuickActionsPanel',
    'InfoPanel',
    'EnhancedChartWidget', 'ChartWidget',

    # Collapsible components
    'CollapsiblePanel',
    'TabbedCollapsiblePanel',
    'CollapsibleLegend',
    'ExpandableSection'
]