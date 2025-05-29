# 📁 app/gui/chart/events.py
"""
Event handlers dla chart widget
"""

import logging
from typing import Optional, Callable, Dict, Any
from .base import ChartDataManager, ChartTooltipBuilder

logger = logging.getLogger(__name__)


class ChartEventHandler:
    """
    Główny handler eventów dla wykresu
    """

    def __init__(self, data_manager: ChartDataManager,
                 tooltip_builder: ChartTooltipBuilder,
                 status_callback: Optional[Callable] = None):
        self.data_manager = data_manager
        self.tooltip_builder = tooltip_builder
        self.status_callback = status_callback

        # Event state
        self._last_click_pos = None
        self._drag_start = None

    def on_scroll(self, event, canvas):
        """Enhanced scroll handling z różnymi trybami"""
        if event.inaxes is None:
            return

        try:
            # Różne prędkości zoom
            if hasattr(event, 'key') and event.key == 'shift':
                scale = 1.05 if event.button == 'up' else 0.95  # Wolniejszy
            else:
                scale = 1.15 if event.button == 'up' else 0.85  # Normalny

            xlim = event.inaxes.get_xlim()
            ylim = event.inaxes.get_ylim()

            xdata = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
            ydata = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2

            x_left = xdata - (xdata - xlim[0]) * scale
            x_right = xdata + (xlim[1] - xdata) * scale
            y_bottom = ydata - (ydata - ylim[0]) * scale
            y_top = ydata + (ylim[1] - ydata) * scale

            event.inaxes.set_xlim(x_left, x_right)
            event.inaxes.set_ylim(y_bottom, y_top)
            canvas.draw_idle()

        except Exception as e:
            logger.error(f"Error in scroll handler: {e}")

    def on_click(self, event):
        """Enhanced click handler z tooltipami"""
        if event.inaxes is None or event.xdata is None:
            return

        try:
            x_index = int(round(event.xdata))
            candle_info = self.data_manager.get_candle_info(x_index)

            if candle_info:
                # Buduj tooltip
                df, indicators = self.data_manager.get_data()
                tooltip_parts = self.tooltip_builder.build_tooltip(candle_info, indicators)

                # Wyświetl w status bar
                tooltip_text = " | ".join(tooltip_parts)
                if self.status_callback:
                    self.status_callback(tooltip_text)

                logger.debug(f"Click on candle {x_index}: {len(tooltip_parts)} info parts")

                # Zapisz pozycję dla innych eventów
                self._last_click_pos = (x_index, event.ydata)

        except Exception as e:
            logger.error(f"Error in click handler: {e}")

    def on_key_press(self, event, chart_controller):
        """Obsługa skrótów klawiszowych"""
        try:
            key_handlers = {
                'r': chart_controller.reset_zoom,
                'l': lambda: chart_controller.toggle_legend(),
                'g': lambda: chart_controller.toggle_grid(),
                's': chart_controller.save_chart,
                '+': chart_controller.zoom_in,
                '-': chart_controller.zoom_out,
                'escape': lambda: self._cancel_operations()
            }

            if event.key in key_handlers:
                key_handlers[event.key]()

        except Exception as e:
            logger.error(f"Error in key press handler: {e}")

    def on_mouse_move(self, event, crosshair_enabled: bool = False):
        """Obsługa ruchu myszy z opcjonalnym crosshair"""
        if not crosshair_enabled or event.inaxes is None:
            return

        try:
            # Placeholder dla crosshair functionality
            # Można dodać tutaj rysowanie linii crosshair
            pass

        except Exception as e:
            logger.error(f"Error in mouse move handler: {e}")

    def on_button_press(self, event):
        """Obsługa naciśnięcia przycisku myszy (start drag)"""
        if event.inaxes is None:
            return

        try:
            if event.button == 1:  # Left mouse button
                self._drag_start = (event.xdata, event.ydata)

        except Exception as e:
            logger.error(f"Error in button press handler: {e}")

    def on_button_release(self, event):
        """Obsługa zwolnienia przycisku myszy (end drag)"""
        try:
            if event.button == 1 and self._drag_start:
                # Zakończ operację drag
                self._drag_start = None

        except Exception as e:
            logger.error(f"Error in button release handler: {e}")

    def _cancel_operations(self):
        """Anuluje bieżące operacje"""
        self._drag_start = None
        self._last_click_pos = None


class ChartZoomHandler:
    """
    Specjalizowany handler dla operacji zoom
    """

    def __init__(self):
        self.zoom_history = []
        self.max_history = 20

    def zoom_in(self, axes_list, factor: float = 0.35):
        """Zoom in na wszystkich osiach"""
        try:
            # Zapisz stan przed zoom
            self._save_zoom_state(axes_list)

            for ax in axes_list:
                if ax:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()

                    x_center = (xlim[0] + xlim[1]) / 2
                    y_center = (ylim[0] + ylim[1]) / 2
                    x_range = (xlim[1] - xlim[0]) * factor
                    y_range = (ylim[1] - ylim[0]) * factor

                    ax.set_xlim(x_center - x_range, x_center + x_range)
                    ax.set_ylim(y_center - y_range, y_center + y_range)

        except Exception as e:
            logger.error(f"Error zooming in: {e}")

    def zoom_out(self, axes_list, factor: float = 0.65):
        """Zoom out na wszystkich osiach"""
        try:
            # Zapisz stan przed zoom
            self._save_zoom_state(axes_list)

            for ax in axes_list:
                if ax:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()

                    x_center = (xlim[0] + xlim[1]) / 2
                    y_center = (ylim[0] + ylim[1]) / 2
                    x_range = (xlim[1] - xlim[0]) * factor
                    y_range = (ylim[1] - ylim[0]) * factor

                    ax.set_xlim(x_center - x_range, x_center + x_range)
                    ax.set_ylim(y_center - y_range, y_center + y_range)

        except Exception as e:
            logger.error(f"Error zooming out: {e}")

    def reset_zoom(self, axes_dict: Dict, df_len: int, price_data=None, cci_data=None):
        """Reset zoom do pełnego widoku"""
        try:
            # Zapisz stan przed reset
            axes_list = [ax for ax in axes_dict.values() if ax is not None]
            self._save_zoom_state(axes_list)

            # Reset X dla wszystkich osi
            for ax in axes_list:
                ax.set_xlim(-1, df_len)

            # Reset Y dla price axis
            if 'price' in axes_dict and axes_dict['price'] and price_data is not None:
                ax = axes_dict['price']
                import numpy as np
                price_min = np.nanmin(price_data)
                price_max = np.nanmax(price_data)
                margin = (price_max - price_min) * 0.03
                ax.set_ylim(price_min - margin, price_max + margin)

            # Reset Y dla CCI axis
            if 'cci' in axes_dict and axes_dict['cci'] and cci_data is not None:
                ax = axes_dict['cci']
                cci_values = [x for x in cci_data if x != 0]
                if cci_values:
                    cci_min = min(cci_values)
                    cci_max = max(cci_values)
                    margin = max(50, (cci_max - cci_min) * 0.1)
                    ax.set_ylim(cci_min - margin, cci_max + margin)

        except Exception as e:
            logger.error(f"Error resetting zoom: {e}")

    def auto_fit(self, axes_list, df_len: int, recent_candles: int = 100):
        """Auto-fit do najnowszych danych"""
        try:
            # Zapisz stan
            self._save_zoom_state(axes_list)

            start_idx = max(0, df_len - recent_candles)

            for ax in axes_list:
                if ax:
                    ax.set_xlim(start_idx, df_len)

        except Exception as e:
            logger.error(f"Error in auto fit: {e}")

    def zoom_back(self, axes_list):
        """Wróć do poprzedniego stanu zoom"""
        if not self.zoom_history:
            return

        try:
            previous_state = self.zoom_history.pop()

            for i, ax in enumerate(axes_list):
                if ax and i < len(previous_state):
                    xlim, ylim = previous_state[i]
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

        except Exception as e:
            logger.error(f"Error in zoom back: {e}")

    def _save_zoom_state(self, axes_list):
        """Zapisuje aktualny stan zoom"""
        try:
            state = []
            for ax in axes_list:
                if ax:
                    state.append((ax.get_xlim(), ax.get_ylim()))
                else:
                    state.append(None)

            self.zoom_history.append(state)

            # Ograniczenie historii
            if len(self.zoom_history) > self.max_history:
                self.zoom_history.pop(0)

        except Exception as e:
            logger.error(f"Error saving zoom state: {e}")


class ChartExportHandler:
    """
    Handler dla eksportu wykresów
    """

    def __init__(self, colors_config: Dict):
        self.colors = colors_config

    def save_chart(self, fig, filename: str, dpi: int = 300):
        """Zapisuje wykres z enhanced ustawieniami"""
        try:
            fig.savefig(
                filename,
                dpi=dpi,
                bbox_inches='tight',
                facecolor=self.colors.get('bg_primary', 'black'),
                edgecolor='none',
                transparent=False,
                pad_inches=0.1
            )
            logger.info(f"Chart saved: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            return False

    def export_data(self, df, indicators: Dict, filename: str):
        """Eksportuje dane z wskaźnikami"""
        try:
            if df.empty:
                return False

            export_df = df.copy()

            # Dodaj TMA
            if 'TMA_Main' in indicators:
                tma_data = indicators['TMA_Main']
                for key in ['tma_center', 'tma_upper', 'tma_lower']:
                    if key in tma_data:
                        col_name = f'TMA_{key.split("_")[1].title()}'
                        export_df[col_name] = self._align_series(
                            tma_data[key], export_df.index
                        )

            # Dodaj CCI
            if 'CCI_Arrows_Main' in indicators:
                cci_data = indicators['CCI_Arrows_Main']
                if 'cci' in cci_data:
                    export_df['CCI'] = self._align_series(
                        cci_data['cci'], export_df.index
                    )

            # Export w odpowiednim formacie
            if filename.endswith('.xlsx'):
                export_df.to_excel(filename, index=True)
            elif filename.endswith('.json'):
                export_df.to_json(filename, orient='index', date_format='iso')
            else:
                export_df.to_csv(filename, index=True)

            logger.info(f"Data exported: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False

    def _align_series(self, data_array, index):
        """Wyrównuje array do pandas Series z indeksem"""
        import pandas as pd
        try:
            return pd.Series(data_array, index=index)
        except:
            # Fallback - dopasuj długość
            if len(data_array) != len(index):
                # Pad lub trim
                if len(data_array) < len(index):
                    padded = [0] * len(index)
                    padded[:len(data_array)] = data_array
                    return pd.Series(padded, index=index)
                else:
                    return pd.Series(data_array[:len(index)], index=index)
            return pd.Series(data_array, index=index)


class ChartInteractionHandler:
    """
    Handler dla zaawansowanych interakcji z wykresem
    """

    def __init__(self):
        self.selection_mode = False
        self.crosshair_enabled = False
        self.annotation_mode = False

        # Selection state
        self._selection_start = None
        self._selection_end = None
        self._selection_rect = None

    def toggle_selection_mode(self):
        """Przełącza tryb selekcji"""
        self.selection_mode = not self.selection_mode
        logger.info(f"Selection mode: {'ON' if self.selection_mode else 'OFF'}")

    def toggle_crosshair(self):
        """Przełącza crosshair"""
        self.crosshair_enabled = not self.crosshair_enabled
        logger.info(f"Crosshair: {'ON' if self.crosshair_enabled else 'OFF'}")

    def start_selection(self, event):
        """Rozpoczyna selekcję obszaru"""
        if not self.selection_mode or event.inaxes is None:
            return

        self._selection_start = (event.xdata, event.ydata)
        logger.debug(f"Selection started at: {self._selection_start}")

    def update_selection(self, event, ax):
        """Aktualizuje selekcję podczas drag"""
        if not self.selection_mode or not self._selection_start or event.inaxes is None:
            return

        # Usuń poprzedni prostokąt selekcji
        if self._selection_rect:
            self._selection_rect.remove()

        # Dodaj nowy prostokąt
        from matplotlib.patches import Rectangle
        start_x, start_y = self._selection_start
        width = event.xdata - start_x
        height = event.ydata - start_y

        self._selection_rect = Rectangle(
            (start_x, start_y), width, height,
            linewidth=1, edgecolor='yellow', facecolor='yellow', alpha=0.3
        )
        ax.add_patch(self._selection_rect)

    def end_selection(self, event):
        """Kończy selekcję"""
        if not self.selection_mode or not self._selection_start:
            return

        self._selection_end = (event.xdata, event.ydata)
        logger.info(f"Selection completed: {self._selection_start} to {self._selection_end}")

        # Cleanup
        if self._selection_rect:
            self._selection_rect.remove()
            self._selection_rect = None

        return self._selection_start, self._selection_end

    def cancel_selection(self):
        """Anuluje selekcję"""
        self._selection_start = None
        self._selection_end = None
        if self._selection_rect:
            self._selection_rect.remove()
            self._selection_rect = None