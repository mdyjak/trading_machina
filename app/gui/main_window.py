# 📁 app/gui/main_window.py
"""
Enhanced główne okno GUI aplikacji z lepszym layoutem
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, Any, Optional
import pandas as pd

from .control_panel import ControlPanel, QuickActionsPanel
from .info_panel import InfoPanel
from .simple_chart_widget import SimpleChartWidget  # Używamy prostego chart widget
from .collapsible_panel import CollapsiblePanel, TabbedCollapsiblePanel
from ..config.settings import AppSettings, COLORS

logger = logging.getLogger(__name__)


class EnhancedMainWindow:
    """
    Enhanced główne okno aplikacji Trading Platform
    - Lepszy layout z możliwością dostosowania
    - Składane panele
    - Dock-able panels
    - Responsive design
    """

    def __init__(self, root: tk.Tk, app):
        self.root = root
        self.app = app
        self.settings = AppSettings()

        # Layout state
        self.layout_config = {
            'sidebar_width': 350,
            'sidebar_collapsed': False,
            'info_panel_height': 80,
            'show_quick_actions': True
        }

        # GUI Components
        self.main_container = None
        self.sidebar = None
        self.chart_area = None

        # Panels
        self.control_panel = None
        self.info_panel = None
        self.chart_widget = None
        self.quick_actions = None

        # Collapsible panels
        self.collapsible_panels = {}

        self._setup_window()
        self._create_enhanced_layout()
        self._setup_menu_bar()
        self._setup_status_bar()

        logger.info("EnhancedMainWindow initialized")

    def _setup_window(self):
        """Enhanced window setup"""
        self.root.title(f"{self.settings.window_title} - Enhanced Edition")
        self.root.geometry(f"{self.settings.window_size[0]}x{self.settings.window_size[1]}")
        self.root.configure(bg=COLORS['bg_primary'])

        # Window icon (jeśli dostępny)
        try:
            # self.root.iconbitmap('icon.ico')  # Dodaj ikonę jeśli masz
            pass
        except:
            pass

        # Minimum size
        self.root.minsize(1200, 800)

        # Enhanced style
        self._setup_enhanced_styles()

    def _setup_enhanced_styles(self):
        """Konfiguruje enhanced style"""
        style = ttk.Style()

        # Wybierz najlepszy dostępny theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')

        # Enhanced dark theme styles
        enhanced_styles = {
            'Enhanced.TFrame': {
                'configure': {
                    'background': COLORS['bg_primary'],
                    'relief': 'flat',
                    'borderwidth': 0
                }
            },
            'Sidebar.TFrame': {
                'configure': {
                    'background': COLORS['bg_secondary'],
                    'relief': 'raised',
                    'borderwidth': 1
                }
            },
            'Enhanced.TLabel': {
                'configure': {
                    'background': COLORS['bg_primary'],
                    'foreground': COLORS['text_primary'],
                    'font': ('Arial', 9)
                }
            },
            'Title.TLabel': {
                'configure': {
                    'background': COLORS['bg_primary'],
                    'foreground': COLORS['accent_gold'],
                    'font': ('Arial', 12, 'bold')
                }
            },
            'Enhanced.TLabelFrame': {
                'configure': {
                    'background': COLORS['bg_primary'],
                    'foreground': COLORS['text_primary'],
                    'borderwidth': 1,
                    'relief': 'solid'
                }
            }
        }

        for style_name, config in enhanced_styles.items():
            if 'configure' in config:
                style.configure(style_name, **config['configure'])

    def _create_enhanced_layout(self):
        """Tworzy enhanced layout"""
        # === GŁÓWNY KONTENER ===
        self.main_container = ttk.Frame(self.root, style='Enhanced.TFrame')
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # === LAYOUT: SIDEBAR + CHART AREA ===
        self._create_sidebar()
        self._create_chart_area()

        # === SPLITTER (możliwość zmiany rozmiaru) ===
        self._setup_splitter()

    def _create_sidebar(self):
        """Tworzy sidebar z składanymi panelami"""
        # Frame sidebar
        self.sidebar = ttk.Frame(self.main_container, style='Sidebar.TFrame')
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2), pady=5)

        # Konfiguruj szerokość
        self.sidebar.config(width=self.layout_config['sidebar_width'])
        self.sidebar.pack_propagate(False)

        # === HEADER SIDEBAR ===
        self._create_sidebar_header()

        # === SCROLLABLE CONTENT ===
        self._create_scrollable_sidebar()

    def _create_sidebar_header(self):
        """Tworzy header sidebar z kontrolkami"""
        header = ttk.Frame(self.sidebar, style='Enhanced.TFrame')
        header.pack(fill=tk.X, padx=5, pady=5)

        # Tytuł
        ttk.Label(header, text="🎛️ Control Center",
                  style='Title.TLabel').pack(side=tk.LEFT)

        # Przycisk collapse sidebar
        collapse_btn = ttk.Button(header, text="◀", width=3,
                                  command=self._toggle_sidebar)
        collapse_btn.pack(side=tk.RIGHT)

        # Separator
        ttk.Separator(self.sidebar, orient='horizontal').pack(fill=tk.X, pady=5)

    def _create_scrollable_sidebar(self):
        """Tworzy scrollable content dla sidebar"""
        # Canvas + Scrollbar dla scroll
        canvas = tk.Canvas(self.sidebar, bg=COLORS['bg_secondary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.sidebar, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas, style='Enhanced.TFrame')

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack scrolling components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # === ZAWARTOŚĆ SIDEBAR ===
        self._populate_sidebar()

        # Bind mouse wheel
        self._bind_mousewheel(canvas)

    def _populate_sidebar(self):
        """Wypełnia sidebar panelami"""
        # === QUICK ACTIONS (jeśli włączone) ===
        if self.layout_config['show_quick_actions']:
            self.quick_actions = QuickActionsPanel(self.scrollable_frame, self.app)
            self.quick_actions.get_widget().pack(fill=tk.X, padx=5, pady=(0, 10))

        # === GŁÓWNY CONTROL PANEL ===
        self.control_panel = ControlPanel(self.scrollable_frame, self.app)
        self.control_panel.get_widget().pack(fill=tk.X, padx=5, pady=(0, 10))

        # === DODATKOWE PANELE (collapsible) ===
        self._create_additional_panels()

    def _create_additional_panels(self):
        """Tworzy dodatkowe składane panele"""
        # === PANEL ANALIZ ===
        analysis_panel = CollapsiblePanel(
            self.scrollable_frame,
            "📈 Analiza techniczna",
            expanded=False
        )
        analysis_panel.get_frame().pack(fill=tk.X, padx=5, pady=(0, 5))
        self.collapsible_panels['analysis'] = analysis_panel

        # Zawartość analysis panel
        analysis_content = analysis_panel.get_content_frame()
        ttk.Label(analysis_content, text="🚧 Poziomy S/R, Fibonacci, etc.",
                  style='Enhanced.TLabel', foreground='gray').pack(pady=10)

        # === PANEL ALERTÓW ===
        alerts_panel = CollapsiblePanel(
            self.scrollable_frame,
            "🔔 Alerty i powiadomienia",
            expanded=False
        )
        alerts_panel.get_frame().pack(fill=tk.X, padx=5, pady=(0, 5))
        self.collapsible_panels['alerts'] = alerts_panel

        # Zawartość alerts panel
        alerts_content = alerts_panel.get_content_frame()
        ttk.Label(alerts_content, text="🚧 Price alerts, signal notifications",
                  style='Enhanced.TLabel', foreground='gray').pack(pady=10)

        # === PANEL STATYSTYK ===
        stats_panel = CollapsiblePanel(
            self.scrollable_frame,
            "📊 Statystyki sesji",
            expanded=False
        )
        stats_panel.get_frame().pack(fill=tk.X, padx=5, pady=(0, 5))
        self.collapsible_panels['stats'] = stats_panel

        # Zawartość stats panel
        stats_content = stats_panel.get_content_frame()
        ttk.Label(stats_content, text="🚧 Win rate, PnL, signals count",
                  style='Enhanced.TLabel', foreground='gray').pack(pady=10)

    def _create_chart_area(self):
        """Tworzy obszar wykresu"""
        # Frame dla chart area
        self.chart_area = ttk.Frame(self.main_container, style='Enhanced.TFrame')
        self.chart_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 5), pady=5)

        # === INFO PANEL (na górze) ===
        self.info_panel = InfoPanel(self.chart_area, self.app)
        self.info_panel.get_widget().pack(fill=tk.X, pady=(0, 5))

        # === CHART WIDGET (główny obszar) ===
        self.chart_widget = SimpleChartWidget(self.chart_area, self.app)
        self.chart_widget.get_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_splitter(self):
        """Konfiguruje splitter między sidebar a chart"""
        # Bind events do resizing
        self.sidebar.bind('<Button-1>', self._start_resize)
        self.sidebar.bind('<B1-Motion>', self._do_resize)

    def _setup_menu_bar(self):
        """Tworzy enhanced menu bar"""
        menubar = tk.Menu(self.root, bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        self.root.config(menu=menubar)

        # === FILE MENU ===
        file_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        menubar.add_cascade(label="Plik", menu=file_menu)
        file_menu.add_command(label="Zapisz wykres...", command=self._save_chart_menu)
        file_menu.add_command(label="Eksportuj dane...", command=self._export_data_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Wyjście", command=self._exit_application)

        # === VIEW MENU ===
        view_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        menubar.add_cascade(label="Widok", menu=view_menu)
        view_menu.add_command(label="Toggle Sidebar", command=self._toggle_sidebar)
        view_menu.add_command(label="Reset Layout", command=self._reset_layout)
        view_menu.add_separator()
        view_menu.add_command(label="Pełny ekran", command=self._toggle_fullscreen)

        # === TOOLS MENU ===
        tools_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        menubar.add_cascade(label="Narzędzia", menu=tools_menu)
        tools_menu.add_command(label="Konfiguracja wskaźników...", command=self._show_indicator_config)
        tools_menu.add_command(label="Presets...", command=self._show_presets_dialog)
        tools_menu.add_separator()
        tools_menu.add_command(label="Ustawienia...", command=self._show_settings_dialog)

        # === HELP MENU ===
        help_menu = tk.Menu(menubar, tearoff=0, bg=COLORS['bg_secondary'], fg=COLORS['text_primary'])
        menubar.add_cascade(label="Pomoc", menu=help_menu)
        help_menu.add_command(label="Skróty klawiszowe", command=self._show_shortcuts)
        help_menu.add_command(label="O programie...", command=self._show_about)

    def _setup_status_bar(self):
        """Tworzy enhanced status bar"""
        self.status_bar = ttk.Frame(self.root, style='Enhanced.TFrame', relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Status text
        self.status_text = tk.StringVar(value="Gotowy")
        status_label = ttk.Label(self.status_bar, textvariable=self.status_text,
                                 style='Enhanced.TLabel')
        status_label.pack(side=tk.LEFT, padx=5, pady=2)

        # Connection status
        self.connection_status = tk.StringVar(value="🔴 Offline")
        connection_label = ttk.Label(self.status_bar, textvariable=self.connection_status,
                                     style='Enhanced.TLabel')
        connection_label.pack(side=tk.RIGHT, padx=5, pady=2)

        # FPS counter (dla chart updates)
        self.fps_text = tk.StringVar(value="FPS: 0")
        fps_label = ttk.Label(self.status_bar, textvariable=self.fps_text,
                              style='Enhanced.TLabel')
        fps_label.pack(side=tk.RIGHT, padx=10, pady=2)

    # === EVENT HANDLERS ===
    def _toggle_sidebar(self):
        """Przełącza widoczność sidebar"""
        if self.layout_config['sidebar_collapsed']:
            # Pokaż sidebar
            self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2), pady=5, before=self.chart_area)
            self.layout_config['sidebar_collapsed'] = False
        else:
            # Ukryj sidebar
            self.sidebar.pack_forget()
            self.layout_config['sidebar_collapsed'] = True

    def _start_resize(self, event):
        """Rozpoczyna resize sidebar"""
        self._resize_start_x = event.x_root

    def _do_resize(self, event):
        """Wykonuje resize sidebar"""
        if hasattr(self, '_resize_start_x'):
            delta = event.x_root - self._resize_start_x
            new_width = max(200, min(500, self.layout_config['sidebar_width'] + delta))
            self.layout_config['sidebar_width'] = new_width
            self.sidebar.config(width=new_width)
            self._resize_start_x = event.x_root

    def _bind_mousewheel(self, canvas):
        """Bind mouse wheel do scrolling"""

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        self.scrollable_frame.bind('<Enter>', _bind_to_mousewheel)
        self.scrollable_frame.bind('<Leave>', _unbind_from_mousewheel)

    # === MENU HANDLERS ===
    def _save_chart_menu(self):
        """Menu handler dla save chart"""
        if self.chart_widget:
            self.chart_widget.save_chart()

    def _export_data_menu(self):
        """Menu handler dla export data"""
        if self.chart_widget:
            self.chart_widget.export_data()

    def _exit_application(self):
        """Zamyka aplikację"""
        self.root.quit()

    def _reset_layout(self):
        """Resetuje layout do domyślnego"""
        self.layout_config = {
            'sidebar_width': 350,
            'sidebar_collapsed': False,
            'info_panel_height': 80,
            'show_quick_actions': True
        }
        # Rebuild layout
        self._setup_window()

    def _toggle_fullscreen(self):
        """Przełącza pełny ekran"""
        current_state = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current_state)

    def _show_indicator_config(self):
        """Pokazuje okno konfiguracji wskaźników"""
        # Placeholder - można dodać dedykowane okno
        self.show_status("🚧 Okno konfiguracji wskaźników - w przygotowaniu")

    def _show_presets_dialog(self):
        """Pokazuje dialog presets"""
        self.show_status("🚧 Dialog presets - w przygotowaniu")

    def _show_settings_dialog(self):
        """Pokazuje dialog ustawień"""
        self.show_status("🚧 Dialog ustawień - w przygotowaniu")

    def _show_shortcuts(self):
        """Pokazuje listę skrótów klawiszowych"""
        shortcuts_text = """
🎹 SKRÓTY KLAWISZOWE:

Wykres:
• R - Reset zoom
• L - Toggle legend
• G - Toggle grid
• S - Save chart
• + - Zoom in
• - - Zoom out

Nawigacja:
• ← → - Pan chart
• Shift + Scroll - Precise zoom
• Ctrl + R - Refresh data

Panele:
• F11 - Toggle fullscreen
• Ctrl + 1-9 - Toggle panels
        """

        # Proste okno info
        import tkinter.messagebox as mb
        mb.showinfo("Skróty klawiszowe", shortcuts_text)

    def _show_about(self):
        """Pokazuje informacje o programie"""
        about_text = f"""
📊 {self.settings.window_title}
Enhanced Edition v2.0

🚀 Professional MT5-style Trading Platform
📈 TMA + CCI Arrows + More indicators
🎯 Optimized for M5/M15 strategies

💻 Built with Python + Tkinter + Matplotlib
🔧 Modular architecture
⚡ Real-time data support

© 2024 Trading Platform Team
        """

        import tkinter.messagebox as mb
        mb.showinfo("O programie", about_text)

    # === PUBLIC METHODS ===
    def update_chart(self, df: pd.DataFrame, indicator_results: Dict):
        """Aktualizuje wykres"""
        if self.chart_widget:
            self.chart_widget.update_chart(df, indicator_results)

    def update_market_info(self, market_stats: Dict):
        """Aktualizuje panel informacyjny"""
        if self.info_panel:
            self.info_panel.update_market_info(market_stats)

        # Update connection status
        if market_stats:
            self.connection_status.set("🟢 Live")
        else:
            self.connection_status.set("🔴 Offline")

    def show_status(self, message: str):
        """Pokazuje status"""
        if hasattr(self, 'status_text'):
            self.status_text.set(message)

        if self.info_panel:
            self.info_panel.set_status(message)

    def update_fps(self, fps: float):
        """Aktualizuje FPS counter"""
        if hasattr(self, 'fps_text'):
            self.fps_text.set(f"FPS: {fps:.1f}")

    # === GETTERS (delegated to control panel) ===
    def get_current_symbol(self) -> str:
        """Zwraca aktualnie wybrany symbol"""
        if self.control_panel:
            return self.control_panel.get_symbol()
        return 'BTC/USDT'

    def get_current_timeframe(self) -> str:
        """Zwraca aktualnie wybrany timeframe"""
        if self.control_panel:
            return self.control_panel.get_timeframe()
        return '5m'

    def get_candles_limit(self) -> int:
        """Zwraca limit świec"""
        if self.control_panel:
            return self.control_panel.get_candles_limit()
        return 200

    def get_collapsible_panel(self, panel_name: str) -> Optional[CollapsiblePanel]:
        """Zwraca collapsible panel po nazwie"""
        return self.collapsible_panels.get(panel_name)


# Backward compatibility
MainWindow = EnhancedMainWindow