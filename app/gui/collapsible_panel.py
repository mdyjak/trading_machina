# 📁 app/gui/collapsible_panel.py
"""
Składany panel - widget z możliwością zwijania/rozwijania
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CollapsiblePanel:
    """
    Składany panel z header i zawartością
    Można zwijać/rozwijać klikając na header
    """

    def __init__(self, parent, title: str, expanded: bool = True,
                 on_toggle: Optional[Callable] = None):
        self.parent = parent
        self.title = title
        self.expanded = expanded
        self.on_toggle = on_toggle

        # Widgets
        self.frame = None
        self.header_frame = None
        self.content_frame = None
        self.toggle_button = None
        self.title_label = None

        # Stan
        self.content_visible = expanded

        self._create_widget()

    def _create_widget(self):
        """Tworzy składany panel"""
        # Główny frame
        self.frame = ttk.Frame(self.parent, relief=tk.RIDGE, borderwidth=1)

        # Header z przyciskiem toggle
        self.header_frame = ttk.Frame(self.frame)
        self.header_frame.pack(fill=tk.X, padx=2, pady=2)

        # Przycisk toggle (▼/▶)
        toggle_text = "▼" if self.expanded else "▶"
        self.toggle_button = ttk.Button(
            self.header_frame,
            text=toggle_text,
            width=3,
            command=self._on_toggle
        )
        self.toggle_button.pack(side=tk.LEFT, padx=(0, 5))

        # Tytuł
        self.title_label = ttk.Label(
            self.header_frame,
            text=self.title,
            font=('Arial', 10, 'bold')
        )
        self.title_label.pack(side=tk.LEFT)

        # Frame na zawartość
        self.content_frame = ttk.Frame(self.frame)
        if self.expanded:
            self.content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        # Bind click na header
        self.header_frame.bind("<Button-1>", lambda e: self._on_toggle())
        self.title_label.bind("<Button-1>", lambda e: self._on_toggle())

    def _on_toggle(self):
        """Obsługa zwijania/rozwijania"""
        self.expanded = not self.expanded
        self.content_visible = self.expanded

        # Zmień tekst przycisku
        self.toggle_button.config(text="▼" if self.expanded else "▶")

        # Pokaż/ukryj zawartość
        if self.expanded:
            self.content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        else:
            self.content_frame.pack_forget()

        # Callback
        if self.on_toggle:
            try:
                self.on_toggle(self.expanded)
            except Exception as e:
                logger.error(f"Error in collapsible panel callback: {e}")

    def get_frame(self) -> ttk.Frame:
        """Zwraca główny frame panelu"""
        return self.frame

    def get_content_frame(self) -> ttk.Frame:
        """Zwraca frame na zawartość"""
        return self.content_frame

    def set_title(self, title: str):
        """Ustawia tytuł panelu"""
        self.title = title
        self.title_label.config(text=title)

    def set_expanded(self, expanded: bool):
        """Programowo ustawia stan rozwinięcia"""
        if self.expanded != expanded:
            self._on_toggle()

    def is_expanded(self) -> bool:
        """Sprawdza czy panel jest rozwinięty"""
        return self.expanded


class TabbedCollapsiblePanel:
    """
    Panel z zakładkami, gdzie każda zakładka może być zwijana
    """

    def __init__(self, parent):
        self.parent = parent
        self.tabs = {}
        self.current_tab = None

        # Widgets
        self.frame = None
        self.tab_buttons_frame = None
        self.content_frame = None

        self._create_widget()

    def _create_widget(self):
        """Tworzy panel z zakładkami"""
        self.frame = ttk.Frame(self.parent)

        # Frame na przyciski zakładek
        self.tab_buttons_frame = ttk.Frame(self.frame)
        self.tab_buttons_frame.pack(fill=tk.X, padx=2, pady=2)

        # Frame na zawartość
        self.content_frame = ttk.Frame(self.frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    def add_tab(self, tab_id: str, title: str, expanded: bool = True):
        """Dodaje nową zakładkę"""
        # Przycisk zakładki
        button = ttk.Button(
            self.tab_buttons_frame,
            text=title,
            command=lambda: self._switch_tab(tab_id)
        )
        button.pack(side=tk.LEFT, padx=2)

        # Panel zawartości
        panel = CollapsiblePanel(
            self.content_frame,
            title,
            expanded=expanded,
            on_toggle=lambda exp: self._on_tab_toggle(tab_id, exp)
        )

        self.tabs[tab_id] = {
            'button': button,
            'panel': panel,
            'title': title
        }

        # Jeśli to pierwsza zakładka, ustaw jako aktywną
        if not self.current_tab:
            self._switch_tab(tab_id)

        return panel.get_content_frame()

    def _switch_tab(self, tab_id: str):
        """Przełącza na wybraną zakładkę"""
        if tab_id not in self.tabs:
            return

        # Ukryj poprzednią zakładkę
        if self.current_tab and self.current_tab in self.tabs:
            self.tabs[self.current_tab]['panel'].get_frame().pack_forget()
            self.tabs[self.current_tab]['button'].state(['!pressed'])

        # Pokaż nową zakładkę
        self.tabs[tab_id]['panel'].get_frame().pack(fill=tk.BOTH, expand=True)
        self.tabs[tab_id]['button'].state(['pressed'])
        self.current_tab = tab_id

    def _on_tab_toggle(self, tab_id: str, expanded: bool):
        """Obsługa zwijania zakładki"""
        # Można dodać logikę specjalną dla zwijania zakładek
        pass

    def get_frame(self) -> ttk.Frame:
        """Zwraca główny frame"""
        return self.frame

    def get_tab_content(self, tab_id: str) -> Optional[ttk.Frame]:
        """Zwraca frame zawartości zakładki"""
        if tab_id in self.tabs:
            return self.tabs[tab_id]['panel'].get_content_frame()
        return None


class CollapsibleLegend:
    def __init__(self, ax, title: str = "Legenda", position: str = 'upper right'):
        self.ax = ax
        self.title = title
        self.position = position
        self.expanded = True
        self.legend_items = []
        self.legend_obj = None
        self.font_size = 8

    def clear_and_rebuild(self, items: List):
        """Czyści i odbudowuje legendę"""
        self.legend_items = []
        self.add_multiple_items(items)

    def add_multiple_items(self, items: List):
        """Dodaje wiele elementów naraz"""
        for item in items:
            if len(item) >= 2:
                handle, label = item[0], item[1]
                color = item[2] if len(item) > 2 else '#FFFFFF'
                # Nie sprawdzaj duplikatów - po prostu dodaj
                self.legend_items.append((handle, label, color))

        # Zaktualizuj legendę po dodaniu wszystkich
        self._update_legend()

    def _update_legend(self):
        """Aktualizuje legendę z enhanced styling"""
        # Usuń starą legendę
        if self.legend_obj:
            try:
                self.legend_obj.remove()
            except:
                pass
            self.legend_obj = None

        # Twórz nową tylko jeśli expanded i są elementy
        if self.expanded and self.legend_items:
            try:
                handles, labels = [], []

                for item in self.legend_items:
                    # Twórz fake handle dla każdego elementu
                    import matplotlib.lines as mlines
                    color = item[2] if len(item) > 2 and item[2] else '#FFFFFF'

                    # Różne markery dla różnych typów
                    if 'Signal' in item[1]:
                        marker = '^' if 'Buy' in item[1] else 'v'
                        handle = mlines.Line2D([], [], color=color, marker=marker,
                                               markersize=6, linestyle='None')
                    elif 'EMA' in item[1] or 'TMA' in item[1]:
                        handle = mlines.Line2D([], [], color=color, linewidth=2)
                    else:
                        handle = mlines.Line2D([], [], color=color, marker='o',
                                               markersize=4, linestyle='None')

                    handles.append(handle)
                    labels.append(item[1])

                # Stwórz legendę z lepszymi ustawieniami
                self.legend_obj = self.ax.legend(
                    handles, labels,
                    loc=self.position,
                    fontsize=self.font_size,
                    framealpha=0.85,
                    fancybox=True,
                    shadow=False,
                    borderpad=0.5,
                    columnspacing=1.0,
                    handlelength=2.0,
                    handletextpad=0.8,
                    frameon=True,
                    facecolor='#2b2b2b',
                    edgecolor='#404040',
                    ncol=1
                )

                # ✅ Ustaw z-order żeby legenda była na wierzchu
                if self.legend_obj:
                    self.legend_obj.set_zorder(1000)

            except Exception as e:
                import logging
                logging.error(f"Error creating legend: {e}")
                self.legend_obj = None

    def toggle(self):
        """Przełącza widoczność legendy"""
        self.expanded = not self.expanded
        self._update_legend()


class ExpandableSection:
    """
    Sekcja którą można rozwijać/zwijać w większym panelu
    Lżejsza wersja CollapsiblePanel do użytku wewnętrznego
    """

    def __init__(self, parent, title: str, expanded: bool = True):
        self.parent = parent
        self.title = title
        self.expanded = expanded

        # Widgets
        self.header_frame = None
        self.content_frame = None
        self.toggle_var = tk.BooleanVar(value=expanded)

        self._create_widget()

    def _create_widget(self):
        """Tworzy sekcję"""
        # Header z checkboxem
        self.header_frame = ttk.Frame(self.parent)
        self.header_frame.pack(fill=tk.X, pady=(5, 0))

        # Checkbox jako toggle
        toggle_cb = ttk.Checkbutton(
            self.header_frame,
            text=self.title,
            variable=self.toggle_var,
            command=self._on_toggle
        )
        toggle_cb.pack(side=tk.LEFT)

        # Frame na zawartość
        self.content_frame = ttk.Frame(self.parent)
        if self.expanded:
            self.content_frame.pack(fill=tk.X, padx=10, pady=2)

    def _on_toggle(self):
        """Obsługa toggle"""
        self.expanded = self.toggle_var.get()

        if self.expanded:
            self.content_frame.pack(fill=tk.X, padx=10, pady=2)
        else:
            self.content_frame.pack_forget()

    def get_content_frame(self) -> ttk.Frame:
        """Zwraca frame na zawartość"""
        return self.content_frame

    def set_expanded(self, expanded: bool):
        """Ustawia stan rozwinięcia"""
        self.toggle_var.set(expanded)
        self._on_toggle()