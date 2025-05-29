# 📁 run.py - Enhanced with better error handling
"""
Multi-Exchange Trading Platform
Professional MT5-style interface with modular indicators
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Dodaj ścieżki do PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# Sprawdź czy wszystkie wymagane pakiety są dostępne
def check_dependencies():
    """Sprawdza dostępność pakietów"""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import tkinter as tk
        from tkinter import ttk
        print("✅ Podstawowe pakiety dostępne")
        return True
    except ImportError as e:
        print(f"❌ Brak wymaganego pakietu: {e}")
        print("Zainstaluj wymagane pakiety: pip install pandas numpy matplotlib")
        return False


def check_ccxt():
    """Sprawdza dostępność ccxt"""
    try:
        import ccxt
        print("✅ CCXT dostępne - połączenia z giełdami włączone")
        return True
    except ImportError:
        print("⚠️ CCXT niedostępne - tylko dane testowe")
        return False


def setup_logging():
    """Konfiguracja logowania"""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "trading_platform.log"),
            logging.StreamHandler()
        ]
    )


def safe_import_app():
    """Bezpieczny import aplikacji"""
    try:
        from app.main import TradingPlatform
        return TradingPlatform
    except ImportError as e:
        print(f"❌ Błąd importu aplikacji: {e}")
        print("Sprawdź czy wszystkie pliki są na miejscu")
        traceback.print_exc()
        return None


def main():
    """Główna funkcja uruchomieniowa z obsługą błędów"""
    try:
        # Sprawdź dependencies
        if not check_dependencies():
            input("Naciśnij Enter aby zakończyć...")
            return

        check_ccxt()
        setup_logging()

        print("🚀 Uruchamianie Professional Trading Platform...")
        print("📊 Obsługiwane giełdy: Binance, Bybit, OKX, Kraken, Coinbase, KuCoin")
        print("⏰ Timeframes: M1, M5, M15, M30, H1, H4, D1, W1")
        print("📈 Wskaźniki: TMA (Triangular Moving Average) + CCI Arrows")
        print("🎯 Strategia: M5/M15 - Professional Scalping & Swing Trading")
        print("=" * 60)
        print("🔧 Funkcje:")
        print("   • TMA z pasmami ATR - sygnały odbicia")
        print("   • CCI Arrows - momentum i punkty zwrotne")
        print("   • Dywergencje i multi-timeframe analysis")
        print("   • Auto-refresh danych w czasie rzeczywistym")
        print("   • Eksport wykresów i danych")
        print("   • Składane panele kontrolne")
        print("   • Skróty klawiszowe i zaawansowane zoom")
        print("=" * 60)

        # Import aplikacji
        TradingPlatform = safe_import_app()
        if TradingPlatform is None:
            input("Naciśnij Enter aby zakończyć...")
            return

        # Stwórz i uruchom aplikację
        print("📱 Inicjalizacja GUI...")
        app = TradingPlatform()

        print("🎨 Ładowanie interfejsu...")
        print("💡 Wskazówka: Naciśnij F11 dla pełnego ekranu")
        print("💡 Kliknij prawym przyciskiem na wykres dla opcji")
        print("💡 Użyj Ctrl+R aby odświeżyć dane")
        print("💡 Skróty: R-reset zoom, S-save, G-grid, L-legend")
        print()
        print("🟢 Aplikacja gotowa! Miłego tradingu! 📈")
        print("=" * 60)

        # Uruchom główną pętlę
        app.run()

    except KeyboardInterrupt:
        print("\n⏹️ Zatrzymano przez użytkownika")

    except Exception as e:
        print(f"\n❌ KRYTYCZNY BŁĄD: {e}")
        print("\n📋 Stack trace:")
        traceback.print_exc()
        print("\n🔧 Troubleshooting:")
        print("1. Sprawdź czy wszystkie pliki są na miejscu")
        print("2. Sprawdź czy masz zainstalowane wymagane pakiety")
        print("3. Sprawdź czy nie ma konfliktu z innymi aplikacjami")
        print("4. Spróbuj uruchomić z uprawnieniami administratora")
        print("5. Sprawdź log w folderze logs/")

        input("\nNaciśnij Enter aby zakończyć...")

    finally:
        print("\n🔚 Zamykanie aplikacji...")
        # Cleanup resources if needed
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass


def run_safe_mode():
    """Tryb bezpieczny - minimalna konfiguracja"""
    print("🔧 TRYB BEZPIECZNY - Minimalna konfiguracja")
    try:
        # Spróbuj uruchomić z podstawowymi ustawieniami
        from app.main import TradingPlatform

        app = TradingPlatform()

        # Wyłącz auto-refresh na początku
        print("⚠️ Auto-refresh wyłączony w trybie bezpiecznym")

        app.run()

    except Exception as e:
        print(f"❌ Błąd nawet w trybie bezpiecznym: {e}")
        traceback.print_exc()


def interactive_setup():
    """Interaktywna konfiguracja przy pierwszym uruchomieniu"""
    print("🛠️ PIERWSZA KONFIGURACJA")
    print("=" * 40)

    # Sprawdź ccxt
    has_ccxt = check_ccxt()

    if not has_ccxt:
        print("\n❓ Czy chcesz zainstalować CCXT dla połączeń z giełdami?")
        print("   (Bez tego będą tylko dane testowe)")
        choice = input("Tak/Nie [T/n]: ").lower()

        if choice in ['t', 'tak', 'y', 'yes', '']:
            print("💻 Instaluj: pip install ccxt")
            print("   Następnie uruchom aplikację ponownie")
            return False

    print("\n📊 Wybierz domyślną konfigurację:")
    print("1. Scalping (M1, M5) - Szybkie sygnały")
    print("2. Day Trading (M15, M30) - Balans")
    print("3. Swing Trading (H1, H4) - Długoterminowe")
    print("4. Custom - Własne ustawienia")

    choice = input("Wybór [1-4]: ")

    configs = {
        '1': {'timeframe': 'M5', 'style': 'aggressive'},
        '2': {'timeframe': 'M15', 'style': 'balanced'},
        '3': {'timeframe': 'H1', 'style': 'conservative'},
        '4': {'timeframe': 'M5', 'style': 'balanced'}  # default
    }

    config = configs.get(choice, configs['2'])
    print(f"✅ Wybrano: {config}")

    return True


if __name__ == "__main__":
    print("🏦 PROFESSIONAL TRADING PLATFORM")
    print("🔥 MT5-Style Multi-Exchange Interface")
    print("⚡ TMA + CCI Arrows - Professional Signals")
    print("=" * 50)

    # Sprawdź argumenty linii komend
    if len(sys.argv) > 1:
        if sys.argv[1] == '--safe':
            run_safe_mode()
        elif sys.argv[1] == '--setup':
            if interactive_setup():
                main()
        elif sys.argv[1] == '--help':
            print("📖 OPCJE URUCHOMIENIA:")
            print("  python run.py          - Normalne uruchomienie")
            print("  python run.py --safe   - Tryb bezpieczny")
            print("  python run.py --setup  - Konfiguracja")
            print("  python run.py --help   - Ta pomoc")
        else:
            print(f"❓ Nieznana opcja: {sys.argv[1]}")
            print("Użyj --help dla listy opcji")
    else:
        # Normalne uruchomienie
        main()