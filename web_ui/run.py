#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skrypt uruchamiający aplikację webową do rozpoznawania zwierząt
"""

import os
import sys
import subprocess
import importlib.util

def check_module(module_name):
    """Sprawdza czy moduł jest zainstalowany"""
    return importlib.util.find_spec(module_name) is not None

def install_requirements():
    """Instaluje wymagane zależności"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        return False

def check_dependencies():
    """Sprawdza wymagane zależności i instaluje je w razie potrzeby"""
    required_modules = ["flask", "torch", "fastai", "pytube", "cv2", "numpy", "PIL"]
    missing_modules = []
    
    for module in required_modules:
        module_name = "opencv-python" if module == "cv2" else module
        if not check_module(module):
            missing_modules.append(module_name)
    
    if missing_modules:
        print(f"Brakujące moduły: {', '.join(missing_modules)}")
        print("Instalacja wymaganych zależności...")
        if install_requirements():
            print("Zależności zostały zainstalowane pomyślnie.")
        else:
            print("Błąd podczas instalacji zależności. Spróbuj zainstalować je ręcznie:")
            print("pip install -r requirements.txt")
            return False
    
    return True

def setup_folders():
    """Tworzy wymagane foldery"""
    folders = ["uploads", "uploads/processed", "models"]
    for folder in folders:
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder), exist_ok=True)

def run_app():
    """Uruchamia aplikację Flask"""
    try:
        print("Uruchamianie aplikacji webowej...")
        import app
        app.app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Błąd podczas uruchamiania aplikacji: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== System Rozpoznawania Zwierząt - Aplikacja webowa ===")
    
    # Sprawdzenie zależności
    if not check_dependencies():
        sys.exit(1)
    
    # Tworzenie folderów
    setup_folders()
    
    # Uruchomienie aplikacji
    if not run_app():
        sys.exit(1) 