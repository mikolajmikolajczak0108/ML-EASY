# System Rozpoznawania Zwierząt - Aplikacja webowa

Ten katalog zawiera aplikację webową Flask do rozpoznawania zwierząt przy użyciu modeli uczenia maszynowego.

## Funkcje

* **Klasyfikacja obrazów** - rozpoznawanie psów, kotów i ptaków na obrazach
* **Detekcja obiektów** - wykrywanie i lokalizacja zwierząt na obrazach i filmach
* **Obsługa filmów z YouTube** - możliwość analizy filmów bezpośrednio z YouTube

## Wymagania

Do uruchomienia aplikacji potrzebne są następujące pakiety:

```
flask==2.0.1
werkzeug==2.0.1
torch>=1.8.0
torchvision>=0.9.0
fastai>=2.4.0
pytube>=12.0.0
opencv-python>=4.5.3
numpy>=1.20.0
Pillow>=8.3.1
```

## Instalacja

1. Zainstaluj wymagane pakiety:

```bash
pip install -r requirements.txt
```

2. Upewnij się, że model klasyfikacyjny znajduje się w folderze `rozwiązanie` (powinien zostać wygenerowany po uruchomieniu notatnika Jupyter `rozpoznawanie_zwierząt.ipynb`).

## Uruchomienie

Aby uruchomić aplikację, wykonaj poniższe polecenie z katalogu `web_ui`:

```bash
python app.py
```

Aplikacja będzie dostępna pod adresem `http://localhost:5000`.

## Obsługa aplikacji

### Klasyfikacja obrazów

1. Przejdź do zakładki "Klasyfikacja Obrazów"
2. Wybierz plik obrazu z komputera
3. Kliknij przycisk "Klasyfikuj"
4. Po chwili zobaczysz wynik klasyfikacji z informacją o klasie oraz poziomie pewności

### Detekcja obiektów

1. Przejdź do zakładki "Detekcja w Filmie"
2. Wybierz jedną z opcji:
   * **Plik Wideo/Obraz** - wybierz plik wideo lub obrazu z komputera
   * **Link YouTube** - podaj link do filmu na YouTube
3. Kliknij przycisk "Analizuj"
4. Po chwili zobaczysz wyniki detekcji - na obrazie/filmie będą zaznaczone wykryte zwierzęta

## Uwagi

* Pierwsze uruchomienie może trwać dłużej, ponieważ aplikacja musi pobrać model YOLOv5
* W przypadku problemów z modelami, kliknij pasek statusu w górnej części aplikacji, aby ponownie załadować modele 