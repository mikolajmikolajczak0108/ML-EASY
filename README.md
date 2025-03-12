# System Rozpoznawania Zwierząt

System do rozpoznawania zwierząt na obrazach i w filmach z wykorzystaniem sieci neuronowych. Projekt umożliwia klasyfikację obrazów oraz detekcję zwierząt w filmach.

## Funkcjonalności

- Klasyfikacja obrazów do kategorii: psy, koty, ptaki
- Detekcja obiektów (zwierząt) na zdjęciach i w filmach
- Intuicyjny interfejs webowy

## Struktura projektu

```
.
├── web_ui/                      # Aplikacja webowa Flask
│   ├── app.py                   # Główny plik aplikacji
│   ├── templates/               # Szablony HTML
│   ├── static/                  # Pliki statyczne
│   ├── models/                  # Folder na wytrenowane modele
│   └── uploads/                 # Folder na przesłane pliki
```

## Technologie

- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Modele ML**:
  - FastAI (klasyfikacja obrazów)
  - YOLOv5 (detekcja obiektów)
- **Biblioteki**: OpenCV, PyTorch, NumPy

## Instalacja i uruchomienie

1. Sklonuj repozytorium:
   ```
   git clone [URL_REPOZYTORIUM]
   ```

2. Zainstaluj wymagane pakiety:
   ```
   pip install -r web_ui/requirements.txt
   ```

3. Uruchom aplikację Flask:
   ```
   cd web_ui
   python app.py
   ```

4. Otwórz przeglądarkę i przejdź pod adres: `http://localhost:5000`

## Korzystanie z aplikacji

1. **Klasyfikacja obrazów**:
   - Przejdź do zakładki "Klasyfikacja Obrazów"
   - Prześlij zdjęcie zawierające psa, kota lub ptaka
   - Uzyskaj wynik klasyfikacji z poziomem pewności

2. **Detekcja w filmach/obrazach**:
   - Przejdź do zakładki "Detekcja w Filmie/Obrazie"
   - Prześlij plik wideo lub zdjęcie
   - Otrzymaj przetworzone wideo/obraz z zaznaczonymi zwierzętami 