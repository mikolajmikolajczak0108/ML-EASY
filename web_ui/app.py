import os
import cv2
import uuid
import shutil
import random
import requests
import zipfile
import io
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np

# Importy dla fastai
from fastai.vision.all import load_learner, PILImage, ImageDataLoaders, vision_learner, error_rate, Resize, RandomResizedCrop, imagenet_stats, models, get_image_files, DataBlock, Categorize, CategoryBlock, ImageBlock

# Konfiguracja aplikacji
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # limit 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov'}
app.config['MODEL_PATH'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models')
app.config['DATASET_PATH'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'datasets')

# Tworzenie katalogów
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_PATH'], exist_ok=True)
os.makedirs(app.config['DATASET_PATH'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'processed'), exist_ok=True)

# Lista dostępnych architektur modeli
AVAILABLE_MODELS = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'mobilenet_v2': models.mobilenet_v2,
    'densenet121': models.densenet121
}

# Standardowe linki do przykładowych datasetów
EXAMPLE_DATASETS = {
    'macbook_air_pro': 'https://storage.googleapis.com/ml-easy-datasets/macbooks_dataset.zip',
    'iphone_models': 'https://storage.googleapis.com/ml-easy-datasets/iphone_dataset.zip'
}

def allowed_file(filename):
    """Sprawdza czy rozszerzenie pliku jest dozwolone"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_video_file(filename):
    """Sprawdza czy plik jest filmem"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'mp4', 'mov'}

# Ścieżka do modelu
fastai_model_path = os.path.join(app.config['MODEL_PATH'], 'klasyfikator_zwierząt.pkl')

# Ładowanie modelu fastai (jeśli istnieje)
fastai_model = None
try:
    if os.path.exists(fastai_model_path):
        fastai_model = load_learner(fastai_model_path)
except Exception as e:
    print(f"Błąd ładowania modelu fastai: {e}")

# Funkcja do pobierania i rozpakowania przykładowego datasetu
def download_and_extract_dataset(dataset_name, dataset_url):
    """Pobiera i rozpakowuje przykładowy dataset"""
    dataset_dir = os.path.join(app.config['DATASET_PATH'], dataset_name)
    
    if os.path.exists(dataset_dir):
        print(f"Dataset {dataset_name} już istnieje")
        return True
    
    try:
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"Pobieranie datasetu {dataset_name}...")
        
        response = requests.get(dataset_url, stream=True)
        if response.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(dataset_dir)
            print(f"Dataset {dataset_name} został pomyślnie rozpakowany")
            return True
        else:
            print(f"Błąd pobierania datasetu: {response.status_code}")
            return False
    except Exception as e:
        print(f"Błąd podczas pobierania/rozpakowywania datasetu: {e}")
        return False

# Pobieranie dostępnych modeli i datasetów
def get_available_resources():
    """Pobiera listę dostępnych modeli i datasetów"""
    models = [f for f in os.listdir(app.config['MODEL_PATH']) 
             if f.endswith('.pkl')]
    
    datasets = []
    if os.path.exists(app.config['DATASET_PATH']):
        datasets = [d for d in os.listdir(app.config['DATASET_PATH']) 
                   if os.path.isdir(os.path.join(app.config['DATASET_PATH'], d))]
    
    return {'models': models, 'datasets': datasets}

@app.route('/')
def index():
    """Strona główna"""
    resources = get_available_resources()
    return render_template('index.html', resources=resources, architectures=list(AVAILABLE_MODELS.keys()))

@app.route('/classify', methods=['POST'])
def classify_image():
    """Klasyfikacja obrazu przy użyciu modelu fastai"""
    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nie wybrano pliku'}), 400
        
    if not allowed_file(file.filename) or is_video_file(file.filename):
        return jsonify({'error': 'Niedozwolony format pliku'}), 400
    
    # Jeśli model nie jest załadowany
    if fastai_model is None:
        return jsonify({
            'error': 'Model nie jest załadowany. Skontaktuj się z administratorem.'
        }), 500
    
    try:
        # Wczytanie obrazu
        img_bytes = file.read()
        img = PILImage.create(img_bytes)
        
        # Klasyfikacja
        pred, pred_idx, probs = fastai_model.predict(img)
        
        # Zapisanie oryginalnego obrazu
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        with open(file_path, 'wb') as f:
            f.write(img_bytes)
        
        return jsonify({
            'prediction': str(pred),
            'probability': float(probs[pred_idx]),
            'filename': unique_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Analiza wideo lub obrazu przy użyciu modelu fastai"""
    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku'}), 400
    
    # Sprawdzenie, czy model jest załadowany
    if fastai_model is None:
        return jsonify({'error': 'Model klasyfikacji nie jest załadowany'}), 500
    
    # Obsługa przesłanego pliku
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nie wybrano pliku'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Niedozwolony format pliku'}), 400
    
    try:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Zapisanie oryginalnego pliku
        file.save(file_path)
        
        # Sprawdzenie czy to wideo
        if is_video_file(file.filename):
            # Przetwarzanie wideo - klasyfikacja losowych klatek
            result = process_video(file_path)
            if result:
                os.remove(file_path)  # Usunięcie oryginalnego wideo
                return jsonify(result)
            else:
                return jsonify({'error': 'Nie udało się przetworzyć wideo'}), 500
        else:
            # Klasyfikacja obrazu
            img = PILImage.create(file_path)
            pred, pred_idx, probs = fastai_model.predict(img)
            
            # Zapisanie obrazu wynikowego z adnotacją
            frame = cv2.imread(file_path)
            if frame is not None:
                # Dodanie etykiety z klasyfikacją
                label = f"{str(pred)} ({probs[pred_idx]*100:.2f}%)"
                cv2.putText(
                    frame, label, 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2
                )
                
                # Zapisanie przetworzonego obrazu
                processed_filename = f"processed_{unique_filename}"
                processed_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], 'processed', processed_filename
                )
                cv2.imwrite(processed_path, frame)
                
                return jsonify({
                    'filename': processed_filename,
                    'prediction': str(pred),
                    'probability': float(probs[pred_idx])
                })
            
            # Jeśli nie udało się przetworzyć obrazu
            return jsonify({
                'prediction': str(pred),
                'probability': float(probs[pred_idx])
            })
    except Exception as e:
        print(f"Błąd podczas przetwarzania: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_video(video_path):
    """Przetwarzanie wideo z klasyfikacją losowych klatek"""
    try:
        # Sprawdzenie czy plik istnieje
        if not os.path.exists(video_path):
            print(f"Plik wideo nie istnieje: {video_path}")
            return None
            
        # Sprawdzenie czy plik ma odpowiedni rozmiar
        if os.path.getsize(video_path) < 1000:  # Mniej niż 1KB
            print(f"Plik wideo jest zbyt mały: {video_path}")
            return None

        # Otwarcie wideo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Nie można otworzyć wideo: {video_path}")
            return None
        
        # Pobranie parametrów wideo
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Domyślna wartość dla problematycznych plików
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if width <= 0 or height <= 0 or total_frames <= 0:
            print("Nieprawidłowe parametry wideo")
            cap.release()
            return None
        
        print(f"Parametry wideo: {width}x{height}, {fps} FPS, {total_frames} klatek")
        
        # Ograniczenie rozmiaru wyjściowego do rozsądnych wartości
        max_dim = 640
        if width > max_dim or height > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            width, height = new_width, new_height
        
        # Utworzenie nazwy pliku wynikowego - zawsze jako MP4
        output_filename = f"processed_{os.path.basename(video_path)}"
        if not output_filename.lower().endswith('.mp4'):
            output_filename = output_filename.rsplit('.', 1)[0] + '.mp4'
            
        output_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'processed', output_filename)
        
        # Utworzenie tymczasowego pliku zamiast bezpośredniego zapisu
        temp_output_path = f"{output_path}.temp.mp4"
        
        # Lista dostępnych kodeków do wypróbowania - tylko MP4 kompatybilne
        codecs = [
            ('mp4v', 'mp4'),  # Standardowy kodek MP4
            ('avc1', 'mp4'),  # H.264
            ('h264', 'mp4')   # Jeszcze jeden wariant H.264
        ]
        
        # Próba użycia różnych kodeków
        out = None
        
        for codec, ext in codecs:
            try:
                print(f"Próba użycia kodeka {codec} dla MP4...")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_path = temp_output_path.replace('.mp4', f'.{ext}')
                out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                
                if out and out.isOpened():
                    temp_output_path = test_path
                    print(f"Udane otwarcie pliku wyjściowego MP4 z kodekiem {codec}")
                    break
                else:
                    if out:
                        out.release()
                    out = None
            except Exception as e:
                print(f"Błąd przy próbie użycia kodeka {codec}: {str(e)}")
                if out:
                    out.release()
                out = None
        
        if out is None or not out.isOpened():
            print("Nie można utworzyć pliku wyjściowego z żadnym kodekiem")
            cap.release()
            return None
        
        # Wybierz 15 losowych indeksów klatek do analizy
        if total_frames <= 15:
            frame_indices = list(range(int(total_frames)))
        else:
            frame_indices = sorted(random.sample(range(int(total_frames)), 15))
        
        # Przygotowanie do zliczania wyników klasyfikacji
        predictions = {}
        processed_frames = 0
        frames_written = 0
        
        # Aktualny indeks klatki
        current_frame = 0
        
        # Lista na zapamiętanie wybranych klatek do późniejszego zapisu
        selected_frames = []
        
        print(f"Rozpoczynam analizę losowych klatek. Wybranych klatek: {len(frame_indices)}")
        
        # Odczytywanie kolejnych klatek
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sprawdź, czy aktualna klatka jest jedną z wybranych do analizy
            if current_frame in frame_indices:
                processed_frames += 1
                print(f"Przetwarzanie klatki {current_frame}")
                
                # Zmiana rozmiaru klatki
                resized_frame = cv2.resize(frame, (width, height))
                
                # Konwersja klatki na format PIL i klasyfikacja
                try:
                    # Zapisz klatkę tymczasowo jako plik
                    temp_frame_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], f"temp_frame_{current_frame}.jpg"
                    )
                    cv2.imwrite(temp_frame_path, resized_frame)
                    
                    # Klasyfikacja klatki
                    img = PILImage.create(temp_frame_path)
                    pred, pred_idx, probs = fastai_model.predict(img)
                    
                    # Aktualizacja słownika predykcji
                    class_name = str(pred)
                    if class_name in predictions:
                        predictions[class_name] = predictions[class_name] + float(probs[pred_idx])
                    else:
                        predictions[class_name] = float(probs[pred_idx])
                    
                    # Dodanie etykiety z klasyfikacją na klatce
                    label = f"{class_name} ({probs[pred_idx]*100:.2f}%)"
                    cv2.putText(
                        resized_frame, label,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2
                    )
                    
                    # Zapamiętanie klatki z etykietą
                    selected_frames.append((current_frame, resized_frame.copy()))
                    
                    # Usunięcie tymczasowego pliku
                    os.remove(temp_frame_path)
                    
                except Exception as e:
                    print(f"Błąd klasyfikacji klatki {current_frame}: {str(e)}")
            
            current_frame += 1
        
        # Jeśli nie przetworzono żadnej klatki
        if processed_frames == 0:
            print("Nie przetworzono żadnej klatki")
            cap.release()
            out.release()
            return None
        
        # Znajdź najczęściej występującą klasę
        best_prediction = max(predictions.items(), key=lambda x: x[1])
        best_class = best_prediction[0]
        best_probability = best_prediction[1] / processed_frames
        
        print(f"Wynik klasyfikacji: {best_class} z prawdopodobieństwem {best_probability:.2f}")
        
        # Zapisz wszystkie wybrane klatki do wideo wynikowego
        for idx, (frame_idx, frame) in enumerate(selected_frames):
            # Dodaj numer klatki i wynik klasyfikacji
            cv2.putText(
                frame, f"Klatka {frame_idx}/{total_frames}",
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )
            
            # Zapisz klatkę do wideo
            out.write(frame)
            frames_written += 1
        
        # Zwolnienie zasobów
        cap.release()
        out.release()
        
        print(f"Zakończono przetwarzanie wideo, zapisano {frames_written} klatek")
        
        # Sprawdź, czy plik wyjściowy został utworzony i ma odpowiedni rozmiar
        if frames_written == 0 or not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) < 100:
            print("Przetwarzanie nie powiodło się: brak klatek lub pusty plik")
            # Utworzenie "awaryjnego" wideo z pojedynczą klatką
            fallback_path = create_fallback_video(output_path, width, height, fps)
            
            # Sprawdź, czy plik awaryjny został pomyślnie utworzony
            if fallback_path and os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 0:
                print(f"Utworzono awaryjne wideo zamiast: {fallback_path}")
                return {
                    'filename': os.path.basename(fallback_path),
                    'prediction': 'błąd',
                    'probability': 0.0
                }
            else:
                print("Nie udało się utworzyć nawet awaryjnego wideo")
                return None
        
        # Przenieś plik tymczasowy do docelowego
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            
            # Sprawdź, czy plik tymczasowy istnieje
            if not os.path.exists(temp_output_path):
                print(f"Plik tymczasowy nie istnieje: {temp_output_path}")
                return None
                
            print(f"Przenoszenie pliku z {temp_output_path} do {output_path}")
            shutil.copy2(temp_output_path, output_path)  # Użyj copy2 zamiast rename
            os.remove(temp_output_path)  # Usuń plik tymczasowy po skopiowaniu
            
            # Sprawdź, czy plik docelowy istnieje
            if not os.path.exists(output_path):
                print(f"Plik docelowy nie został utworzony: {output_path}")
                return None
                
            print(f"Plik został pomyślnie skopiowany do {output_path}")
            
        except Exception as e:
            print(f"Błąd podczas przenoszenia pliku: {e}")
            # Jeśli nie udało się przenieść, spróbuj zwrócić ścieżkę do pliku tymczasowego
            if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                print(f"Zwracam ścieżkę do pliku tymczasowego: {temp_output_path}")
                output_path = temp_output_path
            else:
                print("Nie można zwrócić nawet ścieżki tymczasowej")
                return None
            
        return {
            'filename': os.path.basename(output_path),
            'prediction': best_class,
            'probability': best_probability
        }
        
    except Exception as e:
        print(f"Błąd podczas przetwarzania wideo: {str(e)}")
        import traceback
        traceback.print_exc()  # Drukuj pełny traceback dla lepszego debugowania
        return None

def create_fallback_video(output_path, width, height, fps):
    """Tworzy awaryjne wideo z jedną klatką informującą o problemie"""
    try:
        # Zapewnienie, że szerokość i wysokość są poprawne
        width = max(320, min(width, 1280))
        height = max(240, min(height, 720))
        
        # Zapewnienie, że fps jest poprawne
        fps = max(10, min(fps, 30))
        
        # Upewnij się, że ścieżka wyjściowa ma rozszerzenie MP4
        if not output_path.lower().endswith('.mp4'):
            output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        
        # Utwórz pustą klatkę
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Dodaj tekst informacyjny
        text_lines = [
            "Przetwarzanie wideo nie powiodło się",
            "Spróbuj z innym plikiem",
            f"Rozdzielczość: {width}x{height}, FPS: {fps}"
        ]
        
        y_position = height // 2 - 30
        for line in text_lines:
            cv2.putText(
                frame, line,
                (width//10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2
            )
            y_position += 40
        
        # Dodaj obramowanie
        cv2.rectangle(frame, (20, 20), (width-20, height-20), (0, 120, 255), 2)
        
        # Ustaw tymczasową ścieżkę wyjściową
        temp_path = f"{output_path}.fallback.mp4"
        
        # Lista kodeków MP4 do wypróbowania
        codecs = [
            ('mp4v', 'mp4'),
            ('avc1', 'mp4'),
            ('h264', 'mp4')
        ]
        
        # Wypróbuj różne kodeki
        for codec, ext in codecs:
            try:
                print(f"Próba utworzenia awaryjnego wideo z kodekiem MP4 {codec}...")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_path = temp_path.replace('.mp4', f'.{ext}')
                out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                
                if out and out.isOpened():
                    # Zapisz klatkę wielokrotnie (ok. 5 sekund)
                    for _ in range(int(fps * 5)):
                        out.write(frame)
                    
                    out.release()
                    print(f"Utworzono awaryjne wideo MP4: {test_path}")
                    
                    # Skopiuj do docelowej ścieżki
                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        shutil.copy2(test_path, output_path)
                        os.remove(test_path)
                        print(f"Skopiowano awaryjne wideo MP4 do: {output_path}")
                        return output_path
                    except Exception as e:
                        print(f"Nie udało się skopiować awaryjnego wideo MP4: {e}")
                        return test_path  # Zwróć ścieżkę tymczasową, jeśli kopiowanie się nie udało
                        
                    break
                else:
                    if out:
                        out.release()
            except Exception as e:
                print(f"Błąd przy próbie utworzenia awaryjnego wideo MP4 z kodekiem {codec}: {str(e)}")
        
        print("Nie udało się utworzyć awaryjnego wideo MP4 z żadnym kodekiem")
        return None
    except Exception as e:
        print(f"Błąd podczas tworzenia awaryjnego wideo MP4: {e}")
        return None

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Obsługa wyświetlania przesłanych plików"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/processed/<path:filename>')
def processed_file(filename):
    """Obsługa wyświetlania przetworzonych plików"""
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], 'processed'), filename
    )

@app.route('/status')
def status():
    """Sprawdzenie statusu modeli"""
    fastai_status = 'loaded' if fastai_model is not None else 'not_loaded'
    
    return jsonify({
        'fastai_model': fastai_status
    })

@app.route('/load_models')
def load_models():
    """Ładowanie modeli"""
    global fastai_model
    
    if not os.path.exists(os.path.dirname(fastai_model_path)):
        os.makedirs(os.path.dirname(fastai_model_path), exist_ok=True)
    
    # Sprawdzenie czy model fastai został skopiowany
    if not os.path.exists(fastai_model_path):
        try:
            # Szukanie modelu w folderze rozwiązanie
            solution_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                'rozwiązanie'
            )
            possible_paths = [
                os.path.join(solution_path, 'klasyfikator_zwierząt.pkl'),
                os.path.join(solution_path, 'export.pkl')
            ]
            
            model_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    shutil.copy(path, fastai_model_path)
                    fastai_model = load_learner(fastai_model_path)
                    model_found = True
                    break
            
            if not model_found:
                return jsonify({
                    'error': 'Nie znaleziono modelu klasyfikacyjnego. ' + 
                             'Upewnij się, że model został wytrenowany.'
                }), 404
        except Exception as e:
            return jsonify({'error': f'Błąd kopiowania modelu: {str(e)}'}), 500
    else:
        try:
            if fastai_model is None:
                fastai_model = load_learner(fastai_model_path)
        except Exception as e:
            return jsonify({'error': f'Błąd ładowania modelu: {str(e)}'}), 500
    
    return jsonify({
        'fastai_model': 'loaded'
    })

@app.route('/datasets', methods=['GET'])
def list_datasets():
    """Zwraca listę dostępnych datasetów"""
    datasets = []
    
    if os.path.exists(app.config['DATASET_PATH']):
        for dataset in os.listdir(app.config['DATASET_PATH']):
            dataset_path = os.path.join(app.config['DATASET_PATH'], dataset)
            if os.path.isdir(dataset_path):
                # Pobierz kategorie (podfoldery) w datasecie
                categories = []
                for category in os.listdir(dataset_path):
                    category_path = os.path.join(dataset_path, category)
                    if os.path.isdir(category_path):
                        # Policz obrazy w kategorii
                        image_count = len([f for f in os.listdir(category_path) 
                                         if os.path.isfile(os.path.join(category_path, f)) and 
                                         f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
                        categories.append({
                            'name': category,
                            'image_count': image_count
                        })
                
                datasets.append({
                    'name': dataset,
                    'categories': categories,
                    'total_images': sum(c['image_count'] for c in categories)
                })
    
    return jsonify({'datasets': datasets})

@app.route('/create_dataset', methods=['POST'])
def create_dataset():
    """Tworzy nowy dataset"""
    if 'dataset_name' not in request.form:
        return jsonify({'error': 'Brak nazwy datasetu'}), 400
    
    dataset_name = request.form['dataset_name']
    if not dataset_name or dataset_name.strip() == '':
        return jsonify({'error': 'Nazwa datasetu nie może być pusta'}), 400
    
    # Zabezpieczenie nazwy
    dataset_name = secure_filename(dataset_name)
    dataset_path = os.path.join(app.config['DATASET_PATH'], dataset_name)
    
    if os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset o tej nazwie już istnieje'}), 400
    
    try:
        os.makedirs(dataset_path, exist_ok=True)
        return jsonify({'success': True, 'dataset_name': dataset_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/create_category', methods=['POST'])
def create_category():
    """Tworzy nową kategorię w datasecie"""
    if 'dataset_name' not in request.form or 'category_name' not in request.form:
        return jsonify({'error': 'Brak nazwy datasetu lub kategorii'}), 400
    
    dataset_name = request.form['dataset_name']
    category_name = request.form['category_name']
    
    if not dataset_name or not category_name or dataset_name.strip() == '' or category_name.strip() == '':
        return jsonify({'error': 'Nazwa datasetu i kategorii nie mogą być puste'}), 400
    
    # Zabezpieczenie nazw
    dataset_name = secure_filename(dataset_name)
    category_name = secure_filename(category_name)
    
    dataset_path = os.path.join(app.config['DATASET_PATH'], dataset_name)
    if not os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset nie istnieje'}), 404
    
    category_path = os.path.join(dataset_path, category_name)
    if os.path.exists(category_path):
        return jsonify({'error': 'Kategoria o tej nazwie już istnieje w tym datasecie'}), 400
    
    try:
        os.makedirs(category_path, exist_ok=True)
        return jsonify({'success': True, 'dataset_name': dataset_name, 'category_name': category_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_to_category', methods=['POST'])
def upload_to_category():
    """Dodaje obrazy do kategorii w datasecie"""
    if 'dataset_name' not in request.form or 'category_name' not in request.form:
        return jsonify({'error': 'Brak nazwy datasetu lub kategorii'}), 400
    
    if 'files[]' not in request.files:
        return jsonify({'error': 'Brak plików'}), 400
    
    dataset_name = request.form['dataset_name']
    category_name = request.form['category_name']
    
    # Zabezpieczenie nazw
    dataset_name = secure_filename(dataset_name)
    category_name = secure_filename(category_name)
    
    category_path = os.path.join(app.config['DATASET_PATH'], dataset_name, category_name)
    if not os.path.exists(category_path):
        return jsonify({'error': 'Kategoria nie istnieje'}), 404
    
    files = request.files.getlist('files[]')
    saved_files = []
    
    for file in files:
        if file and allowed_file(file.filename) and not is_video_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(category_path, unique_filename)
            file.save(file_path)
            saved_files.append(unique_filename)
    
    return jsonify({
        'success': True, 
        'saved_files': saved_files,
        'dataset_name': dataset_name,
        'category_name': category_name
    })

@app.route('/download_example_dataset', methods=['POST'])
def download_example_dataset():
    """Pobiera przykładowy dataset"""
    if 'dataset_type' not in request.form:
        return jsonify({'error': 'Nie wybrano typu datasetu'}), 400
    
    dataset_type = request.form['dataset_type']
    
    if dataset_type not in EXAMPLE_DATASETS:
        return jsonify({'error': 'Nieznany typ datasetu'}), 400
    
    dataset_url = EXAMPLE_DATASETS[dataset_type]
    success = download_and_extract_dataset(dataset_type, dataset_url)
    
    if success:
        return jsonify({'success': True, 'dataset_name': dataset_type})
    else:
        return jsonify({'error': 'Błąd podczas pobierania datasetu'}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    """Trenuje model na wybranym datasecie"""
    if 'dataset_name' not in request.form:
        return jsonify({'error': 'Nie wybrano datasetu'}), 400
    
    dataset_name = request.form['dataset_name']
    model_name = request.form.get('model_name', f"model_{dataset_name}_{uuid.uuid4()}")
    architecture = request.form.get('architecture', 'resnet34')
    epochs = int(request.form.get('epochs', 5))
    batch_size = int(request.form.get('batch_size', 16))
    
    # Zabezpieczenie nazw
    dataset_name = secure_filename(dataset_name)
    model_name = secure_filename(model_name)
    
    dataset_path = os.path.join(app.config['DATASET_PATH'], dataset_name)
    if not os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset nie istnieje'}), 404
    
    if architecture not in AVAILABLE_MODELS:
        return jsonify({'error': 'Nieznana architektura modelu'}), 400
    
    # Sprawdź czy dataset zawiera co najmniej 2 kategorie
    categories = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    
    if len(categories) < 2:
        return jsonify({'error': 'Dataset musi zawierać co najmniej 2 kategorie'}), 400
    
    try:
        # Przygotowanie danych
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=parent_label,
            item_tfms=[Resize(224)],
            batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
        )
        
        # Tworzenie dataloadera
        dls = dblock.dataloaders(dataset_path, bs=batch_size)
        
        # Tworzenie modelu
        model_func = AVAILABLE_MODELS[architecture]
        learn = vision_learner(dls, model_func, metrics=error_rate)
        
        # Trenowanie modelu
        learn.fine_tune(epochs)
        
        # Zapisanie modelu
        model_path = os.path.join(app.config['MODEL_PATH'], f"{model_name}.pkl")
        learn.export(model_path)
        
        return jsonify({
            'success': True, 
            'model_name': f"{model_name}.pkl",
            'accuracy': float(1 - learn.validate()[1]),
            'categories': categories
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    """Ładuje wybrany model do użycia"""
    if 'model_name' not in request.form:
        return jsonify({'error': 'Nie wybrano modelu'}), 400
    
    model_name = request.form['model_name']
    
    model_path = os.path.join(app.config['MODEL_PATH'], model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model nie istnieje'}), 404
    
    try:
        global fastai_model
        fastai_model = load_learner(model_path)
        return jsonify({'success': True, 'model_name': model_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Sprawdzenie czy model jest załadowany przy starcie
    if fastai_model is None:
        print("Model fastai nie jest załadowany. " + 
              "Spróbuj załadować ręcznie przez /load_models")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 