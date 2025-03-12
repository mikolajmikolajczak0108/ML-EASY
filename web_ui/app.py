import os
import cv2
import uuid
import shutil
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np

# Importy dla fastai
from fastai.vision.all import load_learner, PILImage

# Konfiguracja aplikacji
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # limit 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}
app.config['MODEL_PATH'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models')

# Tworzenie katalogów
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_PATH'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'processed'), exist_ok=True)

def allowed_file(filename):
    """Sprawdza czy rozszerzenie pliku jest dozwolone"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_video_file(filename):
    """Sprawdza czy plik jest filmem"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

# Ścieżki do modeli
fastai_model_path = os.path.join(app.config['MODEL_PATH'], 'klasyfikator_zwierząt.pkl')
yolo_model_path = os.path.join(app.config['MODEL_PATH'], 'yolov5s.pt')

# Ładowanie modelu fastai (jeśli istnieje)
fastai_model = None
try:
    if os.path.exists(fastai_model_path):
        fastai_model = load_learner(fastai_model_path)
except Exception as e:
    print(f"Błąd ładowania modelu fastai: {e}")

# Ładowanie lub pobieranie modelu YOLOv5
yolo_model = None
def load_yolo_model():
    global yolo_model
    try:
        if not os.path.exists(yolo_model_path):
            print("Pobieranie modelu YOLOv5...")
            yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            # Zapisanie modelu
            torch.save(yolo_model.state_dict(), yolo_model_path)
        else:
            yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return True
    except Exception as e:
        print(f"Błąd ładowania modelu YOLOv5: {e}")
        return False

@app.route('/')
def index():
    """Strona główna"""
    return render_template('index.html')

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
    """Detekcja obiektów przy użyciu modelu YOLOv5"""
    if 'file' not in request.files:
        return jsonify({'error': 'Brak pliku'}), 400
    
    # Ładowanie modelu YOLOv5 (jeśli nie jest załadowany)
    if yolo_model is None:
        if not load_yolo_model():
            return jsonify({'error': 'Nie można załadować modelu YOLOv5'}), 500
    
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
            output_path = process_video(file_path)
            if output_path:
                os.remove(file_path)  # Usunięcie oryginalnego wideo
                filename = os.path.basename(output_path)
                return jsonify({
                    'filename': filename,
                    'source': 'upload'
                })
            else:
                return jsonify({'error': 'Nie udało się przetworzyć wideo'}), 500
        else:
            # Detekcja na obrazie
            results = yolo_model(file_path)
            
            # Zapisanie wyników
            results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'processed')
            results.save(save_dir=results_dir)
            
            # Znalezienie nazwy zapisanego pliku
            results_filename = None
            for f in os.listdir(results_dir):
                if f.startswith(unique_filename.split('.')[0]):
                    results_filename = f
                    break
            
            if not results_filename:
                return jsonify({'error': 'Nie udało się zapisać wyników detekcji'}), 500
            
            # Przefiltrowanie wyników (tylko zwierzęta)
            animals = []
            for *box, conf, cls in results.xyxy[0]:
                class_name = results.names[int(cls)]
                if class_name in ['dog', 'cat', 'bird']:
                    animals.append({
                        'class': class_name,
                        'confidence': float(conf)
                    })
            
            return jsonify({
                'filename': results_filename,
                'source': 'upload',
                'animals': animals
            })
    except Exception as e:
        print(f"Błąd podczas przetwarzania: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_video(video_path):
    """Przetwarzanie wideo z detekcją obiektów"""
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
        
        # Utworzenie nazwy pliku wynikowego
        output_filename = f"processed_{os.path.basename(video_path)}"
        output_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'processed', output_filename)
        output_path = output_path.replace('.mp4', '.avi')  # Zawsze używamy .avi
        
        # Utworzenie tymczasowego pliku zamiast bezpośredniego zapisu
        temp_output_path = f"{output_path}.temp.avi"
        
        # Lista dostępnych kodeków do wypróbowania
        codecs = [
            ('XVID', 'avi'),
            ('MJPG', 'avi'),
            ('MP4V', 'avi'),
            ('DIV3', 'avi'),
            ('X264', 'avi')
        ]
        
        # Próba użycia różnych kodeków
        out = None
        for codec, ext in codecs:
            try:
                print(f"Próba użycia kodeka {codec}...")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_path = temp_output_path.replace('.avi', f'.{ext}')
                out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                
                if out.isOpened():
                    temp_output_path = test_path
                    print(f"Udane otwarcie pliku wyjściowego z kodekiem {codec}")
                    break
                else:
                    out.release()
                    out = None
            except Exception as e:
                print(f"Błąd przy próbie użycia kodeka {codec}: {str(e)}")
                out = None
        
        if out is None or not out.isOpened():
            print("Nie można utworzyć pliku wyjściowego z żadnym kodekiem")
            cap.release()
            return None
        
        # Klasy zwierząt, które nas interesują
        animal_classes = ['dog', 'cat', 'bird']
        animal_class_indices = [
            yolo_model.names.index(cls) if cls in yolo_model.names else -1 
            for cls in animal_classes
        ]
        animal_class_indices = [idx for idx in animal_class_indices if idx != -1]
        
        frame_count = 0
        processed_count = 0
        # Przetwarzamy mniej klatek dla dużych wideo, ale co najmniej 30
        sample_rate = max(1, total_frames // min(300, total_frames))
        
        print(f"Rozpoczynam przetwarzanie wideo, sampling rate: {sample_rate}")
        
        # Zmienna, która zapamięta, czy co najmniej jedna klatka została zapisana
        frames_written = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Przetwarzamy co sample_rate klatek
            if frame_count % sample_rate != 0:
                continue
            
            processed_count += 1
            
            # Zmiana rozmiaru klatki
            frame = cv2.resize(frame, (width, height))
            
            # Detekcja obiektów
            results = yolo_model(frame)
            
            # Rysowanie prostokątów i etykiet
            detections = results.xyxy[0].cpu().numpy()
            animals_detected = False
            
            for *xyxy, conf, cls_idx in detections:
                if int(cls_idx) in animal_class_indices and conf > 0.25:  # Obniżony próg pewności
                    animals_detected = True
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = f"{yolo_model.names[int(cls_idx)]} {conf:.2f}"
                    
                    # Określenie koloru na podstawie klasy
                    if yolo_model.names[int(cls_idx)] == 'dog':
                        color = (0, 255, 0)  # Zielony dla psów
                    elif yolo_model.names[int(cls_idx)] == 'cat':
                        color = (0, 0, 255)  # Czerwony dla kotów
                    elif yolo_model.names[int(cls_idx)] == 'bird':
                        color = (255, 0, 0)  # Niebieski dla ptaków
                    else:
                        color = (255, 255, 0)  # Żółty dla innych
                    
                    # Narysowanie prostokąta i etykiety
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
            
            # Dodanie informacji tekstowej na klatce
            cv2.putText(
                frame, f"Klatka: {frame_count}/{total_frames}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # Zapisanie klatki
            out.write(frame)
            frames_written += 1
            
            # Sprawdzenie co 50 klatek czy zapisywanie działa
            if processed_count % 50 == 0:
                print(f"Przetworzono {processed_count} klatek, zapisano {frames_written}")
            
            # Ograniczenie maksymalnej liczby klatek do 1000 (dla bezpieczeństwa)
            if processed_count >= 1000:
                print("Osiągnięto limit 1000 przetworzonych klatek")
                break
        
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
                return fallback_path
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
                return temp_output_path
            else:
                print("Nie można zwrócić nawet ścieżki tymczasowej")
                return None
            
        return output_path
        
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
        temp_path = f"{output_path}.fallback.avi"
        
        # Lista kodeków do wypróbowania
        codecs = [
            ('XVID', 'avi'),
            ('MJPG', 'avi'),
            ('MP4V', 'avi')
        ]
        
        # Wypróbuj różne kodeki
        for codec, ext in codecs:
            try:
                print(f"Próba utworzenia awaryjnego wideo z kodekiem {codec}...")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_path = temp_path.replace('.avi', f'.{ext}')
                out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                
                if out.isOpened():
                    # Zapisz klatkę wielokrotnie (ok. 5 sekund)
                    for _ in range(int(fps * 5)):
                        out.write(frame)
                    
                    out.release()
                    print(f"Utworzono awaryjne wideo: {test_path}")
                    
                    # Skopiuj do docelowej ścieżki
                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        shutil.copy2(test_path, output_path)
                        os.remove(test_path)
                        print(f"Skopiowano awaryjne wideo do: {output_path}")
                        return output_path
                    except Exception as e:
                        print(f"Nie udało się skopiować awaryjnego wideo: {e}")
                        return test_path  # Zwróć ścieżkę tymczasową, jeśli kopiowanie się nie udało
                        
                    break
                else:
                    out.release()
            except Exception as e:
                print(f"Błąd przy próbie utworzenia awaryjnego wideo z kodekiem {codec}: {str(e)}")
        
        print("Nie udało się utworzyć awaryjnego wideo z żadnym kodekiem")
        return None
    except Exception as e:
        print(f"Błąd podczas tworzenia awaryjnego wideo: {e}")
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
    yolo_status = 'loaded' if yolo_model is not None else 'not_loaded'
    
    return jsonify({
        'fastai_model': fastai_status,
        'yolo_model': yolo_status
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
    
    # Ładowanie modelu YOLOv5
    if not load_yolo_model():
        return jsonify({'error': 'Nie można załadować modelu YOLOv5'}), 500
    
    return jsonify({
        'fastai_model': 'loaded',
        'yolo_model': 'loaded'
    })

if __name__ == '__main__':
    # Sprawdzenie czy modele są załadowane przy starcie
    if fastai_model is None:
        print("Model fastai nie jest załadowany. " + 
              "Spróbuj załadować ręcznie przez /load_models")
    
    if yolo_model is None:
        print("Model YOLOv5 nie jest załadowany. " + 
              "Spróbuj załadować ręcznie przez /load_models")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 