import cv2
import sqlite3
import numpy as np
from deepface import DeepFace
from playsound import playsound
import os
import time
import threading
from typing import Optional

def ensure_arcface():
    try:
        from arcface_onnx import ArcFaceONNX
        import onnxruntime as ort  
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'face_recognition_sface_2021dec.onnx')
        if not os.path.exists(model_path):
            urls = [
                'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
                'https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'
            ]
            import urllib.request
            for u in urls:
                try:
                    urllib.request.urlretrieve(u, model_path)
                    break
                except Exception:
                    continue
            if not os.path.exists(model_path):
                raise RuntimeError('Falha ao baixar modelo SFace ONNX')
        return ArcFaceONNX(model_path)
    except Exception:
        return None

arc = ensure_arcface()
TARGET_WIDTH = 640
DETECTOR_BACKEND = 'mediapipe'
FALLBACK_DETECTORS = ['mtcnn', 'opencv']  # tried only if primary returns 0 faces
DETECT_EVERY_N = 3
SIM_THRESHOLD = 0.65

def _play_beep_async():
    sound_path = os.path.join('sounds', 'beep.wav')
    if os.path.exists(sound_path):
        threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()

def load_encodings_from_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.name, e.encoding
        FROM students s
        JOIN encodings e ON s.id = e.student_id
    ''')
    known_face_encodings = []
    known_face_names = []
    for row in cursor.fetchall():
        name = row[0]
        encoding = np.frombuffer(row[1], dtype=np.float32)
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    conn.close()
    return known_face_encodings, known_face_names

def mark_attendance(student_name):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM students WHERE name = ?", (student_name,))
    student_id_row = cursor.fetchone()
    if student_id_row:
        student_id = student_id_row[0]
        cursor.execute("SELECT * FROM attendance WHERE student_id = ? AND DATE(timestamp) = DATE('now')", (student_id,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO attendance (student_id) VALUES (?)", (student_id,))
            conn.commit()
            print(f"Presença de {student_name} registrada.")
            _play_beep_async()
            return True
    conn.close()
    return False

def main():
    known_face_encodings, known_face_names = load_encodings_from_db()
    if not known_face_encodings:
        print("Nenhum rosto treinado encontrado no banco de dados. Execute o script 'train_model.py' primeiro.")
        return

    # Pre-normalize known embeddings (speed up cosine sim)
    known_vecs = []
    for ke in known_face_encodings:
        ke = ke.astype(np.float32)
        known_vecs.append(ke / (np.linalg.norm(ke) + 1e-10))
    known_mat = np.stack(known_vecs, axis=0) if known_vecs else np.empty((0, 0), dtype=np.float32)

    cv2.setUseOptimized(True)

    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    present_students = set()
    frame_count = 0
    last_faces = []  # list of (x, y, x2, y2, name)
    last_time = time.time()
    fps = 0.0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        current_visible_names = set()
        detected_this_frame = False
        try:
            if frame_count % DETECT_EVERY_N == 0:
                detected_this_frame = True
                h0, w0 = frame.shape[:2]
                if w0 > TARGET_WIDTH:
                    scale = TARGET_WIDTH / float(w0)
                    small = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
                else:
                    scale = 1.0
                    small = frame
                small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                try:
                    faces = DeepFace.extract_faces(small_rgb, detector_backend=DETECTOR_BACKEND, enforce_detection=False) or []
                except Exception:
                    faces = []

                # Fallback detectors if nothing found
                if not faces:
                    for bk in FALLBACK_DETECTORS:
                        try:
                            faces = DeepFace.extract_faces(small_rgb, detector_backend=bk, enforce_detection=False) or []
                            if faces:
                                break
                        except Exception:
                            faces = []

                last_faces = []
                for face_info in faces:
                    fa = face_info.get('facial_area', {})
                    x = int(fa.get('x', 0) / scale)
                    y = int(fa.get('y', 0) / scale)
                    w = int(fa.get('w', 0) / scale)
                    h = int(fa.get('h', 0) / scale)
                    x = max(0, x)
                    y = max(0, y)
                    x2 = min(frame.shape[1], x + w)
                    y2 = min(frame.shape[0], y + h)
                    if x2 <= x or y2 <= y:
                        continue
                    face_img = frame[y:y2, x:x2]
                    name = "Desconhecido"

                    face_encoding: Optional[np.ndarray] = None
                    if arc is not None:
                        face_encoding = arc.get_embedding(face_img).astype(np.float32)
                    else:
                        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        embedding_objs = DeepFace.represent(rgb_face, model_name='Facenet', detector_backend='mtcnn', enforce_detection=False)
                        if embedding_objs:
                            face_encoding = np.array(embedding_objs[0]['embedding'], dtype=np.float32)

                    if face_encoding is not None and known_mat.size > 0:
                        if face_encoding.ndim != 1:
                            face_encoding = face_encoding.flatten()
                        face_vec = face_encoding / (np.linalg.norm(face_encoding) + 1e-10)
                        sims = known_mat @ face_vec
                        best_idx = int(np.argmax(sims))
                        if sims[best_idx] > SIM_THRESHOLD:
                            name = known_face_names[best_idx]
                            if name not in present_students:
                                if mark_attendance(name):
                                    present_students.add(name)
                            current_visible_names.add(name)

                    last_faces.append((x, y, x2, y2, name))

            # draw cached results every frame (cheap)
            for (x, y, x2, y2, name) in last_faces:
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception:
            pass

        # FPS
        now = time.time()
        dt = now - last_time
        if dt > 0:
            inst = 1.0 / dt
            fps = (fps * 0.9 + inst * 0.1) if fps > 0 else inst
        last_time = now
        frame_count += 1

        live_count = len(current_visible_names)
        cv2.putText(frame, f"Na sala agora: {live_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Total presentes: {len(present_students)}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if detected_this_frame and len(last_faces) == 0:
            cv2.putText(frame, "Nenhum rosto detectado", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        provider = 'CPU' if arc is None else (arc.active_providers[0] if getattr(arc, 'active_providers', None) else 'CPU')
        cv2.putText(frame, f"FPS: {fps:.1f} | EP: {provider} | Detect/skip: {DETECT_EVERY_N}", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        cv2.putText(frame, f"Caixas: {len(last_faces)}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists('sounds'):
        os.makedirs('sounds')
    main()
