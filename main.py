import cv2
import sqlite3
import numpy as np
from deepface import DeepFace
from playsound import playsound
import os

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
            
            sound_path = os.path.join('sounds', 'beep.wav')
            if os.path.exists(sound_path):
                playsound(sound_path)
            return True
    conn.close()
    return False

def main():
    known_face_encodings, known_face_names = load_encodings_from_db()
    
    if not known_face_encodings:
        print("Nenhum rosto treinado encontrado no banco de dados. Execute o script 'train_model.py' primeiro.")
        return

    video_capture = cv2.VideoCapture(0)
    present_students = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        current_visible_names = set()
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = []
            for backend in ['retinaface', 'mediapipe', 'mtcnn', 'opencv']:
                try:
                    faces = DeepFace.extract_faces(frame_rgb, detector_backend=backend, enforce_detection=False)
                    if faces:
                        break
                except Exception:
                    continue

            for face_info in faces:
                fa = face_info.get('facial_area', {})
                x = int(fa.get('x', 0))
                y = int(fa.get('y', 0))
                w = int(fa.get('w', 0))
                h = int(fa.get('h', 0))

                x = max(0, x)
                y = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)
                if x2 <= x or y2 <= y:
                    continue

                face_img = frame[y:y2, x:x2]
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                embedding_objs = DeepFace.represent(rgb_face, model_name='Facenet', detector_backend='mtcnn', enforce_detection=False)
                if embedding_objs:
                    face_encoding = np.array(embedding_objs[0]['embedding'], dtype=np.float32)
                    if face_encoding.ndim != 1:
                        face_encoding = face_encoding.flatten()
                    face_norm = np.linalg.norm(face_encoding) + 1e-10
                    face_vec = face_encoding / face_norm

                    sims = []
                    for ke in known_face_encodings:
                        ke = ke.astype(np.float32)
                        ke_vec = ke / (np.linalg.norm(ke) + 1e-10)
                        sims.append(np.dot(face_vec, ke_vec))

                    if sims:
                        best_match_index = int(np.argmax(sims))
                        if sims[best_match_index] > 0.65:
                            name = known_face_names[best_match_index]
                            if name not in present_students:
                                if mark_attendance(name):
                                    present_students.add(name)
                            current_visible_names.add(name)
                        else:
                            name = "Desconhecido"
                    else:
                        name = "Desconhecido"

                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            faces = []

        live_count = len(current_visible_names)
        cv2.putText(frame, f"Na sala agora: {live_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Total presentes: {len(present_students)}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if not faces:
            cv2.putText(frame, "Nenhum rosto detectado", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists('sounds'):
        os.makedirs('sounds')
    
    main()
