from deepface import DeepFace
import os
import sqlite3
import numpy as np
from PIL import Image
import cv2

def ensure_arcface():
    try:
        from arcface_onnx import ArcFaceONNX
        import onnxruntime as ort  # noqa: F401
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

def train_model():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("DELETE FROM encodings")
    cursor.execute("DELETE FROM students")
    conn.commit()

    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        print(f"Diretório '{dataset_path}' não encontrado. Crie o diretório e adicione as imagens dos alunos.")
        return

    arc = ensure_arcface()

    for student_name in os.listdir(dataset_path):
        student_path = os.path.join(dataset_path, student_name)
        if os.path.isdir(student_path):
            cursor.execute("INSERT INTO students (name) VALUES (?)", (student_name,))
            student_id = cursor.lastrowid
            
            for image_name in os.listdir(student_path):
                image_path = os.path.join(student_path, image_name)
                try:
                    if arc is not None:
                        img = cv2.imread(image_path)
                        if img is None:
                            continue
                        embedding = arc.get_embedding(img).astype(np.float32)
                        cursor.execute("INSERT INTO encodings (student_id, encoding) VALUES (?, ?)",
                                       (student_id, embedding.tobytes()))
                    else:
                        embedding_objs = DeepFace.represent(img_path=image_path, model_name='Facenet', detector_backend='mtcnn', enforce_detection=False)
                        if not embedding_objs:
                            continue
                        embedding = np.array(embedding_objs[0]['embedding'], dtype=np.float32)
                        cursor.execute("INSERT INTO encodings (student_id, encoding) VALUES (?, ?)",
                                       (student_id, embedding.tobytes()))
                except Exception as e:
                    print(f"Erro ao processar a imagem {image_path}: {e}")

    conn.commit()
    conn.close()
    print("Treinamento concluído e embeddings armazenados no banco de dados.")

if __name__ == "__main__":
    train_model()
