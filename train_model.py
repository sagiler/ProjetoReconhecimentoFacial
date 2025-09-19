from deepface import DeepFace
import os
import sqlite3
import numpy as np
from PIL import Image

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

    for student_name in os.listdir(dataset_path):
        student_path = os.path.join(dataset_path, student_name)
        if os.path.isdir(student_path):
            cursor.execute("INSERT INTO students (name) VALUES (?)", (student_name,))
            student_id = cursor.lastrowid
            
            for image_name in os.listdir(student_path):
                image_path = os.path.join(student_path, image_name)
                try:
                    embedding_objs = DeepFace.represent(img_path=image_path, model_name='Facenet', detector_backend='mtcnn', enforce_detection=False)
                    
                    if embedding_objs:
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
