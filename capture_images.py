import cv2
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def capture_images_from_webcam():
    student_name = input("Digite seu nome (sem espaços ou caracteres especiais): ")
    if not student_name:
        print("Nome inválido.")
        return

    dataset_path = os.path.join(BASE_DIR, 'dataset')
    student_path = os.path.join(dataset_path, student_name)

    if not os.path.exists(student_path):
        os.makedirs(student_path)
        print(f"Diretório criado em: {student_path}")

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return

    img_counter = 0
    capture_count = 20 
    
    print("\nPrepare-se para a captura!")
    print(f"Serão tiradas {capture_count} fotos. Mova sua cabeça lentamente para capturar diferentes ângulos.")
    time.sleep(3)

    while img_counter < capture_count:
        ret, frame = video_capture.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            break

        cv2.imshow('Capturando Rosto - Pressione "s" para salvar ou "q" para sair', frame)
        
        key = cv2.waitKey(1)

        if key % 256 == ord('q'):
            print("Captura interrompida pelo usuário.")
            break
        
        # Captura automática com um pequeno intervalo
        time.sleep(0.2) # Intervalo de 200ms entre as fotos
        
        img_name = os.path.join(student_path, f"{student_name}_{img_counter}.png")
        cv2.imwrite(img_name, frame)
        print(f"{img_name} salvo!")
        img_counter += 1


    video_capture.release()
    cv2.destroyAllWindows()
    print(f"\n{img_counter} imagens salvas em '{student_path}'.")
    print("Agora, execute o script 'train_model.py' para treinar o modelo com seu rosto.")

if __name__ == "__main__":
    capture_images_from_webcam()
