import cv2
import os
import time

def guided_capture():
    student_name = input("Digite seu nome (sem espaços ou caracteres especiais): ")
    if not student_name:
        print("Nome inválido.")
        return

    dataset_path = 'dataset'
    student_path = os.path.join(dataset_path, student_name)

    if not os.path.exists(student_path):
        os.makedirs(student_path)
        print(f"Diretório criado em: {student_path}")

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    positions = {
        "Centro": False,
        "Direita": False,
        "Esquerda": False,
        "Cima": False,
        "Baixo": False
    }
    
    current_position_idx = 0
    position_keys = list(positions.keys())
    
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    center_x, center_y = frame_width // 2, frame_height // 2
    zone_threshold = 0.25 

    print("\nIniciando captura guiada. Siga as instruções na tela.")
    time.sleep(2)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_flipped = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        target_position = position_keys[current_position_idx]
        instruction_text = f"Posicione o rosto: {target_position}"
        ui_color = (0, 0, 255) # Vermelho

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_center_x = x + w // 2
            
            current_face_pos = "Desconhecida"

            if abs(face_center_x - center_x) < w * zone_threshold:
                current_face_pos = "Centro"
            elif face_center_x < center_x - w * zone_threshold:
                current_face_pos = "Esquerda"
            elif face_center_x > center_x + w * zone_threshold:
                current_face_pos = "Direita"
            
            # Simplificando para cima/baixo (pode ser menos preciso)
            if y < frame_height * 0.2:
                 current_face_pos = "Cima"
            elif y > frame_height * 0.5:
                 current_face_pos = "Baixo"


            if current_face_pos == target_position:
                ui_color = (0, 255, 0) # Verde
                instruction_text = f"{target_position} - Capturado!"
                
                img_name = os.path.join(student_path, f"{student_name}_{target_position}.png")
                cv2.imwrite(img_name, cv2.flip(frame_flipped, 1)) # Salva a imagem original, não a espelhada
                print(f"Imagem para '{target_position}' salva em {img_name}")
                
                positions[target_position] = True
                time.sleep(1) # Pausa para o usuário ver o feedback

                if all(positions.values()):
                    break
                
                current_position_idx += 1

        # Desenha a UI
        cv2.ellipse(frame_flipped, (center_x, center_y), (100, 140), 0, 0, 360, ui_color, 2)
        cv2.putText(frame_flipped, instruction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, ui_color, 2)
        
        progress_text = f"Progresso: {sum(positions.values())}/{len(positions)}"
        cv2.putText(frame_flipped, progress_text, (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Captura Guiada - Pressione "q" para sair', frame_flipped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
    if all(positions.values()):
        print("\nCaptura guiada concluída com sucesso!")
        print("Execute 'train_model.py' para treinar o modelo com as novas imagens.")
    else:
        print("\nCaptura interrompida.")

if __name__ == "__main__":
    guided_capture()
