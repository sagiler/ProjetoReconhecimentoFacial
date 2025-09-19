import cv2
import os
import time
import math
from collections import deque
import numpy as np
from deepface import DeepFace
from playsound import playsound

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
SOUNDS_DIR = os.path.join(BASE_DIR, 'sounds')

# UX/Qualidade
MIN_FACE_RATIO = 0.22   # face_w / frame_w >=
MAX_FACE_RATIO = 0.6    # evita muito perto
MIN_BLUR_VAR = 90.0     # variância do Laplaciano mínima (foco)
BRIGHTNESS_MIN = 70     # média de brilho mínima
BRIGHTNESS_MAX = 200    # média de brilho máxima
STABILITY_FRAMES = 10   # frames consecutivos para capturar
MOVE_TOLERANCE = 12     # tolerância de movimento em px para estabilidade
SECTOR_THRESHOLD = 0.25 # limiar relativo para setores (esq/dir/cima/baixo)
RING_RADIUS_RATIO = 0.33
CAPTURE_DELAY_S = 0.2   # pequena pausa antes de capturar (anti-tremor)

STEPS = [
    ("Centro",      (0,  0)),
    ("Direita",     (1,  0)),
    ("Esquerda",    (-1, 0)),
    ("Cima",        (0, -1)),
    ("Baixo",       (0,  1)),
    ("Cima-Direita",  (1, -1)),
    ("Cima-Esquerda", (-1,-1)),
    ("Baixo-Direita", (1,  1)),
    ("Baixo-Esquerda",(-1, 1)),
]

def _play_beep():
    path = os.path.join(SOUNDS_DIR, 'beep.wav')
    if os.path.exists(path):
        try:
            playsound(path)
        except Exception:
            pass

def _laplacian_var(gray_roi):
    return cv2.Laplacian(gray_roi, cv2.CV_64F).var()

def _brightness(gray_roi):
    return float(np.mean(gray_roi))

def _draw_ring_and_progress(img, center, radius, done_count, total, color_ready):
    cx, cy = center
    # anel base
    cv2.circle(img, (cx, cy), radius, (80, 80, 80), 6)
    # progresso
    if total > 0 and done_count > 0:
        angle = int(360 * done_count / total)
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, -90, -90 + angle, color_ready, 10)

def _in_sector(dx, dy, target):
    tx, ty = target
    # Centro: ambos |dx| e |dy| pequenos
    if tx == 0 and ty == 0:
        return abs(dx) < SECTOR_THRESHOLD/2 and abs(dy) < SECTOR_THRESHOLD/2
    # Direções/diagonais: sinais compatíveis e magnitude suficiente
    okx = (tx == 0) or (dx * tx >  SECTOR_THRESHOLD)
    oky = (ty == 0) or (dy * ty >  SECTOR_THRESHOLD)
    # se diagonal, exigir ambos; se eixo, exigir o respectivo
    if tx != 0 and ty != 0:
        return okx and oky
    return okx or oky

def guided_capture():
    student_name = input("Digite seu nome (sem espaços ou caracteres especiais): ").strip()
    if not student_name:
        print("Nome inválido.")
        return

    student_dir = os.path.join(DATASET_DIR, student_name)
    os.makedirs(student_dir, exist_ok=True)
    if not os.path.exists(SOUNDS_DIR):
        os.makedirs(SOUNDS_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return

    # Dimensões
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * RING_RADIUS_RATIO)

    current_idx = 0
    stability_queue = deque(maxlen=STABILITY_FRAMES)
    last_capture_time = 0.0
    completed = [False] * len(STEPS)

    # Ajuda inicial
    print("\nIniciando captura guiada estilo Face ID.\nDicas: mantenha o rosto centralizado, boa iluminação e mova a cabeça conforme instruções.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Espelhar para UX natural
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecção de rosto (robusta) via DeepFace
        try:
            faces = DeepFace.extract_faces(rgb, detector_backend='mediapipe', enforce_detection=False) or []
        except Exception:
            faces = []

        ui_color = (0, 0, 255) # vermelho por padrão
        status_lines = []

        # Ring e progresso
        _draw_ring_and_progress(frame, (cx, cy), radius, sum(completed), len(STEPS), (50, 215, 50))

        # Instrução ativa
        step_label, target = STEPS[current_idx]
        cv2.putText(frame, f"Passo {current_idx+1}/{len(STEPS)}: {step_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, "Siga o guia e mantenha-se estavel.", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2)

        ready_to_capture = False

        if faces:
            fa = faces[0].get('facial_area', {})
            x, y, fw, fh = int(fa.get('x', 0)), int(fa.get('y', 0)), int(fa.get('w', 0)), int(fa.get('h', 0))
            # Desenhar caixa
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 180), 2)

            # Métricas de qualidade na ROI
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x2, y2 = max(0, x), max(0, y)
            x3, y3 = min(w, x+fw), min(h, y+fh)
            roi = gray[y2:y3, x2:x3]
            if roi.size > 0:
                blur = _laplacian_var(roi)
                bright = _brightness(roi)
            else:
                blur = 0.0
                bright = 0.0

            ratio = fw / max(1, w)
            fx, fy = x + fw//2, y + fh//2

            dx = (fx - cx) / max(1, radius)
            dy = (fy - cy) / max(1, radius)

            # Checks
            too_small = ratio < MIN_FACE_RATIO
            too_big = ratio > MAX_FACE_RATIO
            too_dark = bright < BRIGHTNESS_MIN
            too_bright = bright > BRIGHTNESS_MAX
            out_of_sector = not _in_sector(dx, dy, target)

            # Mensagens de correção
            if too_small:
                status_lines.append("Aproxime-se da câmera")
            if too_big:
                status_lines.append("Afaste-se um pouco")
            if too_dark:
                status_lines.append("Melhore a iluminação")
            if too_bright:
                status_lines.append("Iluminação muito forte")
            if blur < MIN_BLUR_VAR:
                status_lines.append("Mantenha o rosto nítido (pare por um instante)")
            if out_of_sector:
                # Dicas de direção
                tx, ty = target
                if tx > 0: status_lines.append("Gire um pouco para a direita")
                if tx < 0: status_lines.append("Gire um pouco para a esquerda")
                if ty > 0: status_lines.append("Olhe um pouco para baixo")
                if ty < 0: status_lines.append("Olhe um pouco para cima")

            # Estabilidade (movimento reduzido)
            stability_queue.append((fx, fy))
            stable = False
            if len(stability_queue) == STABILITY_FRAMES:
                xs = [p[0] for p in stability_queue]
                ys = [p[1] for p in stability_queue]
                if (max(xs) - min(xs) <= MOVE_TOLERANCE) and (max(ys) - min(ys) <= MOVE_TOLERANCE):
                    stable = True

            # Pronto para capturar?
            ready_to_capture = not (too_small or too_big or too_dark or too_bright or out_of_sector or (blur < MIN_BLUR_VAR)) and stable
            ui_color = (50, 215, 50) if ready_to_capture else (0, 165, 255) if faces else (0,0,255)

            # Centro visual do setor alvo
            tgt_pt = (cx + int(target[0] * radius * 0.7), cy + int(target[1] * radius * 0.7))
            cv2.circle(frame, tgt_pt, 8, (200, 255, 200), -1)
            cv2.line(frame, (fx, fy), tgt_pt, (180, 180, 180), 1)

            # Captura automática quando pronto e com pequena pausa anti-tremor
            now = time.time()
            if ready_to_capture and (now - last_capture_time) > CAPTURE_DELAY_S:
                img_name = os.path.join(student_dir, f"{student_name}_{step_label.replace(' ','-')}_{int(now)}.png")
                # salvar NÃO espelhado (coerente com dataset)
                original = cv2.flip(frame, 1)
                cv2.imwrite(img_name, original)
                _play_beep()
                completed[current_idx] = True
                last_capture_time = now
                # Próximo passo
                if current_idx < len(STEPS) - 1:
                    current_idx += 1
                    stability_queue.clear()
                else:
                    # terminou
                    cv2.putText(frame, "Captura concluída!", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 215, 50), 3)
                    cv2.imshow('Captura Guiada', frame)
                    cv2.waitKey(750)
                    break

        else:
            status_lines.append("Rosto não detectado — alinhe ao círculo")

        # UI final: anel principal
        cv2.circle(frame, (cx, cy), radius, ui_color, 3)

        # Status lines
        y0 = h - 80
        for i, line in enumerate(status_lines[:3]):
            cv2.putText(frame, f"• {line}", (20, y0 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Rodapé com atalhos
        cv2.putText(frame, "[Q] Sair  [R] Recomeçar  [P] Pular passo", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

        cv2.imshow('Captura Guiada', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            current_idx = 0
            completed = [False] * len(STEPS)
            stability_queue.clear()
        elif key == ord('p'):
            # permitir pular em casos problemáticos
            completed[current_idx] = True
            if current_idx < len(STEPS) - 1:
                current_idx += 1
                stability_queue.clear()
            else:
                break

    cap.release()
    cv2.destroyAllWindows()

    if all(completed):
        print("\nCaptura guiada concluída com sucesso! Execute 'train_model.py' para treinar o modelo.")
    else:
        print("\nCaptura interrompida.")

if __name__ == "__main__":
    guided_capture()
