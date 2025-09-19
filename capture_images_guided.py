import cv2
import os
import time
import math
from collections import deque
import numpy as np
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # reduce TF log noise
from deepface import DeepFace
from playsound import playsound
from PIL import Image, ImageDraw, ImageFont
try:
    import mediapipe as mp  # optional
except Exception:
    mp = None

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
SECTOR_THRESHOLD = 0.22 # limiar relativo para eixos (esq/dir/cima/baixo)
DIAG_THRESHOLD = 0.15   # limiar mais permissivo para diagonais
# valores ajustáveis em tempo real
g_axis_thr = SECTOR_THRESHOLD
g_diag_thr = DIAG_THRESHOLD
g_move_thr = 0.06  # limiar de movimento (normalizado pelo raio) para diagonais
g_ang_axis_deg = 40.0   # tolerância angular (graus) para eixos
g_ang_diag_deg = 60.0   # tolerância angular (graus) para diagonais
g_min_mag_axis = 0.10   # magnitude mínima (|v|) normalizada pelo raio
g_min_mag_diag = 0.14   # magnitude mínima para diagonais
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

DETECT_BACKENDS_ORDER = ["mediapipe", "mtcnn", "opencv"]

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'segoeui.ttf'),
        os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'arial.ttf'),
        os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'calibri.ttf'),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

def _draw_texts_pil(frame_bgr, texts):
    # texts: list of dicts: {text, x, y, size, color_bgr, outline}
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for t in texts:
        txt = t.get('text', '')
        x = t.get('x', 0)
        y = t.get('y', 0)
        size = t.get('size', 24)
        color = t.get('color', (255, 255, 255))
        outline = t.get('outline', True)
        font = _get_font(size)
        # Outline for readability
        if outline:
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    if ox == 0 and oy == 0:
                        continue
                    draw.text((x+ox, y+oy), txt, font=font, fill=(0, 0, 0))
        # Main text
        draw.text((x, y), txt, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
    # back to BGR
    frame_bgr[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _detect_face_with_fallback(rgb_frame):
    last_det = None
    for det in DETECT_BACKENDS_ORDER:
        try:
            faces = DeepFace.extract_faces(rgb_frame, detector_backend=det, enforce_detection=False) or []
            last_det = det
            if faces:
                return faces, det
        except Exception:
            last_det = det
            continue
    return [], last_det

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

def _in_direction(dx: float, dy: float, target) -> bool:
    tx, ty = target
    # Centro: ficar próximo do centro
    thr = g_diag_thr if (tx != 0 and ty != 0) else g_axis_thr
    if tx == 0 and ty == 0:
        return abs(dx) < thr/2 and abs(dy) < thr/2
    # Vetor atual normalizado e vetor alvo
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return False
    vx, vy = dx / mag, dy / mag
    tt_mag = math.hypot(tx, ty)
    txn, tyn = (tx/tt_mag, ty/tt_mag) if tt_mag > 0 else (0.0, 0.0)
    dot = max(-1.0, min(1.0, vx*txn + vy*tyn))
    # tolerância angular e magnitude mínima
    if tx != 0 and ty != 0:
        ang_tol = math.radians(g_ang_diag_deg)
        min_mag = g_min_mag_diag
    else:
        ang_tol = math.radians(g_ang_axis_deg)
        min_mag = g_min_mag_axis
    return (mag >= min_mag) and (math.acos(dot) <= ang_tol)

def guided_capture():
    global g_axis_thr, g_diag_thr, g_ang_axis_deg, g_ang_diag_deg, g_min_mag_axis, g_min_mag_diag
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
    current_profile = 'EASY'

    # Ajuda inicial
    print("\nIniciando captura guiada estilo Face ID.\nDicas: mantenha o rosto centralizado, boa iluminação e mova a cabeça conforme instruções.")

    face_mesh = None
    if mp is not None:
        try:
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception:
            face_mesh = None

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                break

            # Espelhar para UX natural (apenas a visualização)
            frame = cv2.flip(raw_frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detecção de rosto com fallback
            faces, used_detector = _detect_face_with_fallback(rgb)

            # Head pose via Face Mesh (pitch/yaw)
            pitch_deg = None
            yaw_deg = None
            mesh_res = None
            if face_mesh is not None:
                try:
                    mesh_res = face_mesh.process(rgb)
                except Exception:
                    mesh_res = None
            if mesh_res and mesh_res.multi_face_landmarks:
                lms = mesh_res.multi_face_landmarks[0].landmark
                # pontos 2D
                lm_ids = {
                    'nose': 1,
                    'chin': 152,
                    'left_eye_outer': 263,
                    'right_eye_outer': 33,
                    'left_mouth': 291,
                    'right_mouth': 61,
                }
                pts_2d = []
                try:
                    for k in ['nose','chin','left_eye_outer','right_eye_outer','left_mouth','right_mouth']:
                        lm = lms[lm_ids[k]]
                        pts_2d.append([lm.x * w, lm.y * h])
                    image_points = np.array(pts_2d, dtype=np.float64)
                    # modelo 3D aproximado (mm)
                    model_points = np.array([
                        [0.0, 0.0, 0.0],          # nose
                        [0.0, -63.6, -12.5],      # chin
                        [-43.3, 32.7, -26.0],     # left eye outer
                        [43.3, 32.7, -26.0],      # right eye outer
                        [-28.9, -28.9, -24.1],    # left mouth
                        [28.9, -28.9, -24.1],     # right mouth
                    ], dtype=np.float64)
                    focal_length = w
                    camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float64)
                    dist_coeffs = np.zeros((4,1))
                    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                    if success:
                        R, _ = cv2.Rodrigues(rvec)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
                        pitch_deg = float(angles[0])
                        yaw_deg = float(angles[1])
                except Exception:
                    pitch_deg = None
                    yaw_deg = None

            ui_color = (0, 0, 255) # vermelho por padrão
            status_lines = []
            text_items = []

            # Ring e progresso
            _draw_ring_and_progress(frame, (cx, cy), radius, sum(completed), len(STEPS), (50, 215, 50))

            # Instrução ativa
            step_label, target = STEPS[current_idx]
            text_items.append({"text": f"Passo {current_idx+1}/{len(STEPS)}: {step_label}", "x": 20, "y": 20+20, "size": 36, "color": (255,255,255)})
            text_items.append({"text": "Siga o guia e mantenha-se estável.", "x": 20, "y": 70, "size": 22, "color": (230,230,230)})
            if pitch_deg is not None and yaw_deg is not None:
                text_items.append({"text": f"Pitch: {pitch_deg:.0f}°  Yaw: {yaw_deg:.0f}°", "x": w-280, "y": h-24, "size": 18, "color": (180,220,255)})
            # Mostrar thresholds atuais
            text_items.append({"text": f"Modo: {current_profile}", "x": 20, "y": h-24, "size": 18, "color": (200,220,200)})
            text_items.append({"text": f"DiagThr: {g_diag_thr:.2f}  AxisThr: {g_axis_thr:.2f}", "x": w-280, "y": 20, "size": 18, "color": (200,200,200)})
            text_items.append({"text": f"AngDiag: {g_ang_diag_deg:.0f}° AngAxis: {g_ang_axis_deg:.0f}°", "x": w-280, "y": 44, "size": 18, "color": (200,200,200)})
            text_items.append({"text": f"MinMagD: {g_min_mag_diag:.2f} MinMagA: {g_min_mag_axis:.2f}", "x": w-280, "y": 68, "size": 18, "color": (200,200,200)})

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
                out_of_sector = not _in_direction(dx, dy, target)

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
                    if tx != 0 and ty != 0:
                        status_lines.append("Dica: mova levemente a cabeça na diagonal indicada")

                # Estabilidade (movimento reduzido)
                stability_queue.append((fx, fy))
                stable = False
                if len(stability_queue) == STABILITY_FRAMES:
                    xs = [p[0] for p in stability_queue]
                    ys = [p[1] for p in stability_queue]
                    if (max(xs) - min(xs) <= MOVE_TOLERANCE) and (max(ys) - min(ys) <= MOVE_TOLERANCE):
                        stable = True

                # Pronto para capturar?
                # Permitir diagonais com mais tolerância via distância ao alvo (mais suave para 'cima-...')
                rx, ry = 0.7, 0.7
                if target[1] < 0:
                    ry = 0.5  # exigir menos deslocamento vertical para cima
                tgt_pt = (cx + int(target[0] * radius * rx), cy + int(target[1] * radius * ry))
                dist_to_target = math.hypot(fx - tgt_pt[0], fy - tgt_pt[1])
                near_target = dist_to_target < (radius * (0.65 if (target[0] != 0 and target[1] != 0 and target[1] < 0) else (0.58 if (target[0] != 0 and target[1] != 0) else 0.45)))

                # Condições de pitch para diagonais
                pitch_ok = True
                if target[1] < 0:   # cima
                    if pitch_deg is not None:
                        pitch_ok = (pitch_deg < -5.0)
                elif target[1] > 0: # baixo
                    if pitch_deg is not None:
                        pitch_ok = (pitch_deg > 5.0)

                out_block = (out_of_sector and not near_target) or (not pitch_ok)
                ready_to_capture = not (too_small or too_big or too_dark or too_bright or out_block or (blur < MIN_BLUR_VAR)) and stable
                ui_color = (50, 215, 50) if ready_to_capture else (0, 165, 255) if faces else (0,0,255)

                # Centro visual do setor alvo
                cv2.circle(frame, tgt_pt, 8, (200, 255, 200), -1)
                cv2.line(frame, (fx, fy), tgt_pt, (180, 180, 180), 1)

                # Captura automática quando pronto e com pequena pausa anti-tremor
                now = time.time()
                if ready_to_capture and (now - last_capture_time) > CAPTURE_DELAY_S:
                    img_name = os.path.join(student_dir, f"{student_name}_{step_label.replace(' ','-')}_{int(now)}.png")
                    # salvar frame limpo, não espelhado
                    cv2.imwrite(img_name, raw_frame)
                    _play_beep()
                    completed[current_idx] = True
                    last_capture_time = now
                    # Próximo passo
                    if current_idx < len(STEPS) - 1:
                        current_idx += 1
                        stability_queue.clear()
                    else:
                        # terminou
                        text_items.append({"text": "Captura concluída!", "x": 20, "y": h-40, "size": 30, "color": (50,215,50)})
                        cv2.imshow('Captura Guiada', frame)
                        cv2.waitKey(750)
                        break

            else:
                status_lines.append("Rosto não detectado — alinhe ao círculo")

            # UI final: anel principal
            cv2.circle(frame, (cx, cy), radius, ui_color, 3)

            # Status lines
            y0 = h - 90
            for i, line in enumerate(status_lines[:3]):
                text_items.append({"text": f"• {line}", "x": 20, "y": y0 + i*28, "size": 22, "color": (255,255,255)})

            # Rodapé e detector em uso
            text_items.append({"text": "[Q] Sair  [R] Recomeçar  [P] Pular  [E/B/S] Modo", "x": 20, "y": h - 48, "size": 20, "color": (220,220,220)})
            if 'used_detector' in locals() and used_detector:
                text_items.append({"text": f"Detector: {used_detector}", "x": w - 220, "y": 20, "size": 18, "color": (180,220,255)})

            # Desenhar textos com suporte a acentos
            _draw_texts_pil(frame, text_items)

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
            elif key == ord('['):
                g_diag_thr = max(0.05, g_diag_thr - 0.02)
            elif key == ord(']'):
                g_diag_thr = min(0.5, g_diag_thr + 0.02)
            elif key == ord(','):
                g_axis_thr = max(0.05, g_axis_thr - 0.02)
            elif key == ord('.'):
                g_axis_thr = min(0.5, g_axis_thr + 0.02)
            elif key in (ord('e'), ord('E')):
                current_profile = 'EASY'
                g_ang_axis_deg, g_ang_diag_deg = 55.0, 75.0
                g_min_mag_axis, g_min_mag_diag = 0.08, 0.10
            elif key in (ord('b'), ord('B')):
                current_profile = 'BALANCED'
                g_ang_axis_deg, g_ang_diag_deg = 40.0, 60.0
                g_min_mag_axis, g_min_mag_diag = 0.10, 0.14
            elif key in (ord('s'), ord('S')):
                current_profile = 'STRICT'
                g_ang_axis_deg, g_ang_diag_deg = 30.0, 45.0
                g_min_mag_axis, g_min_mag_diag = 0.14, 0.20
            elif key == ord('1'):
                g_ang_diag_deg = max(10.0, g_ang_diag_deg - 5.0)
            elif key == ord('2'):
                g_ang_diag_deg = min(80.0, g_ang_diag_deg + 5.0)
            elif key == ord('3'):
                g_min_mag_diag = max(0.05, g_min_mag_diag - 0.02)
            elif key == ord('4'):
                g_min_mag_diag = min(0.6, g_min_mag_diag + 0.02)
            elif key == ord('5'):
                g_ang_axis_deg = max(10.0, g_ang_axis_deg - 5.0)
            elif key == ord('6'):
                g_ang_axis_deg = min(80.0, g_ang_axis_deg + 5.0)
            elif key == ord('7'):
                g_min_mag_axis = max(0.04, g_min_mag_axis - 0.02)
            elif key == ord('8'):
                g_min_mag_axis = min(0.5, g_min_mag_axis + 0.02)
    finally:
        if face_mesh is not None:
            try:
                face_mesh.close()
            except Exception:
                pass

    cap.release()
    cv2.destroyAllWindows()

    if all(completed):
        print("\nCaptura guiada concluída com sucesso! Execute 'train_model.py' para treinar o modelo.")
    else:
        print("\nCaptura interrompida.")

if __name__ == "__main__":
    guided_capture()
