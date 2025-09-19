# Reconhecimento Facial para Presença em Sala (IoT)

![Build](https://github.com/klebr55/ProjetoReconhecimentoFacial/actions/workflows/ci.yml/badge.svg)

Aplicação em Python que usa OpenCV, DeepFace e um modelo ONNX (SFace) acelerado por GPU (DirectML) para reconhecer rostos de alunos e marcar presença em um banco SQLite. A captura é feita via webcam e a interface exibe:

- Na sala agora: quantidade de alunos reconhecidos visíveis no momento
- Total presentes: total de alunos que já tiveram presença registrada no dia

O projeto inclui scripts para:
- Configurar o banco (`database_setup.py`)
- Capturar imagens com webcam (`capture_images.py`) e captura guiada estilo FaceID (`capture_images_guided.py`)
- Treinar embeddings faciais a partir do dataset (`train_model.py`) — usa SFace ONNX por padrão (GPU via DirectML) com fallback para DeepFace/Facenet
- Rodar a aplicação principal de detecção e reconhecimento (`main.py`)

## Requisitos

- Windows 10/11
- Python 3.11+ (testado com 3.13 em venv)
- Webcam funcional
- GPU NVIDIA/AMD/Intel compatível com DirectML (opcional, recomendado)

## Dependências (pip)

Instaladas via `requirements.txt`:

- opencv-python
- deepface
- numpy
- Pillow
- playsound==1.2.2
- imutils
- requests
- onnxruntime-directml (aceleração por DirectML no Windows)

Obs:
- O DeepFace traz dependências opcionais como TensorFlow, keras, mtcnn, retina-face, mediapipe, pandas, etc. Todas são instaladas automaticamente pelo pip.
- onnxruntime-directml habilita o provedor `DmlExecutionProvider` (GPU) no Windows, sem exigir CUDA.

## Estrutura do Projeto

```
ProjetoReconhecimentoFacial/
  database_setup.py
  train_model.py
  main.py
  capture_images.py
  capture_images_guided.py
  requirements.txt
  README.md
  .gitignore
  dataset/              # imagens por aluno (criado por você)
  sounds/               # opcional: beep.wav
  models/               # baixado automaticamente (SFace ONNX)
```

## Instalação

Recomenda-se usar ambiente virtual. No PowerShell (Windows):

```powershell
# Na pasta do projeto
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se já estiver usando um venv existente, adapte o caminho do Python conforme seu ambiente.

Para GPU (Windows): o projeto instala `onnxruntime-directml`. Na primeira execução, validamos os providers e, se disponível, usaremos `DmlExecutionProvider` automaticamente. Se ausente, caímos para CPU.

## Passo a passo de uso

1. Configurar o banco:
```powershell
python database_setup.py
```

2. Capturar imagens do(s) aluno(s):
- Captura rápida:
```powershell
python capture_images.py
```
- Captura guiada estilo FaceID:
```powershell
python capture_images_guided.py
```
As imagens serão salvas em `dataset/NOME_ALUNO/`.

### Captura Guiada (UX estilo Face ID)

O `capture_images_guided.py` guia o usuário por 9 poses (centro, direita, esquerda, cima, baixo e diagonais), com:
- Círculo guia e anel de progresso.
- Dicas contextuais (aproxime/afaste, iluminação, direção do olhar).
- Validação de qualidade: tamanho do rosto, foco (Laplacian), brilho, estabilidade e direção.
- Renderização de textos com acentos via Pillow (melhor legibilidade em PT-BR).
- Detectores com fallback: mediapipe → mtcnn → opencv (se mediapipe não estiver disponível no seu Python, o script funciona com os demais).
- Captura automática quando as condições estão OK (beep de confirmação).

Modos e atalhos:
- Modo de sensibilidade: EASY (padrão), BALANCED, STRICT — teclas E/B/S.
- Ajustes finos ao vivo:
  - DiagThr: [ / ]
  - AxisThr: , / .
  - Tolerância angular: (diagonal) 1/2, (eixo) 5/6
  - Magnitude mínima: (diagonal) 3/4, (eixo) 7/8
  - Pular passo: P, Recomeçar: R, Sair: Q

Observações sobre MediaPipe:
- Em algumas combinações (por exemplo, Python 3.13/Windows), `mediapipe` pode não ter wheel disponível. O script lida com isso e segue com MTCNN/OpenCV.
- Se quiser head-pose mais robusto (pitch/yaw) e sua versão suportar, instale: `pip install mediapipe`.

3. Treinar embeddings:
```powershell
python train_model.py
```

Ao treinar, o script fará download automático do modelo SFace ONNX (OpenCV Zoo) em `models/face_recognition_sface_2021dec.onnx` e usará a GPU via DirectML quando disponível. Os embeddings serão gravados em `database.db`.

4. Executar a aplicação principal:
```powershell
python main.py
```

- Pressione `q` para sair.
- A tela mostra "Na sala agora", "Total presentes" e rótulos nos rostos.
- Também exibimos FPS e o provider ativo (EP: DmlExecutionProvider/CPU) para facilitar diagnóstico.

## Dicas para melhor reconhecimento

- Iluminação frontal e uniforme; evite reflexos fortes nos óculos.
- Use a captura guiada para cobrir centro, esquerda, direita, cima e baixo.
- Garanta 15–30 imagens por aluno com ângulos variados.

Se usar óculos com reflexo, prefira a captura guiada e ângulos com menos glare. 

## Performance e Tuning

O `main.py` foi otimizado para reduzir lag:
- Detecção apenas a cada N frames (padrão 3) com cache das caixas entre frames.
- Processamento em largura-alvo (padrão 640 px) para aliviar CPU.
- Reconhecimento com SFace ONNX (GPU via DirectML) e similaridade vetorizada.
- Beep assíncrono para não travar a UI.

Parâmetros no topo do `main.py` que você pode ajustar:
- `DETECT_EVERY_N`: 2–5. Valores menores detectam com mais frequência, mas usam mais CPU.
- `TARGET_WIDTH`: 480–800. Resoluções maiores melhoram detecção, mas aumentam custo.
- `SIM_THRESHOLD`: 0.60–0.70. Baixar facilita reconhecer; subir reduz falsos positivos.
- Detectores: primário `mediapipe` com fallback `mtcnn` → `opencv` quando necessário.

## Solução de problemas

- "Nenhum rosto detectado": aproxime-se, centralize o rosto, melhore a luz. Hoje a ordem padrão é: mediapipe → mtcnn → opencv (com fallback automático).
- Diagonais para cima difíceis de acionar na captura guiada: use Modo EASY (tecla E) e, se necessário, aumente a tolerância angular (2) e/ou reduza a magnitude mínima (3). O HUD mostra os valores atuais.
- Lento na sua máquina: podemos fixar o detector primário para `mediapipe` e usar `opencv` como fallback. 
- Erro com TensorFlow: atualize pip (`python -m pip install -U pip`) e reinstale `deepface`. Caso veja `tf-keras` ausente, instale com `pip install tf-keras`.

### Sintomas conhecidos (estado atual)
- FPS mostrado alto mas sensação de engasgo: isso ocorre porque detectamos apenas a cada N frames, e alguns decodificadores de webcam variam o pacing. Dicas:
  - Diminua `DETECT_EVERY_N` para 2.
  - Aumente `TARGET_WIDTH` para 800 se a CPU permitir.
  - Garanta que outra aplicação não esteja usando a webcam ao mesmo tempo.
- Rosto detectado mas “Na sala agora” = 0: ajuste `SIM_THRESHOLD` (ex.: 0.60) e/ou gere mais fotos de treino; verifique se o nome do diretório do dataset corresponde ao nome esperado.

### Roadmap curto
- Opcional: tracker (KCF/CSRT) entre detecções para suavizar movimento.
- Cache de embeddings por face enquanto o bounding box muda pouco.
- Botão/tecla para alternar detectores e níveis de qualidade ao vivo.

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
