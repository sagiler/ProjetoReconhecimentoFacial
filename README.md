# Reconhecimento Facial para Presença em Sala (IoT)

![Build](https://github.com/klebr55/ProjetoReconhecimentoFacial/actions/workflows/ci.yml/badge.svg)

Aplicação em Python que usa OpenCV e DeepFace para reconhecer rostos de alunos e marcar presença em um banco SQLite. A captura é feita via webcam e a interface exibe:

- Na sala agora: quantidade de alunos reconhecidos visíveis no momento
- Total presentes: total de alunos que já tiveram presença registrada no dia

O projeto inclui scripts para:
- Configurar o banco (`database_setup.py`)
- Capturar imagens com webcam (`capture_images.py`) e captura guiada estilo FaceID (`capture_images_guided.py`)
- Treinar embeddings faciais a partir do dataset (`train_model.py`)
- Rodar a aplicação principal de detecção e reconhecimento (`main.py`)

## Requisitos

- Windows 10/11
- Python 3.11+ (testado com 3.13 em venv)
- Webcam funcional

## Dependências (pip)

Instaladas via `requirements.txt`:

- opencv-python
- deepface
- numpy
- Pillow
- playsound==1.2.2
- imutils

Obs:
- O DeepFace traz dependências opcionais como TensorFlow, keras, mtcnn, retina-face, mediapipe, pandas, etc. Todas são instaladas automaticamente pelo pip.
- Caso use `face_recognition`/`dlib`, seria necessário CMake e compilador C++; este projeto evita isso usando DeepFace.

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

3. Treinar embeddings:
```powershell
python train_model.py
```

4. Executar a aplicação principal:
```powershell
python main.py
```

- Pressione `q` para sair.
- A tela mostra "Na sala agora", "Total presentes" e rótulos nos rostos.

## Dicas para melhor reconhecimento

- Iluminação frontal e uniforme; evite reflexos fortes nos óculos.
- Use a captura guiada para cobrir centro, esquerda, direita, cima e baixo.
- Garanta 15–30 imagens por aluno com ângulos variados.

## Solução de problemas

- "Nenhum rosto detectado": aproxime-se, centralize o rosto, melhore a luz. O sistema tenta os detectores: retinaface → mediapipe → mtcnn → opencv.
- Lento na sua máquina: podemos fixar o detector primário para `mediapipe` e usar `opencv` como fallback. 
- Erro com TensorFlow: atualize pip (`python -m pip install -U pip`) e reinstale `deepface`. Caso veja `tf-keras` ausente, instale com `pip install tf-keras`.

## Publicando no GitHub

1. Inicialize o Git e faça o commit:
```powershell
git init
git add .
git commit -m "Projeto: reconhecimento facial de presença"
```

2. Crie o repositório no GitHub (via web) e copie a URL (HTTPS). Depois, conecte e envie:
```powershell
git remote add origin https://github.com/SEU_USUARIO/ProjetoReconhecimentoFacial.git
git branch -M main
git push -u origin main
```

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
