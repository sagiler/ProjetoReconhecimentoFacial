# Exemplos IoT com Reconhecimento Facial (Sala de Aula)

Esta pasta traz três abordagens simples para integrar o reconhecimento facial do projeto com dispositivos/serviços IoT em sala de aula. Cada exemplo assume que o `main.py` ou uma lógica similar chame uma função de "callback" quando um aluno é reconhecido (nome) e/ou quando a presença é marcada.

## Requisitos opcionais

Alguns exemplos dependem de pacotes opcionais:

- MQTT: `paho-mqtt`
- Serial: `pyserial`

Instale com:

```
pip install paho-mqtt pyserial
```

## Padrão de callback

Cada exemplo expõe uma função `on_student_present(name: str)` e/ou `on_attendance_marked(name: str)` para ser chamada pelo seu loop principal. Você pode importar e acoplar no `main.py` ou rodar os exemplos de forma independente simulando chamadas (ver bloco `if __name__ == "__main__"`).

## Exemplos

1. `mqtt_attendance.py` — Publica em um broker MQTT sempre que um aluno é reconhecido/teve presença confirmada.
2. `serial_door_access.py` — Controla um microcontrolador via porta serial (ex.: Arduino/ESP32) para abrir uma trava ao reconhecer um aluno autorizado ou acionar um buzzer LED.
3. `http_lighting.py` — Faz requisições HTTP para um serviço de automação (ex.: ESPHome/Tasmota/Home Assistant) alterando iluminação conforme presença/densidade na sala.

Ajuste credenciais/endpoints no topo de cada arquivo.
