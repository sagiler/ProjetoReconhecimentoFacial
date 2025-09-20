import os
import json
import time
from typing import Optional

# Opcional: pip install paho-mqtt
try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

BROKER_HOST = os.environ.get("MQTT_HOST", "localhost")
BROKER_PORT = int(os.environ.get("MQTT_PORT", "1883"))
TOPIC_PRESENCE = os.environ.get("MQTT_TOPIC", "classroom/attendance")
CLIENT_ID = os.environ.get("MQTT_CLIENT_ID", "face-attendance-client")

client: Optional[mqtt.Client] = None

def setup_mqtt():
    global client
    if mqtt is None:
        print("paho-mqtt não instalado. Execute: pip install paho-mqtt")
        return False
    client = mqtt.Client(client_id=CLIENT_ID, clean_session=True)
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()
    print(f"MQTT conectado em {BROKER_HOST}:{BROKER_PORT}, publicando em {TOPIC_PRESENCE}")
    return True


def on_attendance_marked(name: str, when: Optional[float] = None, recognized: bool = True):
    """
    Publique um evento de presença. Chame na hora de registrar presença no DB.
    """
    if client is None:
        return
    payload = {
        "event": "attendance_marked",
        "name": name,
        "timestamp": int((when or time.time())*1000),
        "recognized": recognized,
    }
    client.publish(TOPIC_PRESENCE, json.dumps(payload), qos=1, retain=False)


def on_student_present(name: str, count_visible: int, total_present: int):
    """
    Publique um heartbeat de presença visual atual.
    """
    if client is None:
        return
    payload = {
        "event": "student_present",
        "name": name,
        "visible_now": count_visible,
        "total_present_today": total_present,
        "timestamp": int(time.time()*1000),
    }
    client.publish(TOPIC_PRESENCE, json.dumps(payload), qos=0, retain=False)


if __name__ == "__main__":
    if setup_mqtt():
        # Simulação
        on_student_present("Alice", 1, 1)
        on_attendance_marked("Alice")
        time.sleep(1)
        on_student_present("Bob", 2, 2)
        on_attendance_marked("Bob")
        time.sleep(1)
        print("Exemplo MQTT finalizado.")
