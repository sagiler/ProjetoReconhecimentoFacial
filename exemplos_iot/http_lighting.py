import os
import time
import json
from typing import Optional
from urllib import request, error

BASE_URL = os.environ.get("IOT_LIGHT_URL", "http://192.168.1.50:8123/api/services/light/turn_on")
TOKEN = os.environ.get("IOT_LIGHT_TOKEN", "")  # se usar Home Assistant, gere um Long-Lived Access Token
ENTITY_ID = os.environ.get("IOT_LIGHT_ENTITY", "light.classroom")

# Estratégia: quando a contagem visível aumenta, ilumina mais; quando cai, reduz.

def call_service(brightness: int = 150, color_temp: Optional[int] = None):
    payload = {"entity_id": ENTITY_ID, "brightness": brightness}
    if color_temp is not None:
        payload["color_temp"] = color_temp
    data = json.dumps(payload).encode()
    req = request.Request(BASE_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except error.URLError as e:
        print("Erro HTTP:", e)
        return None, None


def on_visible_count_change(visible_now: int):
    brightness = 80 + min(visible_now, 10) * 17  # 80..250
    status, _ = call_service(brightness=brightness)
    if status:
        print(f"Luz ajustada. brightness={brightness} (visíveis={visible_now})")


if __name__ == "__main__":
    for n in [0, 1, 3, 5, 2]:
        on_visible_count_change(n)
        time.sleep(0.5)
    print("Exemplo HTTP finalizado.")
