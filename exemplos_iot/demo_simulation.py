"""Demo simulation: importa os callbacks dos exemplos e simula um cenário de aula.

Roda em modo dry-run (não necessita hardware). Imprime os payloads que seriam enviados
para MQTT/Serial/HTTP. Use para demonstrações em sala.

Usage:
    python demo_simulation.py
"""
import time
import os
import sys
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Ensure local exemplos_iot folder is importable when running the script directly
this_dir = Path(__file__).resolve().parent
if str(this_dir) not in sys.path:
    sys.path.insert(0, str(this_dir))

import mqtt_attendance as mqtt_attendance
import serial_door_access as serial_door_access
import http_lighting as http_lighting


def simulate_class_session(students: List[str], verbose: bool = False) -> List[Dict[str, Any]]:
    print("[demo] Iniciando simulação de sessão de aula")
    # Setup (tentativas silenciosas)
    try:
        mqtt_ok = mqtt_attendance.setup_mqtt()
    except Exception as e:
        mqtt_ok = False
        if verbose:
            print("[demo] setup_mqtt falhou (silencioso):", e)
    try:
        serial_ok = serial_door_access.setup_serial()
    except Exception as e:
        serial_ok = False
        if verbose:
            print("[demo] setup_serial falhou (silencioso):", e)

    visible_now = 0
    total_present = 0
    records: List[Dict[str, Any]] = []

    for name in students:
        # Simula a pessoa ficando visível
        visible_now += 1
        if verbose:
            print(f"[demo] {name} entrou na sala (visíveis agora: {visible_now})")

        # notificar presença visual (MQTT heartbeat)
        mqtt_attendance.on_student_present(name, visible_now, total_present)
        records.append({
            "type": "student_present",
            "name": name,
            "visible_now": visible_now,
            "total_present": total_present,
            "timestamp": int(time.time()*1000),
        })

        # decide marcar presença e enviar eventos
        time.sleep(0.5)
        total_present += 1
        mqtt_attendance.on_attendance_marked(name)
        records.append({
            "type": "attendance_marked",
            "name": name,
            "timestamp": int(time.time()*1000),
            "recognized": True,
        })

        serial_door_access.on_attendance_marked(name)
        records.append({
            "type": "serial_cmd",
            "cmd": f"GRANT:{name}\\n",
            "timestamp": int(time.time()*1000),
        })

        http_lighting.on_visible_count_change(visible_now)
        records.append({
            "type": "http_light",
            "visible_now": visible_now,
            "brightness": 80 + min(visible_now, 10) * 17,
            "timestamp": int(time.time()*1000),
        })

        time.sleep(0.4)

    # Simula saídas
    for _ in range(2):
        visible_now -= 1
        if verbose:
            print(f"[demo] uma pessoa saiu (visíveis agora: {visible_now})")
        http_lighting.on_visible_count_change(max(0, visible_now))
        records.append({
            "type": "http_light",
            "visible_now": max(0, visible_now),
            "brightness": 80 + min(max(0, visible_now), 10) * 17,
            "timestamp": int(time.time()*1000),
        })
        time.sleep(0.3)

    if verbose:
        print("[demo] Simulação finalizada.")
    return records


def parse_args():
    p = argparse.ArgumentParser(description="Demo simulation for IoT examples")
    p.add_argument("--verbose", action="store_true", help="print verbose logs")
    p.add_argument("--out-file", default="demo_output.json", help="file to write JSON records")
    p.add_argument("--students", nargs="*", default=["Alice", "Bob", "Carol"], help="list of student names to simulate")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Forçar URL de HTTP para httpbin para demo segura (se não especificado)
    os.environ.setdefault('IOT_LIGHT_URL', 'https://httpbin.org/post')
    # Evitar tentativa de conectar MQTT a localhost se não existir
    os.environ.setdefault('MQTT_HOST', 'broker.hivemq.com')
    records = simulate_class_session(args.students, verbose=args.verbose)
    # Salvar output
    out_path = Path(args.out_file)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    if args.verbose:
        print(f"[demo] gravado {len(records)} registros em {out_path}")
