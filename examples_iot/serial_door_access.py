import os
import time
from typing import Optional

# Opcional: pip install pyserial
try:
    import serial
except Exception:
    serial = None

SERIAL_PORT = os.environ.get("SERIAL_PORT", "COM3")
BAUDRATE = int(os.environ.get("SERIAL_BAUD", "115200"))
ALLOWED = set(name.strip() for name in os.environ.get("ALLOWED_STUDENTS", "Alice,Bob").split(","))

ser: Optional["serial.Serial"] = None

def setup_serial():
    global ser
    if serial is None:
        print("pyserial não instalado. Execute: pip install pyserial")
        return False
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"Serial OK em {SERIAL_PORT} @ {BAUDRATE}")
        return True
    except Exception as e:
        print("Falha ao abrir serial:", e)
        return False


def on_attendance_marked(name: str):
    """Ao reconhecer um aluno autorizado, envie comando para abrir trava/acionar LED."""
    if ser is None:
        return
    if name in ALLOWED:
        cmd = f"GRANT:{name}\n".encode()
        ser.write(cmd)
        print("Enviado:", cmd)
    else:
        cmd = f"DENY:{name}\n".encode()
        ser.write(cmd)
        print("Enviado:", cmd)


if __name__ == "__main__":
    if setup_serial():
        on_attendance_marked("Alice")
        time.sleep(1)
        on_attendance_marked("Eve")
        print("Exemplo serial finalizado.")
