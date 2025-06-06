import serial
import time


class UART_CON:

  def __init__(self):
    while True:
      available_ports = list(serial.tools.list_ports.comports())
      if not available_ports:
        print("Could not initialize ESP32")
      print("Available serial ports: ")
      for port in available_ports:
        print(f"Port: {port.device}, Description: {port.description}")

      Serial_Port_Id = available_ports[0].device
      self.Serial_Port = serial.Serial(Serial_Port_Id, 9600, timeout=None)

  def init_connection(self):
    self.Serial_Port.write("[RASPI] READY?")
    current_time = time.time()
    while True:
      if time.time() - current_time > 1:
        self.Serial_Port.write("[RASPI] READY?")
        current_time = time.time()
      if self.Serial_Port.in_waiting():
        str = self.Serial_Port.read_until()
        if str == "[ESP32] READY":
          self.Serial_Port.write("[RASPI] READY CONFIRMED")
