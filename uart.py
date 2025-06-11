import serial
from serial.tools import list_ports
import modules.log
import time

logger = modules.log.get_logger()


class UART_CON:

  def __init__(self):
    while True:
      available_ports = list(list_ports.comports())
      if not available_ports:
        logger.warning("UART device not found")
      for port in available_ports:
        logger.debug(f"Port: {port.device}, Description: {port.description}")

      Serial_Port_Id = available_ports[0].device
      logger.debug(f"Using {available_ports[0].device}")
      self.Serial_Port = serial.Serial(Serial_Port_Id, 9600, timeout=None)
      break

  def init_connection(self):
    self.Serial_Port.write("[RASPI] READY?".encode("utf-8"))
    logger.debug("SEND RASPI READY?")
    current_time = time.time()
    while True:
      if time.time() - current_time > 1:
        self.Serial_Port.write("[RASPI] READY?".encode("utf-8"))
        logger.debug("ESP32 not giving respond, SEND RASPI READY?")
        current_time = time.time()
      if self.Serial_Port.in_waiting > 0:
        str = self.Serial_Port.read_until()
        logger.debug(f"Received {str} from ESP32");
        if str == "[ESP32] READY":
          logger.debug("ESP32 READY!")
          self.Serial_Port.write("[RASPI] READY CONFIRMED".encode("utf-8"))
          logger.debug("RASPI SENT CONFIRMED")
