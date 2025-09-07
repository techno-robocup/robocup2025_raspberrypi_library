import serial
from serial.tools import list_ports
import modules.log
import modules.settings
import time

logger = modules.log.get_logger()


class Message:

  def __init__(self, *args):
    if len(args) == 2:
      self.id = args[0]
      self.message = args[1]
    elif len(args) == 1:
      id_str, message = args[0].split(" ", 1)
      self.id = int(id_str)
      self.message = message.strip()
    else:
      raise ValueError(
          "Message must be initialized with either (id, message) or (combined_string)"
      )

  def __str__(self):
    return f"{self.id} {self.message}\n"

  def getId(self):
    return self.id

  def getMessage(self):
    return self.message


class UART_CON:

  def __init__(self):
    self.Serial_Port = None
    self.connect_to_port()

  def connect_to_port(self):
    while True:
      try:
        available_ports = list(list_ports.comports())
        if not available_ports:
          logger.warning("UART device not found")
          time.sleep(1)
          continue

        for port in available_ports:
          logger.debug(f"Port: {port.device}, Description: {port.description}")
      except Exception as e:
        logger.error(f"Error: {e}")
        time.sleep(1)
        continue

      Serial_Port_Id = available_ports[0].device
      logger.debug(f"Using {available_ports[0].device}")
      self.Serial_Port = serial.Serial(Serial_Port_Id, 9600, timeout=None)
      break

  def send_message(self, message):
    if not self.Serial_Port or not self.Serial_Port.is_open:
      logger.error("Serial port not open")
      return False

    def _send():
      self.Serial_Port.write(str(message).encode("ascii"))
      return True

    timeout = 1
    result, error = modules.settings.timeout_function(_send, timeout=timeout)
    if error:
      if isinstance(error, TimeoutError):
        logger.error(f"Send message timed out after {timeout} seconds")
      else:
        logger.error(f"Serial communication error: {error}")
      return False
    return result

  def receive_message(self):
    if not self.Serial_Port or not self.Serial_Port.is_open:
      logger.error("Serial port not open")
      return False

    def _receive():
      message_str = self.Serial_Port.read_until(b'\n').decode('ascii').strip()
      return Message(message_str)

    timeout = 1
    result, error = modules.settings.timeout_function(_receive, timeout=timeout)
    if error:
      if isinstance(error, TimeoutError):
        logger.error(f"Receive message timed out after {timeout} seconds")
      else:
        logger.error(f"Serial communication error: {error}")
      return False
    return result

  def close(self):
    if self.Serial_Port and self.Serial_Port.is_open:
      self.Serial_Port.close()
      logger.debug("Serial port closed")
