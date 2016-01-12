__author__ = 'Pedro'

import serial
import datetime
import time

class MySerial(serial.Serial):
    encoding = 'utf-8'
    pc_to_3dr = None

    def __init__(self, port=None, baudrate=250000, timeout=60, logfile=None):
        if logfile is None:
            logfile='./logs/serial_{}.log'.format(datetime.datetime.now().strftime("%d%m%y_%H-%M-%S"))

        self.logfile = open(logfile, 'a')
        super().__init__(port, baudrate, timeout=timeout)

    def newline(self):
        d = ""
        if self.pc_to_3dr:
            d = "PC -> 3DR"
        elif not self.pc_to_3dr:
            d = "PC <- 3DR"

        self.logfile.write('\n{} (rx:{}) {} '.format(
            datetime.datetime.now().strftime("%d%m%y %H:%M:%S.%f")[:-3],
            self.inWaiting(),
            d
        ))

    def logline(self, data, received_size, size, pc_to_3dr):
        """
        Logs a data line. 'bytes' is the content to log. 'received_size' is the real size received from the port.
        'size' is the number of bytes requested to read from the port. 'pc_to_3dr' indicates if the data to log
        is going from the pc to the 3dr or viceversa.
        """
        if isinstance(data, bytes):
            data = data.decode(self.encoding, errors='ignore')
        data = data.split('\n')
        #If last access was in the other direction, change and newline
        if self.pc_to_3dr != pc_to_3dr:
            self.pc_to_3dr = pc_to_3dr
            self.newline()
        for i, line in enumerate(data):
            self.logfile.write(line)
            if i < len(data) - 1:
                #If there is more lines, newline
                self.newline()

        #If received less bytes than requested, print timeout
        if received_size != size:
            self.logfile.write("[TIMEOUT]")

        self.logfile.flush()

    def log_external(self, msg):
        self.logfile.write("[[[" + msg + "]]]")
        self.logfile.flush()

    def read(self, size=1):
        bt = super().read(size)
        received_size = len(bt)
        self.logline(bt, received_size, size, False)

        #finally, return read value
        return bt

    def write(self, data):
        if isinstance(data, str):
            data = bytes(data, self.encoding)

        self.logline(data, len(data), len(data), True)
        return super().write(data)

    def wait_until(self, what, max_size=None, timeout=None):
        """ Reads the port until the string 'what' is found, timeout, or max_size bytes have been read.
          Returns True if 'what' was found."""
        buf = bytearray()
        while True:
            c = self.read(1)
            if c:
                buf += c
                if buf[-len(what):] == bytes(what, 'utf-8'):
                    return True
                if max_size is not None and len(buf) >= max_size:
                    break
            else:
                break
        return False

    def wait_until_ack(self):
        self.wait_until('ok\r\n')

