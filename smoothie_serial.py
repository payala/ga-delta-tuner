__author__ = 'Pedro'

import my_serial
import time

class SmoothieSerial(my_serial.MySerial):
    x_endstop = None
    y_endstop = None
    z_endstop = None

    tower_a_angle_corr = None
    tower_b_angle_corr = None
    tower_c_angle_corr = None

    tower_a_radius_corr = None
    tower_b_radius_corr = None
    tower_c_radius_corr = None

    delta_radius = None
    diagonal_rod = None
    z_height = None

    FAST_MOVE_FEEDRATE = 20000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_until(self, what):
        while True:
            ln = self.readline().decode(encoding="utf-8")
            if ln == "":
                #Timeout
                self.log_external("read_until->timeout")
                return None

            if what in ln:
                self.log_external("read_until->'{}' found".format(what))
                return ln
            self.log_external("read_until->continue")

    def start(self):
        self.read_until("Printer is now online.")

    def get_machine_params(self):
        data_found = 0
        self.write(b'M503\n')
        while self.inWaiting():
            ln = self.readline().decode()
            if ln is None:
                break
            if 'M666' in ln:
                #endstop offsets line
                if 'X' in ln:
                    self.x_endstop = float(ln.split('X')[1].split(' ')[0])
                if 'Y' in ln:
                    self.y_endstop = float(ln.split('Y')[1].split(' ')[0])
                if 'Z' in ln:
                    self.z_endstop = float(ln.split('Z')[1].split(' ')[0])
                data_found += 1
            elif 'M665' in ln:
                #Delta geometry line
                if 'A' in ln:
                    self.tower_a_angle_corr = float(ln.split('A')[1].split(' ')[0])
                if 'B' in ln:
                    self.tower_b_angle_corr = float(ln.split('B')[1].split(' ')[0])
                if 'C' in ln:
                    self.tower_c_angle_corr = float(ln.split('C')[1].split(' ')[0])
                if 'D' in ln:
                    self.tower_a_radius_corr = float(ln.split('D')[1].split(' ')[0])
                if 'E' in ln:
                    self.tower_b_radius_corr = float(ln.split('E')[1].split(' ')[0])
                if 'H' in ln:
                    self.tower_c_radius_corr = float(ln.split('H')[1].split(' ')[0])
                if 'R' in ln:
                    self.delta_radius = float(ln.split('R')[1].split(' ')[0])
                if 'L' in ln:
                    self.diagonal_rod = float(ln.split('L')[1].split(' ')[0])
                if 'Z' in ln:
                    self.z_height = float(ln.split('Z')[1].split(' ')[0])
                data_found += 1



    def home(self):
        self.write('G28\n')

    def go(self, x, y, z, f=None):
        if f is None:
            f = self.FAST_MOVE_FEEDRATE
        self.write('G1 X{} Y{} Z{} F{}\n'.format(x, y, z, f))

    def probe(self, x, y, z=10, home=True, timeout=10):
        if home:
            self.home()
        self.flushInput()
        self.go(x, y, z)
        self.write('G30\n')
        ln = self.read_until('Z:')
        z_val = float(ln.split('Z:')[1].split(' ')[0])
        return z_val

if __name__ == '__main__':
    ss = SmoothieSerial('COM10')
    ss.get_machine_params()
    ss.probe(0,0)