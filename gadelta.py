import datetime
import json
import pickle
import random
import threading
import time
import logging
import traceback

logger = logging.getLogger('myLogger')
logger.setLevel(logging.DEBUG)
fm = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch = logging.StreamHandler()
# ch.setFormatter(fm)
# ch.setLevel(logging.INFO)
# logger.addHandler(ch)
fh = logging.FileHandler('evolution-{}.log'.format(time.time()))
fh.setFormatter(fm)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


__author__ = 'Pedro'
import serial
import names


def saturate(min, value, max):
    if value < min:
        value = min
    if value > max:
        value = max
    return value


def serial_flush(ser):
    time.sleep(0.8)
    while ser.inWaiting():
        ser.flushInput()
        time.sleep(0.8)

def send_email(text, subject=""):
    import smtplib

    gmail_user = "grogecito@gmail.com"
    gmail_pwd = "dynamicsink61"
    FROM = 'grogecito@gmail.com'
    TO = ['ppayala@gmail.com'] #must be a list
    SUBJECT = subject
    TEXT = text

    # Prepare actual message
    message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        #server = smtplib.SMTP(SERVER)
        server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        #server.quit()
        server.close()
        logger.info('successfully sent the mail')
    except:
        logger.debug(traceback.format_exc())
        logger.warning("failed to send mail")


class MySerial(serial.Serial):
    encoding = 'utf-8'
    pc_to_3dr = None

    def __init__(self, port=None, baudrate=250000, timeout=60, logfile=None):
        if logfile:
            self.logfile = open(logfile, 'a')
        super().__init__(port, baudrate, timeout=timeout)

    def newline(self):
        if self.pc_to_3dr == True:
            self.logfile.write('\n{} PC -> 3DR '.format(datetime.datetime.now().strftime("%d%m%y %H:%M:%S.%f")[:-3]))
        elif self.pc_to_3dr == False:
            self.logfile.write('\n{} PC <- 3DR '.format(datetime.datetime.now().strftime("%d%m%y %H:%M:%S.%f")[:-3]))

    def logline(self, bytes, received_size, size, pc_to_3dr):
        """
        Logs a data line. 'bytes' is the content to log. 'received_size' is the real size received from the port.
        'size' is the number of bytes requested to read from the port. 'pc_to_3dr' indicates if the data to log
        is going from the pc to the 3dr or viceversa.
        """
        data = bytes.decode(self.encoding)
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
                if buf[-len(what):] == what:
                    return True
                if max_size is not None and len(buf) >= max_size:
                    break
            else:
                break
        return False


class DeltaComm(object):
    ser = None
    def __init__(self, serial_interface):
        self.ser = serial_interface

    def read_until(self, what):
        while True:
            ln = self.ser.readline().decode(encoding="utf-8")
            if ln == "":
                #Timeout
                self.ser.log_external("read_until->timeout")
                return False

            if what in ln:
                self.ser.log_external("read_until->'{}' found".format(what))
                return True
            self.ser.log_external("read_until->continue")

    def start(self):
        self.read_until("SD card ok")

    def get_machine_params(self, delta):
        data_found = 0
        self.ser.write(b'M503\n')
        while (data_found < 2):
            ln = self.ser.readline().decode(encoding='utf-8');
            if 'M666 X' in ln:
                #endstop offsets line
                delta.x_endstop = float(ln.split('X')[1].split(' ')[0])
                delta.y_endstop = float(ln.split('Y')[1].split(' ')[0])
                delta.z_endstop = float(ln.split('Z')[1].split(' ')[0])
                data_found += 1
            elif 'M666 A' in ln:
                #Delta geometry line
                delta.tower_a_angle_corr = float(ln.split('A')[1].split(' ')[0])
                delta.tower_b_angle_corr = float(ln.split('B')[1].split(' ')[0])
                delta.tower_c_angle_corr = float(ln.split('C')[1].split(' ')[0])
                delta.tower_a_radius_corr = float(ln.split('E')[1].split(' ')[0])
                delta.tower_b_radius_corr = float(ln.split('F')[1].split(' ')[0])
                delta.tower_c_radius_corr = float(ln.split('G')[1].split(' ')[0])
                delta.delta_radius = float(ln.split('R')[1].split(' ')[0])
                delta.diagonal_rod = float(ln.split('D')[1].split(' ')[0])
                delta.z_height = float(ln.split('H')[1].split(' ')[0])
                data_found += 1

        self.read_until("M301")


class Delta(object):
    name = ""

    _x_endstop = 0
    _y_endstop = 0
    _z_endstop = 0
    _tower_c_angle_corr = 0
    _tower_c_radius_corr = 0
    _delta_radius = 0
    _diagonal_rod = 0
    _z_height = 0

    reference_delta = None  # If this points to another delta, some values will be checked against it.
    interface = None  # Serial interface to use
    dc = None

    score = 99e99
    z_height_error = 99e99

    _crossable_props = ['x_endstop', 'y_endstop', 'z_endstop', 'tower_c_angle_corr', 'tower_c_radius_corr',
                           'delta_radius', 'diagonal_rod', 'z_height']
    _soft_mutable_props = ['']

    MUTATION_PROBABILITY = 0.05
    SOFT_MUTATION_PROBABILITY = 0.4

    MIN_ENDSTOP_VALUE = -10  # mm
    MAX_TOWER_ANGLE_CORR = 5  # degrees
    MAX_TOWER_RADIUS_CORR = 5  # mm
    MAX_RADIUS_DEV = 10  # percent deviation respect to reference_delta's value
    MAX_ROD_DEV = 0.5  # percent deviation respect to reference_delta's value
    MAX_HEIGHT_DEV = 5  # percent deviation respect to reference_delta's value

    @property
    def x_endstop(self):
        return self._x_endstop

    @x_endstop.setter
    def x_endstop(self, value):
        self._x_endstop = saturate(self.MIN_ENDSTOP_VALUE, value, 0)

    @property
    def y_endstop(self):
        return self._y_endstop

    @y_endstop.setter
    def y_endstop(self, value):
        self._y_endstop = saturate(self.MIN_ENDSTOP_VALUE, value, 0)

    @property
    def z_endstop(self):
        return self._z_endstop

    @z_endstop.setter
    def z_endstop(self, value):
        self._z_endstop = saturate(self.MIN_ENDSTOP_VALUE, value, 0)

    @property
    def tower_c_angle_corr(self):
        return self._tower_c_angle_corr

    @tower_c_angle_corr.setter
    def tower_c_angle_corr(self, value):
        self._tower_c_angle_corr = saturate(-self.MAX_TOWER_ANGLE_CORR, value, self.MAX_TOWER_ANGLE_CORR)

    @property
    def tower_c_radius_corr(self):
        return self._tower_c_radius_corr

    @tower_c_radius_corr.setter
    def tower_c_radius_corr(self, value):
        self._tower_c_radius_corr = saturate(-self.MAX_TOWER_RADIUS_CORR, value, self.MAX_TOWER_RADIUS_CORR)

    @property
    def delta_radius(self):
        return self._delta_radius

    @delta_radius.setter
    def delta_radius(self, value):
        if self.reference_delta is not None:
            dev = self.MAX_RADIUS_DEV / 100
            ref = self.reference_delta.delta_radius
            value = saturate((1 - dev) * ref,
                             value,
                             (1 + dev) * ref)

        self._delta_radius = value

    @property
    def diagonal_rod(self):
        return self._diagonal_rod

    @diagonal_rod.setter
    def diagonal_rod(self, value):
        if self.reference_delta is not None:
            dev = self.MAX_ROD_DEV / 100
            ref = self.reference_delta.diagonal_rod
            value = saturate((1 - dev) * ref,
                             value,
                             (1 + dev) * ref)

        self._diagonal_rod = value

    @property
    def z_height(self):
        return self._z_height

    @z_height.setter
    def z_height(self, value):
        if self.reference_delta is not None:
            dev = self.MAX_HEIGHT_DEV / 100
            ref = self.reference_delta.z_height
            value = saturate((1 - dev) * ref,
                             value,
                             (1 + dev) * ref)

        self._z_height = value

    def __init__(self, interface, deltacomm, reference_delta=None):
        # A new delta is born, isn't it cute!
        self.interface = interface
        self.dc = deltacomm
        self.reference_delta = reference_delta

        # Let's give him some personality
        self.name = names.get_first_name()
        if self.reference_delta is not None:
            #In the beginning, all are mutants
            self.mutate()

    def __repr__(self):
        return "Delta {} [x:{:.2f} y:{:.2f} z:{:.2f} C_ang:{:.2f} C_rad:{:.2f} dt_rad:{:.2f} " \
                "rod:{:.2f} z_height:{:.2f}".format(self.name,
                                                    self.x_endstop,
                                                    self.y_endstop,
                                                    self.z_endstop,
                                                    self.tower_c_angle_corr,
                                                    self.tower_c_radius_corr,
                                                    self.delta_radius,
                                                    self.diagonal_rod,
                                                    self.z_height)

    def _apply_adn(self):
        string = 'M666 X{:.3f} Y{:.3f} Z{:.3f} A0 B0 C{:.3f} I0 J0 K{:.3f} R{:.3f} D{:.3f} H{:.3f}\n'.format(self.x_endstop,
                                                                             self.y_endstop,
                                                                             self.z_endstop,
                                                                             self.tower_c_angle_corr,
                                                                             self.tower_c_radius_corr,
                                                                             self.delta_radius,
                                                                             self.diagonal_rod,
                                                                             self.z_height)
        self.interface.write(string.encode('utf-8'))

    def evaluate(self):

        if self.score < 99e98 and self.z_height_error < 99e98:
            logger.info("skipping {}'s test, already tested.".format(self))
            return
        data_found = 0
        quadratic_error = 0

        zmin = 99e99
        zmax = -99e99



        self._apply_adn()

        self.interface.write(b'G30 T\n')
        #if not self.dc.read_until("Probing bed"):
        #    self.interface.write(b'G30 T\n')

        # obtain quadratic error
        tokens = ['x:', 'y:', 'z:', 'c:', 'ox:', 'oy:', 'oz:']
        while data_found < 7:
            ln = self.interface.readline().decode(encoding='utf-8')
            if ln == "":
                self._apply_adn()
                self.interface.write(b'G30 T\n')
            for token in tokens:
                if token in ln:
                    z = float(ln.split(token)[1])

                    if z > zmax:
                        zmax = z
                    if z < zmin:
                        zmin = z

                    quadratic_error += z ** 2
                    data_found += 1

                    break

        #serial_flush(self.interface)
        self.dc.read_until("test done")
        time.sleep(0.5)

        # add parameter regularization
        # regulariz_params = [(self.x_endstop, 100),
        #                     (self.y_endstop, 100),
        #                     (self.z_endstop, 100),
        #                     (self.tower_c_angle_corr, 10),
        #                     (self.tower_c_radius_corr, 10)]
        #
        # for param, factor in regulariz_params:
        #     quadratic_error += (param ** 2) * factor

        self.z_height_error = zmax - zmin
        self.score = self.z_height_error #quadratic_error
        logger.info("Evaluating {} Score:{:.2f} z height error:{:.4f}".format(self, self.score, self.z_height_error))

    def crossover(self, couple):
        #return the sibling resulting from crossing this delta with the given couple

        parents = [self, couple]
        crossing = random.randint(1, len(self._crossable_props))
        if random.uniform(0,1) > 0.5:
            parents = list(reversed(parents))

        ref_delta = self.reference_delta or self  #If it is the reference delta, then pass itself as ref delta to sibling

        sibling = Delta(self.interface, self.dc, ref_delta)

        if random.uniform(0, 1) < self.MUTATION_PROBABILITY:
            #add some mutation every now and then (do nothing, by default they are mutant)
            logger.info("Mutating {}".format(sibling))
        else:
            #if not a mutant, get parents properties
            for i, prop in enumerate(self._crossable_props):
                if i == crossing:
                    parents = list(reversed(parents))

                value = getattr(parents[0], prop)

                setattr(sibling, prop, value)

            # There is also a soft mutation mechanism which only changes one parameter slightly
            if random.uniform(0, 1) < self.SOFT_MUTATION_PROBABILITY:
                sibling.soft_mutation()
                logging.info("Soft-mutating {}".format(sibling))

        return sibling

    def mutate(self):
        self.x_endstop = random.uniform(self.MIN_ENDSTOP_VALUE, 0)
        self.y_endstop = random.uniform(self.MIN_ENDSTOP_VALUE, 0)
        self.z_endstop = random.uniform(self.MIN_ENDSTOP_VALUE, 0)
        self.tower_c_angle_corr = random.uniform(-self.MAX_TOWER_ANGLE_CORR, self.MAX_TOWER_ANGLE_CORR)
        self.tower_c_radius_corr = random.uniform(-self.MAX_TOWER_RADIUS_CORR, self.MAX_TOWER_RADIUS_CORR)
        self.delta_radius = self.reference_delta.delta_radius * random.uniform(1 - self.MAX_RADIUS_DEV / 100,
                                                                               1 + self.MAX_RADIUS_DEV / 100)
        self.diagonal_rod = self.reference_delta.diagonal_rod * random.uniform(1 - self.MAX_ROD_DEV / 100,
                                                                               1 + self.MAX_ROD_DEV / 100)
        self.z_height = self.reference_delta.z_height * random.uniform(1 - self.MAX_HEIGHT_DEV / 100,
                                                                       1 + self.MAX_HEIGHT_DEV / 100)

    def soft_mutation(self):
        mutate_params = list(self._crossable_props)

        #choose random params to soft mutate
        for i in range(random.randrange(1, len(mutate_params) + 1)):
            pr = mutate_params.pop(random.randrange(len(mutate_params)))

            cur_val = getattr(self, pr)
            cur_val += random.uniform(-0.2, 0.2)
            setattr(self, pr, cur_val)

    def clone(self):
        clone = Delta(self.interface, self.dc, self.reference_delta)
        for prop in self._crossable_props:
            setattr(clone, prop, getattr(self, prop))
        return clone

    @classmethod
    def from_json(cls, json_data, interface, dc, reference_delta=None):
        dt = Delta(interface, dc, reference_delta)
        dt.delta_radius = json_data['_delta_radius']
        dt.diagonal_rod = json_data['_diagonal_rod']
        dt.name = json_data['name']
        dt.tower_c_angle_corr = json_data['_tower_c_angle_corr']
        dt.tower_c_radius_corr = json_data['_tower_c_radius_corr']
        dt.x_endstop = json_data['_x_endstop']
        dt.y_endstop = json_data['_y_endstop']
        dt.z_endstop = json_data['_z_endstop']
        dt.z_height = json_data['_z_height']

        if 'score' in json_data:
            dt.score = json_data['score']
        if 'z_height_error' in json_data:
            dt.z_height_error = json_data['z_height_error']
        return dt


class DeltaOptimizer(object):
    reference_delta = None
    opt_state = 'start'

    delta_swarm = []

    POPULATION = 12
    MAX_POPULATION = 100
    MIN_POPULATION = 9
    SOFT_MUTATION_PROBABILITY = Delta.SOFT_MUTATION_PROBABILITY

    SEND_MAIL_EVERY_X_GEN = 1

    target = 0.05   # z-height variation target

    data_file = 'delta_world.json'

    def __init__(self, serial_object=None):
        if serial_object is None:
            serial_object = MySerial('COM3', 250000, timeout=60,
                                     logfile='serial_{}.log'.format(datetime.datetime.now().strftime("%d%m%y_%H-%M-%S")))
        self.ser = serial_object
        self.dc = DeltaComm(self.ser)

        try:
            self.load_generation()
            logger.info("Found previous delta world, loading contents")
        except:
            logger.warn("Could not find previous delta world, generating a new one")
            self.reference_delta = Delta(self.ser, self.dc)

            #Start by getting machine parameters for the reference delta
            time.sleep(2)
            self.find_machine_params()
            #The reference delta is usually a pretty good start also
            self.reference_delta.name = "Pedrito"
            self.delta_swarm.append(self.reference_delta)
            for i in range(self.POPULATION - 1 ):
                self.delta_swarm.append(Delta(self.ser, self.dc, self.reference_delta))

        self.dc.start()

    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def from_JSON(self, json_dump):
        js_me = json.loads(json_dump)
        self.reference_delta = Delta(self.ser, self.dc)

        self.reference_delta = Delta.from_json(js_me['reference_delta'], self.ser, self.dc)
        for dtjs in js_me['delta_swarm']:
            self.delta_swarm.append(Delta.from_json(dtjs, self.ser, self.dc, self.reference_delta))
        self.POPULATION = js_me.get('POPULATION', self.POPULATION)
        self.SOFT_MUTATION_PROBABILITY = js_me.get('SOFT_MUTATION_PROBABILITY', self.SOFT_MUTATION_PROBABILITY)

    def save_generation(self):
        logging.info("Saving current generation to delta world file")
        f = open(self.data_file, 'wb')
        f.write(self.to_JSON().encode())
        f.close()
        logging.info("generation saved")

    def load_generation(self):
        logging.info("Loading generation from delta world file")
        f = open(self.data_file, 'rb')
        self.from_JSON(f.read().decode())
        f.close()
        logging.info("Generation loaded")

    def find_machine_params(self):
        self.dc.get_machine_params(self.reference_delta)

    def run_optimization(self):
        optimized = False
        generation = 0
        last_notified_gen_error = 99e99
        gen_error = 99e99
        while not optimized:
            if self.opt_state == 'start':

                self.opt_state = 'selection'
            elif self.opt_state == 'selection':
                logger.info("Evaluating generation {}".format(generation))
                for delta in self.delta_swarm:
                    delta.evaluate()
                    time.sleep(0.1)

                self.delta_swarm = sorted(self.delta_swarm, key=lambda delta: delta.score)

                for delta in self.delta_swarm:
                    if delta.z_height_error < self.target:
                        logger.info("The chosen one has been found!! no need to continue searching")
                        logger.info("The chosen one is:")
                        logger.info(delta)

                self.opt_state = 'crossover'
            elif self.opt_state == 'crossover':
                next_generation = []
                #The three best individuals continue
                next_generation.extend(self.delta_swarm[0:3])
                #And the best one is cloned and soft mutated three times
                for i in range(3):
                    sc = self.delta_swarm[0].clone()
                    sc.soft_mutation()
                    next_generation.append(sc)

                i = 0
                while len(next_generation) < self.POPULATION:
                    i += 1
                    for n in range(i):
                        if len(next_generation) >= self.POPULATION:
                            break

                        nr = random.randrange(len(self.delta_swarm))
                        while nr == n:
                            #Don't crossover with himself
                            nr = random.randrange(len(self.delta_swarm))

                        couple = self.delta_swarm[nr]
                        sibling = self.delta_swarm[n].crossover(couple)
                        next_generation.append(sibling)
                        logger.info("Crossing {}(#{}) with {}(#{}), {} was born".format(self.delta_swarm[n].name,
                                                                                  n,
                                                                                  couple.name,
                                                                                  nr,
                                                                                  sibling.name))

                for rocker in self.delta_swarm[0:3]:
                    #Lets do some recognition to the veterans
                    if 'old' not in rocker.name:
                        rocker.name = 'old ' + rocker.name
                prev_gen_error = gen_error
                gen_error = self.delta_swarm[0].z_height_error
                if prev_gen_error <= gen_error:
                    #If there is no improvement, increment the population to increase diversity
                    if self.POPULATION < self.MAX_POPULATION:
                        self.POPULATION += 1

                        #And raise soft mutation probability
                        if self.SOFT_MUTATION_PROBABILITY < 1:
                            self.SOFT_MUTATION_PROBABILITY *= 1.02


                        logger.info("Increasing population to {}, and soft mutation probability to {}".format(
                            self.POPULATION,
                            self.SOFT_MUTATION_PROBABILITY
                        ))

                else:
                    #If there is improvement, reduce population to increase evolution speed
                    if self.POPULATION > self.MIN_POPULATION:
                        self.POPULATION -= 1
                        logger.info("Decreasing population to {}, and soft mutation probability to {}".format(
                            self.POPULATION,
                            Delta.SOFT_MUTATION_PROBABILITY
                        ))

                        #And decrease soft mutation probability
                        self.SOFT_MUTATION_PROBABILITY /= 1.02

                Delta.SOFT_MUTATION_PROBABILITY = self.SOFT_MUTATION_PROBABILITY

                logger.info("GENERATION ERROR: {}".format(gen_error))
                if last_notified_gen_error - gen_error > 0.1 or generation % self.SEND_MAIL_EVERY_X_GEN == 0:
                    txt = "Generation {}: {} achieved a z-error of {}".format(generation,
                                                                              self.delta_swarm[0].name,
                                                                              gen_error)
                    send_email(txt, "GADelta evolution update")
                    last_notified_gen_error = gen_error
                generation += 1
                self.delta_swarm = next_generation

                self.save_generation()
                self.opt_state = 'selection'


if __name__ == "__main__":
    dtop = DeltaOptimizer()
    dtop.run_optimization()
    # ser = serial.Serial('COM3', 250000, timeout=30)
    # while True:
    #     if ser.inWaiting():
    #         ln = ser.readline();
    #         print(ln.decode(encoding='utf-8'))
    #
    #         dtop.find_machine_params(ln);



