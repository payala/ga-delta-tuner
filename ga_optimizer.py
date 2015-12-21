__author__ = 'Pedro'
import datetime
import json
import random
import time
import logging
import traceback
import abc

logger = logging.getLogger('myLogger')
logger.setLevel(logging.DEBUG)
fm = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch = logging.StreamHandler()
# ch.setFormatter(fm)
# ch.setLevel(logging.INFO)
# logger.addHandler(ch)
fh = logging.FileHandler('./log/evolution-{}.log'.format(time.time()))
fh.setFormatter(fm)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


__author__ = 'Pedro'
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


class GaOptimizable(object):
    __metaclass__ = abc.ABCMeta

    rating = 0

    def __init__(self, generation):
        # A baby is born, isn't it cute!
        self.generation = generation
        # Let's give him some personality
        self.name = names.get_first_name()

    @abc.abstractmethod
    def get_adn_limits(self):
        """
            This method indicates the allowable limits for each gene
        :return: A list of tuples (min_val, max_val) for each gene. The order of the list elements is the same
        as used in the get_adn and set_adn methods.
        """

    @abc.abstractmethod
    def get_adn(self):
        """
            This function gets the adn of the current instance
        :return: List of parameters describing the genetic properties of the instance. Same order as passed to set_adn.
        """
        pass

    @abc.abstractmethod
    def set_adn(self, adn):
        """
            This function modifies the current instance with the parameters passed in the list 'adn'.
        :param adn: List of parameters to apply to the instance. Same order as returned by get_adn.
        :return: Nothing
        """
        pass

    @abc.abstractmethod
    def rate_fitness(self):
        """
            This function evaluates the individual giving a single score about his performane.
        :return: A number indicating the rating of the individual, higher is better.
        """

    @abc.abstractclassmethod
    def from_json(cls, js_data):
        """
            This class method returns an instance represented by a dict in js_data
        :param cls:
        :param js_data: a json loaded dict with the info of the instance to return.
        :return:
        """




class GaOptimizer(object):
    POPULATION = 12
    MAX_POPULATION = 100
    MIN_POPULATION = 9
    MUTATION_PROBABILITY = 0.01
    SOFT_MUTATION_PROBABILITY = 0.01

    SEND_MAIL_EVERY_X_GEN = 1

    DATA_FILE = 'delta_world.json'

    def __init__(self, ga_optimizable_class):
        self.fittest = None
        self.opt_state = 'start'

        self.swarm = []

        self.target = 0.05   # z-height variation target

        self.generation = 0
        try:
            self.load_generation()
            logger.info("Found previous delta world, loading contents")
        except:
            logger.warn("Could not find previous delta world, generating a new one")

            self.gao_class = ga_optimizable_class

            for i in range(self.POPULATION - 1 ):
                self.swarm.append(self.gao_class(self.generation))

        self.dc.start()

    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def from_JSON(self, json_dump):
        js_me = json.loads(json_dump)

        for dtjs in js_me['delta_swarm']:
            self.swarm.append(self.gao_class.from_json(dtjs))
        self.POPULATION = js_me.get('POPULATION', self.POPULATION)
        self.SOFT_MUTATION_PROBABILITY = js_me.get('SOFT_MUTATION_PROBABILITY', self.SOFT_MUTATION_PROBABILITY)

    def save_generation(self):
        logging.info("Saving current generation to delta world file")
        f = open(self.DATA_FILE, 'wb')
        f.write(self.to_JSON().encode())
        f.close()
        logging.info("generation saved")

    def load_generation(self):
        logging.info("Loading generation from delta world file")
        f = open(self.DATA_FILE, 'rb')
        self.from_JSON(f.read().decode())
        f.close()
        logging.info("Generation loaded")

    def crossover(self, father, mother):
        """
            Reproduces father with mother and returns a sibling
        :param father:
        :param mother:
        :return:
        """
        #return the sibling resulting from crossing this delta with the given couple

        parents = [father, mother]

        if random.uniform(0,1) > 0.5:
            # Switch positions with P=0.5
            parents = list(reversed(parents))

        sibling = self.gao_class(self.generation)
        adns = [parent.get_adn() for parent in parents]
        crossing_point = random.randint(1, len(adns[0]))
        sibling.set_adn([adns[0][:crossing_point] + adns[1][crossing_point:]])

        self.mutate(sibling, mutation_probability=self.MUTATION_PROBABILITY)

        self.soft_mutation(sibling, mutation_probability=self.SOFT_MUTATION_PROBABILITY)

        return sibling

    def mutate(self, indiv, mutation_probability=0.01, mutation_intensity=1):
        """
            Applies a mutation to an individual.
        :param indiv: Instance to mutate
        :param mutation_intensity: A number from 0 to 1 that indicates the maximum magnitude of the mutation.
        If the number is below 1, then the mutation is applied as a change respect to the current value.
        :return:
        """
        lims = indiv.get_adn_limits()
        dev = [(l[1]-l[0])*mutation_intensity for l in lims]

        adn = indiv.get_adn()
        for n, gene in enumerate(adn):
            mut_prob = random.uniform(0, 1)
            if mut_prob < mutation_probability:
                if mutation_intensity == 1:
                    gene = random.uniform(lims[n][0], lims[n][1])
                else
                    gene += random.uniform(-dev[n]/2, dev[n]/2)

        indiv.set_adn(adn)

    def soft_mutation(self, indiv, mutation_probability=0.01):
        return self.mutate(indiv, mutation_intensity=0.2)

    def run_optimization(self):
        optimized = False

        last_notified_gen_rating = 0
        gen_rating = 0
        while not optimized:
            if self.opt_state == 'start':
                self.opt_state = 'selection'
            elif self.opt_state == 'selection':
                logger.info("Evaluating generation {}".format(self.generation))
                for individual in self.swarm:
                    individual.rating = individual.rate_fitness()
                self.swarm = sorted(self.swarm, key=lambda individual: 1/individual.rating)

                for individual in self.swarm:
                    if individual.rating > self.target:
                        logger.info("The chosen one has been found!! no need to continue searching")
                        logger.info("The chosen one is:")
                        logger.info(individual)

                self.opt_state = 'crossover'
            elif self.opt_state == 'crossover':
                next_generation = []
                #The three best individuals continue
                next_generation.extend(self.swarm[0:3])
                #And the best one is cloned and soft mutated three times
                best_adn = self.swarm[0].get_adn()
                for i in range(3):
                    sc = self.gao_class(self.generation)
                    sc.set_adn(best_adn)
                    sc.soft_mutation()
                    next_generation.append(sc)

                i = 0
                while len(next_generation) < self.POPULATION:
                    i += 1
                    for n in range(i):
                        if len(next_generation) >= self.POPULATION:
                            break

                        nr = random.randrange(len(self.swarm))
                        while nr == n:
                            #Don't crossover with himself
                            nr = random.randrange(len(self.swarm))

                        couple = self.swarm[nr]
                        sibling = self.crossover(self.swarm[n], couple)
                        next_generation.append(sibling)
                        logger.info("Crossing {}(#{}) with {}(#{}), {} was born".format(self.swarm[n].name,
                                                                                  n,
                                                                                  couple.name,
                                                                                  nr,
                                                                                  sibling.name))

                for rocker in self.swarm[0:3]:
                    #Lets do some recognition to the veterans
                    if 'old' not in rocker.name:
                        rocker.name = 'old ' + rocker.name
                prev_gen_rating = gen_rating
                gen_rating = self.swarm[0].rating
                if prev_gen_rating >= gen_rating:
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

                logger.info("GENERATION RATING: {}".format(gen_rating))
                if last_notified_gen_rating - gen_rating < 0.1 or self.generation % self.SEND_MAIL_EVERY_X_GEN == 0:
                    txt = "Generation {}: {} achieved a rating of {}".format(self.generation,
                                                                              self.swarm[0].name,
                                                                              gen_rating)
                    send_email(txt, "GADelta evolution update")
                    last_notified_gen_rating = gen_rating
                self.generation += 1
                self.swarm = next_generation

                self.save_generation()
                self.opt_state = 'selection'


if __name__ == "__main__":
    dtop = GaOptimizer()
    dtop.run_optimization()
    # ser = serial.Serial('COM3', 250000, timeout=30)
    # while True:
    #     if ser.inWaiting():
    #         ln = ser.readline();
    #         print(ln.decode(encoding='utf-8'))
    #
    #         dtop.find_machine_params(ln);



class oldDelta(object):
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