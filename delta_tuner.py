__author__ = 'Pedro'

import smoothie_serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mp
from matplotlib.colors import Normalize
from ga_optimizer import GaOptimizable
import time

from mpl_toolkits.mplot3d import Axes3D


class DeltaTuner(smoothie_serial.SmoothieSerial):
    sensor_offset = (0, 0, 0)#(0, -22.5, 0)
    last_bed_probing = None

    def get_delta_params(self):
        """
            Returns the current params of the machine
        :return: A DeltaParams instance with the parameters currently in use
        """

        self.get_machine_params()
        dp = DeltaParams()
        dp.angle_offset_t1 = self.tower_a_angle_corr
        dp.angle_offset_t2 = self.tower_b_angle_corr
        dp.angle_offset_t3 = self.tower_c_angle_corr
        dp.radius_offset_t1 = self.tower_a_radius_corr
        dp.radius_offset_t2 = self.tower_b_radius_corr
        dp.radius_offset_t3 = self.tower_c_radius_corr
        dp.radius = self.delta_radius
        dp.rodlen_t1 = self.diagonal_rod
        dp.rodlen_t2 = self.diagonal_rod
        dp.rodlen_t3 = self.diagonal_rod
        dp.x_endstop = self.x_endstop
        dp.y_endstop = self.y_endstop
        dp.z_endstop = self.z_endstop

        return dp

    def set_delta_params(self, delta_params):
        """
            Writes a set of params to the machine
        :param delta_params: Instance of a DeltaParams object with the params to write
        :return:
        """
        dp = delta_params
        self.set_machine_params(
            r=dp.radius,
            a=dp.angle_offset_t1,
            b=dp.angle_offset_t2,
            c=dp.angle_offset_t3,
            d=dp.radius_offset_t1,
            e=dp.radius_offset_t2,
            h=dp.radius_offset_t3,
            l=(dp.rodlen_t1 + dp.rodlen_t2 + dp.rodlen_t3)/3
            )

    def probe_bed(self, max_radius=50, num_points_per_circle=6, num_circles=2):
        """
        Probes the bed in a series of concentric circular patterns.
        :param max_radius: Max radius of the probing circle
        :param num_points_per_circle: Number of angular divisions of each circle
        :param num_circles: Number of radial divisions to probe.
        :return: A matrix of [x, y, z] lines with each probe point.
        """
        probe_points = [[0, 0]]
        for r in np.linspace(max_radius / num_circles, max_radius, num_circles):
            for ang in np.linspace(0, 2 * np.pi * (num_points_per_circle - 1) / num_points_per_circle,
                                   num_points_per_circle):
                ang += np.pi / 2  # Start at 90º
                probe_points.append([r * np.cos(ang), r * np.sin(ang)])

        first = True
        for i, point in enumerate(probe_points):
            home = False
            if first:
                home = True
                first = False
            zp = self.probe_with_offset(point[0], point[1], home=home)
            probe_points[i].append(zp)

        self.last_bed_probing = probe_points
        return self.last_bed_probing

    def probe_with_offset(self, x, y, z=5, **kwargs):
        """
            Probes the bed compensating the offset between the z-probe and the nozzle
        :param x: x offset from nozzle to z-probe
        :param y: y offset from nozzle to z-probe
        :param z: z offset from nozzle to z-probe
        :param kwargs: kwargs to self.probe
        :return:
        """
        return self.probe(x - self.sensor_offset[0], y - self.sensor_offset[1], z - self.sensor_offset[2], **kwargs)

    def center_height(self):
        return self.last_bed_probing[0][2]

    def fit_probing_to_plane(self):
        """
            Probes the bed and does a least square fit of the point cloud to a plane.
        :return: The fitted plane
        """
        if self.last_bed_probing is None:
            self.probe_bed()

        XYZ = np.array(self.last_bed_probing).T

        # Plane initial guess, plane equation Ax + By + Cz = D
        p0 = [1, 0.5, 0.7, 0.2]

        def f_min(X, p):
            plane_xyz = p[0:3]
            distance = (plane_xyz*X.T).sum(axis=1) + p[3]
            return distance / np.linalg.norm(plane_xyz)

        def residuals(params, signal, X):
            return f_min(X, params)

        from scipy.optimize import leastsq
        sol = leastsq(residuals, p0, args=(None, XYZ))[0]
        print("Error: {}".format((f_min(XYZ, sol)**2).sum()))
        return sol

    def plot_probe_result(self, fig_axis=None, normalize_to_plane=True):
        if self.last_bed_probing is None:
            self.probe_bed()
        if fig_axis is None:
            fig_axis = plt.gca()
        pps = np.asmatrix(self.last_bed_probing)

        #Normalize the probings respect to the center point
        zdiff = self.center_height()

        x_sc = np.array(pps[:, 0])
        y_sc = np.array(pps[:, 1])
        z_sc = np.array(pps[:, 2]) - zdiff

        if normalize_to_plane:
            #Normalize to the fitted plane
            fplane = self.fit_probing_to_plane()
            # xx, yy = np.meshgrid(np.arange(-50, 50, 10),
            #                      np.arange(-50, 50, 10))
            # zfit = (-fplane[0]*xx -fplane[1]*yy -fplane[3])/fplane[2] - self.center_height()
            z_sc -= (-fplane[0]*x_sc -fplane[1]*y_sc -fplane[3])/fplane[2] -self.center_height()

        fig_axis.scatter(x_sc, y_sc, z_sc, c=z_sc, cmap=cm.seismic, norm=Normalize(vmin=-0.5, vmax=0.5))
        fig_axis.set_xlabel('X axis')
        fig_axis.set_ylabel('Y axis')
        fig_axis.set_zlabel('Z axis')

    def plot_fitted_plane(self):
        fig_axis = plt.gca()
        fplane = self.fit_probing_to_plane()
        xx, yy = np.meshgrid(np.arange(-50, 50, 10),
                             np.arange(-50, 50, 10))
        zfit = (-fplane[0]*xx -fplane[1]*yy -fplane[3])/fplane[2] - self.center_height()
        fig_axis.plot_surface(xx, yy, zfit, cmap=cm.seismic, norm=Normalize(vmin=-0.5, vmax=0.5))

class DeltaParams(object):
    radius = None
    radius_lims = (50, 100)

    rodlen_t1 = 150.85
    rodlen_t2 = 150.85
    rodlen_t3 = 150.85
    rodlen_lims = (140, 160)

    radius_offset_t1 = None
    radius_offset_t2 = None
    radius_offset_t3 = None
    radius_offset_lims = (-2, 2)

    angle_offset_t1 = None
    angle_offset_t2 = None
    angle_offset_t3 = None
    angle_offset_lims = (-4, 4)

    x_endstop = None
    y_endstop = None
    z_endstop = None
    endstop_lims = (-5, 5)

    ser_props = [
        "radius",
        #"rodlen_t1", "rodlen_t2", "rodlen_t3",
        "radius_offset_t1", "radius_offset_t2", "radius_offset_t3",
        "angle_offset_t1","angle_offset_t2", "angle_offset_t3",
        "x_endstop", "y_endstop", "z_endstop"
        ]

    prop_lims = [radius_lims] + [radius_offset_lims]*3 + [angle_offset_lims]*3 + [endstop_lims]*3

    def serialize(self):
        """
            Returns a list of the delta params
        :return:
        """
        retval = []

        for prop in self.ser_props:
            retval.append(getattr(self, prop))

        return retval

    def deserialize(self, ser_params):
        """
            Populates the object with the given serialized parameters
        :param ser_params: Input serialized parameters
        :return:
        """

        for i, prop in enumerate(self.ser_props):
            setattr(self, prop, ser_params[i])

    def get_param_lims(self):
        """
            Gives the range allowed for each parameter.
        :return: an array with tuples (min, max) representing the min and max allowable
            values for each parameter
        """
        return self.prop_lims


class DeltaSimulator(object):
    firmware_params = None
    real_params = None

    def __init__(self, firmware_params, real_params):
        self.firmware_params = firmware_params
        self.real_params = real_params

    def calculate_tower_positions(self, delta_params):
        t1x = (delta_params.radius - delta_params.radius_offset_t1) * np.cos(
            (210 + delta_params.angle_offset_t1) * np.pi / 180)
        t1y = (delta_params.radius - delta_params.radius_offset_t1) * np.sin(
            (210 + delta_params.angle_offset_t1) * np.pi / 180)
        t2x = (delta_params.radius - delta_params.radius_offset_t2) * np.cos(
            (330 + delta_params.angle_offset_t2) * np.pi / 180)
        t2y = (delta_params.radius - delta_params.radius_offset_t2) * np.sin(
            (330 + delta_params.angle_offset_t2) * np.pi / 180)
        t3x = (delta_params.radius - delta_params.radius_offset_t3) * np.cos(
            (90 + delta_params.angle_offset_t3) * np.pi / 180)
        t3y = (delta_params.radius - delta_params.radius_offset_t3) * np.sin(
            (90 + delta_params.angle_offset_t3) * np.pi / 180)

        return [t1x, t1y, t2x, t2y, t3x, t3y]

    def cartesian2delta(self, delta_params, x, y, z):
        [t1x, t1y, t2x, t2y, t3x, t3y] = self.calculate_tower_positions(delta_params)

        dx1 = t1x - x
        dy1 = t1y - y
        dx2 = t2x - x
        dy2 = t2y - y
        dx3 = t3x - x
        dy3 = t3y - y

        delta = [np.sqrt(delta_params.rodlen_t1 ** 2 - dx1 ** 2 - dy1 ** 2) + delta_params.x_endstop,
                 np.sqrt(delta_params.rodlen_t2 ** 2 - dx2 ** 2 - dy2 ** 2) + delta_params.y_endstop,
                 np.sqrt(delta_params.rodlen_t3 ** 2 - dx3 ** 2 - dy3 ** 2) + delta_params.z_endstop, ]

        return delta

    def delta2cartesian(self, delta_params, t1z, t2z, t3z):
        [t1x, t1y, t2x, t2y, t3x, t3y] = self.calculate_tower_positions(delta_params)

        t1z = t1z + delta_params.x_endstop
        t2z = t2z + delta_params.y_endstop
        t3z = t3z + delta_params.z_endstop

        # From Wikipedia (Trilateration)
        p1 = np.array([t1x, t1y, t1z])
        p2 = np.array([t2x, t2y, t2z])
        p3 = np.array([t3x, t3y, t3z])

        # From Wikipedia
        # Transform to get circle 1 at origin
        # Transform to get circle 2 on x axis
        ex = (p2 - p1) / np.linalg.norm(p2 - p1)
        i = np.dot(ex, p3 - p1)
        ey = (p3 - p1 - i * ex) / np.linalg.norm(p3 - p1 - i * ex)
        ez = np.cross(ex, ey)
        d = np.linalg.norm(p2 - p1)
        j = np.dot(ey, p3 - p1)

        # From Wikipedia
        # plug and chug using above values
        x = (delta_params.rodlen_t1 ** 2 - delta_params.rodlen_t2 ** 2 + d ** 2) / (2 * d)
        y = (delta_params.rodlen_t1 ** 2 - delta_params.rodlen_t3 ** 2 + i ** 2 + j ** 2) / (2 * j) - i * x / j

        # Only one case shown here
        z = -np.sqrt(delta_params.rodlen_t1 ** 2 - x ** 2 - y ** 2)

        # tri_pt is an array with ECEF x,y,z of trilateration point
        tri_pt = p1 + x * ex + y * ey + z * ez

        return tri_pt

    def calc_z_errors_at_pos(self, pos, probe_pos=(0,0)):
        """
            Calculates the z-errors at a given set of positions
        :param pos: list of [x, y, z] lists with given x and y positions for probings. z is
        ignored by this method.
        :param probe_pos: optional tuple to indicate the offset of the probe and compensate for it
        :return: list of [x, y, z] lists filled with the calculated z parameter
        """
        points = []
        for p in pos:
            points.append([p[0], p[1], 0])
        for n, p in enumerate(points):
            t = self.cartesian2delta(self.firmware_params, p[0], p[1], 0)
            r = self.delta2cartesian(self.real_params, t[0], t[1], t[2])
            points[n][2] = float(np.real(r[2]))
        return points

    def calc_z_errors(self):
        bed_diam = 117
        br = bed_diam/2
        points_per_mm = 0.5
        yy, xx = np.meshgrid(np.arange(-br, br, 1/points_per_mm),
                             np.arange(-br, br, 1/points_per_mm))
        base = np.sqrt( np.power(yy, 2) + np.power(xx, 2)) <= br
        plane = base * 1.0

        for x in range(len(xx[:,1])):
            for y in range(len(yy[1,:])):
                if base[x, y] == 0:
                    plane[x, y] = np.nan
                else:
                    t = self.cartesian2delta(self.firmware_params, xx[x, 0], yy[0, y], 0)
                    r = self.delta2cartesian(self.real_params, t[0], t[1], t[2])
                    plane[x, y] = np.real(r[2])

        return [xx, yy, plane]

    def plot_z_errors(self, fig_axis=None):
        if fig_axis is None:
            fig_axis = plt.gca()

        [xx, yy, plane] = self.calc_z_errors()

        #Normalize the plane respect to the center point
        plane -= plane[int(len(plane)/2), int(len(plane)/2)]

        surf = fig_axis.plot_surface(xx, yy, plane, rstride=1, cstride=1, cmap=cm.seismic, linewidth=0, norm=Normalize(vmin=-0.5, vmax=0.5))
        fig_axis.set_xlabel('X axis')
        fig_axis.set_ylabel('Y axis')
        fig_axis.set_zlabel('Z axis')


class Delta(GaOptimizable):
    tuner = None
    fittest = None

    real_delta = []

    def __init__(self, generation):
        super(Delta, self).__init__(generation)
        self.delta_params = DeltaParams()
        if Delta.tuner is None:
            Delta.tuner = DeltaTuner('COM10')
        try:
            Delta.real_delta[self.get_reference_delta()]
        except IndexError:

            try:
                ref_params = Delta.fittest.delta_params
            except AttributeError:
                ref_params = Delta.tuner.get_delta_params()

            Delta.tuner.set_delta_params(ref_params)
            time.sleep(1)
            probings = Delta.tuner.probe_bed()

            zcount = 0
            zsum = 0
            for probing in probings:
                zcount += 1
                zsum += probing[2]

            zsum /= zcount

            for i, probing in enumerate(probings):
                probings[i][2] -= zsum

            Delta.real_delta.append({
                'params': ref_params,
                'probing': probings
            })
        self.sim = DeltaSimulator(Delta.real_delta[self.get_reference_delta()]['params'], self.delta_params)

    def get_reference_delta(self):
        #return self.generation
        return 0

    def get_adn_limits(self):
        return self.delta_params.get_param_lims()

    def get_adn(self):
        return self.delta_params.serialize()

    def set_adn(self, adn):
        self.delta_params.deserialize(adn)
        self.sim.real_params = self.delta_params

    def rate_fitness(self):
        """
            Rates the fitness of the delta. The method used to rate the fitness is to calculate the
            least squares error between the simulated parameters and the class' real_delta probing
            results of the current generation. In this way, what is finally found is the real
            physical parameters of the machine.
        :return:
        """

        probings = Delta.real_delta[self.get_reference_delta()]['probing']
        sim_probings = self.sim.calc_z_errors_at_pos(probings)

        error = 0
        for i, probing in enumerate(probings):
            error += (probings[i][2] - sim_probings[i][2])**2

        return 1/error

    def from_json(cls, js_data):
        raise NotImplementedError()

    def __repr__(self):
        return "Delta {} [x:{:.2f} y:{:.2f} z:{:.2f} A_ang:{:.2f} B_ang:{:.2f} C_ang:{:.2f} " \
               "A_rad:{:.2f} B_rad:{:.2f} C_rad:{:.2f} dt_rad:{:.2f} " \
               "A_rod:{:.2f} B_rod:{:.2f} C_rod:{:.2f} ".format(
            self.name,
            self.delta_params.x_endstop,
            self.delta_params.y_endstop,
            self.delta_params.z_endstop,
            self.delta_params.angle_offset_t1,
            self.delta_params.angle_offset_t2,
            self.delta_params.angle_offset_t3,
            self.delta_params.radius_offset_t1,
            self.delta_params.radius_offset_t2,
            self.delta_params.radius_offset_t3,
            self.delta_params.radius,
            self.delta_params.rodlen_t1,
            self.delta_params.rodlen_t2,
            self.delta_params.rodlen_t3,
            )


if __name__ == "__main__":

    firm = DeltaParams()
    firm.radius = 66.8
    firm.rodlen_t2 = firm.rodlen_t3 = firm.rodlen_t1 = 151.24
    firm.radius_offset_t1 = -1.31
    firm.radius_offset_t2 = -0.96
    firm.radius_offset_t3 = 2.58
    firm.angle_offset_t1 = -1.6
    firm.angle_offset_t2 = 2.1
    firm.angle_offset_t3 = 0

    real = DeltaParams()
    real.radius  = firm.radius
    real.rodlen_t1 = real.rodlen_t2 = real.rodlen_t3 = firm.rodlen_t1
    t3off = 0
    t2off = 0
    real.radius_offset_t1 = firm.radius_offset_t1 - t2off - t3off
    real.radius_offset_t2 = firm.radius_offset_t2 + t2off
    real.radius_offset_t3 = firm.radius_offset_t3 + t3off
    real.angle_offset_t1 = firm.angle_offset_t1 - 0.5
    real.angle_offset_t2 = firm.angle_offset_t2
    real.angle_offset_t3 = firm.angle_offset_t3

    ds = DeltaSimulator(firm, real)
    dt = DeltaTuner('COM10')

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    #plt.ion()

    while(1):
        ds.plot_z_errors()
        dt.plot_probe_result(normalize_to_plane=True)
        #dt.plot_fitted_plane()
        plt.draw()
        plt.pause(0.001)
        break

    plt.show()
