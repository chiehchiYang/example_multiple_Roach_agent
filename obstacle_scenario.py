#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk
    O            : set coordinate

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
from multiprocessing import Process
import json
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

from collections import deque

from bird_eye_view.BirdViewProducer import BirdViewProducer, BirdView
from bird_eye_view.Mask import PixelDimensions, Loc, BirdViewMasks

################


from threading import Thread
import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import time
import logging
import math
import random
import re
import weakref
from auto_random_actors import spawn_actor_nearby
import cv2
# from auto_light import *

import datetime

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from carla_gym.core.obs_manager.obs_manager_handler import ObsManagerHandler
from carla_gym.core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from carla_gym.utils import config_utils
from carla_gym.utils.traffic_light import TrafficLightHandler
from navigation.global_route_planner import GlobalRoutePlanner
from multiprocessing import Process, Pool
import copy

import torch.multiprocessing as mp
# import torch.threading as threading
import threading
import gym
from omegaconf import DictConfig, OmegaConf
from importlib import import_module
import wandb
def load_entry_point(name):
    mod_name, attr_name = name.split(":") #agents.rl_birdview.rl_birdview_agent:RlBirdviewAgent
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_o
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from QuinticPolynomialsPlanner.quintic_polynomials_planner import \
    QuinticPolynomial
from CubicSpline import cubic_spline_planner

SIM_LOOP = 500

# Parameter
MAX_SPEED = 30.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 7.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.4  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2  # robot radius [m]
# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0

show_animation = True


class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob):
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path


def generate_target_course(x, y):
    csp = cubic_spline_planner.CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True  # Enables synchronous mode
        self.world.apply_settings(settings)
        self.actor_role_name = args.rolename
        self.args = args
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
            
            
        # layer map remove ...    
        
        self.world.unload_map_layer(carla.MapLayer.Buildings)     
        self.world.unload_map_layer(carla.MapLayer.Decals)     
        self.world.unload_map_layer(carla.MapLayer.Foliage)     
        self.world.unload_map_layer(carla.MapLayer.Ground)     
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)         
        self.world.unload_map_layer(carla.MapLayer.Particles)     
        self.world.unload_map_layer(carla.MapLayer.Props)     
        self.world.unload_map_layer(carla.MapLayer.StreetLights)     
        self.world.unload_map_layer(carla.MapLayer.Walls)     

        self.hud = hud
        self.player = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]
    def get_yaw(self, current_pos, previous_point):
        yaw_vector = (current_pos - previous_point)

        vector = [1, 0]

        unit_vector_1 = yaw_vector / np.linalg.norm(yaw_vector)
        unit_vector_2 = vector / np.linalg.norm(vector)
        
        
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        
        angle = np.arccos(dot_product)
        import math
        
        if yaw_vector[1] < 0:
            yaw = -math.degrees(angle)
        else:
            yaw = math.degrees(angle)
        return yaw

    def restart(self):
        self.player_max_speed = 1.3 #1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        # cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        # cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint = self.world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
        print(blueprint)
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")


       # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()

            obstacle_route = np.load(f'./route/{self.args.town}/route00.npy')
            
            # print('Load' , obstacle_route)
            player_U_turn_spawn = carla.Transform(carla.Location(obstacle_route[0][0], obstacle_route[0][1]))
            player_U_turn_spawn.location.z += 2.0
            player_U_turn_spawn.rotation.roll = 0.0
            player_U_turn_spawn.rotation.pitch = 0.0
            

            player_U_turn_spawn.rotation.yaw = self.get_yaw(obstacle_route[1], obstacle_route[0])
            # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            spawn_point = player_U_turn_spawn
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        
        
        # self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        # self.camera_manager.transform_index = cam_pos_index
        # self.camera_manager.set_sensor(cam_index, notify=False)
        
        
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):

        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        # print(preset[1])
        while ('Night' in preset[1]):
            self._weather_index += -1 if reverse else 1
            self._weather_index %= len(self._weather_presets)
            preset = self._weather_presets[self._weather_index]
            # print(preset[1])
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    # def toggle_radar(self):
    #     if self.radar_sensor is None:
    #         self.radar_sensor = RadarSensor(self.player)
    #     elif self.radar_sensor.sensor is not None:
    #         self.radar_sensor.sensor.destroy()
    #         self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock, frame, display, stored_path):
        end = self.hud.tick(self, clock, self.camera_manager, frame, display, stored_path)
        return end
    def render(self, display):
        # self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        # self.camera_manager.sensor.destroy()
        # self.camera_manager.sensor = None
        # self.camera_manager.index = None
        pass

    def destroy(self):
        # if self.radar_sensor is not None:
        #     self.toggle_radar()

        
        
        sensors = [
            # self.camera_manager.sensor_rgb_front,
            # self.camera_manager.sensor_ss_front,


            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        #self._autopilot_enabled = start_in_autopilot
        self._autopilot_enabled = False
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    
                    # world.toggle_radar()
                    world.hud.record_flag = not world.hud.record_flag
                    if world.hud.record_flag:
                        world.hud.notification("start recording")
                    else:
                        world.hud.notification("End recording")
                    
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key==K_o:
                    xyz=[float(s) for s in input('Enter coordinate: x , y , z  : ').split()]
                    new_location=carla.Location(xyz[0],xyz[1],xyz[2])
                    world.player.set_location(new_location)
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constanft_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    pass
                    #world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height, distance=25.0, town='Town05', stored_path='', v_id=1):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.recording = False
        self.recording_frame = 0
        self.stored_path = stored_path
        self.v_id = int(v_id)
        self.ego_data = {}

        self.d_2_intersection = distance
        self.d_last = distance
        self.jam = 0
        
        self.ss_front = []
        self.ss_left = []
        self.ss_right = []
        self.ss_rear = []
        self.ss_rear_left = []
        self.ss_rear_right = []
        
        self.depth_front = []
        self.depth_left = []
        self.depth_right = []
        self.depth_rear = []
        self.depth_rear_left = []
        self.depth_rear_right = []
        self.counter = 0


        self.data_list = []
        self.frame_list = []
        self.sensor_data_list = []
        self.id_list = []
        self.ego_list = []
        
        self.record_flag = False

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def save_ego_data(self, path):
        with open(os.path.join(path, 'ego_data.json'), 'w') as f:
            json.dump(self.ego_data, f, indent=4)
        self.ego_data = {}

    def record_speed_control_transform(self, world, frame):
        v = world.player.get_velocity()
        c = world.player.get_control()
        t = world.player.get_transform()
        if frame not in self.ego_data:
            self.ego_data[frame] = {}
        self.ego_data[frame]['speed'] = {'constant': math.sqrt(v.x**2 + v.y**2 + v.z**2),
                                         'x': v.x, 'y': v.y, 'z': v.z}
        self.ego_data[frame]['control'] = {'throttle': c.throttle, 'steer': c.steer,
                                           'brake': c.brake, 'hand_brake': c.hand_brake,
                                           'manual_gear_shift': c.manual_gear_shift,
                                           'gear': c.gear}
        self.ego_data[frame]['transform'] = {'x': t.location.x, 'y': t.location.y, 'z': t.location.z,
                                             'pitch': t.rotation.pitch, 'yaw': t.rotation.yaw, 'roll': t.rotation.roll}

    
                                           
    def tick(self, world, clock, camera, frame, display, root_path):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        # print("vehicle height", t)

        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        vehicles = world.world.get_actors().filter('vehicle.*')
        peds = world.world.get_actors().filter('walker.pedestrian.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        moving = False
        acc = world.player.get_acceleration().length()

        if c.throttle == 0:
            self.jam += 1
            # print(acc)
            if self.jam > 100:
                return 0
        else:
            self.jam = 0 

        return 1

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)



# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction, save_mode=True):
        # self.sensor_front = None
        self.sensor_rgb_front = None

        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.save_mode = save_mode

        self.rgb_front = None
        self.ss_front = None



        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                # front view
                (carla.Transform(carla.Location(x=+0.8*bound_x,
                 y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                # front-left view
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=-55)), Attachment.Rigid),
                # front-right view
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=55)), Attachment.Rigid),
                # back view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=180)), Attachment.Rigid),
                # back-left view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=235)), Attachment.Rigid),
                # back-right view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=-235)), Attachment.Rigid),
                # top view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=23*bound_z), carla.Rotation(pitch=18.0)), Attachment.SpringArm),
                # LBC top view
                # (carla.Transform(carla.Location(x=0, y=0,
                #  z=25.0), carla.Rotation(pitch=-90.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=0, y=0,
                 z=20.0), carla.Rotation(pitch=-90.0)), Attachment.SpringArm),
                
                # sensor config for transfuser camera settings 
                #  front view 8
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0)), Attachment.Rigid),
                # left view  9 
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=-60.0)), Attachment.Rigid),
                # right view 10
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=60.0)), Attachment.Rigid),
                # rear 11 
                (carla.Transform(carla.Location(x=-1.3, y=0,
                 z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=180.0)), Attachment.Rigid),
                # rear left 12 
                (carla.Transform(carla.Location(x=-1.3, y=0,
                 z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=-120.0)), Attachment.Rigid),
                # rear right 13
                (carla.Transform(carla.Location(x=-1.3, y=0,
                 z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=120.0)), Attachment.Rigid)
            ]
        else:
            pass
            # self._camera_transforms = [
            #     (carla.Transform(carla.Location(x=-5.5, z=2.5),
            #      carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            #     (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            #     (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)),
            #      Attachment.SpringArm),
            #     (carla.Transform(carla.Location(x=-8.0, z=6.0),
            #      carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            #     (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw,
                'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None,
                'Lidar (Ray-Cast)', {'range': '85', 'rotation_frequency': '25'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.optical_flow', None, 'Optical Flow', {}],
            ['sensor.other.lane_invasion', None, 'Lane lane_invasion', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette,
                'Camera Instance Segmentation (CityScapes Palette)', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()

        # self.bev_bp = bp_library.find('sensor.camera.rgb')
        # self.bev_bp.set_attribute('image_size_x', str(300))
        # self.bev_bp.set_attribute('image_size_y', str(300))
        # self.bev_bp.set_attribute('fov', str(50.0))
        # if self.bev_bp.has_attribute('gamma'):
        #     self.bev_bp.set_attribute('gamma', str(gamma_correction))
        
        
        # self.sensor_rgb_bp = bp_library.find('sensor.camera.rgb')
        # self.sensor_rgb_bp.set_attribute('image_size_x', str(400))
        # self.sensor_rgb_bp.set_attribute('image_size_y', str(300))
        # self.sensor_rgb_bp.set_attribute('fov', str(100.0))
        
        self.sensor_rgb_bp = bp_library.find('sensor.camera.rgb')
        self.sensor_rgb_bp.set_attribute('image_size_x', str(512))
        self.sensor_rgb_bp.set_attribute('image_size_y', str(512))
        self.sensor_rgb_bp.set_attribute('fov', str(60.0))
        
        # self.sensor_ss_bp = bp_library.find('sensor.camera.instance_segmentation')
        self.sensor_ss_bp = bp_library.find('sensor.camera.semantic_segmentation')
        # 
        self.sensor_ss_bp.set_attribute('image_size_x', str(512))
        self.sensor_ss_bp.set_attribute('image_size_y', str(512))
        self.sensor_ss_bp.set_attribute('fov', str(60.0))
        
        self.sensor_depth_bp = bp_library.find('sensor.camera.depth')
        self.sensor_depth_bp.set_attribute('image_size_x', str(1280))
        self.sensor_depth_bp.set_attribute('image_size_y', str(720))
        self.sensor_depth_bp.set_attribute('fov', str(60.0))
        

        self.bev_seg_bp = bp_library.find('sensor.camera.instance_segmentation')
        self.bev_seg_bp.set_attribute('image_size_x', str(400))
        self.bev_seg_bp.set_attribute('image_size_y', str(400))
        self.bev_seg_bp.set_attribute('fov', str(50.0))

        self.front_cam_bp = bp_library.find('sensor.camera.rgb')
        self.front_cam_bp.set_attribute('image_size_x', str(768))
        self.front_cam_bp.set_attribute('image_size_y', str(256))
        self.front_cam_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.front_cam_bp.set_attribute('focal_distance', str(500))
        if self.front_cam_bp.has_attribute('gamma'):
            self.front_cam_bp.set_attribute('gamma', str(gamma_correction))

        self.front_seg_bp = bp_library.find('sensor.camera.instance_segmentation')
        self.front_seg_bp.set_attribute('image_size_x', str(768))
        self.front_seg_bp.set_attribute('image_size_y', str(256))
        self.front_seg_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.front_seg_bp.set_attribute('focal_distance', str(500))
        if self.front_seg_bp.has_attribute('gamma'):
            self.front_seg_bp.set_attribute('gamma', str(gamma_correction))

        self.depth_bp = bp_library.find('sensor.camera.depth')
        self.depth_bp.set_attribute('image_size_x', str(768))
        self.depth_bp.set_attribute('image_size_y', str(256))
        self.depth_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.depth_bp.set_attribute('focal_distance', str(500))
        if self.depth_bp.has_attribute('gamma'):
            self.depth_bp.set_attribute('gamma', str(gamma_correction))

        # self.flow_bp = bp_library.find('sensor.camera.optical_flow')
        # self.flow_bp.set_attribute('image_size_x', str(1236))
        # self.flow_bp.set_attribute('image_size_y', str(256))
        # self.flow_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.flow_bp.set_attribute('focal_distance', str(500))
        # if self.flow_bp.has_attribute('gamma'):
        #     self.flow_bp.set_attribute('gamma', str(gamma_correction))
        for item in self.sensors:
            
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index +
                                1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index]
             [2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor_rgb_front is not None:
                self.sensor_rgb_front.destroy()
                self.surface = None

            # rgb sensor
            if self.save_mode:
                ## setup sensors [ tf sensors  ( total 6 * 3 sensors ) ] 
                # rgb 
                self.sensor_rgb_front = self._parent.get_world().spawn_actor(
                    self.sensor_rgb_bp,
                    self._camera_transforms[8][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])
                
                # ss
                self.sensor_ss_front = self._parent.get_world().spawn_actor(
                    self.sensor_ss_bp,
                    self._camera_transforms[7][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)

            if self.save_mode:
                self.sensor_rgb_front.listen(lambda image: CameraManager._parse_image(weak_self, image, 'rgb_front'))
                self.sensor_ss_front.listen(lambda image: CameraManager._parse_image(weak_self, image, 'ss_front'))
                
                           

                
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)
        
    def render(self, display):
        if self.surface is not None:
            
            # print(self.surface)
            
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, view='top'):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)

        elif view == 'rgb_front':
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (512, 512, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            # render the view shown in monitor
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        #if self.recording and image.frame % 1 == 0:
        if True:
            # if view == 'front':
            #     self.front_img = image

            ## tf sensors  ( total 6 * 3 sensors )
            if view == 'rgb_front':
                self.rgb_front = image
            elif view == 'rgb_left':
                self.rgb_left = image
            elif view == 'rgb_right':
                self.rgb_right = image
                
            elif view == 'rgb_rear':
                self.rgb_rear = image
            elif view == 'rgb_rear_left':
                self.rgb_rear_left = image                
            elif view == 'rgb_rear_right':
                self.rgb_rear_right = image
            elif view == 'depth_front':
                self.depth_front = image
            elif view == 'depth_left':
                self.depth_left = image
            elif view == 'depth_right':
                self.depth_right = image
            elif view == 'depth_rear':
                self.depth_rear = image
            elif view == 'depth_rear_left':
                self.depth_rear_left = image
            elif view == 'depth_rear_right':
                self.depth_rear_right = image
            elif view == 'ss_front':
                self.ss_front = image
            elif view == 'ss_left':
                self.ss_left = image
            elif view == 'ss_right':
                self.ss_right = image
            elif view == 'ss_rear':
                self.ss_rear = image
            elif view == 'ss_rear_left':
                self.ss_rear_left = image
            elif view == 'ss_rear_right':
                self.ss_rear_right = image
            
            
def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []
    
class BEV_MAP():
    def __init__(self, args, world) -> None:
        
        self.args = args
        self.world = world
        
        self.data = None
        self.birdview_producer = BirdViewProducer(
                args.town, 
                # PixelDimensions(width=512, height=512), 
                PixelDimensions(width=304, height=304), 
                pixels_per_meter=5)
        
       
        # init Roach-model 
        self.model = None # load roach model 

        self.vehicle_bbox_1_16 = {}
        self.pedestrain_bbox_1_16 = {}

        self.Y_bbox_1_16 = {}
        self.G_bbox_1_16 = {}
        self.R_bbox_1_16 = {}
        
        
        # find traffic light id , and mapping ids to stop line index 
        town = self.args.town
        with open(f'./bird_eye_view/Stop_Line/{town}.json', 'r') as f:
            stop_line_dict = json.load(f)
        
        traffic_light = {}
            
        lights = self.world.world.get_actors().filter("*traffic_light*")
        for actor in lights:
            
            _id = actor.id
            actor_loc = actor.get_location()
            x = actor_loc.x
            y = actor_loc.y
            
            for key in stop_line_dict.keys():
                stop_line_loc = stop_line_dict[key]["loc"]
                
                if stop_line_loc[0] == -1 and stop_line_loc[1] == -1:
                    continue
                distance = math.sqrt((x-stop_line_loc[0])**2 + (y-stop_line_loc[1])**2)
                if distance < 1:
                    traffic_light[_id] = {}
                    traffic_light[_id]["stop_line_A"] = stop_line_dict[key]["stop_line_A"]
                    traffic_light[_id]["stop_line_B"] = stop_line_dict[key]["stop_line_B"]
                    break
                
        self.traffic_light = traffic_light
        
        
                

    def init_spawn_points_and_planner(self, spawn_points, planner):
        self.spawn_points = spawn_points
        self.planner = planner
        

    def init_policy(self, policy = None):    # prepare policy
        # inti roach model 
        wandb_path = "sonicokuo/cata_train_rl_FixSeed/o6gx5mka"
        if wandb_path[-1] == '\n':
               wandb_path = wandb_path[:-1]
        api = wandb.Api()
        run = api.run(wandb_path)
        all_ckpts = [f for f in run.files() if 'ckpt' in f.name] #load a check points
        f = max(all_ckpts, key=lambda x: int(x.name.split('_')[1].split('.')[0]))
        print(f'Resume checkpoint latest {f.name}')
        f.download(replace=True)
        run.file('config_agent.yaml').download(replace=True)
        cfg = OmegaConf.load('config_agent.yaml')
        cfg = OmegaConf.to_container(cfg)
        self._ckpt = f.name

        ckpt = f.name

        train_cfg = cfg['training'] 

        policy_class = load_entry_point(cfg['policy']['entry_point'])
        policy_kwargs = cfg['policy']['kwargs']
        print(f'Loading wandb checkpoint: {ckpt}')
        policy, train_cfg['kwargs'] = policy_class.load(ckpt)
        policy = policy.eval()

        self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
        self._wrapper_kwargs = cfg['env_wrapper']['kwargs']
        self.policy = policy

        return policy
    def set_policy(self, processed_policy):
        policy = copy.deepcopy(processed_policy)
        self.policy = policy

    def init_vehicle_bbox(self, ego_id):
        self.vehicle_bbox_1_16[ego_id] = []
        self.pedestrain_bbox_1_16[ego_id] = []
        self.Y_bbox_1_16[ego_id] = []
        self.G_bbox_1_16[ego_id] = []
        self.R_bbox_1_16[ego_id] = []
        self.R_bbox_1_16[ego_id] = []

    def collect_actor_data(self, world):
        vehicles_id_list = []
        bike_blueprint = ["vehicle.bh.crossbike","vehicle.diamondback.century","vehicle.gazelle.omafiets"]
        motor_blueprint = ["vehicle.harley-davidson.low_rider","vehicle.kawasaki.ninja","vehicle.yamaha.yzf","vehicle.vespa.zx125"]
        
        def get_xyz(method, rotation=False):

            if rotation:
                roll = method.roll
                pitch = method.pitch
                yaw = method.yaw
                return {"pitch": pitch, "yaw": yaw, "roll": roll}

            else:
                x = method.x
                y = method.y
                z = method.z

                # return x, y, z
                return {"x": x, "y": y, "z": z}

        ego_loc = world.player.get_location()
        data = {}

        vehicles = world.world.get_actors().filter("*vehicle*")
        for actor in vehicles:

            _id = actor.id
            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            rotation = get_xyz(actor.get_transform().rotation, True)

            cord_bounding_box = {}
            bbox = actor.bounding_box
            if actor.type_id in motor_blueprint:
                bbox.extent.x = 1.177870
                bbox.extent.y = 0.381839
                bbox.extent.z = 0.75
                bbox.location = carla.Location(0, 0, bbox.extent.z)
            elif actor.type_id in bike_blueprint:
                bbox.extent.x = 0.821422
                bbox.extent.y = 0.186258
                bbox.extent.z = 0.9
                bbox.location = carla.Location(0, 0, bbox.extent.z)
                
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            distance = ego_loc.distance(actor_loc)


            # if distance < 50:
            vehicles_id_list.append(_id)

            acceleration = get_xyz(actor.get_acceleration())
            velocity = get_xyz(actor.get_velocity())
            angular_velocity = get_xyz(actor.get_angular_velocity())

            v = actor.get_velocity()

            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

            vehicle_control = actor.get_control()
            control = {
                "throttle": vehicle_control.throttle,
                "steer": vehicle_control.steer,
                "brake": vehicle_control.brake,
                "hand_brake": vehicle_control.hand_brake,
                "reverse": vehicle_control.reverse,
                "manual_gear_shift": vehicle_control.manual_gear_shift,
                "gear": vehicle_control.gear
            }


            data[_id] = {}
            data[_id]["location"] = location
            data[_id]["rotation"] = rotation
            data[_id]["distance"] = distance
            data[_id]["acceleration"] = acceleration
            data[_id]["velocity"] = velocity
            data[_id]["vel_x"] = v.x
            data[_id]["vel_y"] = v.y
            data[_id]["speed"] = speed
            data[_id]["angular_velocity"] = angular_velocity
            data[_id]["control"] = control
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "vehicle"

        pedestrian_id_list = []

        walkers = world.world.get_actors().filter("*pedestrian*")
        for actor in walkers:

            _id = actor.id

            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            rotation = get_xyz(actor.get_transform().rotation, True)

            cord_bounding_box = {}
            bbox = actor.bounding_box
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            distance = ego_loc.distance(actor_loc)

            # if distance < 50:
            pedestrian_id_list.append(_id)

            acceleration = get_xyz(actor.get_acceleration())
            velocity = get_xyz(actor.get_velocity())
            angular_velocity = get_xyz(actor.get_angular_velocity())

            walker_control = actor.get_control()
            control = {"direction": get_xyz(walker_control.direction),
                       "speed": walker_control.speed, "jump": walker_control.jump}

            data[_id] = {}
            data[_id]["location"] = location
            # data[_id]["rotation"] = rotation
            data[_id]["distance"] = distance
            data[_id]["acceleration"] = acceleration
            data[_id]["velocity"] = velocity
            data[_id]["angular_velocity"] = angular_velocity
            data[_id]["control"] = control

            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = 'pedestrian'
            
        traffic_id_list = []
        lights = world.world.get_actors().filter("*traffic_light*")
        for actor in lights:
            
            _id = actor.id
            stop_line_A = self.traffic_light[_id]["stop_line_A"]
            stop_line_B = self.traffic_light[_id]["stop_line_B"]
            
            
            traffic_id_list.append(_id)
            traffic_light_state = int(actor.state)  # traffic light state
            cord_bounding_box = {}
            cord_bounding_box["cord_0"] = [stop_line_A[0], stop_line_A[1]]
            cord_bounding_box["cord_1"] = [stop_line_B[0], stop_line_B[1]]

            data[_id] = {}
            data[_id]["state"] = traffic_light_state
            data[_id]["cord_bounding_box"] = cord_bounding_box


        obstacle_id_list = []

        obstacle = world.world.get_actors().filter("*static.prop*")
        # print(obstacle)
        for actor in obstacle:

            _id = actor.id

            actor_loc = actor.get_location()
            # print(actor_loc)
            distance = ego_loc.distance(actor_loc)
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                # print([loc.x, loc.y, loc.z])
                counter += 1
            # if distance < 50:
            obstacle_id_list.append(_id)

            data[_id] = {}
            data[_id]["distance"] = distance
            data[_id]["type"] = "obstacle"
            data[_id]["cord_bounding_box"] = cord_bounding_box


        data["traffic_light_ids"] = traffic_id_list

        data["obstacle_ids"] = obstacle_id_list
        data["vehicles_ids"] = vehicles_id_list
        data["pedestrian_ids"] = pedestrian_id_list

        self.data = data
        return data

    def set_data(self, proccessed_data):
        self.data = proccessed_data

    def run_step(self, result, frame, ego_id, current_route = None):
        start_time = time.time()
        actor_dict =  copy.deepcopy(self.data)

        ego_loc = carla.Location(x=actor_dict[ego_id]["location"]["x"], y=actor_dict[ego_id]["location"]["y"])
        ego_pos = Loc(x=actor_dict[ego_id]["location"]["x"], y=actor_dict[ego_id]["location"]["y"])
        ego_yaw = actor_dict[ego_id]["rotation"]["yaw"]


        obstacle_bbox_list = []
        pedestrian_bbox_list = []
        vehicle_bbox_list = []
        agent_bbox_list = []

        r_traffic_light_list = []
        g_traffic_light_list = []
        y_traffic_light_list = []

        # interactive id 
        vehicle_id_list = list(actor_dict["vehicles_ids"])
        # pedestrian id list 
        pedestrian_id_list = list(actor_dict["pedestrian_ids"])
        # obstacle id list 
        obstacle_id_list = list(actor_dict["obstacle_ids"])

        # traffic light id list 

        traffic_light_id_list = list(actor_dict["traffic_light_ids"])

        for id in traffic_light_id_list:
            traffic_light_loc = carla.Location(x=actor_dict[id]["cord_bounding_box"]["cord_0"][0], y=actor_dict[id]["cord_bounding_box"]["cord_0"][1])
            if not check_close(ego_loc, traffic_light_loc, 45): #35.95m
                continue
            pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
            pos_1 = actor_dict[id]["cord_bounding_box"]["cord_1"]


            if actor_dict[id]["state"] == 0 :
                g_traffic_light_list.append([
                                        Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        ])
            elif actor_dict[id]["state"] == 2:
                g_traffic_light_list.append([
                                        Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        ])  
            elif actor_dict[id]["state"] == 1:
                g_traffic_light_list.append([
                                        Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        ]) 
            



        for id in obstacle_id_list:
            # print('ob',id)
            if actor_dict[id]['distance'] > 25:
                continue
            pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
            pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
            pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
            pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]
            obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
        # print(obstacle_bbox_list)
        for id in vehicle_id_list:
            traffic_light_loc = carla.Location(x=actor_dict[id]['location']['x'], y=actor_dict[id]['location']['y'])
            if not check_close(ego_loc, traffic_light_loc, 36): #35.95m
                continue
            pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
            pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
            pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
            pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]
            

            if int(id) == int(ego_id):
                agent_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
            else:
                vehicle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
                    
        for id in pedestrian_id_list:
            pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
            pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
            pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
            pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]

            pedestrian_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
            


        if len(self.pedestrain_bbox_1_16[ego_id]) < 16:
            self.pedestrain_bbox_1_16[ego_id].append(pedestrian_bbox_list)
        else:
            self.pedestrain_bbox_1_16[ego_id].pop(0)
            self.pedestrain_bbox_1_16[ego_id].append(pedestrian_bbox_list)

        if len(self.vehicle_bbox_1_16[ego_id]) < 16:
            self.vehicle_bbox_1_16[ego_id].append(vehicle_bbox_list)
        else:
            self.vehicle_bbox_1_16[ego_id].pop(0)
            self.vehicle_bbox_1_16[ego_id].append(vehicle_bbox_list)


        if len(self.Y_bbox_1_16[ego_id]) < 16:
            self.Y_bbox_1_16[ego_id].append(y_traffic_light_list)
        else:
            self.Y_bbox_1_16[ego_id].pop(0)
            self.Y_bbox_1_16[ego_id].append(y_traffic_light_list)


        if len(self.G_bbox_1_16[ego_id]) < 16:
            self.G_bbox_1_16[ego_id].append(g_traffic_light_list)
        else:
            self.G_bbox_1_16[ego_id].pop(0)
            self.G_bbox_1_16[ego_id].append(g_traffic_light_list)

        if len(self.R_bbox_1_16[ego_id]) < 16:
            self.R_bbox_1_16[ego_id].append(r_traffic_light_list)
        else:
            self.R_bbox_1_16[ego_id].pop(0)
            self.R_bbox_1_16[ego_id].append(r_traffic_light_list)


        #### initial state for frenet frame calculation (will not change) ####
        c_speed = 3.0 / 3.6  # current speed [m/s]
        c_accel = 0.0  # current acceleration [m/ss]
        c_d = 0.0  # current lateral position [m]
        c_d_d = 0.0  # current lateral speed [m/s]
        c_d_dd = 0.0  # current lateral acceleration [m/s]
        s0 = 0.0  # current course position
   

        #### TODOs: rewrite this part to fit in current_route ###
        # # regenerate a destination when the agent deviates from the current route
        # if not check_close(world.player.get_location(), current_route[0][0].transform.location, 6):
        #     destination = current_route[-1][0].transform.location
        #     current_route = planner.trace_route(world.player.get_location(), destination)

            
        #### Generate new destination if the current one is close ###
        if len(current_route) < 5:
            new_destination = random.choice(self.spawn_points).location
            new_route = self.planner.trace_route(current_route[0], new_destination)
            new_route_locs = []
            for wp in new_route:
                wp = wp[0]
                new_route_locs.append(wp.transform.location)
            temp_route = current_route + new_route_locs
            current_route = temp_route

        #### Delete reached points from current route ###
        #### If Frenet Frame is activated, augment reaching range because the agent may deviate from the original route to avoid obstacles #### 
        if len(obstacle_bbox_list) > 0:
            while check_close(ego_loc, current_route[0][0].transform.location, 5):
                # print(world.player.get_location().distance(current_route[0]))
                current_route.pop(0)
        else:
            while check_close(ego_loc, current_route[0][0].transform.location, 3):
                # print(world.player.get_location().distance(current_route[0]))
                current_route.pop(0)
            #### check if lost reached points ####
            set_check_range = 5
            if len(current_route) > set_check_range:
                if check_close(ego_loc, current_route[set_check_range][0].transform.location, 3):
                    for i in range(set_check_range):
                        current_route.pop(0)

            
        #### Frenet Frame waypoints inputs: (ego_loc, target_point) ####
        wx = []
        wy = []
        wx.append(ego_loc.x)
        wy.append(ego_loc.y)

        #### target: 20th waypoints of current route ####
        set_target = 20
        
        if len(current_route) > set_target+1:
            wx.append(current_route[set_target][0].transform.location.x)
            wy.append(current_route[set_target][0].transform.location.y)     
        else:       
            wx.append(current_route[-1][0].transform.location.x)
            wy.append(current_route[-1][0].transform.location.y)

        #### Frenet Frame obstacle inputs: real obstacle points + virtual obstacle points ####
        #### Obtain real obstacle points from obstacle_bbox_list ####
        ob_points = []
        obs_center_locs = []
        for obstacle in obstacle_bbox_list:
            corner_x = [] 
            corner_y = []  
            for corner in obstacle:
                ob_points.append([corner.x, corner.y])
                corner_x.append(corner.x)
                corner_y.append(corner.y)
            #### add center point ####
            ob_points.append([sum(corner_x)/len(corner_x), sum(corner_y)/len(corner_y)])
            obs_center_locs.append(carla.Location(x=sum(corner_x)/len(corner_x), y=sum(corner_y)/len(corner_y)))

        #### Check left, right, upper, lower points of each obstacle ####
        #### If a point is in none drivable areas, add a virtual obstacle point at the corresponding point ####
        none_drivable_obs = []
        set_range = 20
        pixels_per_meter=5
        for obs_center_loc in obs_center_locs:
            obs_pixel =  self.birdview_producer.masks_generator.location_to_pixel(obs_center_loc, full=True)
            if self.birdview_producer.full_road_cache[obs_pixel.y][obs_pixel.x+set_range] == 0:
                none_drivable_obs.append([obs_center_loc.x+set_range/pixels_per_meter, obs_center_loc.y])
            if self.birdview_producer.full_road_cache[obs_pixel.y][obs_pixel.x-set_range] == 0:
                none_drivable_obs.append([obs_center_loc.x-set_range/pixels_per_meter, obs_center_loc.y])                
            if self.birdview_producer.full_road_cache[obs_pixel.y+set_range][obs_pixel.x] == 0:
                none_drivable_obs.append([obs_center_loc.x, obs_center_loc.y+set_range/pixels_per_meter])
            if self.birdview_producer.full_road_cache[obs_pixel.y-set_range][obs_pixel.x] == 0:
                none_drivable_obs.append([obs_center_loc.x, obs_center_loc.y-set_range/pixels_per_meter])   
        
        #### Append virtual obstacle points ####
        aggregate_ob_points = ob_points + none_drivable_obs
        aggregate_ob_points = np.array(aggregate_ob_points)


        #### if an obstacle is near the ego, use Frenet Frame to calculate new routes
        if len(aggregate_ob_points) > 0:

            #### calculate several curved lines near the waypoints inputs ####
            tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

            #### choose the most ideal curved line that (1) will not run into obstacle points (2) is the nearest to input waypoints ####
            path = frenet_optimal_planning(
                    csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, aggregate_ob_points)
            route_list = []
            for i in range(len(path.x)):
                loc = Loc(path.x[i], path.y[i])
                route_list.append(loc)
        #### if there are no obstacles near the ego, use the original route (from Map_visualizer or A* planner) ####
        else:
            render_route = current_route[0:40]
            route_list = []
            for i in range(len(render_route)):
                loc = Loc(render_route[i][0].transform.location.x, render_route[i][0].transform.location.y)
                route_list.append(loc)


        birdview: BirdView = self.birdview_producer.produce(ego_pos, ego_yaw,
                                                       agent_bbox_list, 
                                                       self.vehicle_bbox_1_16[ego_id],
                                                       self.pedestrain_bbox_1_16[ego_id],
                                                       self.R_bbox_1_16[ego_id],
                                                       self.G_bbox_1_16[ego_id],
                                                       self.Y_bbox_1_16[ego_id],
                                                       obstacle_bbox_list,
                                                       route_list)
    



        topdown = BirdViewProducer.as_rgb(birdview)
        # print(len(topdown))
        # print(obs_center_loc)
        # print(obs_pixel)
        # for i in range(10):
        #     for j in range(10):
        #         topdown[obs_pixel.x+i][obs_pixel.y+j] = (255,255,255)


        # input BEV representation
        roach_obs, new_array =  BirdViewProducer.as_roach_input(birdview)

        # input state   
        throttle =  actor_dict[ego_id]["control"]["throttle"]
        steer =  actor_dict[ego_id]["control"]["steer"]
        brake =  actor_dict[ego_id]["control"]["brake"]
        gear =  actor_dict[ego_id]["control"]["gear"]
        vel_x =  actor_dict[ego_id]["vel_x"]
        vel_y =  actor_dict[ego_id]["vel_y"]
        # throttle, steer, brake, gear/5.0 , vel_x, vel_y

        # combine input


        # birdview = np.expand_dims(birdview, 0)
        # state = np.expand_dims(state, 0)
        
        # obs_dict = {
        #     'state': state.astype(np.float32),
        #     'birdview': birdview
        # }

        policy_input = {}
        policy_input['state'] = np.expand_dims([throttle, steer, brake, gear/5, vel_x, vel_y], 0).astype(np.float32)
        # print(policy_input['state'])
        policy_input['birdview'] = np.expand_dims(roach_obs, 0)

        result["current_route"] = current_route

        # return roach_obs
        result["bev_map_rgb"] = topdown[0:192, 56:248]
        # result["bev_map_rgb"] = topdown

        # result["bev_map_rgb"] = new_array
        result["policy_input"] = policy_input
        # return topdown[0:192, 56:248], policy_input
        # return new_array, policy_input
        self.policy_forward(result, [policy_input])
        # print("finish one both", ego_id ,start_time - time.time(), frame - time.time())
        
    def policy_forward(self, result, policy_inputs_list):    
        control_elements_list = []
        for policy_input in policy_inputs_list:
            actions, values, log_probs, mu, sigma, features = self.policy.forward(
                policy_input, deterministic=True, clip_action=True)
                     
            # print(actions)
            for action in actions:
                acc, steer = action.astype(np.float64)
                if acc >= 0.0:
                    throttle = acc
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = np.abs(acc)

                throttle = np.clip(throttle, 0, 0.5)
                steer = np.clip(steer, -1, 1)
                brake = np.clip(brake, 0, 1)
                control_elements = {}
                control_elements['throttle'] = throttle
                control_elements['steer'] = steer
                control_elements['brake'] = brake
                control_elements_list.append(control_elements)
        result["control_elements_list"] = control_elements_list
        # return control_elements_list
        # return contorl 
        
        
# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def check_close(ev_loc, loc0, distance = 3):

    if ev_loc.distance(loc0) < distance:
        return True
    else:
        return False


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    #try:
    if True:
    
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        # display = pygame.display.set_mode(
        #     (args.width, args.height),
        #     pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        
        display = pygame.display.set_mode(
            (512, 900),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()


        stored_path = os.path.join('data_collection', args.town)
        # if not os.path.exists(stored_path) :
        #     os.makedirs(stored_path)


        hud = HUD(args.width, args.height, args.distance, args.town, stored_path)
        world = World(client.load_world(args.town), hud, args)

        settings = world.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True  # Enables synchronous mode
        world.world.apply_settings(settings)

        controller = KeyboardControl(world, args.autopilot)
        
        ##### set up an obstacle blueprint ####
        obstacles_list = []
        blueprint = world.world.get_blueprint_library().find("static.prop.container")
        print(blueprint)


        ##### set up an obstacle location ####
        obstacle_transform = carla.Transform(carla.Location(-41.0, 96.4, 0.0))

        ##### spawn an obstacle ####
        try:
            obstacle = client.get_world().spawn_actor(blueprint, obstacle_transform)

            id = obstacle.id
            obstacles_list.append(id)
            agent_dict[id] = {}
            agent_dict[id]["agent"] = obstacle
                
        except:
            print("Spawn obstacle failed because of collision at spawn position")

            
        
        vehicles_list = []
        agent_dict = {}
        ##### spawn other agents (currently not support) #####
        map = world.world.get_map()
        spawn_points = map.get_spawn_points()
        waypoint_list = []
        for waypoint in spawn_points:            
            waypoint_list.append(waypoint)
        random.shuffle(waypoint_list)
        blueprints = world.world.get_blueprint_library().filter('vehicle.*')

        actor_numbers = args.n
        # actor_numbers = 4

        # for num_of_vehicles, transform in enumerate(waypoint_list):
        for num_of_vehicles, transform in enumerate(waypoint_list):
            if len(vehicles_list) == actor_numbers:
                break
            blueprint = world.world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')

            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
                
            # blueprint, transform
            
            try:
        
                other_agent = client.get_world().spawn_actor(blueprint, transform)
                id = other_agent.id
                vehicles_list.append(id)
                
                agent_dict[id] = {}
                agent_dict[id]["agent"] = other_agent
                
                # agent = BehaviorAgent(other_agent, behavior='aggressive')
                # destination = random.choice(spawn_points).location
                # agent.set_destination(destination)
                
                # agent_dict[id]["BehaviorAgent"] = agent
            except:
                print("Spawn failed because of collision at spawn position")
      
            
            
        
        # agent = BehaviorAgent(world.player, behavior='aggressive')
        # destination = random.choice(spawn_points).location
        # agent.set_destination(destination)


        planner = GlobalRoutePlanner(map, resolution=1.0)

        clock = pygame.time.Clock()


        #### initialize the ego player ####
        result = {}
        player_bev_map = BEV_MAP(args, world)
        processed_policy = player_bev_map.init_policy()
        player_bev_map.init_vehicle_bbox(world.player.id)
        player_bev_map.init_spawn_points_and_planner(map.get_spawn_points(), planner)


        # obstacle_route = np.load('./route.npy')
        obstacle_route = np.load(f'./route/{args.town}/route00.npy')
        #destination = random.choice(spawn_points).location
        destination = carla.Location(obstacle_route[1][0], obstacle_route[1][1])
        np.delete(obstacle_route, [0,1])
        current_route = planner.trace_route(world.player.get_location(), destination)
        
        
        #print(current_route)
        #### load the route generated from Map_visualizer ###
        #obstacle_route = np.load('/media/hcis-s02/disk2/Map_Visualiser/route/Town10HD/route00.npy')
        
        # current_route = []
        # for point in obstacle_route:
        #     loc = carla.Location(point[0], point[1])
        #     current_route.append(loc)


        #player_bev_map.init_spawn_points_and_planner(carla.Location(obstacle_route[0][0], obstacle_route[0][1]),planner)

        #### initialize a dict to store the information of other agents ####
        agent_bev_maps = {}
        current_routes = {}
        for id in vehicles_list:
            agent_bev_map = BEV_MAP(args)
            agent_bev_map.init_vehicle_bbox(id)
            agent_bev_map.set_policy(processed_policy)
            destination = random.choice(spawn_points).location

            agent_bev_maps[id] = agent_bev_map
            current_routes[id] = []

        while True:

            clock.tick_busy_loop(20)
            frame = world.world.tick()
            if controller.parse_events(client, world, clock):
                return
            view = pygame.surfarray.array3d(display)
            view = view.transpose([1, 0, 2]) 
            image = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                        

            world.tick(clock, frame, image, stored_path)

            
            
            # regenerate a destination when the agent deviates from the current route
            if not check_close(world.player.get_location(), current_route[0][0].transform.location, 6):
                destination = current_route[-1][0].transform.location
                current_route = planner.trace_route(world.player.get_location(), destination)
            
            # Delete reached points from current route
            while check_close(world.player.get_location(), current_route[0][0].transform.location):
                current_route.pop(0)
                if len(current_route) == 0:
                    #new_destination = random.choice(spawn_points).location
                    new_destination = carla.Location(obstacle_route[0][0], obstacle_route[0][1])
                    np.delete(obstacle_route, 0)
                    current_route = planner.trace_route(world.player.get_location(), new_destination)

            
            # Generate new destination if the current one is close 
            if len(current_route) < 10:
                #new_destination = random.choice(spawn_points).location
                new_destination = carla.Location(obstacle_route[0][0], obstacle_route[0][1])
                np.delete(obstacle_route, 0)
                new_route = planner.trace_route(current_route[-1][0].transform.location, new_destination)
                temp_route = current_route + new_route
                current_route = temp_route


            while check_close(world.player.get_location(), current_route[0][0].transform.location):
                #print(world.player.get_location().distance(current_route[0][0]))
                current_route.pop(0)
                if len(current_route) == 0:
                    #new_destination = random.choice(spawn_points).location
                    new_destination = carla.Location(obstacle_route[0][0], obstacle_route[0][1])
                    np.delete(obstacle_route, 0)
                    current_route = planner.trace_route(world.player.get_location(), new_destination)

            route_trace = current_route[0:80]


            # collect data for all agents
            proccessed_data = player_bev_map.collect_actor_data(world)
            # route_list = []
            # # for wp in route_trace:
            # #     wp = wp[0]
            # #     route_list.append(Loc(x=wp.transform.location.x, y=wp.transform.location.y))
 
            # for i in range(len(path.x)):
            #     loc = Loc(path.x[i], path.y[i])
            #     route_list.append(loc)
            # # print(start_time - time.time())

            ###################player control#####################
            # start_time = time.time()
            # print("player_runstep")
            

            player_bev_map.run_step(result, frame, world.player.id, current_route)
            bev_map_rgb = result["bev_map_rgb"]
            current_route = result["current_route"]
            # policy_input = result["policy_input"]
            # print(start_time - time.time())
            # start_time = time.time()
            # print("player_forward")            
            # player_bev_map.policy_forward(result, [policy_input])
            control_elements_list = result["control_elements_list"]
            # print("player_all", start_time - time.time())
            # start_time = time.time()
            # print("player_render")

            # for render
            surface = pygame.surfarray.make_surface(bev_map_rgb)
            surface = pygame.transform.flip(surface, True, False)
            surface = pygame.transform.rotate(surface, 90)
            display.blit(surface, (256, 0))
            # print(start_time - time.time())
            # start_time = time.time()
            # print("player_control")
            control_elements = control_elements_list[0]
            control = carla.VehicleControl(throttle=control_elements['throttle'], steer=control_elements['steer'], brake=control_elements['brake'])
        
            world.player.apply_control( control )
            ###################player control#####################

            # print(start_time - time.time())
            # print("+++++++actors++++++++++")
            route_trace = {}
            count = 1
            inputs = []
            start_time_r = time.time()

            for num_of_vehicles, id in enumerate(vehicles_list):
                if len(current_routes[id]) == 0:
                    new_destination = random.choice(spawn_points).location
                    print(id)
                    print(new_destination)
                    current_routes[id] = planner.trace_route(agent_dict[id]['agent'].get_location(), new_destination)
                
                # regenerate a route when the agent deviates from the current route
                if not check_close(agent_dict[id]["agent"].get_location(), current_routes[id][0][0].transform.location, 6):
                    destination = current_routes[id][-1][0].transform.location
                    current_routes[id] = planner.trace_route(agent_dict[id]["agent"].get_location(), destination)
                
                # Delete reached points from current route
                while check_close(agent_dict[id]["agent"].get_location(), current_routes[id][0][0].transform.location):
                    current_routes[id].pop(0)
                    if len(current_routes[id]) == 0:
                        new_destination = random.choice(spawn_points).location
                        current_routes[id] = planner.trace_route(agent_dict[id]["agent"].get_location(), new_destination)
                # Generate new destination if the current one is close 
                if len(current_routes[id]) < 10:
                    new_destination = random.choice(spawn_points).location
                    new_route = planner.trace_route(current_routes[id][-1][0].transform.location, new_destination)
                    temp_route = current_routes[id] + new_route
                    current_routes[id] = temp_route       

                route_trace[id] = current_routes[id][0:60]
                # bev_map_rgb, control = bev_map.run_step(frame, id, route_trace[id]) 
                # agent_dict[id]["agent"].apply_control(control)

                route_list = []
                for wp in route_trace[id]:
                    wp = wp[0]
                    route_list.append(Loc(x=wp.transform.location.x, y=wp.transform.location.y))
                # #original

                # bev_map_rgb, control_elements = bev_map.run_step(frame, id, route_list) 
                # control = carla.VehicleControl(throttle=control_elements['throttle'], steer=control_elements['steer'], brake=control_elements['brake'])
                # agent_dict[id]["agent"].apply_control(control)
                start_time = time.time()
                inputs.append([start_time, id, route_list])



            #     result = {}
            #     agent_bev_maps[id].set_data(processed_data)
            #     agent_bev_maps[id].run_step(result, start_time, id, route_list)

            #     bev_map_rgb = result["bev_map_rgb"]
            #     policy_input = result["policy_input"]     
            #     control_elements_list = agent_bev_maps[id].policy_forward([policy_input])   
            #     control_elements = control_elements_list[0]
            #     control = carla.VehicleControl(throttle=control_elements['throttle'], steer=control_elements['steer'], brake=control_elements['brake'])
            #     agent_dict[id]["agent"].apply_control(control)

            #     if count < 4:
            #         surface = pygame.surfarray.make_surface(bev_map_rgb)
            #         surface = pygame.transform.flip(surface, True, False)
            #         surface = pygame.transform.rotate(surface, 90)
            #         display.blit(surface, (256, 210*count))                
            #     count += 1

            # print("finish all",start_time_r - time.time())
            # print("finish routelist",start_time_r - time.time())

            # start_time_run = time.time()
            t_list = []
            results = [{} for i in range(actor_numbers)]
            for i in range(actor_numbers):
                input = [results[i]]
                input = input + inputs[i]
                agent_bev_maps[vehicles_list[i]].set_data(proccessed_data)
                t = Thread(target=agent_bev_maps[vehicles_list[i]].run_step, args=tuple(input))
                t_list.append(t)
                t.start()
            for t in t_list:
                t.join()

            # print("finish both",start_time_run - time.time())
            # print("finish runstep",start_time_run - time.time())
            
            
            # start_time = time.time()
            # t2_list = []
            # for i in range(actor_numbers):
            #     policy_input = [results[i]]
            #     policy_input = policy_input + [[results[i]["policy_input"]]]
            #     t = Thread(target=agent_bev_maps[vehicles_list[i]].policy_forward, args=tuple(policy_input))
            #     t2_list.append(t)
            #     t.start()
            # for t in t2_list:
            #     t.join()
            # print("finish infer",start_time - time.time())

            for i in range(1, (min(4, actor_numbers))):
                bev_map_rgb = results[i]["bev_map_rgb"]
                surface = pygame.surfarray.make_surface(bev_map_rgb)
                surface = pygame.transform.flip(surface, True, False)
                surface = pygame.transform.rotate(surface, 90)
                display.blit(surface, (256, 210*i))  

            start_time = time.time()
            for i in range(actor_numbers):
                control_elements_list = results[i]["control_elements_list"] 
                control_elements = control_elements_list[0]
                control = carla.VehicleControl(throttle=control_elements['throttle'], steer=control_elements['steer'], brake=control_elements['brake'])
                agent_dict[vehicles_list[i]]["agent"].apply_control(control)
                

            # print("finish all",start_time_r - time.time())





            # policy_inputs_list = []
            # for bev_map_rgb, policy_input in image_inputs:

            #     policy_inputs_list.append(policy_input)

            #     # Visualize other agents
            #     if count < 4:
            #         surface = pygame.surfarray.make_surface(bev_map_rgb)
            #         surface = pygame.transform.flip(surface, True, False)
            #         surface = pygame.transform.rotate(surface, 90)
            #         display.blit(surface, (256, 210*count))                
            #     count += 1
                
            # # print("finish control and render",start_time - time.time())
            # # start_time = time.time()
            #     # print("finish one",start_time - time.time())
            #     # start_time = time.time()
            #     # agent_dict[id]["agent"].apply_control()
            
            # # print(policy_inputs_list)
            # start_time = time.time()
            # control_elements_list = bev_map.policy_forward(policy_inputs_list)
            # print("finish all forward",start_time - time.time())
            # start_time = time.time()
            # i = 0
            # for control_elements in control_elements_list:
            #     control = carla.VehicleControl(throttle=control_elements['throttle'], steer=control_elements['steer'], brake=control_elements['brake'])
            #     agent_dict[vehicles_list[i]]["agent"].apply_control(control)
            #     i+=1
            # print("finish all control",start_time - time.time())
            # start_time = time.time()
            world.hud.render(display)
            
            pygame.display.flip()

            # print("++++end++++")

            
            
            
            # world.render(display)
            #print("location: ",world.player.get_location())
            
            # if agent.done():
            #     agent.set_destination(random.choice(spawn_points).location)
            # control = agent.run_step()
            # control.manual_gear_shift = False
            # world.player.apply_control(control)
            
    # except Exception as e:
    #       print(e)
    # finally:
        settings = world.world.get_settings()
        settings.synchronous_mode = False 
        world.world.apply_settings(settings)

        print('destroying vehicles')
        
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        



        time.sleep(0.5)


        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='512x512',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--distance',
        default=25.0,
        type=float,
        help='distance to intersection for toggling camera)')
    argparser.add_argument(
        '--town',
        default='Town05',
        type=str,
        help='map)')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--n',
        default='4',
        type=int,
        help='number of Roach agents)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()


