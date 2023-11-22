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

# for BEV maps 
from bird_eye_view.BirdViewProducer import BirdViewProducer, BirdView
from bird_eye_view.Mask import PixelDimensions, Loc
from six.moves import cPickle as pickle #for performance
from scipy.spatial import distance 

################
import carla
from carla import ColorConverter as cc
import argparse
import datetime
import time
import logging
import math
import random
import re
import weakref
# from auto_random_actors import spawn_actor_nearby
import cv2
from navigation.global_route_planner import GlobalRoutePlanner
import copy
from omegaconf import OmegaConf
from importlib import import_module
import wandb
import collections

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
    def __init__(self, carla_world, hud, args, ego_transform):
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

        self.hud = hud
        self.player = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        
        self.collision_sensor = None
        
        
        # set ego position 
        self.restart(ego_transform)
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

    def restart(self, ego_transform ):
        self.player_max_speed = 1.3 #1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        # cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        # cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        # blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        
        
        blueprint = self.world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
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
        # if self.player is not None:
            # spawn_point = self.player.get_transform()
            # spawn_point.location.z += 2.0
            # spawn_point.rotation.roll = 0.0
            # spawn_point.rotation.pitch = 0.0
            # self.destroy()
        self.player = self.world.try_spawn_actor(blueprint, ego_transform) # spawn_point)
        
        
        print(self.player)
        self.show_vehicle_telemetry = False
        self.modify_vehicle_physics(self.player)            
        # while self.player is None:
        #     if not self.map.get_spawn_points():
        #         print('There are no spawn points available in your map/town.')
        #         print('Please add some Vehicle Spawn Point to your UE4 scene.')
        #         sys.exit(1)

        #     # random spawn points 
        #     spawn_points = self.map.get_spawn_points()
        #     spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        #     self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            
        #     self.show_vehicle_telemetry = False
        #     self.modify_vehicle_physics(self.player)
            
        # Set up the sensors.
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        
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

    def tick(self, clock, frame, display):
        end = self.hud.tick(self, clock, self.camera_manager, frame, display)
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
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor,
            self.collision_sensor.sensor
            
            ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
        print("collision ")


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
                    pass
                    # if self._autopilot_enabled:
                    #     world.player.set_autopilot(False)
                    #     world.restart()
                    #     world.player.set_autopilot(True)
                    # else:
                    #     world.restart()
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
    def __init__(self, width, height, distance=25.0, v_id=1):
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

    
                                           
    def tick(self, world, clock, camera, frame, display):
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

        # moving = False
        # acc = world.player.get_acceleration().length()

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
        
# ==============================================================================
# --BEV MAP Manager -------------------------------------------------------------
# ==============================================================================

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
        self.obstacle_ids = []
        self.obstacle_flag = False
        self.obstacle_target_point = Loc(x=-1, y=-1)
        self.destination = None
        self.obstacle_route_list = []
        
        
    # For data collection     
    def set_end_point(self, end_point):
        self.end_point = end_point
    
    def set_destination(self, destination):
        self.destination = destination
            
    def set_obstacle_ids(self, ids):
        self.obstacle_ids = ids

    def set_obstacle_route_list(self, obstacle_route):
        
        self.obstacle_route_list = obstacle_route
        print("set obstalce route list ")

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
        # policy_kwargs = cfg['policy']['kwargs']
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
        
    def _get_forward_speed(self, transform, velocity):
        """ Convert the vehicle transform directly to forward speed """

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def get_relative_transform(self, ego_matrix, vehicle_matrix):
        """
        return the relative transform from ego_pose to vehicle pose
        """
        relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
        rot = ego_matrix[:3, :3].T
        relative_pos = rot @ relative_pos
        
        # transform to right handed system
        relative_pos[1] = - relative_pos[1]

        # transform relative pos to virtual lidar system
        rot = np.eye(3)
        trans = - np.array([1.3, 0.0, 2.5])
        relative_pos = rot @ relative_pos + trans

        return relative_pos

    def normalize_angle(self, x):
        x = x % (2 * np.pi)  # force in range [0, 2 pi)
        if x > np.pi:  # move to [-pi, pi)
            x -= 2 * np.pi
        return x
        
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
            
            # for plant 
            vehicle_transform = actor.get_transform()
            vehicle_matrix = np.array(vehicle_transform.get_matrix())
            vehicle_extent = actor.bounding_box.extent
            dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.
            vehicle_rotation = vehicle_transform.rotation
            yaw =  vehicle_rotation.yaw/180*np.pi
            vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=actor.get_velocity()) # In m/s
            data[_id]["yaw"] = yaw
            data[_id]["foward_speed"] = vehicle_speed
            data[_id]["extent"] = [dx[2], dx[0], dx[1]]
            data[_id]["matrix"] = vehicle_matrix
            
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
            
            # for plant 
            vehicle_transform = actor.get_transform()
            vehicle_matrix = np.array(vehicle_transform.get_matrix())
            vehicle_extent = actor.bounding_box.extent
            dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.

            vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=actor.get_velocity()) # In m/s

            data[_id]["foward_speed"] = vehicle_speed
            data[_id]["extent"] = [dx[2], dx[0], dx[1]]
            data[_id]["matrix"] = vehicle_matrix
            
            
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
            data[_id]["state"] = 2 #traffic_light_state # set all traffic light to green
            data[_id]["cord_bounding_box"] = cord_bounding_box

        obstacle_id_list = []
        obstacle = world.world.get_actors().filter("*static.prop*")
        for actor in obstacle:
            _id = actor.id
            cord_bounding_box = {}
            actor_loc = actor.get_location()
            # print(actor_loc)
            distance = ego_loc.distance(actor_loc)
            bbox = actor.bounding_box
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1
            obstacle_id_list.append(_id)
            
            data[_id] = {}
            data[_id]["distance"] = distance
            data[_id]["type"] = "obstacle"
            data[_id]["cord_bounding_box"] = cord_bounding_box
            
            # for plant 
            vehicle_transform = actor.get_transform()
            vehicle_matrix = np.array(vehicle_transform.get_matrix())
            vehicle_extent = actor.bounding_box.extent
            dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.
            vehicle_rotation = vehicle_transform.rotation
            yaw =  vehicle_rotation.yaw/180*np.pi
            vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=actor.get_velocity()) # In m/s


            data[_id]["yaw"] = yaw
            data[_id]["foward_speed"] = vehicle_speed
            data[_id]["extent"] = [dx[2], dx[0], dx[1]]
            data[_id]["matrix"] = vehicle_matrix
            
            
            
        data["traffic_light_ids"] = traffic_id_list
        data["obstacle_ids"] = obstacle_id_list
        data["vehicles_ids"] = vehicles_id_list
        data["pedestrian_ids"] = pedestrian_id_list

        self.data = data
        return data

    def set_data(self, proccessed_data):
        self.data = proccessed_data

    def run_step(self, result, ego_id, current_route = None):
        
        # result["update_route"] = False

        actor_dict = copy.deepcopy(self.data)
        
        ego_pos = Loc(x=actor_dict[ego_id]["location"]["x"], y=actor_dict[ego_id]["location"]["y"])
        ego_yaw = actor_dict[ego_id]["rotation"]["yaw"]

        obstacle_bbox_list = []
        pedestrian_bbox_list = []
        vehicle_bbox_list = []
        agent_bbox_list = []

        r_traffic_light_list = []
        g_traffic_light_list = []
        y_traffic_light_list = []

        vehicle_id_list = list(actor_dict["vehicles_ids"])
        
        pedestrian_id_list = list(actor_dict["pedestrian_ids"])
        
        # obstacle_id_list = list(actor_dict["obstacle_ids"])
        
        traffic_light_id_list = list(actor_dict["traffic_light_ids"])

        for id in traffic_light_id_list:
            traffic_light_loc = Loc(x=actor_dict[id]["cord_bounding_box"]["cord_0"][0], y=actor_dict[id]["cord_bounding_box"]["cord_0"][1]) 
            if not check_close(ego_pos, traffic_light_loc, 45): #35.95m
                continue
            pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
            pos_1 = actor_dict[id]["cord_bounding_box"]["cord_1"]


            if actor_dict[id]["state"] == 0 :
                r_traffic_light_list.append([
                                        Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        ])
            elif actor_dict[id]["state"] == 2:
                g_traffic_light_list.append([
                                        Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        ])  
            elif actor_dict[id]["state"] == 1:
                y_traffic_light_list.append([
                                        Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        ]) 
            
        for id in self.obstacle_ids:#obstacle_id_list:
        
        # print(self.obstacle_ids)
        # print(obstacle_id_list)
        # print("--")
        
        # for id in obstacle_id_list:
            # print(id)
            
            # if actor_dict[id]['distance'] > 25:
            #     continue
            pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
            pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
            pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
            pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]
            obstacle_bbox_list.append([ Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
            
        for id in vehicle_id_list:
            
            if id in self.obstacle_ids:
                # check if vehicle is obstacle 
                continue
            
            traffic_light_loc = Loc(x=actor_dict[id]['location']['x'], y=actor_dict[id]['location']['y']) 
            # carla.Location(x=actor_dict[id]['location']['x'], y=actor_dict[id]['location']['y'])
            if not check_close(ego_pos, traffic_light_loc, 36): #35.95m
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

        route_list = current_route[:100]
        
        # ####### check if obsacle exist 
        
        # if not self.obstacle_flag:
        #     if len(self.obstacle_route_list) != 0:
                
        #         # print(len(self.obstacle_route_list))
        #         # Obstacle Exits 
                
        #         init_point = self.obstacle_route_list[0]
        #         end_point = self.obstacle_route_list[-1]
        
        #         desired_index = -1
        #         #for index in range(len(route_list)):
        #         for index in range(30):    
        #             route_point = route_list[index]
        #             dis =  distance.euclidean([route_point.x, route_point.y], [init_point[0], init_point[1]])
        #             if dis < 1.0:
        #                 desired_index = index
                        
        #         if desired_index != -1:
        #             # set obstacle target point 
        #             self.obstacle_target_point = Loc(x=end_point[0], y=end_point[1])    
        #             route_list = route_list[:desired_index]
                    
        #             # concate with obstacle route 
        #             for i in range(len(self.obstacle_route_list)):
        #                 loc = Loc(x=self.obstacle_route_list[i][0], y=self.obstacle_route_list[i][1])
        #                 route_list.append(loc)
        #             # print(len(self.obstacle_route_list))
        #             # print("concate route ")
                        
        #             route = self.planner.trace_route(carla.Location(route_list[-1].x, route_list[-1].y), self.destination )
        #             new_route = []
        #             for i in range(len(route)):
        #                 new_route.append(Loc(route[i][0].transform.location.x, route[i][0].transform.location.y))
        #             route_list += new_route
                    
        #             self.obstacle_flag = True
                    
        #             result["update_route"] = True
        #             # uppdte route 
                    
                        
        ####
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

        policy_input = {}
        policy_input['state'] = np.expand_dims([throttle, steer, brake, gear/5, vel_x, vel_y], 0).astype(np.float32)
        policy_input['birdview'] = np.expand_dims(roach_obs, 0)

        result["current_route"] = route_list 

        # return roach_obs
        result["bev_map_rgb"] = topdown[0:192, 56:248]
        result["policy_input"] = policy_input
        
        self.policy_forward(result, [policy_input])
        
    def policy_forward(self, result, policy_inputs_list):    
        control_elements_list = []
        for policy_input in policy_inputs_list:
            actions, values, log_probs, mu, sigma, features = self.policy.forward(
                policy_input, deterministic=True, clip_action=True)
                     
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
        
# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def check_close(ev_loc, loc0, distance = 3):
    
    dis = math.sqrt((ev_loc.x - loc0.x)**2 + (ev_loc.y - loc0.y)**2)
    if dis < distance:
        return True
    else:
        return False

def load_dict( filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def spawn_obstacle(scenario_obstacles, world, client):
    
    # TODOs: Find z from waypoints 
    obstacles_id_list = []
    for obstacle_info in scenario_obstacles:
        
        obstacle_type = obstacle_info['obstacle_type']
        pos = obstacle_info['pos']
        yaw = obstacle_info['yaw']
        
        obstacle_transform = carla.Transform(carla.Location(x=pos[0], y=pos[1], z=0.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=yaw))

        if obstacle_type == "illegal_parking":
            
            vehicle_blueprints = ["vehicle.tesla.cybertruck", "vehicle.nissan.patrol_2021", "vehicle.mini.cooper_s", "vehicle.ford.mustang", "vehicle.ford.crown", 
                                "vehicle.dodge.charger_police_2020", "vehicle.dodge.charger_police", "vehicle.citroen.c3", "vehicle.audi.etron"]
            bp_lib = world.world.get_blueprint_library()
            
            blueprint = random.choice(vehicle_blueprints)
            blueprint = bp_lib.find(blueprint) 
            
            obstacle_transform = carla.Transform(carla.Location(x=pos[0], y=pos[1], z=2.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=yaw))
        else:
            blueprint = world.world.get_blueprint_library().find(f"static.prop.{obstacle_type}")
        
        try:
            obstacle = client.get_world().spawn_actor(blueprint, obstacle_transform)
        
            id = obstacle.id
            obstacles_id_list.append(id)
            
        except:
            pass    
            
    return obstacles_id_list

def spawn_actor_nearby(world, client, spawn_points, agent_dict, route_list, ego_route_index, planner, args, processed_policy):
    
    num_of_vehicle = len(route_list) # -1 
    print(f"spawn {num_of_vehicle} actors")
    
    vehicles_list = []
    blueprints = world.world.get_blueprint_library().filter('vehicle.*')
    
    for i in range(num_of_vehicle):
        if i == ego_route_index:
            continue
        route = route_list[i]
            
        point1 = route[0]
        point2 = route[1]
        yaw = get_yaw(point1, point2)
        transform = carla.Transform(carla.Location(x=point1[0], y=point1[1], z=2.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=yaw))
        
        # blueprint = world.world.get_blueprint_library().find('vehicle*') 
                                                             #lincoln.mkz_2017')
        blueprint = random.choice(blueprints)
        
        try:
        
            # 
            
            agent = client.get_world().spawn_actor(blueprint, transform)
            
            
            id = agent.id
            vehicles_list.append(id)
            agent_dict[id] = {}
            agent_dict[id]["agent"] = agent
            agent_dict[id]["result"] = {}
            
            destination = random.choice(spawn_points).location
            current_route = planner.trace_route(carla.Location(route[-1][0], route[-1][1]), destination)
            
            agent_route = []
            for i in range(len(route)):
                loc = Loc(x=route[i][0], y=route[i][1])
                agent_route.append(loc)
                
            for i in range(len(current_route)):
                loc = Loc(x=current_route[i][0].transform.location.x, y=current_route[i][0].transform.location.y)
                agent_route.append(loc)
            
            agent_bev_map = BEV_MAP(args, world)
            agent_bev_map.init_vehicle_bbox(id)
            agent_bev_map.set_policy(processed_policy)
            agent_dict[id]["bev_map"] = agent_bev_map
            agent_dict[id]["route"] = agent_route
        except:
            pass
            
    return vehicles_list, agent_dict

def get_yaw(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    yaw_vector = (point2 - point1)
    vector = [1, 0]
    unit_vector_1 = yaw_vector / np.linalg.norm(yaw_vector)
    unit_vector_2 = vector / np.linalg.norm(vector)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    if yaw_vector[1] < 0:
        yaw = -math.degrees(angle)
    else:
        yaw = math.degrees(angle)
    return yaw

def game_loop(args):
    
    pygame.init()
    pygame.font.init()
    world = None

    # scenario     
    obstacle_data = load_dict(f"./obstacle_scenarios/{args.town}.pkl")
    print(len(obstacle_data))
    print("")
    
    obstacle_scneario_index = 22
    ego_route_index = 1
    
    # SAVING FLAG
    saving_flag = False
    saving_flag = True
        
    use_threshold = False
    # use_threshold = True
    
    scenario_info = obstacle_data[obstacle_scneario_index]
    ego_counter_threshold_min = 40
    ego_counter_threshold_max = 50
    obstacle_route_list = scenario_info[2] 
    obstacle_info = scenario_info[1] # All obstacle type and transforms  
    # get ego transform 
    # get other vehicle transform 
    # Ego vehicle take the first route 
    point1 = obstacle_route_list[ego_route_index][0]
    point2 = obstacle_route_list[ego_route_index][1]
    yaw = get_yaw(point1, point2)
    
    ego_transform = carla.Transform(carla.Location(x=point1[0], y=point1[1], z=3.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=yaw))
    # ego_transform = carla.Transform(carla.Location(x=point1[0], y=point1[1], z=5.0), carla.Rotation(roll=0.0, pitch=0.0, yaw=yaw))

    
    ############################################################
    
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

    hud = HUD(args.width, args.height, args.distance)
    world = World(client.load_world(args.town), hud, args, ego_transform)

    settings = world.world.get_settings()
    settings.fixed_delta_seconds = 0.05
    settings.synchronous_mode = True  # Enables synchronous mode
    world.world.apply_settings(settings)
    controller = KeyboardControl(world, args.autopilot)
    
    map = world.world.get_map()
    spawn_points = map.get_spawn_points()
    waypoint_list = []
    for waypoint in spawn_points:            
        waypoint_list.append(waypoint)
        
    planner = GlobalRoutePlanner(map, resolution=1.0)
    clock = pygame.time.Clock()

    #### initialize the ego player ####
    result = {}
    player_bev_map = BEV_MAP(args, world)
    processed_policy = player_bev_map.init_policy()
    player_bev_map.init_vehicle_bbox(world.player.id)
    player_bev_map.init_spawn_points_and_planner(map.get_spawn_points(), planner)
    
    destination = random.choice(spawn_points).location
    current_route = planner.trace_route(carla.Location(obstacle_route_list[ego_route_index][-1][0], obstacle_route_list[ego_route_index][-1][1]), destination)
    player_bev_map.set_destination(destination)
    player_bev_map.set_end_point( Loc(x=obstacle_route_list[ego_route_index][-1][0], y=obstacle_route_list[ego_route_index][-1][1]))
    
    ego_route = []
    for i in range(len(obstacle_route_list[ego_route_index])):
        loc = Loc(x=obstacle_route_list[ego_route_index][i][0], y=obstacle_route_list[ego_route_index][i][1])
        ego_route.append(loc)
        
    for i in range(len(current_route)):
        loc = Loc(x=current_route[i][0].transform.location.x, y=current_route[i][0].transform.location.y)
        ego_route.append(loc)
        
        
    ####################################################################################################
    # spawn other vehicle
    vehicles_list = []
    agent_dict = {} 
    
    vehicles_list, agent_dict = spawn_actor_nearby(world, client, spawn_points, agent_dict, obstacle_route_list, ego_route_index, planner, args, processed_policy)
        
        
    ####################################################################################################
    # spawn obstacles 
    
    obstacles_ids = spawn_obstacle(obstacle_info, world, client)
    print("spawn obstacle ")
    # all spawn fails
    if len(obstacles_ids) == 0:
        print( "Obstacle spawn fail")
    
    ## set obstacle id 
    player_bev_map.set_obstacle_ids(obstacles_ids)  
    for id in vehicles_list:
        agent_dict[id]["bev_map"].set_obstacle_ids(obstacles_ids)  
            
    ####################################################################################################
    
    start_frame = world.world.tick()
    
    
    # For data collection     
    # if os.path.exists("./dataset/"):
    current_dataset = os.listdir("./dataset")  
    # filter by Town
    current_dataset = list(filter(lambda x: (args.town in x), current_dataset))  
    
    n = len(current_dataset) + 1
    saving_path = f"./dataset/{args.town}_obstacle_{n}"
    if saving_flag:
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

    world.world.tick()
    
    
    counter_for_ego = 0
    while True:
        
        random.seed(time.time())
        random.shuffle(waypoint_list)
        
        clock.tick_busy_loop(20)
        frame = world.world.tick()
        if controller.parse_events(client, world, clock):
            return
        view = pygame.surfarray.array3d(display)
        view = view.transpose([1, 0, 2]) 
        image = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        world.tick(clock, frame, image)

        ego_current_location = world.player.get_location()
        ego_current_location = Loc(x=ego_current_location.x, y=ego_current_location.y)
        
        # Deal with ego_route
        # Delete reached points from current route
        while check_close(ego_current_location, ego_route[0]):
            ego_route.pop(0)
    
        # Collect data for all agents
        processed_data = player_bev_map.collect_actor_data(world)

        ###################Ego vehicle control#####################
        player_bev_map.run_step(result, world.player.id, ego_route)
        bev_map_rgb = result["bev_map_rgb"] # to vis
        # if result["update_route"]:
        #     ego_route = result["current_route"]

        control_elements_list = result["control_elements_list"]
        # for render
        surface = pygame.surfarray.make_surface(bev_map_rgb)
        surface = pygame.transform.flip(surface, True, False)
        surface = pygame.transform.rotate(surface, 90)
        display.blit(surface, (256, 0))

        control_elements = control_elements_list[0]
        control = carla.VehicleControl(throttle=control_elements['throttle'], steer=control_elements['steer'], brake=control_elements['brake'])
        
        if use_threshold and  counter_for_ego < ego_counter_threshold_max and  counter_for_ego > ego_counter_threshold_min:  
            control = carla.VehicleControl(throttle=0, steer=control_elements['steer'], brake=1)
            
        counter_for_ego +=1
        
        world.player.apply_control( control )
        ###################Ego car control#####################

        ###################Other vehicle control#####################
        
        for _, id in enumerate(vehicles_list):
            agent_current_location = agent_dict[id]['agent'].get_location()
            agent_current_location = Loc(x=agent_current_location.x, y=agent_current_location.y)

            while check_close(agent_current_location, agent_dict[id]["route"][0]):
                agent_dict[id]["route"].pop(0)

            agent_dict[id]["bev_map"].set_data(processed_data)
            agent_dict[id]["bev_map"].run_step(agent_dict[id]["result"], id, agent_dict[id]["route"]) 
            
            # if agent_dict[id]["result"]["update_route"]:
            #     agent_dict[id]["route"] = agent_dict[id]["result"]["current_route"]
            
            control_elements_list = agent_dict[id]["result"]["control_elements_list"]
            control_elements = control_elements_list[0]
            control = carla.VehicleControl(throttle=control_elements['throttle'], steer=control_elements['steer'], brake=control_elements['brake'])
            agent_dict[id]["agent"].apply_control(control)
            
        # for render 
        for i, id in enumerate(vehicles_list):
            if i == 3 :
                break
            bev_map_rgb = agent_dict[id]["result"]["bev_map_rgb"]
            surface = pygame.surfarray.make_surface(bev_map_rgb)
            surface = pygame.transform.flip(surface, True, False)
            surface = pygame.transform.rotate(surface, 90)
            display.blit(surface, (256, 210*(i+1)))  
        
            
        # DATA saving        
        if saving_flag :
            route_list = []
            for wp in ego_route:
                route_list.append([wp.x, wp.y])
            processed_data["ego_id"] = world.player.id 
            processed_data["ego_route"] = route_list
            processed_data["obstacle_ids"] = player_bev_map.obstacle_ids # id including illege parking vehicles 

            current_frame = frame - start_frame
            np.savez_compressed( f"{saving_path}/" + "%04d" % current_frame, np.array(processed_data), allow_pickle=True) # store with npz in compressed format will save much of space on disk
        
        
        world.hud.render(display)
        pygame.display.flip()
        
        if check_close(ego_current_location, player_bev_map.end_point , distance=2):
            break

    # destroy obstacles 
    client.apply_batch([carla.command.DestroyActor(x) for x in obstacles_ids])
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    print("Destory obstacles and other vehicles ")
    
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

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()