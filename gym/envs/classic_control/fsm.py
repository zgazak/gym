"""
Place a spot in a CCD field and maintain the position of that spot
against external perturbations
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt


def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


class FSMEnv(gym.Env):
    """
    Description:
        A spot (target on CCD) position is perturbed, goal is to maintain position of spot.

    Source:
        novel

    Observation:
        Type: numpy array (row, col, channel)
        each element simulates a CCD pixel.  A spot is placed on the pixel

    Actions:
        Type: continuous(2)
        Num   Action
        0     Apply X directional (voltage) to FSM
        1     Apply Y directional (voltage) to FSM

        Note: 0, 0 means returning the input observation, while [-1, 1] would shift the FSM
              mirror to it's extreme bottom (x) and left (Y)

    Reward:
        Reward is 1./distance, where distance is the geometric distance between where
        the action puts the spot and where the spot is supposed to be (center)

    Starting State:
        Spot is placed slightly off center with no external force.

    Termination:
        Spot is moved off frame
        Episode length is greater than 500
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 25,
    }

    def __init__(self, ccdx=500, ccdy=500, Vmin=-1, Vmax=1, noise_sig=1, obj_sig=10, bkg_level=50):
        self.wind_fcn = None
        self.noise_fcn = None
        self.target_xy = [0.5 * ccdx, 0.5 * ccdy]  # ideally, spot centered
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.ccd_shape = (ccdy, ccdx)

        self.noise_sig = noise_sig
        self.bkg_level = bkg_level
        self.obj_sig = obj_sig

        self.action_space = spaces.Box(low=self.Vmin, high=self.Vmax, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.ccd_shape, dtype=np.uint8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state_xy = self.np_random.uniform(low=0, high=min(self.ccd_shape[0:1]), size=(2,))
        self.done = False

    def render_ccd(self):
        noise_frame = self.noise_sig * self.np_random.normal(size=self.ccd_shape) + self.background_level

        x = np.linspace(0, self.ccd_shape[1] - 1, num=self.ccd_shape[1])
        y = np.linspace(0, self.ccd_shape[0] - 1, num=self.ccd_shape[0])
        x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D
        spot_frame = gaus2d(x, y, mx=self.state_xy[1], my=self.state_xy[0])
        spot_frame /= np.max(spot_frame)
        spot_frame *= self.obj_sig

        self.state = spot_frame + noise_frame


    def render(self, mode='human'):
        plt.imshow(self.observation)
