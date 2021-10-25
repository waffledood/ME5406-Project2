import sys

import cv2
import numpy as np
import pygame

sys.path.append("../")
from environment import Environment


class MyRaceTrack(Environment):
    def __init__(self):
        super().__init__()
        self.debug = False
        self.car = pygame.image.load("racecar.png")

    def is_done(self):
        """
        Returns True if agent falls went out of track or if the agent is moving in the opposite direction
        """
        pixel = self.race_track[int(self.y), int(self.x)]
        if np.sum(pixel) == 0 or np.abs(self.d_angle) >= 120 or np.abs(self.d_center) > 125:
            # if np.sum(pixel) == 0:
            return True
        else:
            return False

    def compute_reward(self):
        """
        Rewards the agent
        -0.1 + pixels covered in direction
        TODO: Override this method to customize reward
        """
        pt2 = np.array([self.x, self.y])
        pt1 = self.ckpt
        v = pt2 - pt1

        # projects vector onto line segment
        uv = self.vectors[self.ckpt_idx]
        d = np.dot(v, uv) / np.dot(uv, uv)
        pt3 = pt1 + uv * d

        # computes distance from center lane
        self.d_center = np.linalg.norm(pt2 - pt3)

        # computes angle from lane direction
        d_angle = self.angle - self.angles[self.ckpt_idx]
        d_angle = (d_angle + np.pi) % (2 * np.pi) - np.pi
        self.d_angle = d_angle / np.pi * 180

        # tracks the last covered distance
        dx = d - self.prev_d
        self.prev_d = d

        reward = -0.1 + dx

        if self.debug:
            self.race_track = cv2.circle(
                self.race_track, (int(pt3[0]), int(pt3[1])), 5, (255, 0, 0), -1
            )

        next_ckpt_idx = (self.ckpt_idx + 1) % len(self.checkpoints)
        next_ckpt = self.checkpoints[next_ckpt_idx]
        d_to_goal = np.linalg.norm(next_ckpt - pt3)
        # Note that euclidean distance does not work because the coverage smaller than the lane thickness
        if d_to_goal < self.min_d_to_goal:
            self.ckpt_idx += 1
            self.ckpt_idx %= len(self.checkpoints)
            self.prev_d = 0

            # Move the actual checkpoint back to prevent negative rewards
            self.ckpt = next_ckpt

        if np.abs(self.d_center) < 10:
            reward = reward
        elif np.abs(self.d_center) < 20:
            reward = 0.9 * reward
        elif np.abs(self.d_center) < 30:
            reward = 0.8 * reward
        elif np.abs(self.d_center) < 40:
            reward = 0.7 * reward
        elif np.abs(self.d_center) < 50:
            reward = 0.6 * reward
        elif np.abs(self.d_center) < 60:
            reward = 0.5 * reward
        elif np.abs(self.d_center) < 70:
            reward = 0.4 * reward
        elif np.abs(self.d_center) < 80:
            reward = 0.3 * reward
        elif np.abs(self.d_center) < 90:
            reward = 0.2 * reward
        elif np.abs(self.d_center) < 100:
            reward = 0.1 * reward
        else:
            reward = -0.01
        return reward
