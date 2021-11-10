import os

import cv2
import numpy as np
import pygame

from race_track import create_map

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
WHITE = (255, 255, 255)

# Uncomment this to run headless
# os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.display.set_caption("ME5406 Race Track")


class Environment:
    def __init__(self):
        # For game settings
        self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.map_size = 4000
        self.map_padding = 1000 * 3
        self.lane_thickness = 250
        self.car = pygame.image.load("racecar.png")
        self.fps = 60
        self.dt = 1 / self.fps
        self.debug = False

        # For setting car in the middle of screen
        self.car_x = SCREEN_WIDTH // 2  # DO NOT CHANGE
        self.car_y = SCREEN_HEIGHT // 2  # DO NOT CHANGE

        # For end of episode
        self.crashed = False

        # For keyboard inputs
        self.is_up = False
        self.is_down = False
        self.is_left = False
        self.is_right = False

        # For controlling car
        self.x = 1000
        self.y = 1000
        self.angle = 0
        self.v = 0
        self.gas_a = 20
        self.brake_a = 50
        self.friction_a = 5
        self.max_v = 10
        self.min_v = 0
        self.min_d_to_goal = 50
        self.d_center = 0
        self.d_angle = 0

        # For computing reward
        self.prev_d = 0
        self.ckpt_idx = 0
        self.ckpt = None

        # For race track
        self.start_x = 1000
        self.start_y = 1000
        self.start_angle = 0
        self.vectors = []
        self.checkpoints = []
        self.angles = []
        self.generate_race_track()

    def generate_race_track(self):
        """
        Returns randomly generated race track
        """
        map_info = create_map(self.map_size, self.map_padding, self.lane_thickness)
        race_track, start_point, angle, vectors, checkpoints, angles = map_info
        self.ckpt_idx = 0
        self.race_track = race_track
        self.x = self.start_x = start_point[0]
        self.y = self.start_y = start_point[1]
        self.angle = self.start_angle = angle
        self.vectors = vectors
        self.checkpoints = checkpoints
        self.angles = angles
        self.ckpt = checkpoints[self.ckpt_idx]

    def radian_to_degree(self, radian):
        """
        Converts angle from radian to degree
        """
        return radian / np.pi * 180

    def degree_to_radian(self, degree):
        return degree / 180 * np.pi

    def capture_frame(self):
        """
        Returns frame in RGB format
        """
        screen = pygame.display.get_surface()
        capture = pygame.surfarray.pixels3d(screen)
        capture = capture.transpose([1, 0, 2])
        return capture

    def rotate_image(self, image, angle):
        """
        Rotates image along 2D plane
        """
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def observation_space(self, angle):
        """
        Returns observation space in front of the car
        """
        color_image = self.capture_frame()
        color_image = self.rotate_image(color_image, self.radian_to_degree(-angle))
        color_image = color_image[
            self.car_x : self.car_x + 400, self.car_y - 200 : self.car_y + 200, :
        ]
        color_image = cv2.rotate(color_image, cv2.ROTATE_180)
        return color_image

    def render_top_view(self, x, y):
        """
        Renders top view of racetrack
        """
        int_y, int_x = int(y), int(x)
        img = self.race_track[
            int_y - self.car_x : int_y + self.car_x, int_x - self.car_y : int_x + self.car_y
        ]
        self.display.fill(WHITE)
        surf = pygame.surfarray.make_surface(img)
        self.display.blit(surf, (0, 0))

    def render_car(self, x, y, angle):
        """
        Renders car in middle of the screen
        """
        rotated_image = pygame.transform.rotate(self.car.copy(), angle / np.pi * 180 + 180)
        h, w = rotated_image.get_rect().bottomright
        # computes offsets to positions object center instead of top left corner
        x_offset, y_offset = x - h // 2, y - w // 2
        self.display.blit(rotated_image, (x_offset, y_offset))
        if self.debug:
            pygame.draw.circle(self.display, (255, 0, 0), (self.car_x, self.car_y), 5)
            pygame.draw.rect(
                self.display, (255, 0, 0), (x_offset, y_offset, *rotated_image.get_size()), 2
            )

    def render(self, x, y, angle):
        """
        Renders car and racetrack
        """
        self.render_top_view(x, y)
        self.render_car(self.car_x, self.car_y, angle)

    def keyboard_control(self):
        """
        Controls car via keyboard
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.crashed = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    self.is_right = True
                elif event.key == pygame.K_LEFT:
                    self.is_left = True
                elif event.key == pygame.K_DOWN:
                    self.is_down = True
                elif event.key == pygame.K_UP:
                    self.is_up = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_RIGHT:
                    self.is_right = False
                elif event.key == pygame.K_LEFT:
                    self.is_left = False
                elif event.key == pygame.K_DOWN:
                    self.is_down = False
                elif event.key == pygame.K_UP:
                    self.is_up = False

    def compute_car_pos(self, x, y, angle, action):
        """
        Computes the car positions
        """
        gas_a, brake_a, w = action

        if gas_a:  # gas acceleration
            self.v += self.gas_a * self.dt
            self.v = np.min([self.max_v, self.v])
        if brake_a:  # brake acceleration
            self.v -= self.brake_a * self.dt
            self.v = np.max([self.min_v, self.v])
        # friction acceleration
        self.v -= self.friction_a * self.dt
        self.v = np.max([self.min_v, self.v])

        # prevents turning on the spot if velocity is too low
        if np.abs(self.v) > 0.01:
            angle += w

        x += self.v * np.cos(angle)
        y += self.v * np.sin(angle)
        return x, y, angle

    def policy(self):
        """
        Returns actions from keyboard input
        TODO: Override this method to customize policy
        """
        self.keyboard_control()
        gas_a = 1 if self.is_up else 0
        brake_a = 1 if self.is_down else 0
        w = (
            self.degree_to_radian(1)
            if self.is_left
            else -self.degree_to_radian(1)
            if self.is_right
            else 0
        )
        action = [gas_a, brake_a, w]
        return action

    def is_done(self):
        """
        Returns True if agent falls went out of track or if the agent is moving in the opposite direction
        """
        pixel = self.race_track[int(self.y), int(self.x)]
        if np.sum(pixel) == 0 or self.d_angle >= 120 or self.d_angle <= -120:
        #if np.sum(pixel) == 0:
            return True
        else:
            return False

    def reset(self):
        """
        Resets the agent
        """
        self.generate_race_track()
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.v = 0
        self.prev_d = 0
        self.ckpt_idx = 0
        self.ckpt = self.checkpoints[self.ckpt_idx]
        self.render(self.x, self.y, self.angle)
        pygame.display.update()
        self.clock.tick(self.fps)
        obs = self.observation_space(self.angle)
        info = {
            # "x": self.x,
            # "y": self.y,
            # "angle": self.angle,
            "velocity": self.v,
            "d_center": self.d_center,
            "d_angle": self.d_angle,
        }
        return obs, info

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
		
        if reward > 0:
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
                reward = 0.05 * reward


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

        return reward

    def step(self, action):
        """
        Handles the agent action per frame
        """
        self.x, self.y, self.angle = self.compute_car_pos(self.x, self.y, self.angle, action)
        self.render(self.x, self.y, self.angle)
        pygame.display.update()
        self.clock.tick(self.fps)

        obs = self.observation_space(self.angle)
        done = self.is_done()
        reward = self.compute_reward()
        reward = -1000 if done else reward
        info = {
            # "x": self.x,
            # "y": self.y,
            # "angle": self.angle,
            "velocity": self.v,
            "d_center": self.d_center,
            "d_angle": self.d_angle,
        }
        return obs, reward, done, info

    def run(self):
        """
        Runs the game engine
        TODO: Runs episode in this format
        """
        while not self.crashed:
            action = self.policy()

            obs, reward, done, info = self.step(action)

            # print(reward, info)

            if done:
                self.reset()
            cv2.imshow("Observation Space", obs)
            cv2.waitKey(1)

        self.close()

    def close(self):
        """
        Closes the game
        """
        pygame.quit()
        #quit()


if __name__ == "__main__":
    env = Environment()
    env.run()