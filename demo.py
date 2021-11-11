import cv2

from common.environment import Environment


class MyRaceTrack(Environment):
    def __init__(self):
        super().__init__()
        self.debug = True

    def run(self):
        while not self.crashed:
            action = self.policy()

            obs, reward, done, info = self.step(action)

            print(reward, info)

            if done:
                self.reset()
            cv2.imshow("Observation Space", obs)
            cv2.waitKey(1)

        self.close()


if __name__ == "__main__":
    env = MyRaceTrack()
    env.run()
