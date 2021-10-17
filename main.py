import cv2

from environment import Environment


class MyRaceTrack(Environment):
    def __init__(self):
        super().__init__()
        self.debug = False

    def run(self):
        while not self.crashed:
            action = self.policy()

            obs, reward, done, info = self.step(action)

            print("d_center", info["d_center"])

            if done:
                self.reset()
            cv2.imshow("Observation Space", obs)
            cv2.waitKey(1)

        self.close()


if __name__ == "__main__":
    env = MyRaceTrack()
    env.run()
