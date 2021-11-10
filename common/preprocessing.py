import cv2
import numpy as np


def preprocessing(obs, info):
    # convert to grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    # resize to [40,40]
    obs = cv2.resize(obs, (40, 40), interpolation=cv2.INTER_AREA)
    # add new axis to [1,40,40]
    obs = obs[np.newaxis, :]
    # extract values
    info = np.array(list(info.values()))

    info = info / 360  # Doesn't converge without this
    obs = obs / 255  # Doesn't converge without this
    return obs, info
