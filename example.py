#!/bin/python
from utils import resize_image, XboxController
from termcolor import cprint

import gym, gym_mupen64plus
from train import create_model
import numpy as np

power = 0

class Actor(object):
    def __init__(self):
        # Load in model from train.py and load in the trained weights
        self.model = create_model(keep_prob=1) # no dropout
        self.model.load_weights('gym-mupen64plus/model_weights.h5')

        cprint(self.model.summary(), 'green')

        # Init contoller for manual override
        #self.real_controller = XboxController()

    def get_action(self, obs):

        ## Look
        vec = resize_image(obs)
        vec = np.expand_dims(vec, axis=0) # expand dimensions for pree:\Development\wrkshpz\03-2024\TensorKart\model_weights.h5dict, it wants (1,66,200,3) not (66, 200, 3)
        ## Think
        joystick = self.model.predict(vec, batch_size=1)[0]
       

        ## Act

        ### calibration
        output = [
            int(joystick[0] * 80),
            int(joystick[1] * 80),
           1,
            int(round(joystick[3])),
            int(round(joystick[4])),
        ]

        ### print to console
        cprint("AI: " + str(output), 'green')

        return output
    

env = gym.make('Mario-Kart-Luigi-Raceway-v0')
env.reset()

actor = Actor()
print('actor ready!')

print("NOOP waiting for green light")
for i in range(18):
    (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light

end_episode = False
while not end_episode:
    action = actor.get_action(obs)

    (obs, rew, end, info) = env.step(action) # Drive as model predicts

raw_input("Press <enter> to exit... ")

env.close()
