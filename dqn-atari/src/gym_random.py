#!/usr/bin/env python

import random

import gym
import numpy as np
import PIL.Image
import PIL.ImageTk
import tkinter as tk


RANDOM_SEED = 0
CANVAS_WIDTH = 480
CANVAS_HEIGHT = 630

env = gym.make('MsPacman-v0')
env.seed(RANDOM_SEED)
env.reset()

root = tk.Tk()
root.title('ms. pacman')
root.geometry('{0}x{1}'.format(CANVAS_WIDTH, CANVAS_HEIGHT))
root.configure(background='black')

label = tk.Label(root)
label.pack(side='bottom', fill='both', expand='yes')


game_state = []
episode_frames = 0
total_reward = 0
current_action = 0


def execute(_):
    global game_state
    global episode_frames
    global total_reward
    global current_action

    if len(game_state) == 4:
        #predict_actions = model.predict_on_batch(np.array([game_state]) / 255)[0]
        #current_action = np.argmax(predict_actions)
        #print(current_action, predict_actions)
        current_action = random.randrange(env.action_space.n)

    observation, reward, game_over, _ = env.step(current_action)
    image = (np
             .dot(observation, [0.2989, 0.5870, 0.1140])
             .reshape(observation.shape[:2])
             [5:165:2,::2])
    game_state = game_state[-3:] + [image]
    reward = np.clip(reward, -1., 1.)
    total_reward += reward

    image = (PIL.Image
             #.fromarray(observation, mode='RGB')
             .fromarray(image.astype('uint8'), mode='L')
             .resize((CANVAS_WIDTH, CANVAS_HEIGHT)))

    label.image = PIL.ImageTk.PhotoImage(image)
    label.configure(image=label.image)

    episode_frames += 1

    if game_over:
        print("Episode ended with score: {0}".format(total_reward))
        episode_frames = 0
        total_reward = 0
        env.reset()

    root.after(1, execute, None)


root.after(500, execute, None)
tk.mainloop()
