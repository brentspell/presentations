import threading
import time

from PIL import Image, ImageTk
import numpy as np
import gym
import tkinter as tk

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
from keras.utils import plot_model
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import Callback



INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
CANVAS_WIDTH = 480
CANVAS_HEIGHT = 630


root = tk.Tk()
root.title('ms. pacman')
root.geometry('{0}x{1}'.format(CANVAS_WIDTH, CANVAS_HEIGHT))
root.configure(background='black')

label = tk.Label(root)
label.pack(side='bottom', fill='both', expand='yes')

last_observation = None
total_frames = 0
total_rewards = 0


class AtariProcessor(Processor):
    def process_observation(self, observation):
        global total_frames, total_rewards
        global last_observation
        total_frames += 1
        last_observation = observation
        time.sleep(0.02)
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        global total_frames, total_rewards
        if reward:
            total_rewards += 1
        return np.clip(reward, -1., 1.)


def simulator():
    # Get the environment and extract the number of actions.
    env = gym.make('MsPacman-v0')
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build our model. We use the same model that was described by Mnih et al. (2015).
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = Sequential()
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    plot_model(model, to_file='/Users/brent/tmp/test.png')


    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    # The trade-off between exploration and exploitation is difficult and an on-going research topic.
    # If you want, you can experiment with the parameters or use a different policy. Another popular one
    # is Boltzmann-style exploration:
    # policy = BoltzmannQPolicy(tau=1.)
    # Feel free to give it a try!

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    weights_filename = 'dqn_weights.h5f'
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=1000000, visualize=False)


thread = threading.Thread(target=simulator)
thread.start()


def renderer(*_):
    if last_observation is not None:
        observation = (np
                       .dot(last_observation, [0.2989, 0.5870, 0.1140])
                       .reshape(last_observation.shape[:2])
                       .astype('uint8')
                       [5:165:2,::2])
        image = (Image
                 .fromarray(last_observation, mode='RGB')
                 #.fromarray(observation, mode='L')
                 .resize((CANVAS_WIDTH, CANVAS_HEIGHT)))

        label.image = ImageTk.PhotoImage(image)
        label.configure(image=label.image)

    root.after(1, renderer, None)


root.after(1, renderer, None)
tk.mainloop()
