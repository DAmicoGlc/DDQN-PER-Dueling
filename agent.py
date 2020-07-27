import os
import imageio
import random
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

from sumtree import SumTree

from gym.envs.classic_control import rendering
from PIL import Image
import time

# Initilize radomness
random.seed(1)
np.random.seed(1)

# Network parameters
WIDTH = 84                      # Width of the images feeded to the network
HEIGHT = 84                     # Height of the images feeded to the network
NUM_TRAINING_STEPS = 4000000    # Number of steps for the training mode
NUM_TESTING_EPISODES = 30       # Number of episode for the testing mode
STACKED_FRAME = 4               # Number of consecutive frame to create the input of the network
GAMMA = 0.99                    # Discount Factor
EXPLORATION_STEPS = 1000000     # Number of steps in which the agent perform exploration
INITIAL_EPSILON = 1.0           # Starting value of epsilon
FINAL_EPSILON = 0.01            # Final value of espsilon
INITIAL_MEMORY_SIZE = 20000     # Starting number of experienced inserted in the memory
MEMORY_SIZE = 40000             # Max number of experiences saved in the memory
BATCH_DIM = 32                  # Mini batch size
TARGET_UPDATE_FREQUENCY = 1000  # Numer of steps between two update of the target network
ONLINE_UPDATE_FREQUENCY = 4     # Number of action between two update of the online network
SAVE_FREQUENCY = 500000         # Number of step between two save of the model weights

LEARNING_RATE = 0.0001          # Learning rate for RMSProp
MIN_GRADIENT = 0.001            # Constant to squared gradient in RMSProp
MOMENTUM = 0.95                 # Momentum for RMSProp
NO_OP_STEPS = 5                 # Max number of no op steps perfomed at start of each episodes

ALPHA_MEMORY = 0.6              # Trade-off between drawn experience base on priority and drawn experience randomly
EPSILON_MEMORY = 0.01           # Value added to do not have experience with 0 probabilty to be drawn
BETA_MEMORY = 0.4               # Importance sampling factor
BETA_GROWTH = 0.0001            # Step increment of beta for each sampling

class AgentDDQN:
    """
        Agent of the environment

        Input:
            env: environment information such as number of action or states dimension
            args: Indicate which type of DDQN must be run and which type of optimizer
    """
    def __init__(self, env, args):
        # State parameters
        self.width = WIDTH
        self.height = HEIGHT
        self.stacked_frame = STACKED_FRAME  # number of stacked frame
        if "SpaceInvaders" in args.game:
            self.stacked_frame = 3

        # Episode parameters
        self.max_steps = NUM_TRAINING_STEPS  # max steps for the whole game in training mode
        self.steps_counter = 0

        # Epsilon parameters
        self.explore_steps = EXPLORATION_STEPS                                  # number of steps in Exploration mode
        self.init_eps = INITIAL_EPSILON                                         # starting epsilon value
        self.final_eps = FINAL_EPSILON                                          # final epsilon value
        self.eps_decay = (self.init_eps - self.final_eps)/self.explore_steps    # deacrising epsilon factor

        # DDqn hyperparameters
        self.gamma = GAMMA
        self.epsilon = self.init_eps

        self.init_memory_steps = INITIAL_MEMORY_SIZE    # number of steps to fill the memory at the beginning
        self.memory_count = 0                           # acutal number of experiences in the memory
        self.max_memory_size = MEMORY_SIZE              # max number of experiences saved in the memory

        self.batch_dim = BATCH_DIM
        self.target_update_frequency = TARGET_UPDATE_FREQUENCY  # steps between target network update
        self.online_update_frequency = ONLINE_UPDATE_FREQUENCY    # number of actions between two training steps of the q network

        # Dqqn definition
        self.dueling = args.dueling
        self.PER = args.PER

        self.optimizer = keras.optimizers.RMSprop(lr=LEARNING_RATE, decay=0, rho=0.99, epsilon=MIN_GRADIENT)

        # Environment parameters
        self.env = env
        self.num_actions = self.env.action_space.n
        self.no_op_num = NO_OP_STEPS                # Number of steps with no operation at the beginning
        self.render = args.render

        # Memory initialization
        if self.PER:
            self.a_memory = ALPHA_MEMORY
            self.b_memory = BETA_MEMORY
            self.b_growth = BETA_GROWTH
            self.e_memory = EPSILON_MEMORY
            self.replay_memory = SumTree(self.max_memory_size)
        else:
            self.replay_memory = deque()

        # TODO add parameters for summary and saving
        self.total_reward = 0.0
        self.total_loss = 0.0
        self.duration = 0
        self.episode = 0
        self.last_30_reward = deque()
        self.time_start = time.clock()

        self.save_network_path = 'model/'
        self.save_summary_path = 'summary/'
        if not os.path.exists(self.save_network_path):
            os.makedirs(self.save_network_path)
        if not os.path.exists(self.save_summary_path):
            os.makedirs(self.save_summary_path)

        # Creates double network
        self.online_net = self.build_net()
        self.target_net = self.build_net()

        self.save_frequency = SAVE_FREQUENCY

        # load model for testing, train a new one otherwise
        if args.mode == "Testing":
            self.test_dqn_model_path = args.testing_path
            self.online_net.load_weights(self.test_dqn_model_path)
        else:
            self.log = open(self.save_summary_path+'game.log','w')

        self.target_net.set_weights(self.online_net.get_weights())

    def build_net(self):
        """
            Build the network

            Return:
                model: builded network
        """
        input_layer = keras.Input(shape=(self.width, self.height, self.stacked_frame))

        conv1 = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_layer)
        conv2 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)

        flatten = layers.Flatten()(conv3)

        if self.dueling:
            # Q = V + (A - (mean of A))
            dense1 = layers.Dense(512, activation='relu')(flatten)
            action_advantage = layers.Dense(self.num_actions)(dense1)
            action_advantage = layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(self.num_actions,))(action_advantage)            

            dense2 = layers.Dense(1, activation='relu')(flatten)
            state_value = layers.Dense(1)(dense2)
            state_value = layers.Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.num_actions,))(state_value)
            
            output_layer = layers.Add()([state_value, action_advantage])
        else:
            dense = layers.Dense(512, activation='relu')(flatten)

            output_layer = layers.Dense(self.num_actions)(dense)

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer=self.optimizer)

        return model

    def test_loop(self):
        """
            Main testing loop
        """
        episodes = 30
        viewer = rendering.SimpleImageViewer()
        
        game = 0

        episode_images = []
        game_reward = []

        episode_reward = 0
        episode_images.append([])

        state = self.env.reset()

        for i in range(episodes*self.env.lives):
            terminal = False

            # Let's play one game
            while(not terminal):
                action = self.act(state, act_mode="test")
                state, reward, terminal, info = self.env.step(action)

                episode_reward += reward

                rgb_array = self.env.render('rgb_array')
                episode_images[game].append(Image.fromarray(rgb_array))

                upscaled = self.repeat_upsample(rgb_array, 3, 3)
                # plot using the default viewer
                viewer.imshow(upscaled)

            if self.env.lives == 0:
                game_reward.append(episode_reward)
                episode_reward = 0

                game += 1
                episode_images.append([])

            state = self.env.reset()

        # Save a GIF
        episode_images[np.argmax(game_reward)][0].save('summary/game.gif', save_all=True, append_images=episode_images[np.argmax(game_reward)][1:], duration=20, loop=0)

        print('Run episodes: ', episodes)
        print('Mean:', np.mean(game_reward))
        viewer.close()

    def train_loop(self):
        """
            Main training loop
        """
        # Until the max steps are reached
        while self.steps_counter <= self.max_steps:
            # Extract the first state e set terminal to False
            terminal = False
            state = self.env.reset()

            # Make a random no_op aciotns before the game starts
            for _ in range(random.randint(1, self.no_op_num)):
                if self.render:
                    self.env.render()
                next_state, _, _, _ = self.env.step(0)
                state = next_state

            # Until the game ends
            while not terminal:
                # Select action
                
                action = self.act(state, "train")

                if self.render:
                    self.env.render()

                # Perform the action and save ouput state, reward gained and if it is a terminal state
                next_state, reward, terminal, _ = self.env.step(action)

                # Execute network updating
                self.update(state, action, reward, terminal, next_state)

                # Update actual state
                state = next_state

    def act(self, state, act_mode):
        """
            Execute an action based on epsilon

            Input:
                state:      state in which the agent is at moment
                act_mode:   testing or training mode

            Output:
                action:     action selected by the agent
        """
        state = np.expand_dims(state, axis=0)
        
        if act_mode == "train":
            # If random generate a number lower than epsilon, or the initial memory is not filled
            if self.epsilon >= random.random() or self.steps_counter < self.init_memory_steps:
                # EXPLORE STEP
                action = random.randrange(self.num_actions)
            else:
                # EXPLOIT STEP
                action = np.argmax(self.online_net.predict(state))

            # Reduce epsilon after the initial memory is filled
            if self.epsilon > self.final_eps and self.steps_counter >= self.init_memory_steps:
                self.epsilon -= self.eps_decay

        elif act_mode == "test":
            # If random generate a number lower than a fixed epsilon
            if 0.005 >= random.random():
                # EXPLORE STEP
                action = random.randrange(self.num_actions)
            else:
                # EXPLOIT STEP
                action = np.argmax(self.online_net.predict(state))

        return action

    def update(self, state, action, reward, terminal, next_state):
        """
            Memorize the experience
            Update networks weights training the main network and updating the target one

            Input:
                state:          state in which the agent performed the action
                action:         action perfomed
                reward:         reward gained from the performed action
                terminal:       is the next state terminal?
                next_state:     next_state produced by the performed action
        """
        # Save information in the memory
        self.store_in_memory(state, action, reward, terminal, next_state)

        # If the intial memory is already filled, evaluate to update the networks
        if self.steps_counter >= self.init_memory_steps:
            # If the agent has performed a fixed number of action, train the main network
            if self.steps_counter % self.online_update_frequency == 0:
                self.train_online_network()

            # Update the target network
            if self.steps_counter % self.target_update_frequency == 0:
                self.target_net.set_weights(self.online_net.get_weights())

            # Save network model
            if self.steps_counter % self.save_frequency == 0:
                save_path = self.save_network_path + '/game_'+str(self.steps_counter)+'.h5'
                self.online_net.save(save_path)
                print('Successfully saved: ' + save_path)


        # Update summary variable
        self.total_reward += reward
        self.duration += 1

        # if terminal reset
        if terminal:
            # Observe the mean of rewards on last 30 episode
            self.last_30_reward.append(self.total_reward)
            if len(self.last_30_reward)>30:
                self.last_30_reward.popleft()

            # Log message
            if self.steps_counter < self.init_memory_steps:
                mode = 'random'
            elif self.init_memory_steps <= self.steps_counter < self.init_memory_steps + self.explore_steps:
                mode = 'explore'
            else:
                mode = 'exploit'

            hours, rem = divmod(time.clock()-self.time_start, 3600)
            minutes, seconds = divmod(rem, 60)

            # print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / AVG_REWARD: {4:2.3f} / AVG_LOSS: {5:.5f} / TIME: {6:0>2}:{7:0>2}:{8:05.2f} / MODE: {9}'.format(
            #     self.episode + 1, self.steps_counter, self.duration, self.epsilon,
            #     np.mean(self.last_30_reward), self.total_loss / (float(self.duration) / float(self.online_update_frequency)),
            #     int(hours), int(minutes), int(seconds), mode))
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / AVG_REWARD: {4:2.3f} / AVG_LOSS: {5:.5f} / TIME: {6:0>2}:{7:0>2}:{8:05.2f} / MODE: {9}'.format(
                self.episode + 1, self.steps_counter, self.duration, self.epsilon,
                np.mean(self.last_30_reward), self.total_loss / (float(self.duration) / float(self.online_update_frequency)),
                int(hours), int(minutes), int(seconds), mode), file=self.log)

            # Init for new game
            self.total_reward = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.steps_counter += 1

    def train_online_network(self):
        """
            Train the main online Q network
        """
        # Extract a bacth of experience from the memory
        state_exp, action_exp, reward_exp, terminal_exp, next_state_exp, idx_exp, IS_weights = self.sample_experience()
        batch_size = action_exp.shape[0]

        # Compute Q values o state and next state from online network
        q_value = self.online_net.predict(state_exp)
        next_q_value = self.online_net.predict(next_state_exp)

        # Compute Q values of next state from target network
        q_target = self.target_net.predict(next_state_exp)

        td_error = np.zeros(batch_size, dtype=float)

        # Compute the Q target from the Bellman equation
        for i in range(batch_size):
            old_q_value = q_value[i][action_exp[i]]
            best_action = np.argmax(next_q_value[i])

            if not terminal_exp[i]:
                # Compute new q value of the best action
                q_value[i][action_exp[i]] = reward_exp[i] + self.gamma * q_target[i][best_action]
            else:
                q_value[i][action_exp[i]] = reward_exp[i]

            # If PER compute the new priority value of the experience
            if self.PER:
                td_error[i] = abs(q_value[i][action_exp[i]] - old_q_value)

        # Update priority values and Train the online network
        if self.PER:
            self.update_experience(idx_exp, td_error)
            loss = self.online_net.train_on_batch(state_exp, q_value, sample_weight=IS_weights)
        else:
            loss = self.online_net.train_on_batch(state_exp, q_value)

        self.total_loss += loss

    ################################## REPLAY MEMORY ##################################
    def store_in_memory(self, state, action, reward, terminal, next_state):
        """
            Save actual experience in the memory
        """
        experience = (state, action, reward, terminal, next_state)
        if self.PER:
            # Exctract the maximum priority value in the tree
            priority = np.max(self.replay_memory.tree[-self.replay_memory.capacity:])

            # If the max priority is 0, e.g. the memory is empty, assign 1 to the priority
            if priority == 0:
                priority = 1.0

            # Add the experience with the extracte priority to the tree
            self.replay_memory.add(priority, experience)

            # Count the number of memorized experiences
            if self.memory_count < self.max_memory_size:
                self.memory_count += 1
        else:
            # Add to the deque, and pop in case max size is reached
            self.replay_memory.append(experience)
            self.memory_count += 1

            if self.memory_count >= self.max_memory_size:
                self.replay_memory.popleft()

    def sample_experience(self):
        """
            Extract a sample batch from the memory
        """
        # Create the batch of experience
        exp_batch = []
        sample_size = self.batch_dim

        IS_weights = np.zeros(sample_size, dtype=float)
        idx_exp_list = np.zeros(sample_size, dtype=int)

        # If per extract a sample using priorities
        if self.PER:
            # Calculate the priority segment
            priority_segment = self.replay_memory.total_priority / sample_size

            for i in range (sample_size):
                # Calculate the extreme values of the actual priority segment
                a = priority_segment * i
                b = priority_segment * (i+1)
                # Uniformily sample a leaf from the segment
                value = np.random.uniform(a, b)

                # Extract the corrisponding experience
                idx_exp, priority, experience = self.replay_memory.get_leaf(value)

                idx_exp_list[i] = idx_exp

                # Calculate the related probability
                sampling_prob = priority / self.replay_memory.total_priority

                # Compute the IS weights
                IS_weights[i] = np.power((self.memory_count*sampling_prob), -self.b_memory)

                exp_batch.append(experience)

            # Increase beta
            self.b_memory = np.min([1., self.b_memory + self.b_growth])
        # Otherwise extract experience from deque at random
        else:
            num_sample = self.memory_count if self.memory_count < sample_size else sample_size

            exp_batch = random.sample(self.replay_memory, num_sample)

        # Return experiens as different numpy arrays
        state_exp       = np.array([np.array(exp[0]) for exp in exp_batch])
        action_exp      = np.array([exp[1] for exp in exp_batch])
        reward_exp      = np.array([exp[2] for exp in exp_batch])
        terminal_exp    = np.array([exp[3] for exp in exp_batch])
        next_state_exp  = np.array([np.array(exp[4]) for exp in exp_batch])

        # Normalize the IS_weights for stability
        if self.PER:
            max_w = np.max(IS_weights)
            IS_weights = IS_weights/max_w

        return state_exp, action_exp, reward_exp, terminal_exp, next_state_exp, idx_exp_list, IS_weights

    def update_experience(self, idx_exp, new_td_error):
        """
            Update the priority of the idx experience in the tree
        """
        # Compute the new priority value
        new_td_error += self.e_memory
        # Clip the error and compute the a power
        priorities = np.minimum(new_td_error, 1.0)
        priorities = np.power(priorities, self.a_memory)

        for idx, priority in zip(idx_exp, priorities):
            self.replay_memory.update(idx, priority)

    ################################## INCREASE SCREEN RENDERING TEST ##################################
    def repeat_upsample(self, rgb_array, k=1, l=1, err=[]):
        # repeat kinda crashes if k/l are zero
        if k <= 0 or l <= 0:
            k = 2
            l = 2

        # repeat the pixels k times along the y axis and l times along the x axis
        # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)
        return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)