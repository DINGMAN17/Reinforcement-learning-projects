# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:56:23 2020

@author: DINGMAN
"""
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import numpy as np

#TO DO: class, replay buffer
#TO DO: class, actor network, critic network
#TO DO: class, agent
#TO DO: class, noise (optional)
        

class ReplayBuffer():
    def __init__(self, memory_limit, observation_shape, action_shape):
        self.counter = 0
        self.memory_limit = memory_limit
        self.obs_memory = np.zeros((self.memory_limit, *observation_shape))
        self.obs_new_memory = np.zeros((self.memory_limit, *observation_shape))
        self.act_memory = np.zeros((self.memory_limit, action_shape))
        self.reward_memory = np.zeros(self.memory_limit)
        self.status_memory = np.zeros(self.memory_limit, dtype=np.bool)
        
    def store_transition(self, obs, action, reward, obs_new, done):
        
        current_index = self.counter % self.memory_limit
        self.obs_memory[current_index] = obs
        self.obs_new_memory[current_index] = obs_new
        self.act_memory[current_index] = action
        self.reward_memory[current_index] = reward
        self.status_memory[current_index] = done
        self.counter += 1
        
    def sample(self, BATCH_SIZE):
        #counter might be bigger than the memory size limit
        mem_size = min(self.counter, self.memory_limit)
        batch_idx = np.random.choice(mem_size, BATCH_SIZE)
        
        states = self.obs_memory[batch_idx]
        new_states = self.obs_new_memory[batch_idx]
        rewards = self.reward_memory[batch_idx]
        actions = self.act_memory[batch_idx]
        dones = self.status_memory[batch_idx]
        
        return states, actions, rewards, new_states, dones
    
class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=400, fc2_dims=300,
            name='critic', chkpt_dir=''):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=400, fc2_dims=300, n_actions=2, name='actor',
            chkpt_dir=' '):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu
    
    
class Agent:
    def __init__(self, input_dims, model_dir, alpha=0.001, beta=0.002, env=None,
            gamma=0.99, n_actions=2, max_size=100000, tau=0.005, 
            fc1=400, fc2=300, batch_size=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = -1
        self.min_action = 1
        
        self.actor = ActorNetwork(fc1_dims=fc1, fc2_dims=fc2, n_actions=n_actions, 
                                  name='actor', chkpt_dir=model_dir)
        self.critic = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, name='critic', chkpt_dir=model_dir)
        self.target_actor = ActorNetwork(fc1_dims=fc1, fc2_dims=fc2, n_actions=n_actions, name='target_actor', chkpt_dir=model_dir)
        self.target_critic = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, name='target_critic', chkpt_dir=model_dir)
        
        self.opt_a = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.opt_c = tf.keras.optimizers.Adam(learning_rate=beta)
            
        self.actor.compile(optimizer=self.opt_a)
        self.critic.compile(optimizer=self.opt_c)
        self.target_actor.compile(optimizer=self.opt_a)
        self.target_critic.compile(optimizer=self.opt_c)

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor(observation[None, :], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],
                    mean=0.0, stddev=self.noise)

        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def learn(self):
        if self.memory.counter < self.batch_size:
            return 
        state, action, reward, new_state, done = \
                self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = reward + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                            self.critic.trainable_variables)
        #loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, 
                                    self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
        return critic_loss, actor_loss, critic_network_gradient, actor_network_gradient