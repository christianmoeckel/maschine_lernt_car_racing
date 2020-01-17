import gym
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from scipy.signal import lfilter
from torch.autograd.gradcheck import gradcheck
import copy

discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]  # discounted rewards one liner





class Net(nn.Module):  # an act.or-critic neural network
	def __init__(self, h = 96, w = 96):

		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=10, stride=10)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		#self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		#self.bn3 = nn.BatchNorm2d(32)

		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		def conv2d_size_out(size, kernel_size = 7, stride = 2):
			return (size - (kernel_size - 1) - 1) // stride  + 1
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		linear_input_size = convw * convh * 32

		self.linear = nn.Linear(512,3)
		self.linear2 = nn.Linear(512,1)

	def forward(self, x):

		t = nn.Tanh()		
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		#x = F.relu(self.bn3(self.conv3(x)))
		#print(x.size())
		x = x.view(x.size(0), -1)
		#print(self.linear(x)[0][0])
		#print(self.linear(x)[0][0].item())
		#print(x.size())
		actions = self.linear(x)
	
		action_1 = F.sigmoid(actions[0][0])

		action_2 = F.sigmoid(actions[0][1])

		action_3 = F.sigmoid(actions[0][2])

		
		value_est = F.relu(self.linear2(x))
		return value_est, [action_1, action_2, action_3]


def cost_func(values, actions, rewards, gamma):

	values.append(torch.tensor(0))
	np_values = np.asarray(values)
	delta_t = np.asarray(rewards) + gamma * np_values[1:] - np_values[:-1]

	gen_adv_est = discount(delta_t, gamma)
	
	policy_loss = -(np.asarray(actions) * np.tile(gen_adv_est.copy(), (3,1)).transpose()).sum()

	# l2 loss over value estimator
	rewards[-1] += gamma * np_values[-1]
	discounted_r = discount(np.asarray(rewards), gamma)
	 
	dif = discounted_r - np_values[:-1]
	value_loss = .5 * (dif * dif).sum()
	
	return policy_loss + 0.5 * value_loss

def train(net, env, game_length, i, loss_list, gamma = 0.999):
	env.seed()
	env.reset()
	optimizer.zero_grad()
	state = env.render(mode='rgb_array')
	state = state[:96,:96,]

	values, actions, rewards = [], [], []

	for step in range(game_length):
		if step  == 0: print("game", i +1, "starts")

		value, action = net(torch.from_numpy(np.flip(state,axis=0).copy()).unsqueeze(0).transpose(1,3).float())

		action = np.asarray([single_action.item() for single_action in action])
		
		action[0] = (action[0] * 2) - 1 # stretch steering from 0 to 1   to   -1 to 1
		
		if step % 100 == 0: print('steer, gas, brake values:', action)
		
		state, reward, done, _ = env.step(action)

		done = done or step

		values.append(value)
		actions.append(action)
		rewards.append(reward)

	loss = cost_func(values, actions, rewards, gamma)

	if i % 1 == 0: 
		#print(np.multiply(actions, reward))
		if i % 5 == 0: loss_list.append((i, loss.item()))
		print("training loss in game", i + 1, ":", loss.item())

	loss.backward()
	optimizer.step()


def play(net, env, game_length):
	state = env.reset()
	for _ in range(game_length):
  		env.render()
  		value, action = net(torch.from_numpy(np.flip(state,axis=0).copy()).unsqueeze(0).transpose(1,3).float())
  		#print(np.asarray([single_action.item() for single_action in action]))
  		state, reward, done, info = env.step(np.asarray([single_action.item() for single_action in action]))		
  		print('reward:', reward)
  		if done:
  			state = env.reset()
	env.close()
	return



GAME_LENGTH = 1000
TRAIN_LENGTH = 1000

net = Net()

optimizer = optim.Adam(net.parameters())

env = gym.make('CarRacing-v0')

loss_list = []
for i in range(TRAIN_LENGTH):
	train(net, env, loss_list = loss_list, game_length = GAME_LENGTH, i = i)


play(net, env, game_length = GAME_LENGTH)
print(loss_list)

