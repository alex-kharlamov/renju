from random import randint

import numpy as np


class Model:
    """
    Creates basic Model environment:
    Actions: |A|.
    Exp_action_val: Expected value of distribution in actions.
    Reward_exp: Probability to get a reward.

    """
    def __init__(self, actions):
        self.actions = actions
    def get_reward():
        pass


class Agent:

    def __init__(self, Model):
        self.model = Model
        self.q_function = np.zeros((self.model.actions))

    def argmax(self, policy):
        return np.argwhere(policy == np.max(policy)).flatten()

    def get_reward(self, action):
        return self.model.get_reward(action)

    def choose_action(self, time):
        policy_argmax = self.argmax(self.get_policy())
        return policy_argmax[np.random.randint(0, policy_argmax.size)]

    def correct_strategy(self, cur_reward, action):
        pass

    def get_Q(self):
        return self.q_function

    def get_policy(self):
        pass

    def update_Q(self, time, cur_reward, action):
        self.q_function = self.q_function * (time - 1)
        self.q_function[action] += cur_reward
        self.q_function /= time

    def step(self, time):
        action = self.choose_action(time)
        cur_reward = self.get_reward(action)
        self.update_Q(time, cur_reward, action)
        self.correct_strategy(cur_reward, action)
        return cur_reward

class TicTacToeModel(Model):
    def __init__(self):
        self.zeros_place = 0
        self.x_place = 0
        self.winner_positions = [14, 112, 896, 146, 292, 584, 546, 168]
        self.times = 0
        self.x_wins = 0
        self.zero_wins = 0


    def get_reward(self, mode):
        self.times += 1
        winner = 0
        for elem in self.winner_positions:
            if (self.x_place & elem) == elem:
                winner = 1
                self.x_wins += 1
                break
            if (self.zeros_place & elem) == elem:
                winner = -1
                self.zero_wins += 1
                break
        reward = 0
        if mode == 1:
            if winner == 1:
                reward = 1
            if winner == 0:
                reward = 0
            if winner == -1:
                reward = -1
        if mode == -1:
            if winner == 1:
                reward = -1
            if winner == 0:
                reward = 0
            if winner == -1:
                reward = 1

        if self.times == 2:
            self.times = 0
            if reward != 0:
                self.reset_game()
                return reward
        if (self.x_place | self.zeros_place) == 1022:
            self.reset_game()
        return reward

    def print_game(self):
        space = ['_' for i in range(9)]
        i = 1
        for degree in range(1, 10):
            temp_num = i << degree
            if self.zeros_place & temp_num:
                space[degree - 1] = 'O'
            if self.x_place & temp_num:
                space[degree - 1] = 'X'
        space = np.array(space)
        space = space.reshape((3,3))
        print(space)

    def reset_game(self):
        self.zeros_place = 0
        self.x_place = 0



class TicTacToeQPlayer(Agent):
    def __init__(self,game,  mode, epsilon, alpha, gamma):

        self.model = game
        self.q_function = np.zeros((1023))
        self.mode = mode
        self.policy = np.zeros((9))
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.pool_activity = []

    def update_Q(self, time, cur_reward, action):
        current_state = 1022 ^ (game.zeros_place | game.x_place)
        action = self.choose_action(time)
        q_next_state = 0#self.pool_activity[action]
        for elem in self.pool_activity:
            q_next_state = max(q_next_state, self.q_function[elem])
        self.q_function[current_state] += self.alpha * (game.get_reward(self.mode) + self.gamma * q_next_state  - self.q_function[current_state] )

    def get_policy(self):
        global game
        available_positions = 1022 ^ (game.zeros_place | game.x_place)

        pool_activity = []
        i = 1
        for degree in range(1, 10):
            temp_num = i << degree
            if available_positions & temp_num:
                pool_activity.append(available_positions & (~temp_num))

        available_q = []
        for elem in pool_activity:
            available_q.append(self.q_function[elem])
        argmax_states = self.argmax(available_q)

        policy = np.full((len(pool_activity)), self.epsilon / len(pool_activity))
        np.put(policy, argmax_states, (1 - self.epsilon) / len(argmax_states))
        self.pool_activity = pool_activity
        return policy

    def do_action(self, action):
        cur_state = self.pool_activity[action]
        available_positions = 1022 ^ (game.zeros_place | game.x_place)
        diff = cur_state ^ available_positions
        if self.mode == 1:
            game.x_place |= diff
        if self.mode == -1:
            game.zeros_place |= diff
        return diff

    def step(self, time):
        global game
        action = self.choose_action(time)
        diff = self.do_action(action)
        cur_reward = game.get_reward(self.mode)
        self.update_Q(time, cur_reward, action)
        self.correct_strategy(cur_reward, action)
        return cur_reward



game = TicTacToeModel()

ex = TicTacToeQPlayer(game, 1, 0.2, 0.01, 0.99)

zero = TicTacToeQPlayer(game, -1, 0.2, 0.01, 0.99)
for iteration in range(20000):
    ex.step(1)
    zero.step(1)

print('X win = ', game.x_wins, ' Zero win = ', game.zero_wins)
import math
class Ai:
    def __init__(self, choice):
        self.choice = choice
        print(self.choice)
        global game
        game.reset_game()

    # Main function for bot making a move
    def make_move(self, board, real_position):
        global game
        if self.choice == "O":
            game.x_place |= 1 << real_position + 1
        else:
            game.zeros_place |= 1 << real_position + 1
        self.make_random_move(board) # Do a random move


    # Makes a random move by generating a random integer and checking if the place is free
    def make_random_move(self, board):
        """
        free_buttons = []
        free_id = []
        for i in range(len(board)): # Simple for loop to extract the free buttons
            if len(board[i].text.strip()) < 1: # If the button has no mark, stripping spaces...
                free_buttons.append(board[i])
                free_id.append(i)


        if len(free_buttons) > 1: # If any free buttons left... let's make the move
            rand = randint(0, len(free_buttons) - 1) # Generate a random integer from 0 to the length of the array
            free_buttons[rand].text = self.choice
        """
        global zero
        global game
        global ex
        game.print_game()
        if self.choice == "O":
            action = zero.choose_action(1)
            diff = zero.do_action(action)
            game.zeros_place |= diff
        if self.choice == "X":
            action = ex.choose_action(1)
            diff = ex.do_action(action)
            game.x_place |= diff

        demap = [6,7,8,3,4,5,0,1,2]
        print(int(math.log(diff, 2)))
        board[int(math.log(diff, 2)) - 1].text = self.choice
