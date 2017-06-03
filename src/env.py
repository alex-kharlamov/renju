
import numpy as np


class RenjuTEST(object): 
    def __init__(self, player, mode):
        """
        player : 1 for black, 2 for white
        """
        self.cur_pos = np.zeros((15,15, 3))
        self.cur_player = 1
        self.player = player
        self.action_space = 225
        self.moves_done = 0
        self.ext_pos = np.zeros((25, 25))
        self.ext_pos_lr = np.zeros((625), dtype=np.double)
        self.lr_pos = np.zeros(625)
        self.mode = mode
        self.obl_action = None
        self.win_action = None
        self.latest_move = (7,7)
        
        
    def in_step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        self.moves_done += 1
        if self.cur_player == 1:
            self.cur_pos[action % 15][action // 15][0] = 1
            self.lr_pos[action] = 1
        else:
            self.cur_pos[action % 15][action // 15][1] = 1
            self.lr_pos[action] = 2
            
            
            
        w = (action % 15) + 5
        h = (action // 15) + 5
        
        self.latest_move = (w - 5, h - 5)
        
        self.ext_pos[w][h] = self.cur_player
        self.ext_pos_lr[(h * 25) + w] = self.cur_player
        
        """
        for i in range(15):
            for j in range(15):
                if self.cur_pos[i][j][0] > 0 or self.cur_pos[i][j][1] > 0:
                    self.cur_pos[i][j][2] += 1
        """
        
        
        self.cur_pos[:, :, 2:] = self.cur_pos.sum(2).reshape((15,15,1))
        
        reward = 0
        
        rdiag = [0, 0, 0]
        rdiaginv = [0, 0, 0]
        ldiag = [0, 0, 0]
        ldiaginv = [0, 0, 0]
        rrow = [0, 0, 0]
        lrow = [0, 0, 0]
        rcol = [0, 0, 0]
        lcol = [0, 0, 0]
        
        broken = [0,0,0,0,0,0,0,0]
        
        for i in range(5):
            
            if self.ext_pos[w + i][h + i]:
                if not broken[0]:
                    rdiag[int(self.ext_pos[w + i][h + i])] += 1
                else:
                    broken[0] = 1
                
            if self.ext_pos[w][h + i]:
                if not broken[1]:
                    rcol[int(self.ext_pos[w][h + i])] += 1
                else:
                    broken[1] = 1
                
            if self.ext_pos[w - i][h + i]:
                if not broken[2]:
                    ldiag[int(self.ext_pos[w - i][h + i])] += 1
                else:
                    broken[2] = 1
                
            
            if self.ext_pos[w - i][h]:
                if not broken[3]:
                    lrow[int(self.ext_pos[w - i][h])] += 1
                else:
                    broken[3] = 1
            
            if self.ext_pos[w - i][h - i]:
                if not broken[4]:
                    rdiaginv[int(self.ext_pos[w - i][h - i])] += 1
                else:
                    broken[4] = 1
            
            if self.ext_pos[w][h - i]:
                if not broken[5]:
                    lcol[int(self.ext_pos[w][h - i])] += 1
                else:
                    broken[5] = 1
            
            if self.ext_pos[w + i][h - i]:
                if not broken[6]:
                    ldiaginv[int(self.ext_pos[w + i][h - i])] += 1
                else:
                    broken[6] = 1
            
            if self.ext_pos[w + i][h]:
                if not broken[7]:
                    rrow[int(self.ext_pos[w + i][h])] += 1
                else:
                    broken[7] = 1
                    
        
        self.obl_action = None
        first_obl_action = None
        second_obl_action = None
        
        rightdiag = rdiag[self.cur_player] + rdiaginv[self.cur_player]
        
        if rightdiag >= 4:
            if rightdiag >= 6:
                if self.player == self.cur_player:
                    reward = 1
                else:
                    reward = -1
                       
            
        leftdiag = ldiag[self.cur_player] + ldiaginv[self.cur_player]
        
        if leftdiag >= 4:
            if leftdiag >= 6:
                if self.player == self.cur_player:
                    reward = 1
                else:
                    reward = -1
                    
        
        row = rrow[self.cur_player] + lrow[self.cur_player]
        
        if row >= 4:
            if row >= 6:
                if self.player == self.cur_player:
                    reward = 1
                else:
                    reward = -1
        
        
        col = lcol[self.cur_player] + rcol[self.cur_player]
        
        if col >= 4:
            if col >= 6:
                if self.player == self.cur_player:
                    reward = 1
                else:
                    reward = -1
                    
                    
        #if self.win_action != None:
        #    print("WARNING, WIN POSITION:", self.win_action)
            
        #print("pure", first_obl_action, second_obl_action)
        #print(self.win_action)
        
        if self.cur_player == 1:
            self.cur_player = 2
        else:
            self.cur_player = 1
        
        done = True if (self.moves_done == 225 or reward != 0) else False
        cur_pos = self.cur_pos
        if self.moves_done == 225:
            self.reset()
        info = dict()
        return (cur_pos, reward, done, info)
    
    def net_ans(self, model, mode = 'all'):
        s = model.predict(np.array([[self.cur_pos]]))[0]
        if mode == 'one':
            return np.argmax(s)
        else:
            #return sorted(range(len(s)), key=lambda k: s[k], reverse=True)
            return np.argsort(s)
    
    def simulation(self, tree):
        fake_env = deepcopy(self)
        reward = 0
        cur_tree = tree
        while reward == 0:
            
            #cur_pos = self.root.env.cur_pos
        #ext_pos = self.root.env.ext_pos
            #cur_player = 1 if fake_env.cur_player == 2 else 2
            action = cur_tree.search(fake_env.cur_pos, fake_env.ext_pos, fake_env.cur_player, mode = 'small')
            cur_pos, reward_new, done, info = fake_env.in_step(action)
            reward = reward_new
            cur_tree = UCT(cur_pos, model, 'neural')
            #ext_render(cur_pos)
            #print("Simulation print checker")
            
        if reward == 1:
            return 1
        else:
            return 0
    

    
    def step(self, action, mode = 'kn'):
        """
        cur_pos, reward, done, info = self.in_step(action)
        
        if self.win_action != None:
            #print("WIN ACTION")
            if self.win_action[0]:
                if self.ext_pos[(self.win_action[0] % 15) + 5][(self.win_action[0] // 15) + 5] == 0:
                    action = self.win_action[0]
                    cur_pos, reward, done, info = self.in_step(self.win_action[0])
                    return cur_pos, reward, done, action
            if self.win_action[1]:
                if self.ext_pos[(self.win_action[1] % 15) + 5][(self.win_action[1] // 15) + 5] == 0:
                    action = self.win_action[1]
                    cur_pos, reward, done, info = self.in_step(self.win_action[1])
                    return cur_pos, reward, done, action
            
        
        if self.obl_action:
            action = self.obl_action
            cur_pos, reward, done, info = self.in_step(self.obl_action)
            return cur_pos, reward, done, action
        
        if reward != 0:
            #self.render()
            if done:
                self.reset()
            return cur_pos, reward, done, info
        else:
            if self.mode == 'neural':
                s = model.predict(np.array([[self.cur_pos]]))[0]
            else:
                s = clf.predict_proba([self.lr_pos])[0]
            #plt.figure()
            #k = s.reshape((15,15))
            #plt.imshow(k, cmap='hot', interpolation='nearest')
            #plt.show()
            action = np.argmax(s)
            if self.cur_pos[action % 15][action // 15][0] != 0 or self.cur_pos[action % 15][action // 15][1] != 0:
                net_move = np.argsort(s)[::-1]
                
                action = 0
                #print(net_move)
                for act in net_move:
                    if self.cur_pos[act % 15][act // 15][0] == 0 and self.cur_pos[act % 15][act // 15][1] == 0:
                        action = act
                        break
                #print("Net:", action)

            rnd = np.random.randint(1, 100)
            if rnd < -10:
                action = np.random.randint(0, 224)
                while self.cur_pos[action % 15][action // 15][0] != 0 or self.cur_pos[action % 15][action // 15][1] != 0:
                    action = np.random.randint(0, 224)
            print('Net action:', action)
            cur_pos, reward, done, info = self.in_step(action)
            return cur_pos, reward, done, action
        """
        cur_pos, reward, done, info = self.in_step(action)
        tree = UCT(self, model, 'neural')
        act =  tree.search(0.01)
        cur_pos, reward, done, info = self.in_step(act)
        if abs(reward) < 1:
            return cur_pos, max(tree.point_rl_attack - tree.point_rl_def) / 100000, done, info
        return cur_pos, reward, done, info
    
    def learning(self, opponent):
        if self.moves_done % 2 == 0:
            s = opponent.predict(np.array([[self.cur_pos]]))[0]
        else:
            s = model.predict(np.array([[self.cur_pos]]))[0]
        action = np.argmax(s)
        if self.cur_pos[action % 15][action // 15][0] != 0 or self.cur_pos[action % 15][action // 15][1] != 0:
            net_move = np.argsort(s)[::-1]

            action = 0
            #print(net_move)
            for act in net_move:
                if self.cur_pos[act % 15][act // 15][0] == 0 and self.cur_pos[act % 15][act // 15][1] == 0:
                    action = act
                    break
            #print("Net:", action)
        rnd = np.random.randint(1, 100)
        if rnd < 5:
            action = np.random.randint(0, 224)
            while self.cur_pos[action % 15][action // 15][0] != 0 or self.cur_pos[action % 15][action // 15][1] != 0:
                action = np.random.randint(0, 224)

        cur_pos, reward, done, info = self.in_step(action)
        return cur_pos, reward, done, action
            
            
    
    def render(self, mode='human'):
        if mode == 'human':
            for j in reversed(range(15)):
                for i in range(15):
                    flag = 0
                    if self.cur_pos[i][j][0] == 1:
                        flag = 1
                        print("X", end=' ')
                    if self.cur_pos[i][j][1] == 1:
                        flag = 1
                        print("O", end=' ')
                    if not flag:
                        print("_", end=' ')
                print('\n', end='')
            print("------------------------------------------------\n")
        if mode == 'debug':
            for j in reversed(range(25)):
                for i in range(25):
                    flag = 0
                    if self.ext_pos[i][j] == 1:
                        flag = 1
                        print("X", end=' ')
                    if self.ext_pos[i][j] == 2:
                        flag = 1
                        print("O", end=' ')
                    if not flag:
                        print("_", end=' ')
                print('\n', end='')
            print("------------------------------------------------\n")
        
        
    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.cur_pos = np.zeros((15,15,3))
        self.cur_player = 1
        self.moves_done = 0
        self.ext_pos = np.zeros((25, 25))
        self.lr_pos = np.zeros(225)

        return self.cur_pos
    
