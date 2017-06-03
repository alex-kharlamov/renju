import math
from time import time
import numpy as np
import copy
from copy import deepcopy


from python_api import c_get_policy, c_simulation

class MCSTNode():
    def __init__(self, env, len_nodes = 0):
        self.wins = 0
        self.all_games = 0
        self.childs = []
        self.env = env
        self.parent = None
        self.action = None
        self.index = len_nodes
        self.child_actions = []
        

class UCT():
    def __init__(self, env, model, mode):        
        self.root = MCSTNode(env)
        self.all_games = 0
        self.constant = math.sqrt(2)
        self.cur_node = self.root
        self.model = model
        self.cur_pos = self.root
        self.mode = mode
        self.len_nodes = 1
        self.edges = []
        
    
    def get_child_UCT_list(self, node):
        return list(map(self.get_UCT_stat, node.childs))
        
    def explore(self, number):
        self.cur_node = self.root
        temp_look = np.zeros((15,15))
        if number == 0:
            #print("EXPLORE ZERO NUM")
            overview = 3
            w, h = self.root.env.latest_move
            w_plus = min(15 - w, overview)
            w_minus = min(w, overview)
            
            h_plus = min(15 - h, overview)
            h_minus = min(h, overview)
            
            #print("original", w, h)
            #print("w-range", w_minus, w_plus)
            #print("h-range", h_minus, h_plus)
            
            for j in range(h - h_minus, h + h_plus):
                for i in range(w - w_minus, w + w_plus):
                    if ((i,j) != (w,h)) and self.cur_node.env.ext_pos[i + 5][j + 5] == 0:
                        temp_look[i][j] = 1
                        new_node_action = (j * 15) + i
                        #print('new_node action', new_node_action)

                        new_env = deepcopy(self.cur_node.env)
                        new_env.cur_player = 1 if new_env.cur_player == 2 else 2
                        cur_pos, reward, done, action = new_env.in_step(new_node_action)
                        #print("REWARD NEW ROOT NODE", reward)
                        #new_env.render()

                        new_node = MCSTNode(new_env, self.len_nodes)
                        
                       
                        
                        new_node.env.cur_player = 1 if new_env.cur_player == 2 else 2

                        new_node.parent = self.cur_node
                        new_node.action = new_node_action
                        self.cur_node.child_actions.append(new_node_action)
                        self.len_nodes += 1

                        self.edges.append((self.cur_node.index, new_node.index))

                        #run = new_node.env.simulation(self
                        run = c_simulation(new_node.env.ext_pos_lr,new_node.env.cur_player )

                        self.all_games += 1

                        self.cur_node.childs.append(new_node)
                        self.cur_node = new_node

                        while(self.cur_node.parent != None):
                            self.cur_node.wins += run
                            self.cur_node.all_games += 1
                            self.cur_node = self.cur_node.parent
                        else:
                            self.cur_node.wins += run
                            self.cur_node.all_games += 1
            #look(temp_look)                    
        UCT_list = self.get_child_UCT_list(self.cur_node)
        
        while len(UCT_list) > 0 and  max(UCT_list) > self.get_UCT_stat(self.cur_node):
            #print(UCT_list, "I am at ", self.cur_node.index)
            #print(UCT_list)
            best_node = np.argmax(UCT_list)
            self.cur_node = self.cur_node.childs[best_node]
            UCT_list = self.get_child_UCT_list(self.cur_node)
    
    def expand(self):
    
        
        #if self.mode == 'kn':
        #    s = clf.predict_proba([self.cur_node.env.lr_pos])[0]
        #else:
        """
        s = model.predict(np.array([[self.cur_node.env.cur_pos]]))[0]
        action = np.argmax(s)
        if self.cur_node.env.cur_pos[action % 15][action // 15][0] != 0 or self.cur_node.env.cur_pos[action % 15][action // 15][1] != 0:
            net_move = np.argsort(s)[::-1]
            action = 0
            for act in net_move:
                if act not in self.cur_node.child_actions and self.cur_node.env.cur_pos[act % 15][act // 15][0] == 0 and self.cur_node.env.cur_pos[act % 15][act // 15][1] == 0:
                    action = act
                    break        

        new_node_action = action
        #print('New Node!')
        """
        
        cur_pos = self.root.env.cur_pos
        ext_pos = self.root.env.ext_pos
        cur_player = 2 if self.root.env.cur_player == 2 else 1
        #new_node_action = self.search(cur_pos, ext_pos, cur_player, mode = 'small')
        
        
        policy = c_get_policy(self.cur_node.env.ext_pos_lr, self.cur_node.env.cur_player)
        
        
        for i in range(225):
            if self.cur_node.env.ext_pos[(i % 15) + 5][(i // 15) + 5] != 0:
                policy[i] = -1
                
        for elem in self.cur_node.child_actions:
            policy[elem] = -1
            
        #print(policy)
        #new_node_action = c_get_move(self.cur_node.env.ext_pos_lr, self.cur_node.env.cur_player)
        new_node_action = np.argmax(policy[:225])

        new_env = deepcopy(self.cur_node.env)
        cur_pos, reward, done, action = new_env.in_step(new_node_action)

        new_node = MCSTNode(new_env, self.len_nodes)

        if reward != 0:
            new_node.wins = 999999999999999999
        new_node.parent = self.cur_node
        new_node.action = new_node_action
        self.cur_node.child_actions.append(new_node_action)
        self.len_nodes += 1

        self.edges.append((self.cur_node.index, new_node.index))

        #run = new_node.env.simulation(self)
        run = c_simulation(new_node.env.ext_pos_lr, new_node.env.cur_player)
        self.all_games += 1

        self.cur_node.childs.append(new_node)
        self.cur_node = new_node

        while(self.cur_node.parent != None):
            self.cur_node.wins += run
            self.cur_node.all_games += 1
            self.cur_node = self.cur_node.parent
        else:
            self.cur_node.wins += run
            self.cur_node.all_games += 1

            
            
    def search(self, cur_pos, ext_pos, cur_player, time_limit = 1, mode = 'Full'):
        begin = time()
        self.point_action = np.zeros((225))
        #cur_pos = self.root.env.cur_pos
        #ext_pos = self.root.env.ext_pos
        #cur_player = 1 if self.root.env.cur_player == 2 else 2
        """
        w,h = self.root.env.latest_move
        w += 5
        h += 5
        """ 
        
        for h_iter in range(15):
            for w_iter in range(15):
                w = w_iter + 5
                h = h_iter + 5
                rdiag = ""
                rdiaginv = ""
                ldiag = ""
                ldiaginv = ""
                rrow = ""
                lrow = ""
                rcol = ""
                lcol = ""


                rdiag_def = ""
                rdiaginv_def = ""
                ldiag_def = ""
                ldiaginv_def = ""
                rrow_def = ""
                lrow_def = ""
                rcol_def = ""
                lcol_def = ""


                rdiag_pos = []
                rdiaginv_pos = []
                ldiag_pos = []
                ldiaginv_pos = []
                rrow_pos = []
                lrow_pos = []
                rcol_pos = []
                lcol_pos = []

                rdiaginv_pos.append((w,h))
                ldiaginv_pos.append((w,h))
                lrow_pos.append((w,h))
                lcol_pos.append((w,h))


                for i in range(5):            
                    if ext_pos[w + i][h + i] == 0:
                        rdiag += "0"
                        rdiag_def += "0"
                    else:
                        if ext_pos[w + i][h + i] == cur_player:
                            rdiag += "1"
                            rdiag_def += "0"
                        else:
                            rdiag_def += "1"
                            rdiag += "0"
                    rdiag_pos.append((w + i, h + i))

                    if ext_pos[w][h + i] == 0:
                        rcol += "0"
                        rcol_def += "0"
                    else:
                        if ext_pos[w][h + i] == cur_player:
                            rcol += "1"
                            rcol_def += "0"
                        else:
                            rcol_def += "1"
                            rcol += "0"

                    rcol_pos.append((w, h + i))        

                    if ext_pos[w - i][h + i] == 0:
                        ldiag += "0"
                        ldiag_def += "0"
                    else:
                        if ext_pos[w - i][h + i] == cur_player:
                            ldiag += "1"
                            ldiag_def += "0"
                        else:
                            ldiag_def += "1"
                            ldiag += "0"
                    ldiag_pos.append((w - i, h + i))

                    if ext_pos[w - i][h] == 0:
                        lrow += "0"
                        lrow_def += "0"
                    else:
                        if ext_pos[w - i][h] == cur_player:
                            lrow += "1"
                            lrow_def += "0"
                        else:
                            lrow_def += "1"
                            lrow += "0"

                    lrow_pos.append((w - i, h))

                    if ext_pos[w - i][h - i] == 0:
                        rdiaginv += "0"
                        rdiaginv_def += "0"
                    else:
                        if ext_pos[w - i][h - i] == cur_player:
                            rdiaginv += "1"
                            rdiaginv_def += "0"
                        else:
                            rdiaginv_def += "1"
                            rdiaginv += "0"
                    rdiaginv_pos.append((w - i, h - i))

                    if ext_pos[w][h - i] == 0:
                        lcol += "0"
                        lcol_def += "0"
                    else:
                        if ext_pos[w][h - i] == cur_player:
                            lcol += "1"
                            lcol_def += "0"
                        else:
                            lcol_def += "1"
                            lcol += "0"

                    lcol_pos.append((w, h - i))

                    if ext_pos[w + i][h - i] == 0:
                        ldiaginv += "0"
                        ldiaginv_def += "0"
                    else:
                        if ext_pos[w + i][h - i] == cur_player:
                            ldiaginv += "1"
                            ldiaginv_def += "0"
                        else:
                            ldiaginv_def += "1"
                            ldiaginv += "0"

                    ldiaginv_pos.append((w + i, h - i))
                    if ext_pos[w + i][h] == 0:
                        rrow += "0"
                        rrow_def += "0"
                    else:
                        if ext_pos[w + i][h] == cur_player:
                            rrow += "1"
                            rrow_def += "0"
                        else:
                            rrow_def += "1"
                            rrow += "0"
                    rrow_pos.append((w + i, h))



                left_diag_full = ldiag[::-1] + ldiaginv[1:]
                left_diag_full_pos = ldiag_pos[::-1] + ldiaginv_pos[1:]
                left_diag_full_def = ldiag_def[::-1] + ldiaginv_def[1:]

                right_diag_full = rdiag[::-1] + rdiaginv[1:]
                right_diag_full_pos = rdiag_pos[::-1] + rdiaginv_pos[1:]
                right_diag_full_def = rdiag_def[::-1] + rdiaginv_def[1:]

                col_full = rcol[::-1] + lcol[1:]
                col_full_pos = rcol_pos[::-1] + lcol_pos[1:]
                col_full_def = rcol_def[::-1] + lcol_def[1:]


                row_full = lrow[::-1] + rrow[1:]
                row_full_pos = lrow_pos[::-1] + rrow_pos[1:]
                row_full_def = lrow_def[::-1] + rrow_def[1:]

                #print("left_diag", left_diag_full)
                #print("right_diag", right_diag_full)
                #print("column", col_full)
                #print("row", row_full)

                def find_occurances(what, where):
                    import re
                    return [m.start() for m in re.finditer(what, where)]

                pattern = [
                ['011110', 7000], 
                ['01111', 4000], 
                ['11110', 4000],
                ['010111', 2500],
                ['011011', 2500],
                ['011101', 2500],
                ['111010', 2500],
                ['110110', 2500],
                ['101110', 2500],
                ['01110', 3000],
                ['0111', 1500],
                ['1110', 1500],
                ['01101', 2000],
                ['01011', 2000],
                ['11010', 2000],
                ['10110', 2000],
                ['0110', 200],
                ['10', 50],
                ['01', 50]
                ]


                comb_set = [[left_diag_full, left_diag_full_pos, left_diag_full_def], 
                            [right_diag_full, right_diag_full_pos, right_diag_full_def],
                            [col_full, col_full_pos, col_full_def],
                            [row_full, row_full_pos, row_full_def]]

                try:
                    for match in pattern:
                        for comb in comb_set:
                            #print("I FOUND", match[0], "IN", comb[0], "AND", comb[2])
                            positions_attack = find_occurances(match[0], comb[0])
                            positions_def = find_occurances(match[0], comb[2])
                            #print("FOUND", positions_attack), positions_def

                            #print("pos_attack", positions_attack, match[0])
                            #print("pos_def", positions_def, match[0])
                            self.point_rl_attack = np.zeros((225))
                            self.point_rl_def = np.zeros((225))
                            for left in positions_attack:
                                for i in range(left, left + len(match[0]) + 1):
                                    #print("BONUSES AT", comb[1][i])
                                    self.point_action[comb[1][i][0] - 5 + (comb[1][i][1] - 5) * 15] += match[1]
                                    self.point_rl_attack[comb[1][i][0] - 5 + (comb[1][i][1] - 5) * 15] += match[1]

                            for left in positions_def:
                                for i in range(left, left + len(match[0]) + 1):
                                    self.point_action[comb[1][i][0] - 5 + (comb[1][i][1] - 5) * 15] += match[1] * 1.1
                                    self.point_rl_def[comb[1][i][0] - 5 + (comb[1][i][1] - 5) * 15] += match[1]
                except:
                    pass

        for i in range(225):
            if ext_pos[(i % 15) + 5][(i // 15) + 5]:
                self.point_action[i] = -1
        
        """
        for j in reversed(range(15)):
            for i in range(15):
                flag = 0
                if point_action[j * 15 + i] != 0:
                    flag = 1
                    print(point_action[j * 15 + i], end=' ')
                if not flag:
                    print("_", end=' ')
            print('\n', end='')
        """
        
        number = 0
        #print("Search time", time() - begin)
        if mode == 'Full':
            while (time() - begin) < time_limit * 0.95:
                self.explore(number)
                self.expand()
                number += 1
                #print(number)
            root_values = list(map(self.get_stat, self.root.childs))
            #print(root_values)
            #print(root_UCT_values, len(self.root.childs))
            if len(root_values) != 0:
                best_child = np.argmax(root_values)
            else:
                self.root = self.root.childs[0]
                return self.root.action
            #print("I CHOOSE", self.root.childs[best_child].action)
            self.root = self.root.childs[best_child]

            #if max(point_action) < 100:
            if ext_pos[(self.root.action % 15) + 5][(self.root.action // 15) + 5] == 0:
                return self.root.action
        
        
        else:
            return np.argmax(self.point_action)
        
    def do_mcst(self, time_limit):
        begin = time()
        number = 1
        
        while (time() - begin) < time_limit * 0.95:
            self.explore(number)
            self.expand()
            number += 1
        root_values = list(map(self.get_stat, self.root.childs))
        
        if len(root_values) != 0:
            best_child = np.argmax(root_values)
        else:
            self.root = self.root.childs[0]
            return self.root.action
        best_act = self.root.childs[best_child].action
        #print("I CHOOSE", best_act)
        self.root = self.root.childs[best_child]
        return best_act

        #if max(point_action) < 100:
        if self.root.env.ext_pos[(self.root.action % 15) + 5][(self.root.action // 15) + 5] == 0:
            return self.root.action

                    
    def get_UCT_stat(self, node):
        if node.all_games == 0:
            return 0
        else:
            if (node.parent == None):
                return float(node.wins) / float(node.all_games) + self.constant * math.sqrt(math.log(self.all_games) / node.all_games)
            else:
                return float(node.wins) / float(node.all_games) + self.constant * math.sqrt(math.log(node.parent.all_games) / node.all_games)

    
    def get_stat(self, node):
        if node.all_games == 0:
            return 0
        else:
            return float(node.wins) / float(self.all_games)
        
