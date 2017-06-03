from ctypes import *
import numpy as np
import ctypes

my_lib=CDLL('./ctest.so')

def c_get_move(a, player):
    return my_lib.get_best_move(a.ctypes.data_as(c_void_p), int(player))


def c_get_policy(a, player):
    policy = np.zeros((625))
    ans = my_lib.get_policy(a.ctypes.data_as(c_void_p), int(player), ctypes.c_void_p(policy.ctypes.data))
    return policy



def c_get_move_safe(a, player):
    policy = c_get_policy(a, player)
    
    for i in range(15):
        for j in range(15):
            if a[j + 5][i + 5] != 0:
                policy[(i * 15) + j] = -1
    return np.argmax(policy[:225])



def c_simulation(a, player):
    return my_lib.simulation(a.ctypes.data_as(c_void_p), int(player))
