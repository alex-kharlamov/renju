import logging
import random

import backend
import renju
import util
import numpy as np
import tensorflow as tf
import keras

import pandas as pd
import copy
import random
import ctypes
import networkx as nx


from sklearn.preprocessing import label_binarize
from sklearn.externals import joblib
clf = joblib.load('LinearClass')
from keras.models import load_model
model_policy = load_model("zeros_policy_44")
model = model_policy


from env import RenjuTEST
from tree import UCT

def convert_move(move):
    w = move[0]
    w = ord(w) - ord('a')
    h = int(move[1:]) - 1
    return (h * 15) + w

def inverse_convert(move):
    POS_TO_LETTER = 'abcdefghjklmnop'
    w = move % 15
    h = move // 15
    w = POS_TO_LETTER[w]
    h += 1
    ans = "".join([str(w), str(h)])
    return ans

def conv_tup(tupik):
    return (tupik[0] * 15) + tupik[1]

time_for_search = 9

#mv = 0

def main():
    logging.basicConfig(filename='dummy.log', level=logging.DEBUG)
    logging.debug("Start dummy backend...")


    env = RenjuTEST(1, 'me')
    tree = UCT(env, model, 'kn' if random.randint(0,1) == 0 else 'neuron')
    st = 0
    
    try:
        while True:
            logging.debug("Wait for game update...")
            game = backend.wait_for_game_update()
            pos = game.positions()

            if (len(pos) > 0):
                env.in_step(conv_tup(pos[-1]))
                st += 1
			
			
            tree = UCT(env, model, 'kn' if random.randint(0,1) == 0 else 'neuron')
            tree_act = tree.do_mcst(time_for_search)

            if st == 0:
                tree_act = 112
            st += 1
            env.in_step(tree_act)


            tree_act = inverse_convert(tree_act)
            backend.move(tree_act)

    except:
        logging.debug('Error!', exc_info=True, stack_info=True)


if __name__ == "__main__":
    main()
