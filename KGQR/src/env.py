## a semi-synthetic interacitve recommendation environment based on https://github.com/chenhaokun/TPGR/blob/master/src/env.py

from tqdm import tqdm
import numpy as np
import utils
import os
import copy



class Env():
    def __init__(self,episode_length=32,alpha=0.0,boundary_rating = 3.5,max_rating=5,min_rating=0, ratingfile='movielens'):

        self.episode_length = episode_length
        self.alpha = alpha
        self.boundary_rating = boundary_rating
        self.a = 2.0 / (float(max_rating) - float(min_rating))
        self.b = - (float(max_rating) + float(min_rating)) / (float(max_rating) - float(min_rating))
        self.positive = self.a * self.boundary_rating + self.b
        #self.init_num = init_num

        env_object_path = '../data/run_time/%s_env_objects' % ratingfile
        if os.path.exists(env_object_path):
            objects= utils.pickle_load(env_object_path)
            self.r_matrix = objects['r_matrix']
            self.user_num = objects['user_num']
            self.item_num = objects['item_num']
            self.rela_num = objects['rela_num']
            #print (self.user_num,self.item_num)

        else:
            utils.get_envobjects(ratingfile=ratingfile)
            objects= utils.pickle_load(env_object_path)
            self.r_matrix = objects['r_matrix']
            self.user_num = objects['user_num']
            self.item_num = objects['item_num']
            self.rela_num = objects['rela_num']


    def get_init_data(self):
         return self.user_num, self.item_num, self.rela_num

    def reset(self, user_id):
        self.user_id = user_id
        self.step_count = 0
        self.con_neg_count = 0
        self.con_pos_count = 0
        self.con_zero_count = 0
        self.con_not_neg_count = 0
        self.con_not_pos_count = 0
        self.all_neg_count = 0
        self.all_pos_count = 0
        self.history_items = set()
        self.state = []
        
        return self.state

    def step(self, item_id):
        reward = [0.0, False]
        r = self.r_matrix[self.user_id, item_id]

        # normalize the reward value
        reward[0] = self.a * r + self.b

        self.step_count += 1
        sr = self.con_pos_count - self.con_neg_count
        if reward[0] < 0:
            self.con_neg_count += 1
            self.all_neg_count += 1
            self.con_not_pos_count += 1
            self.con_pos_count = 0
            self.con_not_neg_count = 0
            self.con_zero_count = 0
        elif reward[0] > 0:
            self.con_pos_count += 1
            self.all_pos_count += 1
            self.con_not_neg_count += 1
            self.con_neg_count = 0
            self.con_not_pos_count = 0
            self.con_zero_count = 0
        else:
            self.con_not_neg_count += 1
            self.con_not_pos_count += 1
            self.con_zero_count += 1
            self.con_pos_count = 0
            self.con_neg_count = 0

        self.history_items.add(item_id)

        if self.step_count == self.episode_length or len(self.history_items) == self.item_num:
            reward[1] = True

        reward[0] += self.alpha * sr

        if reward[0]>self.positive:
            self.state.append(item_id)

        curs = copy.deepcopy(self.state)
        
        return curs, reward[0], reward[1]

