import torch
import json
import random
import numpy as np
from tqdm import tqdm
from env import Env
from dqn import DQN
import utils
from qnet import QNet
from gcn import GraphEncoder
import time
import os

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)

class Recommender(object):
    def __init__(self,args):

        set_global_seeds(28)

        #Env
        self.dataset = args.dataset
        self.max_rating = args.max_rating
        self.min_rating = args.min_rating
        self.boundary_rating = args.boundary_rating
        self.alpha = args.alpha
        self.episode_length = args.episode_length
        self.env = Env(episode_length=self.episode_length,alpha=self.alpha, boundary_rating = self.boundary_rating, max_rating=self.max_rating ,min_rating=self.min_rating, ratingfile=self.dataset)
        self.user_num, self.item_num, self.rela_num = self.env.get_init_data()
        self.boundary_userid = int(self.user_num*0.8)

        #DQN
        self.tau = args.tau
        self.gamma = args.gamma
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.memory_size = args.memory_size

        self.topk = args.pop
        self.candi_num = args.candi_num

        self.fix_emb = args.fix_emb
        self.duling = args.duling
        self.double_q = args.double_q

        self.gcn_layer = args.gcn_layer

        #train
        self.max_training_step = args.max_step
        self.log_step = args.log_step
        self.target_update_step = args.update_step
        self.sample_times = args.sample
        self.update_times = args.update
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.l2_norm = args.l2

        self.load = args.load
        self.load_path = '../data/model/%s_model_%s' % (self.dataset, args.modelname)
        self.save_step = args.save_step
        self.save_path = '../data/model/%s_model_v%d' % (self.dataset, args.version)

        embedding_path = '../data/processed_data/'+self.dataset+'/embedding.vec.json'
        embeddings = torch.FloatTensor(json.load(open(embedding_path, 'r'))['ent_embeddings'])
        print("load embedding complete!")
        entity = embeddings.shape[0]
        emb_size = embeddings.shape[1]

        self.eval_net = QNet(candi_num=self.candi_num, emb_size=emb_size).cuda()
        self.target_net = QNet(candi_num=self.candi_num, emb_size=emb_size).cuda()

        self.gcn = GraphEncoder(entity,emb_size, embeddings=embeddings, max_seq_length=32, max_node=40, hiddim = 50, layers=self.gcn_layer,
                                cash_fn='../data/processed_data/'+self.dataset+'/cache_graph-40.pkl.gz',fix_emb=self.fix_emb).cuda()

        if self.load:
            self.load_model()

        self.dqn = DQN(self.item_num, self.learning_rate, self.l2_norm, self.gcn, self.eval_net, self.target_net, self.memory_size, self.eps_start, self.eps_end, self.eps_decay, self.batch_size,
                  self.gamma, self.target_update_step, self.tau, self.double_q)


        hot_items_path = '../data/run_time/'+self.dataset+'_pop%d' % self.topk
        if os.path.exists(hot_items_path):
            self.hot_items = utils.pickle_load('../data/run_time/'+self.dataset+'_pop%d' % self.topk).tolist()
        else:
            utils.popular_in_train_user(self.dataset,self.topk,self.boundary_rating)
            self.hot_items = utils.pickle_load('../data/run_time/'+self.dataset+'_pop%d' % self.topk).tolist()
        self.candi_dict = utils.pickle_load('../data/processed_data/'+self.dataset+'/neighbors.pkl')
        self.result_file_path = '../data/result/' + time.strftime('%Y%m%d%H%M%S') + '_' + self.dataset+ '_%f' % self.alpha
        self.storage = []


    def candidate(self,obs,mask):
        tmp = []
        tmp+=self.hot_items
        for s in obs:
            if s in self.candi_dict:
                tmp+=self.candi_dict[s]
        tmp = set(tmp)-set(mask)
        candi = random.sample(tmp, self.candi_num)

        return candi


    def train(self):
        for itr in tqdm(range(self.sample_times),desc='sampling'):
            cumul_reward, done  =  0, False
            user_id = random.randint(0,self.boundary_userid)
            cur_state = self.env.reset(user_id)
            mask = []
            candi = self.candidate(cur_state,mask)
            while not done:
                if len(cur_state)==0:
                    action_chosen = random.choice(self.hot_items)
                else:
                    action_chosen = self.dqn.choose_action(cur_state,candi)
                new_state,r,done = self.env.step(action_chosen)
                mask.append(action_chosen)
                candi = self.candidate(new_state, mask)
                if len(cur_state) !=0:
                    self.dqn.memory.push(cur_state,action_chosen, r, new_state, candi)
                cur_state = new_state

        for itr in tqdm(range(self.update_times),desc='updating'):
            self.dqn.learn()

    def evaluate(self):
        ave_reward = []
        tp_list = []
        for itr in tqdm(range(self.user_num),desc='evaluate'):
            cumul_reward, done = 0, False
            cur_state = self.env.reset(itr)
            step = 0
            mask = []
            while not done:
                cur_candi = self.candidate(cur_state,mask)
                if len(cur_state)==0:
                    action_chosen = random.choice(self.hot_items)
                else:
                    action_chosen = self.dqn.choose_action(cur_state, cur_candi, is_test=True)
                new_state,r,done = self.env.step(action_chosen)
                cur_state = new_state
                cumul_reward+=r
                step+=1
                mask.append(action_chosen)
            ave = float(cumul_reward)/float(step)
            tp = float(len(cur_state))
            ave_reward.append(ave)
            tp_list.append(tp)
    
        train_ave_reward = np.mean(np.array(ave_reward[:self.boundary_userid]))
        test_ave_reward = np.mean(np.array(ave_reward[self.boundary_userid:]))
    
        precision = np.array(tp_list)/self.episode_length
        recall = np.array(tp_list)/ (self.rela_num + 1e-20)
        #f1 = (2 * precision * recall) / (precision + recall + 1e-20)
    
        train_ave_precision = np.mean(precision[:self.boundary_userid])
        train_ave_recall = np.mean(recall[:self.boundary_userid])
        #train_ave_f1 = np.mean(f1[:self.boundary_userid])
        test_ave_precision = np.mean(precision[self.boundary_userid:self.user_num])
        test_ave_recall = np.mean(recall[self.boundary_userid:self.user_num])
        #test_ave_f1 = np.mean(f1[self.boundary_userid:self.user_num])
    
        self.storage.append([train_ave_reward, train_ave_precision, train_ave_recall, test_ave_reward, test_ave_precision, test_ave_recall])
        utils.pickle_save(self.storage, self.result_file_path)

    
        print('\ttrain average reward over step: %2.4f, precision@%d: %.4f, recall@%d: %.4f' % (train_ave_reward, self.episode_length, train_ave_precision, self.episode_length, train_ave_recall))
        print('\ttest  average reward over step: %2.4f, precision@%d: %.4f, recall@%d: %.4f' % (test_ave_reward, self.episode_length, test_ave_precision, self.episode_length, test_ave_recall))


    def run(self):
        for i in range(0, self.max_training_step):
            self.train()
            if i % self.log_step == 0:
                self.evaluate()
            if i !=0 and i % self.save_step == 0:
                self.save_model(i)

    def save_model(self,epoch):
        torch.save({
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'gcn_net': self.gcn.state_dict(),
        }, self.save_path+ 's%d' % epoch)

    def load_model(self):
        checkpoint = torch.load(self.load_path)
        self.eval_net.load_state_dict(checkpoint['eval_net']),
        self.target_net.load_state_dict(checkpoint['target_net']),
        self.gcn.load_state_dict(checkpoint['gcn_net'])

    
    #def case_study(self):

    #def significance_test(self):
