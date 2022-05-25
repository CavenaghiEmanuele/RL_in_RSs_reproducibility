import torch
from collections import namedtuple
import torch.optim as optim
import random
import torch.nn as nn
import numpy as np
import itertools

Transition = namedtuple('Transition', ('state','action', 'reward','next_state', 'next_candi'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(object):
    def __init__(self, n_action, learning_rate, l2_norm, gcn_net, eval_net, target_net, memory_size, eps_start, eps_end, eps_decay,
                 batch_size, gamma, target_update_step, tau=0.01,  double_q=True):
        self.eval_net = eval_net
        self.target_net = target_net
        self.gcn_net = gcn_net
        self.memory = ReplayMemory(memory_size)
        self.global_step = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_action = n_action
        self.batch_size = batch_size
        self.gamma = gamma
        self.start_learning = 10000
        self.target_update_step = target_update_step
        self.tau = tau
        self.double_q = double_q
        self.optimizer = optim.Adam(itertools.chain(self.eval_net.parameters(),self.gcn_net.parameters()), lr=learning_rate, weight_decay = l2_norm)
        self.loss_func = nn.MSELoss()
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose_action(self, state, candi, mask = None, is_test = False):
        state_emb = self.gcn_net([state]).cuda()
        state_emb = torch.unsqueeze(state_emb,0)
        candi_emb = self.gcn_net.embedding(torch.unsqueeze(torch.LongTensor(candi).cuda(), 0))
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.global_step * self.eps_decay)
        if is_test or random.random() > eps_threshold:
            actions_value = self.eval_net(state_emb,candi_emb)
            action = candi[actions_value.argmax().item()]
        else:
            action = random.randrange(self.n_action)
        return action

    def learn(self):
        if len(self.memory) < self.start_learning:
            return

        #hard assign
        # if self.global_step % self.target_update_step == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())

        #soft assign
        for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

        self.global_step += 1

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))


        b_s = self.gcn_net(list(batch.state))       #[N*max_node*emb_dim]
        b_s_ = self.gcn_net(list(batch.next_state))
        b_a = torch.LongTensor(np.array(batch.action).reshape(-1, 1)).cuda()  #[N*1]
        b_a_emb =self.gcn_net.embedding(b_a)       #[N*1*emb_dim]
        b_r = torch.FloatTensor(np.array(batch.reward).reshape(-1, 1)).cuda()
        next_candi = torch.LongTensor(list(batch.next_candi)).cuda()
        next_candi_emb = self.gcn_net.embedding(next_candi)    #[N*k*emb_dim]

        q_eval = self.eval_net(b_s, b_a_emb,choose_action=False)


        if self.double_q:
            best_actions = torch.gather(input=next_candi, dim=1, index=self.eval_net(b_s_,next_candi_emb).argmax(dim=1).view(self.batch_size,1).cuda())
            best_actions_emb = self.gcn_net.embedding(best_actions)
            q_target = b_r + self.gamma *( self.target_net(b_s_,best_actions_emb,choose_action=False).detach())
            #best_actions = next_candi[self.eval_net(b_s_,next_candi_emb.permute(0,2,1)).argmax(dim=1)]
            #q_target = b_r + self.gamma * q_next.gather(1, best_actions.view(self.batch_size, 1))
        else:
            q_target = b_r + self.gamma*((self.target_net(b_s_,next_candi_emb).detach()).max(dim=1).view(self.batch_size,1))
            #q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
