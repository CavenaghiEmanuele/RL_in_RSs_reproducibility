import argparse
import time
from recommender import Recommender

parser = argparse.ArgumentParser()

#movielens1M for demo
#Env
parser.add_argument('--dataset', type=str, default='movielens', help='which dataset to use')
parser.add_argument('--max_rating', type=float, default=5)
parser.add_argument('--min_rating', type=float, default=0)
parser.add_argument('--boundary_rating', type=float, default=3.5)
parser.add_argument('--alpha', type=float, default=0.0, help='sequential pattern')
parser.add_argument('--episode_length', type=int, default=32)

#DQN
parser.add_argument('--pop', type=int, default=300)
parser.add_argument('--candi_num', type=int, default=200)
parser.add_argument('--tau', type=float, default=0.01)
parser.add_argument('--gamma',type=float, default=0.7)
parser.add_argument('--eps_start', type=float, default=0.9)
parser.add_argument('--eps_end', type=float, default=0.1)
parser.add_argument('--eps_decay', type=float, default=0.0001)
parser.add_argument('--memory_size',type=int, default=300000)


parser.add_argument('--fix_emb',type=bool,default=False)
parser.add_argument('--duling', type=bool, default=True)
parser.add_argument('--double_q', type=bool, default=True)
parser.add_argument('--gcn_layer', type=int, default=2)

# train
parser.add_argument('--max_step', type=int, default=200)
parser.add_argument('--log_step', type=int, default=1)
parser.add_argument('--update_step', type=int, default=100)
parser.add_argument('--sample', type=int, default=2000, help='sample user num')
parser.add_argument('--update', type=int, default=100, help='update times')
parser.add_argument('--batch_size',type=int, default=2000)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--l2', type=float, default=1e-6)

parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--modelname', type=str, default=None)
parser.add_argument('--save_step', type=int, default=3)
parser.add_argument('--version', type=int, default=0)


#book-crossing
#Env
# parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
# parser.add_argument('--max_rating', type=float, default=1)
# parser.add_argument('--min_rating', type=float, default=0)
# parser.add_argument('--boundary_rating', type=float, default=0.5)
# parser.add_argument('--alpha', type=float, default=0.1, help='sequential pattern')
# parser.add_argument('--episode_length', type=int, default=32)

# #DQN
# parser.add_argument('--pop', type=int, default=1000)
# parser.add_argument('--candi_num', type=int, default=2000)
# parser.add_argument('--tau', type=float, default=0.01)
# parser.add_argument('--gamma',type=float, default=0.6)
# parser.add_argument('--eps_start', type=float, default=0.9)
# parser.add_argument('--eps_end', type=float, default=0.1)
# parser.add_argument('--eps_decay', type=float, default=0.00001)
# parser.add_argument('--memory_size',type=int, default=500000)

# parser.add_argument('--fix_emb',type=bool,default=False)
# parser.add_argument('--duling', type=bool, default=True)
# parser.add_argument('--double_q', type=bool, default=True)
# parser.add_argument('--gcn_layer', type=int, default=2)

# # train
# parser.add_argument('--max_step', type=int, default=200)
# parser.add_argument('--log_step', type=int, default=1)
# parser.add_argument('--update_step', type=int, default=100)
# parser.add_argument('--sample', type=int, default=2000, help='sample user num')
# parser.add_argument('--update', type=int, default=100, help='update times')
# parser.add_argument('--batch_size',type=int, default=2000)
# parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--l2', type=float, default=1e-6)

# parser.add_argument('--load', type=bool, default=False)
# parser.add_argument('--modelname', type=str, default=None)
# parser.add_argument('--save_step', type=int, default=3)
# parser.add_argument('--version', type=int, default=0)

'''
# ml-20m
# Env
parser.add_argument('--dataset', type=str, default='movielens20M', help='which dataset to use')
parser.add_argument('--max_rating', type=float, default=5)
parser.add_argument('--min_rating', type=float, default=0)
parser.add_argument('--boundary_rating', type=float, default=3.5)
parser.add_argument('--alpha', type=float, default=0.1, help='sequential pattern')
parser.add_argument('--episode_length', type=int, default=32)

# DQN
parser.add_argument('--pop', type=int, default=1000)
parser.add_argument('--candi_num', type=int, default=3000)
parser.add_argument('--tau', type=float, default=0.01)
parser.add_argument('--gamma',type=float, default=0.7)
parser.add_argument('--eps_start', type=float, default=0.9)
parser.add_argument('--eps_end', type=float, default=0.1)
parser.add_argument('--eps_decay', type=float, default=0.00001)
parser.add_argument('--memory_size',type=int, default=500000)


parser.add_argument('--fix_emb',type=bool,default=False)
parser.add_argument('--duling', type=bool, default=True)
parser.add_argument('--double_q', type=bool, default=True)
parser.add_argument('--gcn_layer', type=int, default=2)

# train
parser.add_argument('--max_step', type=int, default=300)
parser.add_argument('--log_step', type=int, default=1)
parser.add_argument('--update_step', type=int, default=100)
parser.add_argument('--sample', type=int, default=2000, help='sample user num')
parser.add_argument('--update', type=int, default=200, help='update times')
parser.add_argument('--batch_size',type=int, default=3000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--l2', type=float, default=1e-6)

parser.add_argument('--load', type=bool, default=True)
parser.add_argument('--modelname', type=str, default=None)
parser.add_argument('--save_step', type=int, default=3)
parser.add_argument('--version', type=int, default=0)
'''
args = parser.parse_args()

# with open('../data/result/' + time.strftime('%Y%m%d%H%M%S') + '_' + args.dataset + 'args', 'w') as f:
#     f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()

rec = Recommender(args)
rec.run()
