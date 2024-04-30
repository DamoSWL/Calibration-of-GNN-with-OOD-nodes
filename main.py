import torch
from dqn_agent import DQNAgent
from env_gcn import gcn_env_gcn
import numpy as np
import random
import argparse
import logging
from utils.util import manual_seed
import scipy.sparse as sp
import time



def main(args):

    #  create the gcn enviroment
    env = gcn_env_gcn(dataset=args.dataset,datapath=args.datapath,args=args)

    # create the agent which consists of the deep Q network
    agent = DQNAgent(state_shape = env.nfeat,
                     memory_size=10000,
                     dataset=env.dataset,model_path=args.model_path,weight=args.weight)

    env.policy = agent

    max_episodes = 30
    best_acc = 0
    best_ece = 0
    best_result = 100

    if args.Q_training:
        logging.info('Training Q network')
        logging.info(f'weight {args.weight}')


        for i_episode in range(1,max_episodes+1):  

            env.init_model()
            env.init_candidate_adj()  # obtain the sampled edges for adjustment of the weights
            logging.info('Training model')
            env.train(env.adj,args.total_epochs,True)  
               
            agent.learn(env, env.total_timesteps) 

            env.update_adj()  # update the edge weight and investigate the performacne of the GCN
            env.init_model()
            env.train(env.adj,args.total_epochs,True)      
            acc, ece, loss = env.validate(env.adj) 
                
            result = loss
            if result < best_result and i_episode > 0:
                best_result = result
                best_acc = acc
                best_ece = ece
                agent.save_model() 

            logging.info(f'Training DDPG episode: {i_episode}/{max_episodes}  Best_acc: {best_acc} Best_ece : {best_ece} acc: {acc} ece: {ece}')
        



        agent.load_model()
        env.update_adj()
    



if __name__ == '__main__':
   

    parser = argparse.ArgumentParser(description='RL graph OOD')

    parser.add_argument('--total_epochs', '-totep', type=int, default=20, help='number of training epochs to run for (warmup epochs are included in the count)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=40, help='Random seed.')
    parser.add_argument('--dataset', default="cora", help="The name of the dataset")
    parser.add_argument('--model_path', default="cora_HyperU", help="The path of the saved model checkpoint")
    parser.add_argument('--datapath', default="./data/", help="The path of the graph data.")
    parser.add_argument('--log', default="log_cora.txt", help="The log file name.")
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument("--normalization", default="BingGeNormAdj", help="The normalization on the adj matrix.")
    parser.add_argument('--test_times', type=int, default=1, help='how many time to run the test and results are averaged')
    parser.add_argument('--Q_training', action='store_true', default=False,help='Enable taining of Q network')
    parser.add_argument('--weight', type=float, default=0.5, help='the adjusted weight of the edges')

    args = parser.parse_args()
    manual_seed(args.seed)
    logging.basicConfig(filemode='a',filename=args.log, level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')

    torch.cuda.empty_cache()
    main(args)


