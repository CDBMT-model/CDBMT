import pandas as pd
import torch
import time
import pickle
import random
import time as tt
import os

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
import argparse

from scrips.mydata import MyData
from scrips.model import PSM
import config


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2))) 
        else:
            raise ValueError('method must be 0 or 1.')
    return 0

def mean_reciprocal_rank(r):
    return np.sum(r / np.arange(1, r.size + 1))

def hit_rate(r):
    if (np.sum(r) >= 0.9):
        return 1
    else:
        return 0

def get_query_laten(q_linear, query, query_len, max_query_len):
    '''
    input size: (batch, maxQueryLen)
    对query处理使用函数
    tanh(W*(mean(Q))+b)
    '''
    query_len = torch.tensor(query_len).view(1,-1).float()
    # size: ((batch, maxQueryLen))) ---> (batch, len(query[i]), embedding)
    # query len mask 使得padding的向量为0
    len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(max_query_len-int(i.item())) for i in query_len]).unsqueeze(2)
    query = query.mul(len_mask)
    query = query.sum(dim=1).div(query_len)
    query = q_linear(query).tanh()

    return query

def train(my_model, data_set, device):

    # timestamp
    current_time = tt.localtime()
    # set the dataloader
    data_gen = DataLoader(data_set, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # set the optimizer
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.00001)

    my_model.train()
    total_epoch = config.epoch
    total_batch = len(data_gen)
    epochs, all_hrs, all_mrrs, all_ndbgs = [], [], [], []

    for e in range(total_epoch+1):
        my_model.to(device)
        my_model.train()
        for i, data in enumerate(data_gen):
            
            user_mean, user_std, query, item_mean_pos, item_std_pos, items_mean_neg, items_std_neg,\
            user_sample, product_sample, product_sample_neg, loss, dis_pos, dis_neg, word_mean_pos,\
            word_std_pos, word_mean_neg, word_std_neg \
            = my_model(
            data[0][0].to(device), data[0][1].to(device),
            torch.stack(data[0][2]).t().to(device), data[0][3].to(device),
            torch.stack(data[0][4]).t().to(device), data[0][5].to(device), data[0][6].to(device),
            torch.stack(data[0][7]).t().to(device), data[0][8].to(device), 
            torch.stack(data[0][9]).t().to(device), data[0][10].to(device),
            torch.stack(data[0][11]).t().to(device), data[0][12].to(device),
            torch.stack(data[0][13]).t().to(device), data[0][14].to(device),
            torch.stack(data[1][0]).t().to(device),
            torch.stack(data[1][1]).t().to(device))
            
            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()
            if (i % 20 == 0):
                print('E: {}/{} | B: {}/{} | Loss: {} | POS: {} | NEG: {}'.format(e, total_epoch, i, total_batch,loss[0].item(), dis_pos.item(), dis_neg.item()))
                print('Loss:{} | Main:{} | Word:{} | KL:{} | PRIVATE:{} | PUBLIC:{} | RECENT_I:{} | RECENT_U:{}'.format(loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item(), loss[4].item(), loss[5].item(), loss[6].item(), loss[7].item()))
        
        # output the middle result
        if (e % 10 == 0) :
            hr, mrr, ndcg = test(my_model, data_set, device)

            if (len(all_hrs) == 0):
                torch.save(my_model.state_dict(), 'out/{}_{}_{}.pkl'.format(config.dataset, tt.strftime("%Y-%m-%d-%H-%M-%S", current_time),'ok'))
            elif (hr > all_hrs[-1]):
                torch.save(my_model.state_dict(), 'out/{}_{}_{}.pkl'.format(config.dataset, tt.strftime("%Y-%m-%d-%H-%M-%S", current_time),'ok'))

            epochs.append(e)
            all_hrs.append(hr)
            all_mrrs.append(mrr)
            all_ndbgs.append(ndcg)
            print('E: {}/{} | HR: {} | MRR: {} | NDCG: {}'.format(e, total_epoch, hr, mrr, ndcg))

            res = pd.DataFrame()
            res['epoch'] = epochs
            res['hr'] = all_hrs
            res['mrr'] = all_mrrs
            res['ndcg'] = all_ndbgs
            if not os.path.exists(config.out_path):
                os.mkdir(config.out_path)
            res.to_csv('out/result_{}_{}_{}_{}_ran{}_{}.pkl'.format(config.dataset, tt.strftime("%Y-%m-%d-%H-%M-%S", current_time), config.private_parameter, config.recent_user_parameter, config.random_seed, config.batch_size))


def test(my_model, data_set, device):
    my_model.eval()
    device=torch.device('cpu')
    my_model.to(device)
    all_p_m = torch.empty(data_set.time_num, data_set.productNum, config.embedding_dim)
    for ii in range(data_set.time_num):
        for i in range(data_set.productNum):
            p_mean = my_model.item_mean(torch.tensor([i], device=device)).squeeze(1)
            time= my_model.time_embdding(torch.tensor([ii], device=device)+torch.tensor(1, device=device)).squeeze(1)
            p_mean = my_model.time2mean_i(torch.cat([p_mean, time], 1)).squeeze()
            all_p_m[ii][i] = p_mean

    eval_dataset = data_set.test_data
    test_counter = 0
    all_hr = 0
    all_ndcg = 0
    all_mrr = 0
    for ii in trange(len(eval_dataset)):
        td = eval_dataset[ii]
        '''
        应该定义一个训练过的user， 这里简单的先取训练过的时间段的用户
        '''
        if (td[6] >= 0):
            user = my_model.user_mean(torch.tensor([td[0]], device=device)).squeeze(1)
            time= my_model.time_embdding(torch.tensor([td[6]], device=device)+torch.tensor(1, device=device)).squeeze(1)
            user = my_model.time2mean_u(torch.cat([user, time], 1)).squeeze()
            
            query_len = td[3]
            query = torch.cat(tuple([my_model.wordEmbedding_mean(torch.tensor([i], device=device).squeeze(0)) for i in td[2]])).view(1,-1,config.embedding_dim)
            query = get_query_laten(my_model.queryLinear, query, query_len, data_set.max_query_len)
            user_query = user+query
    #         uq_i = torch.empty(datasets.productNum)
            user_query.squeeze_(0)
            uq_i = (user_query - all_p_m[td[6]]).norm(2, dim=1)*(-1.)
    #         for i in range(datasets.productNum):
    #             p_mean = product_time_latent[td[6]+1][i][0]
    #             uq_i[i] = -1*(user_query-p_mean).norm(2).item()
            ranks_order = uq_i.topk(20)[1]
            r = torch.eq(ranks_order, td[1]).numpy()
            all_hr += hit_rate(r)
            all_mrr += mean_reciprocal_rank(r)
            all_ndcg += dcg_at_k(r, 20, 1)
            test_counter += 1
    hr = all_hr / float(test_counter+1e-6)
    mrr = all_mrr / float(test_counter+1e-6)
    ndcg = all_ndcg / float(test_counter+1e-6)
    # print(hr, mrr, ndcg)
    return hr, mrr, ndcg
    

def main():
    # get the paremeter
    parser = argparse.ArgumentParser()

    parser.add_argument("--contrastive_para", 
		type=float,
		default=0.1, 
		help="contrastive_para")
    parser.add_argument("--recent_parameter",
        type=float,
        default=1e-2,
        help="the recent review parameter")
    parser.add_argument("--random_seed",
        type=int,
        default=3,
        help="the random seed")
    parser.add_argument("--batch_size",
        type=int,
        default=256,
        help="the batch size")
    parser.add_argument("--gpu_id",
        type=int,
        default=6,
        help="the_id_of_gpu")
    parser.add_argument("--dataset",
        type=int,
        default=0,
        help="choose the dataset")
    parser.add_argument("--embedding_dim",
        type=int,
        default=50,
        help="choose the embedding dim")

    FLAGS = parser.parse_args()
    config.private_parameter = FLAGS.contrastive_para
    config.public_parameter = FLAGS.contrastive_para
    config.recent_user_parameter = FLAGS.recent_parameter
    config.recent_item_parameter = FLAGS.recent_parameter
    config.random_seed = FLAGS.random_seed
    config.batch_size = FLAGS.batch_size
    config.gpu_id = FLAGS.gpu_id

    # set the dataset
    dataset_type = FLAGS.dataset
    if dataset_type == 1:
        config.dataset = "Clothing_Shoes_and_Jewelry"
    elif dataset_type == 2:
        config.dataset = "Cell_Phones_and_Accessories"
    elif dataset_type == 3:
        config.dataset = "Electronics"
    elif dataset_type == 4:
        config.dataset = "Beauty"
    # set the random seed
    setup_seed(config.random_seed)

    # get the dataset
    with open(os.path.join(config.processed_path, 'dataset_time_'+config.dataset+'.bin'), 'rb') as f:
        data_set = pickle.load(f)

    # set the hyperparameter
    config.neg_sample_num = data_set.neg_sample_num
    config.dataLen = len(data_set.train_data)
    config.batch_num = int(config.dataLen/config.batch_size)
    config.full_len = config.batch_num*config.batch_size
    config.time_bin_num = len(data_set.time_data)
    device  = torch.device("cuda:"+str(config.gpu_id) if torch.cuda.is_available() else "cpu")

    # get the model
    dbml = PSM(data_set.userNum,
           data_set.productNum,
           data_set.wordNum,
           config.embedding_dim,
           data_set.max_query_len,
           data_set.max_review_len,
           config.batch_size,
           data_set.time_num + 1,
           neg_num=5,
           sample_num=1,
           transfer_hidden_dim=100,
           max_private_len=3,
           max_public_len=10,
           sigma_parameter=1e-3,
           kl_parameter=1e-3,
           word_parameter=1e0,
           contrastive_private_parameters=config.private_parameter,
           contrastive_public_parameters=config.public_parameter,
           user_recent_word_parameter=config.recent_user_parameter,
           item_recent_word_parameter=config.recent_item_parameter,
           device=device)
    dbml.to(device)

    # train the model
    train(dbml, data_set, device)
    # test(dbml, data_set, device)

if __name__ == '__main__':
    main()
