import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''
product search model
'''
'''
product search model
'''
class PSM(nn.Module):
    def __init__(self, user_size, item_size, word_size, embedding_dim,\
                 max_query_len, max_review_len, batch_size, time_num,\
                 neg_num=5,sample_num=1,transfer_hidden_dim=100,\
                 max_private_len = 3, max_public_len = 10,\
                 item_log_len = 3, user_log_len = 3, \
                 user_word_max = 5, item_word_max = 5, \
                 sigma_parameter=1e0, kl_parameter=1e0,\
                 contrastive_private_parameters = 1e-1, contrastive_public_parameters = 1e-1,\
                 item_word_parameter = 1e-1, user_word_parameter = 1e-1,\
                 item_recent_word_parameter = 1e-2, user_recent_word_parameter = 1e-2,\
                 word_parameter=1e0, device=torch.device('cpu')):
        super(PSM, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.word_size = word_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.max_query_len = max_query_len
        self.max_review_len = max_review_len
        self.item_log_len = item_log_len
        self.user_log_len = user_log_len
        self.user_word_max = user_word_max
        self.item_word_max = item_word_max
        self.sample_num = sample_num
        self.transfer_hidden_dim = transfer_hidden_dim
        self.max_private_len = max_private_len
        self.max_public_len = max_public_len
        self.kl_parameter = kl_parameter
        self.sigma_parameter = sigma_parameter
        self.word_parameter = word_parameter
        self.item_recent_word_parameter = item_recent_word_parameter
        self.user_recent_word_parameter = user_recent_word_parameter
        self.contrastive_private_parameters = contrastive_private_parameters,
        self.contrastive_public_parameters = contrastive_public_parameters,
        self.item_word_parameter = item_word_parameter
        self.user_word_parameter = user_word_parameter
        self.device = device
        self.neg_num = neg_num
        self.time_num = time_num
        
        self.esp = 1e-10
        
        
        
        self.time_embdding = nn.Embedding(self.time_num, self.embedding_dim)
        self.time2mean_u = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.time2mean_i = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.time2mean_w = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.time2std_i = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.time2std_u = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.time2std_w = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        
        self.user_mean = nn.Embedding(self.user_size, self.embedding_dim, _weight=torch.ones(self.user_size, self.embedding_dim))
        self.user_std = nn.Embedding(self.user_size, self.embedding_dim, _weight=torch.zeros(self.user_size, self.embedding_dim))
        
        self.item_mean = nn.Embedding(self.item_size, self.embedding_dim, _weight=torch.ones(self.item_size, self.embedding_dim))
        self.item_std = nn.Embedding(self.item_size, self.embedding_dim, _weight=torch.zeros(self.item_size, self.embedding_dim))
        
        
        self.wordEmbedding_mean = nn.Embedding(self.word_size, self.embedding_dim, padding_idx=0, _weight=torch.ones(self.word_size, self.embedding_dim))
        self.wordEmbedding_std = nn.Embedding(self.word_size, self.embedding_dim, padding_idx=0, _weight=torch.zeros(self.word_size, self.embedding_dim))
        self.queryLinear = nn.Linear(self.embedding_dim, self.embedding_dim)
    
    
        self.transfer_linear_u = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
        self.transfer_linear_i = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
        self.transfer_linear_ni = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
        self.transfer_linear_w = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
        self.transfer_linear_nw = nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
#         self.transfer_linear = {
#             "u":nn.Linear(self.embedding_dim, self.transfer_hidden_dim),
#             "i":nn.Linear(self.embedding_dim, self.transfer_hidden_dim),
#             "ni":nn.Linear(self.embedding_dim, self.transfer_hidden_dim),
#             "w":nn.Linear(self.embedding_dim, self.transfer_hidden_dim),
#             'nw':nn.Linear(self.embedding_dim, self.transfer_hidden_dim)
#         }
        self.transfer_mean_u = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_mean_i = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_mean_ni = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_mean_w = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_mean_nw = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
#         self.transfer_mean = {
#             "u":nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             "i":nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             "ni":nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             'w':nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             'nw':nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
#         }
        self.transfer_std_u = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_std_i = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_std_ni = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_std_w = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
        self.transfer_std_nw = nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
#         self.transfer_std = {
#             "u":nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             "i":nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             "ni":nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             'w':nn.Linear(self.transfer_hidden_dim, self.embedding_dim),
#             'nw':nn.Linear(self.transfer_hidden_dim, self.embedding_dim)
#         }
        

        
#         self.userDecoder = nn.Linear(self.embedding_dim, self.embedding_dim)
#         self.itemDecoder = nn.Linear(self.embedding_dim, self.embedding_dim)

        # contrastive module
        self.private_mean_transfer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.private_std_transfer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.public_mean_transfer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.public_std_transfer = nn.Linear(self.embedding_dim, self.embedding_dim)

        # word contrastive module
        self.review_user_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.review_user_std = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.review_item_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.review_item_std = nn.Linear(self.embedding_dim, self.embedding_dim)
        
    '''
    (uid, pid_pos, qids_pos, len_pos, time_bin_pos)
    [( uid, pid, qids_neg, len_neg, time_bin_pos),..,( uid, pid, qids_neg, len_neg, time_bin_pos)]*neg_sample_num
    '''
    
    def forward(self, user, item_pos, query, query_len, word, word_len, times, private_bought, current_private_len, public_bought, current_public_len, 
                user_recent_review, user_recent_review_len, item_recent_review, item_recent_review_len, items_neg, word_neg):
        self.batch_size = user.shape[0]
        '''
        time embedding
        '''
        time_laten = self.time_embdding(times+torch.tensor(1).to(self.device)).squeeze(1)
        pri_time_laten =self.time_embdding(times)
        
        '''
        user
        '''
        user_mean = self.user_mean(user).squeeze(1) # (batch, out_size)
        user_mean_pri = self.time2mean_u(torch.cat([user_mean, pri_time_laten], 1))
        user_mean = self.time2mean_u(torch.cat([user_mean, time_laten], 1))
        
        user_std = self.user_std(user).squeeze(1) # (batch, out_size)
        user_std_pri = self.time2std_u(torch.cat([user_std, pri_time_laten], 1)).mul(0.5).exp()
        user_std = self.time2std_u(torch.cat([user_std, time_laten], 1)).mul(0.5).exp()
        

        
        '''
        query
        '''
        query = self.get_train_query_tanh_mean(query, query_len)# ((batch, maxQueryLen))) ---> ((batch, embedding)
       
        
        '''
        word
        '''
        word_mean_pos = self.wordEmbedding_mean(word)
        word_mean_pos_pri = self.time2mean_w(torch.cat([word_mean_pos, pri_time_laten.unsqueeze(1).expand_as(word_mean_pos)], 2))
        word_mean_pos = self.time2mean_w(torch.cat([word_mean_pos, time_laten.unsqueeze(1).expand_as(word_mean_pos)], 2))
        
        word_std_pos = self.wordEmbedding_std(word)
        word_std_pos_pri = self.time2std_w(torch.cat([word_std_pos, pri_time_laten.unsqueeze(1).expand_as(word_std_pos)], 2)).mul(0.5).exp()
        word_std_pos = self.time2std_w(torch.cat([word_std_pos, time_laten.unsqueeze(1).expand_as(word_std_pos)], 2)).mul(0.5).exp()
        
        '''
        recent review
        '''
        recent_user_word_mean = self.wordEmbedding_mean(user_recent_review)
        recent_user_word_mean = self.time2mean_w(torch.cat([recent_user_word_mean, time_laten.unsqueeze(1).expand_as(recent_user_word_mean)], 2))

        recent_user_word_std = self.wordEmbedding_std(user_recent_review)
        recent_user_word_std = self.time2std_w(torch.cat([recent_user_word_std, time_laten.unsqueeze(1).expand_as(recent_user_word_std)], 2)).mul(0.5).exp()
        
        recent_item_word_mean = self.wordEmbedding_mean(item_recent_review)
        recent_item_word_mean = self.time2mean_w(torch.cat([recent_item_word_mean, time_laten.unsqueeze(1).expand_as(recent_item_word_mean)], 2))

        recent_item_word_std = self.wordEmbedding_std(item_recent_review)
        recent_item_word_std = self.time2std_w(torch.cat([recent_item_word_std, time_laten.unsqueeze(1).expand_as(recent_item_word_std)], 2)).mul(0.5).exp()
        
        '''
        neg word
        '''
        word_mean_neg = self.wordEmbedding_mean(word_neg)
        word_mean_neg_pri = self.time2mean_w(torch.cat([word_mean_neg, pri_time_laten.unsqueeze(1).expand_as(word_mean_neg)], 2))
        word_mean_neg = self.time2mean_w(torch.cat([word_mean_neg, time_laten.unsqueeze(1).expand_as(word_mean_neg)], 2))
        
        word_std_neg = self.wordEmbedding_std(word_neg)
        word_std_neg_pri = self.time2std_w(torch.cat([word_std_neg, pri_time_laten.unsqueeze(1).expand_as(word_std_neg)], 2)).mul(0.5).exp()
        word_std_neg = self.time2std_w(torch.cat([word_std_neg, time_laten.unsqueeze(1).expand_as(word_std_neg)], 2)).mul(0.5).exp()    
        
        '''
        pos product
        '''
        item_mean_pos = self.item_mean(item_pos).squeeze(1) # (batch, out_size)
        item_mean_pos_pri = self.time2mean_i(torch.cat([item_mean_pos, pri_time_laten], 1))
        item_mean_pos = self.time2mean_i(torch.cat([item_mean_pos, time_laten], 1))
        
        item_std_pos = self.item_std(item_pos).squeeze(1) # (batch, out_size)
        item_std_pos_pri = self.time2std_i(torch.cat([item_std_pos, pri_time_laten], 1)).mul(0.5).exp()
        item_std_pos = self.time2std_i(torch.cat([item_std_pos, time_laten], 1)).mul(0.5).exp()

        
        '''
        neg product
        '''
        items_mean_neg = self.item_mean(items_neg)# (batch, neg_sample_num, out_size)
        items_mean_neg_pri = self.time2mean_i(torch.cat([items_mean_neg, pri_time_laten.unsqueeze(1).expand_as(items_mean_neg)], 2))
        items_mean_neg = self.time2mean_i(torch.cat([items_mean_neg, time_laten.unsqueeze(1).expand_as(items_mean_neg)], 2))
        
        items_std_neg = self.item_std(items_neg)# (batch, neg_sample_num, out_size)
        items_std_neg_pri = self.time2std_i(torch.cat([items_std_neg, pri_time_laten.unsqueeze(1).expand_as(items_std_neg)], 2)).mul(0.5).exp()
        items_std_neg = self.time2std_i(torch.cat([items_std_neg, time_laten.unsqueeze(1).expand_as(items_std_neg)], 2)).mul(0.5).exp()
        
        '''
        the product that are private interest in 
        '''
        items_mean_private = self.item_mean(private_bought)
        #items_mean_private_pri = self.time2mean_i(torch.cat([items_mean_private, pri_time_laten.unsqueeze(1).expand_as(items_mean_private)], 2))
        items_mean_private = self.time2mean_i(torch.cat([items_mean_private, time_laten.unsqueeze(1).expand_as(items_mean_private)], 2))

        items_std_private = self.item_std(private_bought)
        #items_std_private_pri = self.time2std_i(torch.cat([items_std_private, pri_time_laten.unsqueeze(1).expand_as(items_std_private)], 2))
        items_std_private = self.time2std_i(torch.cat([items_std_private, time_laten.unsqueeze(1).expand_as(items_std_private)], 2))

        '''
        the product that are public interset in
        '''
        items_mean_public = self.item_mean(public_bought)
        items_mean_public_pri = self.time2mean_i(torch.cat([items_mean_public, pri_time_laten.unsqueeze(1).expand_as(items_mean_public)], 2))
        items_mean_public = self.time2mean_i(torch.cat([items_mean_public, time_laten.unsqueeze(1).expand_as(items_mean_public)], 2))

        items_std_public = self.item_std(public_bought)
        items_std_public_pri = self.time2std_i(torch.cat([items_std_public, pri_time_laten.unsqueeze(1).expand_as(items_std_public)], 2))
        items_std_public = self.time2std_i(torch.cat([items_std_public, time_laten.unsqueeze(1).expand_as(items_std_public)], 2))

        '''
        the prediction of private and public interest
        '''
        user_private_mean = self.private_mean_transfer(user_mean)
        user_private_std = self.private_std_transfer(user_std)
        user_public_mean = self.public_mean_transfer(user_mean)
        user_public_std = self.public_std_transfer(user_std)

        '''
        the prediction of review on user and item
        '''
        user_review_mean = self.review_user_mean(user_mean)
        user_review_std = self.review_user_std(user_std)
        item_review_mean = self.review_item_mean(item_mean_pos)
        item_review_std = self.review_item_std(item_std_pos)

        '''
        用户和product word的隐变量采样
        '''
        user_sample = self.reparameter(user_mean, user_std)
        product_sample = self.reparameter(item_mean_pos, item_std_pos)
        product_sample_neg = self.reparameter(items_mean_neg, items_std_neg)
        product_sample_neg_contrastive = self.reparameter(items_mean_neg, items_std_neg)
        product_sample_private = self.reparameter(items_mean_private, items_std_private)
        product_sample_public = self.reparameter(items_mean_public, items_std_public)
        word_sample = self.reparameter(word_mean_pos, word_std_pos)
        word_sample_neg = self.reparameter(word_mean_neg, word_std_neg)
        recent_user_word_sample = self.reparameter(recent_user_word_mean, recent_user_word_std)
        recent_item_word_sample = self.reparameter(recent_item_word_mean, recent_item_word_std)
        recent_item_word_mean = self.reparameter(recent_item_word_mean, recent_item_word_std)
        user_private_sample = self.reparameter(user_private_mean, user_private_std)
        user_public_sample = self.reparameter(user_public_mean, user_public_std)
        user_review_sample = self.reparameter(user_review_mean, user_review_std)
        item_review_sample = self.reparameter(item_review_mean, item_review_std)

        private_product = self.get_train_product_tanh_mean(product_sample_private, current_private_len, self.max_private_len)
        public_product = self.get_train_product_tanh_mean(product_sample_public, current_public_len, self.max_public_len)
        user_recent_words = self.get_train_product_tanh_mean(recent_user_word_sample, user_recent_review_len, self.user_word_max)
        item_recent_words = self.get_train_product_tanh_mean(recent_item_word_sample, item_recent_review_len, self.item_word_max)
        
#         query_sample

        '''
        loss 计算
        '''
        # 主要的损失u+q-i 采样得到的uqi 计算重构误差
        loss_main, dis_pos, dis_neg = self.lossF_sigmod_ml(user_sample, query, product_sample, product_sample_neg)
        # 计算uw和iw的损失
        user_word_loss = self.word_loss(user_sample, word_sample, word_len, word_sample_neg)
        item_word_loss = self.word_loss(product_sample, word_sample, word_len, word_sample_neg)

        # contrastive loss
        private_loss = self.loss_contrastive_product(user_private_sample, private_product, product_sample_neg_contrastive)
        public_loss = self.loss_contrastive_product(user_public_sample, public_product, product_sample_neg_contrastive)
        user_recent_loss = self.loss_contrastive_product(user_review_sample, user_recent_words, word_sample_neg)
        item_recent_loss = self.loss_contrastive_product(item_review_sample, item_recent_words, word_sample_neg)

        # 转移损失(KL损失) -->
        # 转移概率 loss current_mean, current_std, prior_mean, prior_std
        user_trans_loss = self.transfer_kl_loss(user_mean, user_std, user_mean_pri, user_std_pri, False, 'u')
        product_trans_pos_loss = self.transfer_kl_loss(item_mean_pos, item_std_pos, item_mean_pos_pri, item_std_pos_pri, False, 'i')
        product_trans_neg_loss = self.transfer_kl_loss(items_mean_neg, items_std_neg, items_mean_neg_pri, items_std_neg_pri, True, 'ni')
        word_trans_pos_loss = self.transfer_kl_loss(word_mean_pos, word_std_pos, word_mean_pos_pri, word_std_pos_pri, True, 'w')
        word_trans_pos_neg_loss = self.transfer_kl_loss(word_mean_neg, word_std_neg, word_mean_neg_pri, word_std_neg_pri, True, 'nw')

        
        
        
        # query_trans_loss
        loss = loss_main+\
        (user_word_loss+item_word_loss)*torch.tensor(self.word_parameter).to(self.device)+\
        private_loss*torch.tensor(self.contrastive_private_parameters).to(self.device)+\
        public_loss*torch.tensor(self.contrastive_public_parameters).to(self.device)+\
        user_recent_loss*torch.tensor(self.user_recent_word_parameter).to(self.device)+\
        item_recent_loss*torch.tensor(self.item_recent_word_parameter).to(self.device)+\
        (user_trans_loss+product_trans_pos_loss+product_trans_neg_loss+word_trans_pos_loss+word_trans_pos_neg_loss)*\
        torch.tensor(self.kl_parameter).to(self.device)
        
        loss = (loss, loss_main, user_word_loss+item_word_loss, \
                user_trans_loss+product_trans_pos_loss+product_trans_neg_loss+word_trans_pos_loss+word_trans_pos_neg_loss, \
                private_loss, public_loss, item_recent_loss, user_recent_loss)
        
        return user_mean, user_std, query, \
                item_mean_pos, item_std_pos,\
                items_mean_neg, items_std_neg, \
                user_sample, product_sample, product_sample_neg, \
                loss, dis_pos, dis_neg,\
                word_mean_pos, word_std_pos, word_mean_neg, word_std_neg
    
    def word_loss(self, itemOrUser, word_pos, word_len, word_neg):
        len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(self.max_review_len-int(i.item())) for i in word_len]).unsqueeze(2).to(self.device)
        word_pos = word_pos.mul(len_mask)
        itemOrUser.unsqueeze_(1)
        dis_pos = (itemOrUser - word_pos).norm(2, dim=2).mean(dim=1)
        dis_neg = (itemOrUser - word_neg).norm(2, dim=2).mean(dim=1)
        wl = torch.log(torch.sigmoid(dis_neg-dis_pos)).mean()*(-1.0)
        itemOrUser.squeeze_(1)
        return wl
        
    def reparameter(self, mean, std):
#         sigma = torch.exp(torch.mul(0.5,log_var))
        std_z = torch.randn(std.shape, device=self.device)
        return mean + torch.tensor(self.sigma_parameter).to(self.device)*std* Variable(std_z)  # Reparameterization trick
    
    
    def get_train_query_tanh_mean(self, query, query_len):
        '''
        input size: (batch, maxQueryLen)
        对query处理使用函数
        tanh(W*(mean(Q))+b)
        
        '''
        query = self.wordEmbedding_mean(query) # size: ((batch, maxQueryLen))) ---> (batch, len(query[i]), embedding)
        # query len mask 使得padding的向量为0
        len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(self.max_query_len-int(i.item())) for i in query_len]).unsqueeze(2).to(self.device)
        query = query.mul(len_mask)

        query = query.sum(dim=1).div(query_len.unsqueeze(1).float())
        query = self.queryLinear(query).tanh()

        return query

    def get_train_product_tanh_mean(self, products, product_len, max_len):
        '''
        get the mean of the product list
        '''
        len_mask = torch.tensor([ [1.]*int(i.item())+[0.]*(max_len-int(i.item())) for i in product_len]).unsqueeze(2).to(self.device)
        products = products.mul(len_mask)

        products = products.sum(dim=1).div(product_len.unsqueeze(1).float())
        
        # whether add the tanh one

        return products

    def transfer_mlp(self, prior, aim='u'):
        transfer_linear = getattr(self, 'transfer_linear_'+aim)
        current_hidden = transfer_linear(prior)
        transfer_mean = getattr(self, 'transfer_mean_'+aim)
        transfer_std = getattr(self, 'transfer_std_'+aim)
        return transfer_mean(current_hidden), transfer_std(current_hidden).mul(0.5).exp()

    
    def transfer_kl_loss(self, current_mean, current_std, prior_mean, prior_std, dim3=False, aim='u'):
        dim2 = current_mean.shape[1]
        if (dim3 == False):
            current_transfer_mean = torch.zeros((self.batch_size, self.embedding_dim), device=self.device)
            current_transfer_std = torch.zeros((self.batch_size, self.embedding_dim), device=self.device)
            for i in range(self.sample_num):
                prior_instance = self.reparameter(prior_mean, prior_std)
                cur_instance = self.transfer_mlp(prior_instance, aim)
                current_transfer_mean += cur_instance[0]
                current_transfer_std += cur_instance[1]

            # 取多个采样的Q(Zt-1)分布的均值为最终的loss 计算使用的P(Zt|B1:t-1)分布
            current_transfer_mean = current_transfer_mean.div(self.sample_num)
            current_transfer_std = current_transfer_std.div(self.sample_num**2)

            kl_loss = self.DKL(current_mean, current_std, current_transfer_mean, current_transfer_std)
        else:
            current_transfer_mean = torch.zeros((self.batch_size, dim2, self.embedding_dim), device=self.device)
            current_transfer_std = torch.zeros((self.batch_size, dim2, self.embedding_dim), device=self.device)
            for i in range(self.sample_num):
                prior_instance = self.reparameter(prior_mean, prior_std)
                cur_instance = self.transfer_mlp(prior_instance, aim)
                current_transfer_mean += cur_instance[0]
                current_transfer_std += cur_instance[1]

            # 取多个采样的Q(Zt-1)分布的均值为最终的loss 计算使用的P(Zt|B1:t-1)分布
            current_transfer_mean = current_transfer_mean.div(self.sample_num)
            current_transfer_std = current_transfer_std.div(self.sample_num)

            kl_loss = self.DKL(current_mean, current_std, current_transfer_mean, current_transfer_std, True)
        
        return kl_loss
    
    
    '''
    KL 误差
    KL(Q(Zt)||P(Zt|B1:t-1))
    P(Zt|B1:t-1) 使用采样计算～～1/K sum_{i=1}^K(P(Zt|Z_{i}t-1))
    '''
    def DKL(self, mean1, std1, mean2, std2, neg = False):
        var1 = std1.pow(2) + self.esp
        var2 = std2.pow(2) + self.esp
        mean_pow2 = (mean2-mean1)*(torch.tensor(1.0, device=self.device)/var2)*(mean2-mean1)
        tr_std_mul = (torch.tensor(1.0,  device=self.device)/var2)*var1
        if (neg == False):
            dkl = (torch.log(var2/var1)-1+tr_std_mul+mean_pow2).mul(0.5).sum(dim=1).mean()
        else:
            dkl = (torch.log(var2/var1)-1+tr_std_mul+mean_pow2).mul(0.5).sum(dim=2).sum(dim=1).mean()
        return dkl
    
    '''
    主损失 重构误差
    -Eq(log{P(Bt|Zt)})
    '''
    def lossF_sigmod_ml(self, user, query, item_pos, items_neg):
        u_plus_q = user+query
        dis_pos = (u_plus_q - item_pos).norm(2, dim=1).mul(5.)
        u_plus_q.unsqueeze_(1)
        dis_neg = (u_plus_q - items_neg)
        dis_neg = dis_neg.norm(2,dim=2)
        dis_pos = dis_pos.view(-1,1)
        batch_loss = torch.log(torch.sigmoid(dis_neg-dis_pos)).sum(dim=1)*(-1.0)
        return batch_loss.mean() , dis_pos.mean(), dis_neg.mean()

    def loss_contrastive_product(self, user, item, neg_item):
        '''
        calculate the distance between the user and the private interset or the public interest
        '''

        dis_pos = (user - item).norm(2, dim=1).mul(5.)
        dis_pos = dis_pos.view(-1,1)

        user.unsqueeze_(1)
        dis_neg = (user - neg_item)
        dis_neg = dis_neg.norm(2,dim=2)
        dis_pos = dis_pos.view(-1,1)
        batch_loss = torch.log(torch.sigmoid(dis_neg-dis_pos)).sum(dim=1)*(-1.0)
        
        return batch_loss.mean()

