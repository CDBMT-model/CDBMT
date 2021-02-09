from pickle import FALSE
from numpy import random
from torch.utils.data import DataLoader, Dataset
import numpy as np
import multiprocessing as mp
from tqdm import trange, tqdm

class MyData(Dataset):
    def __init__(self, reviewData, metaData, neg_sample_num, max_query_len, max_review_len, time_num, weights = True, item_seq_length = 3, item_all_length = 10, 
    item_recent_review_windows = 3, item_recent_review_len = 5, user_recent_review_windows = 3, user_recent_review_len = 5):
        
        # userID and the one-hot id
        self.id2user = dict()
        self.user2id = dict()
        
        # productID and one-hot id
        self.id2product = dict()
        self.product2id = dict()
        
        # [asin, query]
        self.product2query = dict()
        
        # query
        self.word2id = dict()
        self.id2word = dict()
        
        self.userReviews = dict()
        self.userReviewsCount = dict()
        self.userReviewsCounter = dict()
        self.userReviewsTest = dict()
        
        
        self.nes_weight = []
        self.word_weight = []
        self.max_review_len = max_review_len
        self.max_query_len = max_query_len
        self.neg_sample_num = neg_sample_num
        self.item_seq_length = item_seq_length
        self.item_all_length = item_all_length
        self.item_recent_review_len = item_recent_review_len
        self.user_recent_review_len = user_recent_review_len
        self.item_recent_review_windows = item_recent_review_windows
        self.user_recent_review_windows = user_recent_review_windows
        
        self.time_num = time_num
        self.time_data = []
        
        self.init_dict(reviewData, metaData)
        

        self.train_data = []
        self.test_data = []
        self.eval_data = []

        self.init_dataset(reviewData, weights)
        self.init_sample_table()
        

    def init_dict(self, reviewData, metaData):
        for i in range(self.time_num):
            self.time_data.append([])
        
        uid = 0
        us = set(reviewData['reviewerID'])
        pr = set()
        words = set()
        
        for u in tqdm(us):
            # 只有两个购买记录 不够验证和测试
            asins = list(reviewData[reviewData['reviewerID'] == u]['asin'])
            if (len(asins) <= 2):
                continue

            self.id2user[uid] = u
            self.user2id[u] = uid

            # 得到每个用户购买物品记录
            pr.update(asins)
            self.userReviews[uid] = asins
            #　最后一个购买的物品做测试集
            self.userReviewsTest[uid] = asins[-1]
            words.update(set(' '.join(list(reviewData[reviewData['reviewerID'] == u]['reviewText'])).split()))
#             reviewTexts += list(reviewData[reviewData['reviewerID'] == u]['reviewText'])
            uid += 1
            #if uid > 500: break
            if uid % 100 == 0:
                with open (r'out.txt','a+') as ff:
                    ff.write(str(len(us))+' uid: '+str(uid)+'\n')

        self.userNum = uid
        
        pid = 0
#         words = set()
        for p in tqdm(pr):
            if pid % 300 == 0:
                with open (r'out.txt','a+') as ff:
                    ff.write(str(len(pr)) + ' pid:'+str(pid)+'\n')
            try:
                '''
                判断这个product是否有query
                '''
                if (len(metaData.loc[p]['query']) > 0):
                    self.product2query[p] = metaData.loc[p]['query']
                    words.update(' '.join(metaData.loc[p]['query']).split(' '))
            except:
                pass
            self.id2product[pid] = p
            self.product2id[p] = pid
            pid += 1
            
        self.productNum = pid
        self.nes_weight = np.zeros(self.productNum)
        self.queryNum = len(self.product2query)
        
        wi = 0
        self.word2id['<pad>'] = wi
        self.id2word[wi] = '<pad>'
        wi += 1
        for w in tqdm(words):
            if(w==''):
                continue
            self.word2id[w] = wi
            self.id2word[wi] = w
            wi += 1
            
        self.wordNum = wi
        self.word_weight = np.zeros(wi)


    def init_dataset(self, reviewData,weights=True):
        try:
            self.data_X = []
            recent_buffer = []
            current_count = 0
            for r in trange(len(reviewData)):
                recent_item_seq = []
                recent_all_seq = []
                if r % 100 == 0:
                    with open (r'out.txt','a+') as ff:
                        ff.write(str(len(reviewData))+ ' review: '+str(r) + '\n')
                rc = reviewData.iloc[r]
                try:
                    uid = self.user2id[rc['reviewerID']]
                    pid_pos = self.product2id[rc['asin']]
                    time_bin_pos = int(rc['timeBin'])
                except:
                    # 这个user没有加入到字典，购买次数不到3次
                    continue
                
                try:
                    # add the recent bought records
                    item_seq = reviewData[:r]
                    item_seq = item_seq[item_seq['reviewerID'] == rc['reviewerID']]
                    # item_seq = item_seq[item_seq['unixReviewTime'] < rc['unixReviewTime']]

                    if len(item_seq) > self.item_seq_length:
                        seq_len = self.item_seq_length
                    else:
                        seq_len = len(item_seq)

                    for i in range(seq_len):
                        recent_rev = item_seq.iloc[-i-1]
                        recent_item_seq.append(self.product2id[recent_rev['asin']])

                    if seq_len == 0:
                        recent_item_seq.append(pid_pos)
                        seq_len += 1
                    for _ in range(self.item_seq_length - seq_len):
                        recent_item_seq.append(0)

                    # add what all the users are interested in
                    for one in recent_buffer:
                        recent_all_seq.append(self.product2id[one])
                    
                    public_seq_len = len(recent_all_seq)

                    if public_seq_len == 0:
                        recent_all_seq.append(pid_pos)
                        public_seq_len += 1
                    for _ in range(self.item_all_length - public_seq_len):
                        recent_all_seq.append(0)

                    if current_count < self.item_all_length:
                        recent_buffer.append(rc['asin'])
                        current_count += 1
                    else:
                        recent_buffer[current_count%self.item_all_length] = rc['asin']
                        current_count += 1

                except :
                    print('error')
                    break

                text = rc['reviewText']

                try:
                    # 得到product的query数组
                    q_text_array_pos = self.product2query[self.id2product[pid_pos]]
                except:
                    '''
                    没有对应的query
                    '''
                    continue
                try:
                    text_ids, len_r= self.trans_to_ids(text, self.max_review_len)
                    # 设置product的负采样频率
                    self.nes_weight[pid_pos] += 1
                except:
                    continue
                
                try:
                    item_seq = reviewData[:r+1]
                    item_review_seq = item_seq[item_seq['asin'] == rc['asin']]
                    if len(item_review_seq) > self.item_recent_review_windows:
                        item_review_seq = item_review_seq[-self.item_recent_review_windows:]

                    user_review_seq = item_seq[item_seq['reviewerID'] == rc['reviewerID']]
                    if len(user_review_seq) > self.user_recent_review_windows:
                        user_review_seq = user_review_seq[-self.user_recent_review_windows:]

                    # add the recent review of the item
                    word_list = list(' '.join(list(item_review_seq['reviewText'])).split())
                    if len(word_list) == 0: word_list.append('<pad>')
                    item_recent_words = self.get_random_word(word_list, self.item_recent_review_len)

                    # add the recent review of the user
                    word_list = list(' '.join(list(user_review_seq['reviewText'])).split())
                    if len(word_list) == 0: word_list.append('<pad>')
                    user_recent_words = self.get_random_word(word_list, self.user_recent_review_len)

                except Exception as e :
                    print(e)
                    print(r)
                    print(rc)
                    break

                # 遍历每个物品的每个query 得到一个(u, p, q, r)元组
                for qi in range(len(q_text_array_pos)):
                    try:
                        qids_pos, len_pos = self.trans_to_ids(q_text_array_pos[qi], self.max_query_len)
                    except:
                        break
                    self.data_X.append((uid, pid_pos, qids_pos, len_pos, text_ids, len_r, \
                                        time_bin_pos, recent_item_seq, seq_len, recent_all_seq, public_seq_len, \
                                        item_recent_words, self.item_recent_review_len, user_recent_words, self.user_recent_review_len))
                    try:
                        self.userReviewsCount[uid] += 1
                        self.userReviewsCounter[uid] += 1
                    except:
                        self.userReviewsCount[uid] = 1
                        self.userReviewsCounter[uid] = 1

            '''
            数据集合划分 ---> 取每个用户购买过的item的最后一个
            '''
            for r in self.data_X:
                # 只考虑有3个以上（uqi）三元组的user
                if self.userReviewsCount[r[0]] > 2:
                    t = self.userReviewsCounter[r[0]]
                    if (t == 0):
                        continue
                    elif (t == 2): # 倒数第二个
                        self.eval_data.append(r)
                    elif (t == 1): # 倒数第一个
                        self.test_data.append(r)
                    else:
                        self.train_data.append(r)
                        self.time_data[r[6]].append(r)
                    self.userReviewsCounter[r[0]] -= 1

            if weights is not False:
                wf = np.power(self.nes_weight, 0.75)
                wf = wf / wf.sum()
                self.weights = wf
                wf = np.power(self.word_weight, 0.75)
                wf = wf / wf.sum()
                self.word_weight = wf
        except e:
            with open (r'out.txt','a+') as ff:
                ff.write(str(e)+ '\n')


    def trans_to_ids(self, query, max_len, weight_cal = True):
        query = query.split(' ')
        qids = []
        for w in query:
            if w == '':
                continue
            try:
                qids.append(self.word2id[w])
            except:
                continue
            # 需要统计词频
            if weight_cal:
                self.word_weight[self.word2id[w]-1] += 1
        for _ in range(len(qids), max_len):
            qids.append(self.word2id['<pad>'])
        return qids, len(query)

    def get_random_word(self, word_list, sample_len):
        random_words = []
        while len(random_words) < sample_len:
            one_word = word_list[np.random.randint(len(word_list))]
            try:
                random_words.append(self.word2id[one_word])
            except:
                continue

        return random_words

    def neg_sample(self, pos_pid, pos_words):
        neg_items = []
        neg_words = []

        for ii in range(self.neg_sample_num):

            neg_item = self.sample_table_item[np.random.randint(self.table_len_item)]
            while((neg_item in neg_items) or (neg_item == pos_pid)):
                neg_item = self.sample_table_item[np.random.randint(self.table_len_item)]
            neg_items.append(neg_item)

            neg_word = self.sample_table_word[np.random.randint(self.table_len_word)]
            while((neg_word in neg_words) or (neg_word in pos_words)):
                neg_word = self.sample_table_word[np.random.randint(self.table_len_word)]
            neg_words.append(neg_word)
            
        return neg_items,neg_words
    

    def init_sample_table(self):
        table_size = 1e6
        count = np.round(self.weights*table_size)
        self.sample_table_item = []
        for idx, x in enumerate(count):
            self.sample_table_item += [idx]*int(x)
        self.table_len_item = len(self.sample_table_item)
        
        count = np.round(self.word_weight*table_size)
        self.sample_table_word = []
        for idx, x in enumerate(count):
            self.sample_table_word += [idx]*int(x)
        self.table_len_word = len(self.sample_table_word)
    
    def __getitem__(self, i):
        
        pos = self.train_data[i]
        neg = self.neg_sample(pos[1], pos[4])
        
        return pos, neg

    def get_time_data(self, time_bin, i):
        pos = self.time_data[time_bin][i]
        neg = self.neg_sample(pos[1], pos[4])
        return pos, neg


    def getTestItem(self, i):
        return self.test_data[i]

        
    def __len__(self):
        return len(self.train_data)