---
layout: post
title: Implementing Matrix Factorisation using Tensorflow.
date: 2015-03-15
description: My quora response
---

Neural networks are powerful machine learning architecture that allows us to express various machine learning models. In this post, we go through step by step process to implement matrix factorisation model using tensorflow library.

<h3>Install Dependencies</h3>


```python
! pip install progressbar --upgrade
```


```python
! pip install envoy --upgrade
```

<h3> Data loader </h3>


```python
import scipy.sparse
import numpy as np
import envoy
import progressbar
import sys

class Data(object):

    def __init__(self):
        self.users = {}
        self.items = {}
        self.nusers = 0
        self.nitems = 0
        self.include_time = False

    def update_user_item(self, user, item):
        if user not in self.users:
            self.users[user] = self.nusers
            self.nusers += 1
        if item not in self.items:
            self.items[item] = self.nitems
            self.nitems += 1

    def import_data(self, filename, parser, shape=None,
                    contains_header=False, debug=False):
        r = envoy.run('wc -l {}'.format(filename))
        num_lines = int(r.std_out.strip().partition(' ')[0])
        bar = progressbar.ProgressBar(maxval=num_lines,
                                      widgets=["Loading data: ",
                                     progressbar.Bar(
                                         '=', '[', ']'),
                                     ' ', progressbar.Percentage(),

                                     ' ', progressbar.ETA()]).start()
        I, J, V = [], [], []
        with open(filename) as f:
            if contains_header:
                f.readline()
            for i, line in enumerate(f):
                if (i % 1000) == 0:
                    bar.update(i % bar.maxval)
                try:
                    userid, itemid, rating = parser.parse(line)
                    self.update_user_item(userid, itemid)
                    uid = self.users[userid]
                    iid = self.items[itemid]
                    I.append(uid)
                    J.append(iid)
                    V.append(float(rating))
                except:
                    if debug:
                        print "Ignoring Input: ", line,
                    continue
        bar.finish()
        if shape is not None:
            _shape = (self.nusers if shape[0] is None else shape[0],
                      self.nitems if shape[1] is None else shape[1])
            R = scipy.sparse.coo_matrix(
                (V, (I, J)), shape=_shape)
        else:
            R = scipy.sparse.coo_matrix(
                (V, (I, J)), shape=(self.nusers, self.nitems))
        self.R = R.tocsr()
        self.R.eliminate_zeros()
        sys.stdout.flush()
        return self.R

    def filter(self, n_users=5, n_items=5, iscount=False):
        while True:
            if iscount:
                Rcp = self.R.copy()
                Rcp.data[:] = 1.0
            else:
                Rcp = self.R
            user_stats = Rcp.sum(axis=1)
            item_stats = Rcp.sum(axis=0)
            filter_user = np.ravel((user_stats < n_users) * 1)
            filter_user_cum = np.cumsum(filter_user)
            filter_item = np.ravel((item_stats < n_items) * 1)
            filter_item_cum = np.cumsum(filter_item)
            if (filter_user_cum[-1] == 0) and (filter_item_cum[-1] == 0):
                break

            m, n = self.R.shape

            # filter User item
            I, J, V = [], [], []
            data, ri, rptr = self.R.data, self.R.indices, self.R.indptr
            for i in xrange(m):
                indices = range(rptr[i], rptr[i + 1])
                items = ri[indices]
                ratings = data[indices]
                for j, item in enumerate(items):
                    if (filter_user[i] == 0) and (filter_item[item] == 0):
                        I.append(i - filter_user_cum[i])
                        J.append(item - filter_item_cum[item])
                        V.append(ratings[j])
            R = scipy.sparse.coo_matrix((V, (I, J)),
                                        shape=(m - filter_user_cum[-1],
                                               n - filter_item_cum[-1]))
            self.R = R.tocsr()

            inv_users = {v: k for k, v in self.users.items()}
            inv_items = {v: k for k, v in self.items.items()}

            for i in range(m):
                if filter_user[i] == 1:
                    del self.users[inv_users[i]]
                else:
                    self.users[inv_users[i]] -= filter_user_cum[i]

            for i in range(n):
                if filter_item[i] == 1:
                    del self.items[inv_items[i]]
                else:
                    self.items[inv_items[i]] -= filter_item_cum[i]

def loadDataset(filename, usermap, itemmap, parser, shape=None):
    r = envoy.run('wc -l {}'.format(filename))
    num_lines = int(r.std_out.strip().partition(' ')[0])
    bar = progressbar.ProgressBar(maxval=num_lines,
                                  widgets=["Loading data: ",
                                  progressbar.Bar(
                                     '=', '[', ']'),
                                  ' ', progressbar.Percentage(),

                                  ' ', progressbar.ETA()]).start()
    I, J, V = [], [], []
    cold = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if (i % 1000) == 0:
                bar.update(i % bar.maxval)
            userid, itemid, rating = parser.parse(line)
            if userid not in usermap or itemid not in itemmap:
                cold.append((userid, itemid, rating))
                continue
            uid = usermap[userid]
            iid = itemmap[itemid]
            I.append(uid)
            J.append(iid)
            V.append(float(rating))
    bar.finish()
    if shape is not None:
        R = scipy.sparse.coo_matrix((V, (I, J)), shape=shape)
    else:
        R = scipy.sparse.coo_matrix(
            (V, (I, J)), shape=(len(usermap), len(itemmap)))
    R = R.tocsr()

    return R, cold
```


```python
#Line parser
class UserItemRatingParser:
    def __init__(self, delim=",", threshold = 60):
        self.delim = delim
    def parse(self, line):
        user, item, rating = line.strip().split(self.delim)
        return (user, item, rating)
```

<h3>Biased matrix factorisation</h3>


```python
import tensorflow as tf
class TensorflowMF:
    """
    Biased matrix factorisation model using TensorFlow
    r_ui = b + b_u + b_i + < U_u, V_i >
    """
    def __init__(self, num_users, num_items, rank, reg):
        self.rank = rank
        self.num_users = num_users
        self.num_items = num_items
        self.reg = reg
        self.initialize_values()

    def initialize_values(self):
        self.b =  tf.Variable(0.0, name="global_bias")
        self.b_u =  tf.Variable(tf.truncated_normal([self.num_users, 1],
                                                    stddev=0.01, mean=0),
                                                    name="user_bias")
        self.b_i =  tf.Variable(tf.truncated_normal([self.num_items, 1],
                                                    stddev=0.01, mean=0),
                                                    name="item_bias")
        self.U = tf.Variable(tf.truncated_normal([self.num_users, rank],
                                                  stddev=0.01, mean=0),
                                                  name="users")
        self.V = tf.Variable(tf.truncated_normal([self.num_items, rank],
                                                 stddev=0.01, mean=0),
                                                 name="items")


    def predict(self, users, items):
        U_ = tf.squeeze(tf.nn.embedding_lookup(self.U, users))
        V_ = tf.squeeze(tf.nn.embedding_lookup(self.V, items))
        prediction = tf.nn.sigmoid((tf.reduce_sum(tf.mul(U_, V_),
                                                  reduction_indices=[1])))
        ubias = tf.squeeze(tf.nn.embedding_lookup(self.b_u, users))
        ibias = tf.squeeze(tf.nn.embedding_lookup(self.b_i, items))
        prediction =   self.b + ubias + ibias + tf.squeeze(prediction)
        return prediction

    def regLoss(self):
        reg_loss = 0
        reg_loss +=  tf.reduce_sum(tf.square(self.U))
        reg_loss +=  tf.reduce_sum(tf.square(self.V))
        reg_loss += tf.reduce_sum(tf.square(self.b_u))
        reg_loss += tf.reduce_sum(tf.square(self.b_i))
        return reg_loss * self.reg

    def loss(self, users_items_ratings):
        users, items, ratings = users_items_ratings
        prediction = self.predict(users, items)
        err_loss = tf.nn.l2_loss(prediction - ratings)
        reg_loss = self.regLoss()
        self.total_loss = err_loss + reg_loss
        tf.scalar_summary("loss", self.total_loss)
        return self.total_loss

    def fit(self, users_items_ratings, test_users_items_ratings=None, n_iter=10):
        cost = self.loss(users_items_ratings)
        optimiser = tf.train.AdamOptimizer(0.01).minimize(cost)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            users, items, ratings = users_items_ratings
            for i in range(n_iter):
                sess.run(optimiser)
                if i%20 == 0:
                    print self.evalTestError(test_users_items_ratings).eval()

    def evalTestError(self, test_user_items_ratings):
        testusers, testitems, testratings = test_user_items_ratings
        testprediction = self.predict(testusers, testitems)
        return tf.sqrt(tf.nn.l2_loss(testprediction -
                                     testratings) * 2.0 / len(testusers))
```

<h3>Experiments</h3>


```python
# input data file in format <userid>\t<itemid>\t<rating>

train_path = INSERT_TRAIN_DATA_PATH
test_path = INSERT_TEST_DATA_PATH
```


```python
import numpy as np
def sparseMatrix2UserItemRating(_mat):
    temp = _mat.tocoo()
    user = temp.row.reshape(-1,1)
    item = temp.col.reshape(-1,1)
    rating = temp.data
    return user, item, rating
```


```python
#load data
parser = UserItemRatingParser("\t")
d = Data()
d.import_data(train_path, parser)
train = d.R
test, cold_start_user_item_ratings = loadDataset(test_path,
                                                 d.users,
                                                 d.items,
                                                 parser)
```


```python
num_users, num_items = train.shape
rank = 5
reg = 1.0
n_iter = 400
t = TensorflowMF(num_users, num_items, rank, reg)
users_items_ratings = sparseMatrix2UserItemRating(train)
test_users_items_ratings = sparseMatrix2UserItemRating(test)
t.fit(users_items_ratings, test_users_items_ratings, n_iter)
```
