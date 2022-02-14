import numpy as np

# 单例模式
class Singleton(type):
    _instance = {}

    def __call__(cls,*args,**kwargs):
        if cls not in Singleton._instance:
            Singleton._instance[cls] = type.__call__(cls,*args,**kwargs)
        return Singleton._instance[cls]


# 1. para-server
# self.params_server格式：{key:feature_embedding}
# self.dim
class PS(metaclass = Singleton):
    def __init__(self, embedding_dim):
        np.random.seed(0)
        self.params_server = dict()
        self.dim = embedding_dim
        print("PS inited...")

    # input: mat[batch_size,feature_num]
    # output: mat[batch_size,feature_num,embedding_dim]
    def pull(self, keys):
        values = []
        for k in keys:
            tmp = []
            for arr in k:
                value = self.params_server.get(arr,None)
                if value is None:
                    value = np.random.rand(self.dim)
                    self.params_server[arr] = value
                tmp.append(value)
            values.append(tmp)
        return np.asarray(values, dtype = 'float32')

    # input/output: mat[batch_size,feature_num,embedding_dim]
    def push(self, keys, values):
        for i in range(len(keys)):
            for j in range(len(keys[i])):
                self.params_server[keys[i][j]] = values[i][j]

    def delete(self, keys):
        for k in keys:
            self.params_server.pop(k)

    def save(self, path):
        print("总共包含 %d 个隐向量." % len(self.params_server))
        writer = open(path,"w")
        for k,v in self.params_server.items():
            writer.write(str(k) + '\t' + ','.join(['%.8f' % _ for _ in v]) + '\n')
        writer.close()


# 2. input：读取、解析tfrecord格式文件，拉取ps中隐向量，设置数据集循环和分批等，返回数据集迭代器
import tensorflow as tf
import os
tf.compat.v1.disable_eager_execution()

class InputFn:
    def __init__(self,local_ps,batch_size):
        self.feature_len = 2
        self.label_len = 1
        self.n_parse_threads = 4
        self.shuffle_buffer_size = 1024
        self.prefetch_buffer_size = 1
        self.batch_size = batch_size
        self.local_ps = local_ps
    
    def input_fn(self,data_dir,is_test = False):
        # 定义单条数据解析方法
        def _parse_example(example):
            features = {
                "feature":tf.io.FixedLenFeature(self.feature_len,tf.int64),
                "label":tf.io.FixedLenFeature(self.label_len,tf.float32)
            }
            return tf.io.parse_single_example(example,features)

        # 定义取ps向量方法
        def _get_embedding(parsed):
            keys = parsed["feature"] # mat(8,2)
            keys_array = tf.compat.v1.py_func(self.local_ps.pull,[keys],tf.float32)
            result = {
                "feature":parsed["feature"],
                "label":parsed["label"],
                "feature_embedding":keys_array
            }
            return result

        # 读取所有.tfrecords文件
        file_list = os.listdir(data_dir)
        files = []
        for i in range(len(file_list)):
            files.append(os.path.join(data_dir,file_list[i]))
        dataset = tf.compat.v1.data.Dataset.list_files(files)

        # 设置训练集循环
        if is_test:
            dataset = dataset.repeat(1)
        else:
            dataset = dataset.repeat()

        # 读取tfrecords数据
        dataset = dataset.interleave(
            lambda _: tf.compat.v1.data.TFRecordDataset(_),
            cycle_length = 1
        )

        # 解析tfrecords数据
        dataset = dataset.map(
            _parse_example,
            num_parallel_calls = self.n_parse_threads
        )
        
        # 数据分批
        dataset = dataset.batch(
            self.batch_size,drop_remainder = True
        )

        # 拉取ps中embedding
        dataset = dataset.map(
            _get_embedding,
            num_parallel_calls = self.n_parse_threads
        )

        # 训练数据打散
        if not is_test:
            dataset.shuffle(self.shuffle_buffer_size)

        # 数据预加载
        dataset = dataset.prefetch(buffer_size = self.prefetch_buffer_size)

        # 迭代器
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)

        return iterator,iterator.get_next()


# 3. model部分

# 3.1 定义mf_fn方法读取feature_embedding,计算预测值和loss
def mf_fn(inputs, embedding_dim, is_test):
    embed_layer = inputs["feature_embedding"] # mat(batch_size, feature_num, embedding_dim)
    embed_layer = tf.reshape(embed_layer, shape = [-1,2,embedding_dim])
    # print("embed_layer size:",embed_layer.shape)
    label = inputs["label"] # mat(batch_size, 1)
    embed_layer = tf.split(embed_layer, num_or_size_splits = 2,axis = 1)
    user_id_embedding = tf.reshape(embed_layer[0], shape = [-1,embedding_dim]) # mat(batch_size, embedding_dim)
    movie_id_embedding = tf.reshape(embed_layer[1], shape = [-1,embedding_dim]) # mat(batch_size, embedding_dim)
    
    # 计算预测值
    # TODO: 只保留movie_id,user_id之前均存在的结果.tf熟练再说
    out_ = tf.reduce_mean(user_id_embedding * movie_id_embedding, axis = 1)
    # print("output size:",out_.shape)
    label_ = tf.reshape(label,[-1])

    # 验证集上结果
    out_tmp = tf.sigmoid(out_)
    if is_test:
        tf.compat.v1.add_to_collections("input_tensor",embed_layer)
        tf.compat.v1.add_to_collections("output_tensor",out_tmp)

    # 计算loss
    loss_ = tf.reduce_sum(tf.square(label_ - out_))

    out_dic = {
        "loss" : loss_,
        "ground_truth" : label_,
        "prediction" : out_
    }
    return out_dic


# 3.2 定义模型图结构:调用mf_fn计算预测值和loss，使用SGD算出新para
def setup_graph(inputs, embedding_dim, learning_rate, is_test = False):
    result = {}
    with tf.compat.v1.variable_scope("net_graph",reuse = is_test):
        net_out_dic = mf_fn(inputs, embedding_dim, is_test)
        loss = net_out_dic["loss"]
        result["out"] = net_out_dic

        if is_test:
            return result
        
        embedding_grad = tf.gradients(loss, [inputs["feature_embedding"]], \
            name = "feature_embedding")[0]
        result["feature"] = inputs["feature"] 
        result["feature_new_embedding"] = inputs["feature_embedding"] - \
            learning_rate * embedding_grad
        result["feature_embedding"] = inputs["feature_embedding"]
        return result
        

# 3.3 定义模型评估类
from sklearn.metrics import roc_auc_score
class AUCUtils(object):
    def __init__(self):
        self.reset()

    def add(self, loss, g = np.array([]),p = np.array([])):
        self.loss.append(loss)
        self.ground_truth += g.flatten().tolist()
        self.prediction += p.flatten().tolist()

    def calc(self):
        return {
            "loss_num":  len(self.loss),
            "loss": np.array(self.loss).mean(),
            "auc_num": len(self.ground_truth),
            "auc": roc_auc_score(self.ground_truth,self.prediction) if \
                len(self.ground_truth) > 0 else 0,
            "pcoc": sum(self.prediction) / sum(self.ground_truth)
        }

    def calc_str(self):
        res = self.calc()
        return "loss: %f(%d), auc: %f(%d), pcoc: %f" % (res["loss"], \
            res["loss_num"], res["auc"], res["auc_num"], res["pcoc"])

    def reset(self):
        self.loss = []
        self.prediction = []
        self.ground_truth = []


# 3.4 训练流程
def train():
    embedding_dim = 8
    learning_rate = 0.0001
    local_ps = PS(embedding_dim)
    saved_embedding_dir = 'data/saved_embedding'

    batch_size = 8
    inputs = InputFn(local_ps,batch_size)
    train_file_dir = 'data/train'
    test_file_dir = 'data/val'

    train_iter, train_inputs = inputs.input_fn(train_file_dir, is_test = False)
    # print("train_dic:")
    train_dic = setup_graph(train_inputs, embedding_dim, learning_rate, is_test = False)
    # print("test_dic:")
    test_iter, test_inputs = inputs.input_fn(test_file_dir, is_test = True)
    test_dic = setup_graph(test_inputs, embedding_dim, learning_rate, is_test = True)

    train_metric = AUCUtils()
    test_metric = AUCUtils()

    max_steps = 1000000  # 理解：模拟batch流实时更新，所以不设置epoch。这里37525可以保证跑完一遍训练集
    train_log_steps = 100000
    last_train_auc = 0
    test_log_steps = 37525
    last_test_auc = 0.5

    def _valid_step(sess, test_iter, test_dic):
        test_metric.reset()
        sess.run(test_iter.initializer)
        nonlocal last_test_auc
        while True:
            try:
                out = sess.run(test_dic["out"])
                test_metric.add(
                    out["loss"],
                    out["ground_truth"],
                    out["prediction"])
            except tf.errors.OutOfRangeError:
                print("Test: %s" % test_metric.calc_str())
                if test_metric.calc()['auc'] > last_test_auc:
                    last_test_auc = test_metric.calc()['auc']
                    local_ps.save(saved_embedding_dir)
                break

    _step = 0
    with tf.compat.v1.Session() as sess:
        sess.run([tf.compat.v1.global_variables_initializer(),\
            tf.compat.v1.local_variables_initializer()])
        sess.run(train_iter.initializer)
        while(_step < max_steps):
            feature_old_embedding, feature_new_embedding, keys, out = \
                sess.run([
                    train_dic["feature_embedding"],
                    train_dic["feature_new_embedding"],
                    train_dic["feature"],
                    train_dic["out"]
                ])
            train_metric.add(out["loss"], out["ground_truth"], out["prediction"])
            local_ps.push(keys,feature_new_embedding)

            _step += 1
            if _step % train_log_steps == 0:
                print("Train at step %d: %s" % (_step,train_metric.calc_str())) 
                train_metric.reset()
                print("总共包含 %d 个隐向量." % len(local_ps.params_server))
            if _step % test_log_steps == 0:
                _valid_step(sess,test_iter,test_dic)


# if __name__ == "__main__":
#     train()
 

