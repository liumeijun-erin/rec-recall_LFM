# coding = utf-8

# 0.检查原始数据格式:uid, mid, score, ts
data_file = "data/ratings.dat"
lines = open(data_file)
count = 0
for i,line in enumerate(lines):
    # if i < 10:
    #     print(line.strip())
    # else :
    #     break
    count += 1
print("查看源文件, 总记录数: ",count)  # 10000054

# 1.读取数据, 转换为 {uid:{(mid,score,ts)}}格式
def read_raw_data(file_path):
    user_info = dict()
    lines = open(file_path)
    for line in lines:
        tmp = line.strip().split("::")
        if len(tmp) < 4:
            continue
        ui = user_info.get(tmp[0],None)
        if ui is None:
            user_info[tmp[0]] = [(tmp[1],tmp[2],tmp[3])]
        else:
            user_info[tmp[0]].append((tmp[1],tmp[2],tmp[3]))
    return user_info

user_info = read_raw_data(data_file)

# 2.统计每个用户的行为数,基于行为数过滤异常用户
user_action_num = {}
for k,v in user_info.items():
    user_action_num[k] = len(v)

import numpy as np
user_stat = np.asarray(list(user_action_num.values()))
# print("总用户数: ",len(user_stat)) # 69878
max_num = np.max(user_stat) # 7359
min_num = np.min(user_stat) # 20
median_num = np.median(user_stat) # 69.0
average_num = np.average(user_stat) # 143.10732991785684
# print(max_num,min_num,median_num,average_num)

# filter_user_num = 0
# for n in user_stat:
#     if n > 2000:
#         filter_user_num += 1
# print(filter_user_num)

# 筛掉用户行为过多的异常用户
def extract_valid_user(user_info):
    user_info_filter = {}
    count_filter = 0
    for k,v in user_info.items():
        if len(v) > 2000:
            continue
        user_info_filter[k] = v
        count_filter += len(v)
    print("过滤异常用户记录, 剩余记录数: ",count_filter)  # 9746146
    return user_info_filter

user_info = extract_valid_user(user_info)

# 3.利用时间戳将每个用户的行为数据排序，划分训练集和测试集(每个用户时间排序后两条数据)
# 数据格式: "uid_str,mid_str,score_str"
def split_train_test(user_info): # 
    train_set = []
    test_set = []
    pos_cnt = 0
    neg_cnt = 0
    # test_set_filtered = []
    # movie_dic = dict()  # 改进：test_set中只保留出现过至少一次的movie
    for k,v in user_info.items():
        tmp = sorted(v,key = lambda _:_[2])
        if len(tmp) <= 2: 
            continue
        for i in range(len(tmp)):
            if i < len(tmp) - 2:
                train_set.append(str(k) + "," + tmp[i][0] + ","+ tmp[i][1])
                if float(tmp[i][1]) >= 4:# 分值阈值是4：4837189 4769395
                    pos_cnt += 1
                else:
                    neg_cnt += 1
                # if movie_dic.get(tmp[i][0]):
                #     movie_dic[tmp[i][0]] += 1
                # else:
                #     movie_dic[tmp[i][0]] = 1
            else:
                test_set.append(str(k) + "," + tmp[i][0] + "," + tmp[i][1])
    # for rate in test_set:
    #     tmp = rate.split(",")
    #     if len(tmp) < 2:
    #         print(tmp)
    #         break
    #     if movie_dic.get(tmp[1]):
    #         test_set_filtered.append(rate)
    # 9606584, 139562
    print("划分后训练集中有%d条记录，测试集有%d条记录." % (len(train_set), len(test_set)))
    # print("过滤后测试集有%d条记录." % (len(test_set_filtered))) // 影响不大
    print(pos_cnt,neg_cnt)  # 阈值是3的时候7937832 1668752
    return train_set,test_set  

train_set, test_set = split_train_test(user_info)

# 模仿真实数据打散
def save_data(train_set, test_set,save_path_dir):
    import random 
    random.shuffle(train_set) # 打散，模仿真实数据
    random.shuffle(test_set)
    with open(save_path_dir + "train_set","w") as f:
        for line in train_set:
            f.write(line + '\n')
    with open(save_path_dir + "test_set","w") as f:
        for line in test_set:
            f.write(line + '\n')

save_path = "data/"
save_data(train_set,test_set,save_path)
        
# 4.特征哈希化:提高稀疏数据存储的空间利用率，方便模型部署

# 使用字符串hash方法，hash(feature_name+feature_value) -> int_64
def bkdr2hash64(str):
    mask60 = 0x0fffffffffffffff
    seed = 131
    hash = 0
    for s in str:
        hash = hash* seed + ord(s)
    return hash & mask60

# print("user_id == 1 :",bkdr2hash64("UserID=1"))
# print("movie_id == 1 :",bkdr2hash64("MovieID=1"))

def tohash(file,save_path):
    wfile = open(save_path,"w")
    with open(file) as f:
        for line in f:
            tmp = line.strip().split(",")
            user_id = bkdr2hash64("UserID="+tmp[0])
            item_id = bkdr2hash64("ItemID="+tmp[0])
            wfile.write(str(user_id) + "," + str(item_id) + "," +tmp[2] + "\n")
    wfile.close()

train_file_path = "data/train_set"
train_tohash = "data/train_set_tohash"
test_file_path = "data/test_set"
test_tohash = "data/test_set_tohash"
tohash(train_file_path,train_tohash)
tohash(test_file_path,test_tohash)
print("user_id, movie_id统一hash化完成.")

# 5.转化为tensorflow模型可接收的tfrecords格式
import tensorflow as tf

# 定义单条记录的转换方式
def get_tfrecords_example(feature,label):
    tfrecords_features = {
        'feature':  tf.train.Feature(int64_list=tf.train.Int64List(value=feature)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
    }
    return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))

# 将label值按3划分，转化为二分类问题
# 将记录转换成tf.train.Example格式，保存到tfrecord文件内
# 模拟实际情况，多文件保存(定义200000条数据为一个文件大小上限)
def totfrecords(file,save_dir):
    print("Process to tfrecord File: %s..."% file)
    num = 0
    writer = tf.io.TFRecordWriter(save_dir + "/" + "part-0000" + str(num) +".tfrecords")
    lines = open(file)
    for i,line in enumerate(lines):
        tmp = line.strip().split(",")
        feature = [int(tmp[0]),int(tmp[1])]
        label = [float(1) if float(tmp[2]) >= 4 else float(0)]
        example = get_tfrecords_example(feature, label)
        writer.write(example.SerializeToString())
        if (i + 1) % 200000 == 0:
            writer.close()
            num += 1
            writer = tf.io.TFRecordWriter(save_dir + "/" + "part-0000" + str(num) + ".tfrecords")
    print("Process To tfrecord File:%s End" % file)
    writer.close()

import os 
train_file_path = "data/train_set_tohash"
train_totfrecord = "data/train"
test_file_path = "data/test_set_tohash"
test_totfrecord = "data/val"

os.mkdir(train_totfrecord)
os.mkdir(test_totfrecord)
totfrecords(train_file_path, train_totfrecord)
totfrecords(test_file_path, test_totfrecord)
print("tfrecords文件保存完成.")