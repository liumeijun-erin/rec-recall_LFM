# coding = utf-8
import numpy as np

# 0. 预处理
# 读取embedding隐向量文件
def read_embedding_file(file):
    dic = {}
    with open(file) as f:
        for line in f:
            tmp = line.split("\t")
            embedding = [float(_) for _ in tmp[1].split(",")]
            dic[tmp[0]] = embedding
    return dic

def bkdr2hash64(str):
    mask60 = 0x0fffffffffffffff
    seed = 131
    hash = 0
    for s in str:
        hash = hash* seed + ord(s)
    return hash & mask60

# 设置字典保存hash_id-raw_id映射
def get_hash2id(file):
    movie_dict = {}
    user_dict = {}
    with open(file) as f:
        for line in f:
            tmp = line.split(',')
            movie_dict[str(bkdr2hash64("UserID="+tmp[1]))] = tmp[1]
            movie_dict[str(bkdr2hash64("UserID="+tmp[1]))] = tmp[0]
    return user_dict, movie_dict

# 将movie_raw_id movie_embedding和movie_raw_id movie_embedding分开记录
def split_user_movie(embedding_file, train_file):
    user_dict, movie_dict = get_hash2id(train_file)
    embedding_dict = read_embedding_file(embedding_file)

    movie_embedding = {}
    user_embedding ={}
    for k,v in embedding_dict.items():
        m_id = movie_dict.get(k,None)
        if m_id is not None:
            movie_embedding[m_id] = v
        u_id = user_dict.get(k,None)
        if u_id is not None:
            user_embedding[u_id] = v

    return movie_embedding,user_embedding

# 1.召回策略-i2i
# # 向量点积计算电影相似度,暂存前200
def col_sim(movie_sim_movie_file, movie_embedding):
    with open(movie_sim_movie_file,'w') as f:
        for m,vec1 in movie_embedding.items():
            sim_movie_tmp = {}
            for n, vec2 in movie_embedding.items():
                if m == n:
                    continue
                sim_movie_tmp[n] = np.dot(np.asarray(vec2),np.asarray(vec1))
        
            sim_movie = sorted(sim_movie_tmp.items(),key = lambda _:_[1], reverse = True)
            sim_movie = [str(_[0]) for _ in sim_movie][: 200]

            f.write(m + "\t" + ",".join(sim_movie) + '\n')

# 2.召回策-u2i
# 分别保存用户、电影向量到文件中
def write_user_movie_embedding(movie_embedding_file, user_embedding_file, movie_embedding\
    ,user_embedding) :
    m_file = open(movie_embedding_file,"w")
    for k,v in movie_embedding.items():
        m_file.write(k + "\t" + ",".join([str(_) for _ in v]) + '\n')
    m_file.close()
    u_file = open(user_embedding_file,"w")
    for k,v in user_embedding.items():
        u_file.write(k + "\t" + ",".join([str(_) for _ in v]) + '\n')
    u_file.close()

if __name__ == '__main__':
    embedding_file = './data/saved_embedding'
    train_file = './data/train_set'
    movie_embedding, user_embedding = split_user_movie(embedding_file, train_file)
    
    # u2i模式：保存user和movie隐向量到文件中，然后召回user movie相似度高的
    movie_embedding_file = './data/movie_embedding_file'
    user_embedding_file = './data/user_embedding_file'
    write_user_movie_embedding(movie_embedding_file, user_embedding_file, movie_embedding, user_embedding)
    print("movie_embedding_file, user_embedding_file saved.")

    # i2i模式：召回与用户实时点击相似电影
    movie_sim_movie_file = './data/movie_sim_movie_file'
    col_sim(movie_sim_movie_file, movie_embedding)
    print("movie_sim_movie_file saved.")



