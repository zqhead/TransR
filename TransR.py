'''
在TransH_pytorch的基础上，更进一步采用pytorch.nn.embedding class来实现transH模型

'''

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import codecs
import numpy as np
import copy
import time
import random
import json

entity2id = {}
relation2id = {}
relation_tph = {}
relation_hpt = {}

def data_loader(file):
    print("load file...")
    file1 = file + "train.txt"
    file2 = file + "entity2id.txt"
    file3 = file + "relation2id.txt"

    with open(file2, 'r') as f1, open(file3, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity2id[line[0]] = line[1]

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation2id[line[0]] = line[1]

    entity_set = set()
    relation_set = set()
    triple_list = []
    relation_head = {}
    relation_tail = {}

    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = int(entity2id[triple[0]])
            t_ = int(entity2id[triple[1]])
            r_ = int(relation2id[triple[2]])

            triple_list.append([h_, t_, r_])

            entity_set.add(h_)
            entity_set.add(t_)

            relation_set.add(r_)
            if r_ in relation_head:
                if h_ in relation_head[r_]:
                    relation_head[r_][h_] += 1
                else:
                    relation_head[r_][h_] = 1
            else:
                relation_head[r_] = {}
                relation_head[r_][h_] = 1

            if r_ in relation_tail:
                if t_ in relation_tail[r_]:
                    relation_tail[r_][t_] += 1
                else:
                    relation_tail[r_][t_] = 1
            else:
                relation_tail[r_] = {}
                relation_tail[r_][t_] = 1

    for r_ in relation_head:
        sum1, sum2 = 0, 0
        for head in relation_head[r_]:
            sum1 += 1
            sum2 += relation_head[r_][head]
        tph = sum2/sum1
        relation_tph[r_] = tph

    for r_ in relation_tail:
        sum1, sum2 = 0, 0
        for tail in relation_tail[r_]:
            sum1 += 1
            sum2 += relation_tail[r_][tail]
        hpt = sum2/sum1
        relation_hpt[r_] = hpt

    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))

    if len(entity_set) != len(entity2id):
        raise ValueError("The number of entities is not equal")
    if len(relation_set) != len(relation2id):
        raise ValueError("The number of relations is not equal")

    return entity_set, relation_set, triple_list

def norm_l1(h, r, t):
    return np.sum(np.fabs(h + r - t))


class TransR(nn.Module):
    def __init__(self, entity_num, relation_num, ent_dim, rel_dim, margin):
        super(TransR, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.margin = margin


        self.ent_embedding = torch.nn.Embedding(num_embeddings=self.entity_num,
                                                          embedding_dim=self.ent_dim)
        self.rel_embedding = torch.nn.Embedding(num_embeddings=self.relation_num,
                                                           embedding_dim=self.rel_dim)
        self.rel_matrix = torch.nn.Embedding(num_embeddings= self.relation_num,
                                                           embedding_dim=self.ent_dim*self.rel_dim)
        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="sum")
        # pairwiseDIstance 用于计算成批成对的两个向量之间的距离（差值），具体的距离为 Lp范数，参数P定义了使用第几范数，默认为L2
        self.distance_function = nn.PairwiseDistance(p=2)

        self.__data_init()

    def __data_init(self):
        # embedding.weight (Tensor) -形状为(num_embeddings, embedding_dim)的嵌入中可学习的权值
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        identity = torch.zeros(self.ent_dim, self.rel_dim)
        for i in range(min(self.ent_dim, self.rel_dim)):
            identity[i][i] = 1
        identity = identity.view(self.ent_dim * self.rel_dim)
        for i in range(self.relation_num):
            self.rel_matrix.weight.data[i] = identity

    def input_pre_transe(self, ent_vector, rel_vector):
        for i in range(self.entity_num):
            self.ent_embedding.weight.data[i] = torch.from_numpy(np.array(ent_vector[str(i)]))
        for i in range(self.relation_num):
            self.rel_embedding.weight.data[i] = torch.from_numpy(np.array(rel_vector[str(i)]))

    # def normalization_norm_relations(self):
    #     norm = self.relation_norm_embedding.weight.detach().cpu().numpy()
    #     norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
    #     self.relation_norm_embedding.weight.data.copy_(torch.from_numpy(norm))

    def transfer(self, e, rel_mat):
        # view 的作用是重新将一个tensor转化成另一个形状
        # 数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的tensor
        # -1 表示根据其他维度来自动计算这一维的数量
        rel_matrix = rel_mat.view(-1, self.ent_dim, self.rel_dim)
        e = e.view(-1, 1, self.ent_dim)
        e = torch.matmul(e, rel_matrix)

        return e.view(-1, self.rel_dim)

    def distance(self, h, r, t):
        # 在 tensor 的指定维度操作就是对指定维度包含的元素进行操作，如果想要保持结果的维度不变，设置参数keepdim=True即可
        # 如 下面sum中 r_norm * h 结果是一个1024 *50的矩阵（2维张量） sum在dim的结果就变成了 1024的向量（1位张量） 如果想和r_norm对应元素两两相乘
        # 就需要sum的结果也是2维张量 因此需要使用keepdim= True报纸张量的维度不变
        # 另外关于 dim 等于几表示张量的第几个维度，从0开始计数，可以理解为张量的最开始的第几个左括号，具体可以参考这个https://www.cnblogs.com/flix/p/11262606.html
        head = torch.squeeze(self.ent_embedding(h), dim=1)
        rel = torch.squeeze(self.rel_embedding(r), dim=1)
        rel_mat = torch.squeeze(self.rel_matrix(r), dim=1)
        tail = torch.squeeze(self.ent_embedding(t), dim=1)

        head = self.transfer(head, rel_mat)
        tail = self.transfer(tail, rel_mat)

        head = F.normalize(head, 2, -1)
        rel = F.normalize(rel, 2, -1)
        tail = F.normalize(tail, 2, -1)
        distance = head + rel - tail
        # dim = -1表示的是维度的最后一维 比如如果一个张量有3维 那么 dim = 2 = -1， dim = 0 = -3

        score = torch.norm(distance, p = 2, dim=1)
        return score

    def forward(self, current_triples, corrupted_triples):
        h, t, r = torch.chunk(current_triples, 3, dim=1)
        h_c, t_c, r_c = torch.chunk(corrupted_triples, 3, dim=1)

        # torch.nn.embedding类的forward只接受longTensor类型的张量

        pos = self.distance(h, r, t)
        neg = self.distance(h_c, r_c, t_c)

        # loss_F = max(0, -y*(x1-x2) + margin)
        loss1 = torch.sum(torch.relu(pos - neg + self.margin))
        y = Variable(torch.Tensor([-1]))
        loss = self.loss_F(pos, neg, y)

        return loss



class TransR_Training:
    def __init__(self, entity_set, relation_set, triple_list, embedding_dim=50, lr=0.01, margin=1.0, norm=1, C=1.0, epsilon = 1e-5):
        self.entities = entity_set
        self.relations = relation_set
        self.triples = triple_list
        self.dimension = embedding_dim
        self.learning_rate = lr
        self.margin = margin
        self.norm = norm
        self.loss = 0.0
        self.entity_embedding = {}
        self.norm_relations = {}
        self.hyper_relations = {}
        self.C = C
        self.epsilon = epsilon


    def data_initialise(self, transe_ent = None, transe_rel = None):
        self.model = TransR(len(self.entities), len(self.relations), self.dimension, self.dimension, self.margin)
        self.optim = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        if transe_ent != None and transe_rel != None:
            entity_dic = {}
            relation_dic = {}

            with codecs.open(transe_ent, 'r') as f1, codecs.open(transe_rel, 'r') as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()
                for line in lines1:
                    line = line.strip().split('\t')
                    if len(line) != 2:
                        continue
                    entity_dic[line[0]] = json.loads(line[1])

                for line in lines2:
                    line = line.strip().split('\t')
                    if len(line) != 2:
                        continue
                    relation_dic[line[0]] = json.loads(line[1])
            self.model.input_pre_transe(entity_dic, relation_dic)


    def training_run(self, epochs=2, nbatches=100):

        batch_size = int(len(self.triples) / nbatches)
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0.0


            for batch in range(nbatches):
                batch_samples = random.sample(self.triples, batch_size)

                current = []
                corrupted = []
                change = False
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    p = relation_tph[int(corrupted_sample[2])] / (
                                relation_tph[int(corrupted_sample[2])] + relation_hpt[int(corrupted_sample[2])])
                    if pr > p:
                        # change the head entity
                        corrupted_sample[0] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities, 1)[0]
                    else:
                        # change the tail entity
                        corrupted_sample[1] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[1] == sample[1]:
                            corrupted_sample[1] = random.sample(self.entities, 1)[0]

                    current.append(sample)
                    corrupted.append(corrupted_sample)
                current = torch.from_numpy(np.array(current)).long()
                corrupted =  torch.from_numpy(np.array(corrupted)).long()
                self.update_triple_embedding(current, corrupted)
            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("running loss: ", self.loss)

        # .detach()的作用就是返回一个新的tensor，和原来tensor共享内存，但是这个张量会从计算途中分离出来，并且requires_grad=false
        # 由于 能被grad的tensor不能直接使用.numpy(), 所以要是用。detach().numpy()
        with codecs.open("TransR_pytorch_entity_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f1:

            for i, e in enumerate(self.model.ent_embedding.weight):
                f1.write(str(i) + "\t")
                f1.write(str(e.detach().numpy().tolist()))
                f1.write("\n")

        with codecs.open("TransR_pytorch_reltion_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f2:

            for i, e in enumerate(self.model.rel_embedding.weight):
                f2.write(str(i) + "\t")
                f2.write(str(e.detach().numpy().tolist()))
                f2.write("\n")

        with codecs.open("TransR_pytorch_rel_matrix_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f3:
            for i, e in enumerate(self.model.rel_matrix.weight):
                f3.write(str(i) + "\t")
                f3.write(str(e.detach().numpy().tolist()))
                f3.write("\n")

    def update_triple_embedding(self, correct_sample, corrupted_sample):
        self.optim.zero_grad()
        loss = self.model(correct_sample, corrupted_sample)
        self.loss += loss
        loss.backward()
        self.optim.step()



if __name__ == '__main__':
    file1 = "FB15k\\"
    entity_set, relation_set, triple_list = data_loader(file1)

    transR = TransR_Training(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.01, margin=1.0, norm=1)
    transR.data_initialise("transE_entity_vector_50dim", "transE_relation_vector_50dim")
    transR.training_run()







# 关于叶节点的说明， 整个计算图中，只有叶节点的变量才能进行自动微分得到梯度，任何变量进行运算操作后，再把值付给他自己，这个变量就不是叶节点了，就不能进行自动微分








