import json
import operator # operator模块输出一系列对应Python内部操作符的函数
import time

import numpy as np
import codecs
import torch
import torch.nn as nn

from TransR import data_loader,entity2id,relation2id


def test_data_loader(entity_embedding_file, relation_embedding_file, relation_matrix_file, test_data_file):
    print("load data...")
    file1 = entity_embedding_file
    file2 = relation_embedding_file
    file3 = relation_matrix_file
    file4 = test_data_file

    entity_dic = {}
    relation_dic = {}
    relation_matrix = {}
    triple_list = []

    with codecs.open(file1, 'r') as f1, codecs.open(file2, 'r') as f2, codecs.open(file3, 'r') as f3:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        lines3 = f3.readlines()
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

        for line in lines3:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation_matrix[line[0]] = json.loads(line[1])

    with codecs.open(file4, 'r') as f4:
        content = f4.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            head = entity2id[triple[0]]
            tail = entity2id[triple[1]]
            relation = relation2id[triple[2]]

            triple_list.append([head, tail, relation])

    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_dic), len(relation_dic), len(triple_list)))

    return entity_dic, relation_dic, relation_matrix, triple_list

class testTransH:
    def __init__(self, entities_dict, relation, relation_matirx, test_triple_list, train_triple_list, filter_triple=False, n=2500 ,norm=1):
        self.entities = entities_dict
        self.relations = relation
        self.rel_matrix = relation_matirx
        self.test_triples = test_triple_list
        self.train_triples = train_triple_list
        self.filter = filter_triple
        self.norm = norm
        self.n = n
        self.mean_rank = 0
        self.hit_10 = 0
        self.ent_dim = 50
        self.rel_dim = 50

    # def test_theading(self, test_triple):
    #     hits = 0
    #     rank_sum = 0
    #     num = 0
    #
    #     for triple in test_triple:
    #         num += 1
    #         print(num, triple)
    #         rank_head_dict = {}
    #         rank_tail_dict = {}
    #         #
    #         for entity in self.entities.keys():
    #             head_triple = [entity, triple[1], triple[2]]
    #             if self.filter:
    #                 if head_triple in self.train_triples:
    #                     continue
    #             head_embedding = self.entities[head_triple[0]]
    #             tail_embedding = self.entities[head_triple[1]]
    #             norm_relation = self.norm_relation[head_triple[2]]
    #             hyper_relation = self.hyper_relation[head_triple[2]]
    #             distance = self.distance(head_embedding, norm_relation,hyper_relation, tail_embedding)
    #             rank_head_dict[tuple(head_triple)] = distance
    #
    #         for tail in self.entities.keys():
    #             tail_triple = [triple[0], tail, triple[2]]
    #             if self.filter:
    #                 if tail_triple in self.train_triples:
    #                     continue
    #             head_embedding = self.entities[tail_triple[0]]
    #             tail_embedding = self.entities[tail_triple[1]]
    #             norm_relation = self.norm_relation[tail_triple[2]]
    #             hyper_relation = self.hyper_relation[tail_triple[2]]
    #             distance = self.distance(head_embedding, norm_relation, hyper_relation, tail_embedding)
    #             rank_tail_dict[tuple(tail_triple)] = distance
    #
    #         # itemgetter 返回一个可调用对象，该对象可以使用操作__getitem__()方法从自身的操作中捕获item
    #         # 使用itemgetter()从元组记录中取回特定的字段 搭配sorted可以将dictionary根据value进行排序
    #         # sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
    #         '''
    #         sorted(iterable, cmp=None, key=None, reverse=False)
    #         参数说明：
    #         iterable -- 可迭代对象。
    #         cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
    #         key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    #         reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
    #         '''
    #         rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
    #         rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1), reverse=False)
    #
    #         # calculate the mean_rank and hit_10
    #         # head data
    #         for i in range(len(rank_head_sorted)):
    #             if triple[0] == rank_head_sorted[i][0][0]:
    #                 if i < 10:
    #                     hits += 1
    #                 rank_sum = rank_sum + i + 1
    #                 break
    #
    #         # tail rank
    #         for i in range(len(rank_tail_sorted)):
    #             if triple[1] == rank_tail_sorted[i][0][1]:
    #                 if i < 10:
    #                     hits += 1
    #                 rank_sum = rank_sum + i + 1
    #                 break
    #     return hits, rank_sum


    def test_run(self):
        hits = 0
        rank_sum = 0
        num = 0

        for triple in self.test_triples:
            start = time.time()
            num += 1
            print(num, triple)
            rank_head_dict = {}
            rank_tail_dict = {}
            #
            head_embedding = []
            tail_embedding = []
            r_matrix = []
            rel = []
            tamp = []


            for i, entity in enumerate(self.entities.keys()):

                head_triple = [entity, triple[1], triple[2]]
                if self.filter:
                    if head_triple in self.train_triples:
                        continue
                head_embedding.append(self.entities[head_triple[0]])
                tail_embedding.append(self.entities[head_triple[1]])
                rel.append(self.relations[head_triple[2]])
                r_matrix.append(self.rel_matrix[head_triple[2]])
                tamp.append(tuple(head_triple))


            s = time.time()
            distance = self.distance(head_embedding, rel, r_matrix, tail_embedding)
            e = time.time()
            print("cost time: %s" % (round((e - s), 3)))

            for i in range(len(tamp)):
                rank_head_dict[tamp[i]] = distance[i]
            head_embedding = []
            tail_embedding = []
            r_matrix = []
            rel = []
            tamp = []

            for i, tail in enumerate(self.entities.keys()):

                tail_triple = [triple[0], tail, triple[2]]
                if self.filter:
                    if tail_triple in self.train_triples:
                        continue
                head_embedding.append(self.entities[tail_triple[0]])
                tail_embedding.append(self.entities[tail_triple[1]])
                rel.append(self.relations[tail_triple[2]])
                r_matrix.append(self.rel_matrix[tail_triple[2]])
                tamp.append(tuple(tail_triple))

            distance = self.distance(head_embedding, rel, r_matrix, tail_embedding)
            for i in range(len(tamp)):
                rank_tail_dict[tamp[i]] = distance[i]

            # for entity in self.entities.keys():
            #
            #     head_triple = [entity, triple[1], triple[2]]
            #     if self.filter:
            #         if head_triple in self.train_triples:
            #             continue
            #     head_embedding = self.entities[head_triple[0]]
            #     tail_embedding = self.entities[head_triple[1]]
            #     rel = self.relations[head_triple[2]]
            #     r_matrix = self.rel_matrix[head_triple[2]]
            #     distance = self.distance(head_embedding, rel, r_matrix, tail_embedding)
            #     rank_head_dict[tuple(head_triple)] = distance
            #
            #
            # for tail in self.entities.keys():
            #     tail_triple = [triple[0], tail, triple[2]]
            #     if self.filter:
            #         if tail_triple in self.train_triples:
            #             continue
            #     head_embedding = self.entities[tail_triple[0]]
            #     tail_embedding = self.entities[tail_triple[1]]
            #     rel = self.relations[tail_triple[2]]
            #     r_matrix = self.rel_matrix[tail_triple[2]]
            #     distance = self.distance(head_embedding, rel, r_matrix, tail_embedding)
            #     rank_tail_dict[tuple(tail_triple)] = distance



            # itemgetter 返回一个可调用对象，该对象可以使用操作__getitem__()方法从自身的操作中捕获item
            # 使用itemgetter()从元组记录中取回特定的字段 搭配sorted可以将dictionary根据value进行排序
            # sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
            '''
            sorted(iterable, cmp=None, key=None, reverse=False)
            参数说明：
            iterable -- 可迭代对象。
            cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
            key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
            reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
            '''
            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1), reverse=False)

            # calculate the mean_rank and hit_10
            # head data
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            # tail rank
            for i in range(len(rank_tail_sorted)):
                if triple[1] == rank_tail_sorted[i][0][1]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break
            end = time.time()
            print("epoch: ", num, "cost time: %s" % (round((end - start), 3)), str(hits / (2 * num)),
                  str(rank_sum / (2 * num)))
        self.hit_10 = hits / (2 * len(self.test_triples))
        self.mean_rank = rank_sum / (2 * len(self.test_triples))

        return self.hit_10, self.mean_rank


    def distance(self, h, r, r_mat, t):
        head = torch.from_numpy(np.array(h)).cuda()
        rel = torch.from_numpy(np.array(r)).cuda()
        rel_mat = torch.from_numpy(np.array(r_mat)).cuda()
        tail = torch.from_numpy(np.array(t)).cuda()

        head = self.transfer(head, rel_mat)
        tail = self.transfer(tail, rel_mat)

        distance = head + rel - tail
        score = torch.norm(distance, p=2, dim=1)
        return score.cpu().numpy()

    def transfer(self, e, rel_mat):
        # view 的作用是重新将一个tensor转化成另一个形状
        # 数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的tensor
        # -1 表示根据其他维度来自动计算这一维的数量
        rel_matrix = rel_mat.view(-1, self.ent_dim, self.rel_dim)
        e = e.view(-1, 1, self.ent_dim)
        e = torch.matmul(e, rel_matrix)

        return e.view(-1, self.rel_dim)



if __name__ == "__main__":
    _, _, train_triple = data_loader("FB15k\\")

    entity, relation, relation_matrix, test_triple = test_data_loader("TransR_pytorch_entity_50dim_batch4831",
                                                               "TransR_pytorch_reltion_50dim_batch4831",
                                                               "TransR_pytorch_rel_matrix_50dim_batch4831",
                                                               "FB15k\\test.txt")

    test = testTransH(entity, relation, relation_matrix, test_triple, train_triple, filter_triple=False, n=2500, norm=2)
    hit10, mean_rank = test.test_run()
    print("raw entity hits@10: ", hit10)
    print("raw entity meanrank: ",mean_rank)

    # test2 = testTransH(entity, norm_relation, hyper_relation, test_triple, train_triple, filter_triple=True, n=2500, norm=2)
    # filter_hit10, filter_mean_rank = test2.test_run()
    # print("filter entity hits@10: ", filter_hit10)
    # print("filter entity meanrank: ", filter_mean_rank)
