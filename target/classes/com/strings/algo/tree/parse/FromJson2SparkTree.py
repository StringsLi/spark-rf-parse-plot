
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:39:53 2020

@author: lixin
从 json 树转化成spark tree python版本 并且实现了把spark的tree画出来
"""


import os
import json


class Node:
    def __init__(self):
        self.data = 0.0  # 节点的预测值
        self.impurity_stats = []  # 不纯量的统计值，用于后面预测类别概率等
        self.gain = 0.0  # 信息增益量
        self.left_node = None  # 左子树节点
        self.right_node = None  # 右子树节点
        self.feature_i = -1  # 特征的索引
        self.threshold = ""  # 连续变量的阈值，离散变量的左 分裂取值点
        self.split_type = ""  # 是否有离散变量，大于0表示有离散变量，小于0表示变量全部为连续变量
        self.is_leaf = False  # 改节点是否为叶子节点


class ParseFromJson():
    """
      :param file_name: 模型的存储路径
      :param num_classes: 模型处理的是几分类问题
      """
    def __init__(self,file_name,num_classes = 2):
        self.treeNodeList = []
        self.num_classes = num_classes
        self.load_model(file_name)

    def load_model(self, file_path):
        filenames = os.listdir(file_path)  
        filenames.sort(key=lambda x: int(x.replace(".json", "").split("_")[1]))
        for file_name in filenames:
            path = file_path + "/" + file_name
            print(path)
            with open(path) as load_f:
                d = json.load(load_f)
                self.treeNodeList.append(self.walk(d))

    def walk(self,data):
        root = Node()
        if type(data) is dict:
            if data["nodeType"] == "leaf":
                root.is_leaf = True
                root.data = data["prediction"]
                root.impurity_stats = data["impurityStats"]
            else:
                root.data = data["prediction"]
                root.feature_i = data["featureIndex"]
                root.gain = data["gain"]
                root.impurity_stats = data["impurityStats"]
                root.is_leaf = False
                root.split_type = data["splitType"]
                root.threshold = data["threshold"] if data["splitType"] == "continuous" else data["leftCategories"]

                root.left_node = self.walk(data["leftChild"])
                root.right_node = self.walk(data["rightChild"])
        return root


    def predict_pro_one(self,feature_map):
        raw_predict = self.raw_predict(feature_map)
        predict_sum = sum(raw_predict)
        res = list(map(lambda x:x/predict_sum,raw_predict))
        return res

    def predict_one(self,feature_map):
        score = 0.0
        for spark_tree in self.treeNodeList:
            cur_tmp = spark_tree
            while not cur_tmp.is_leaf:
                if cur_tmp.feature_i in feature_map:
                    ## 连续性变量
                    if cur_tmp.split_type == "continuous":
                        if float(feature_map[cur_tmp.feature_i]) <= cur_tmp.threshold:
                            cur_tmp = cur_tmp.left_node
                        else:
                            cur_tmp = cur_tmp.right_node
                    ## 离散型变量情形
                    else:
                        if float(feature_map[cur_tmp.feature_i]) in cur_tmp.threshold:
                            cur_tmp = cur_tmp.left_node
                        else:
                            cur_tmp = cur_tmp.right_node
            score += cur_tmp.data

        return score / len(self.treeNodeList)

    def raw_predict(self,feature_map):
        raw_ret = [0.0 for _ in range(self.num_classes)]
        for spark_tree in self.treeNodeList:
            cur_tmp = spark_tree
            while not cur_tmp.is_leaf:
                if cur_tmp.feature_i in feature_map:
                    ## 连续性变量
                    if cur_tmp.split_type == "continuous":
                        if float(feature_map[cur_tmp.feature_i]) <= cur_tmp.threshold:
                            cur_tmp = cur_tmp.left_node
                        else:
                            cur_tmp = cur_tmp.right_node
                    ## 离散型变量情形
                    else:
                        if float(feature_map[cur_tmp.feature_i]) in cur_tmp.threshold:
                            cur_tmp = cur_tmp.left_node
                        else:
                            cur_tmp = cur_tmp.right_node
            impurity_stats = cur_tmp.impurity_stats
            # print(impurity_stats)
            total = sum(impurity_stats)
            i = 0
            while i < self.num_classes:
                raw_ret[i] += impurity_stats[i] / total
                i += 1
        return raw_ret

    def predict(self, data_lst):
        res = []
        for feature_map in data_lst:
            res.append(self.predict_pro_one(feature_map))
        return res

    """
    columns2index:
    比如 
    {"sepal length (cm)": 0,"sepal width (cm)": 1,"petal length (cm)": 2,"petal width (cm)": 3}
    """
    def read_data(self, file_name,columns2index):
        res = []
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            for line in f:
                dict_data = json.loads(line)
                for key, value in columns2index.items():
                    dict_data.update({value: dict_data.pop(key)})
                res.append(dict_data)
        return res

    def dotgraph(self, decisionTree, columnsDict=None):
        from collections import defaultdict
        dcNodes = defaultdict(list)
        def toString(iSplit, decisionTree, bBranch, szParent="null", indent=''):
            if decisionTree.is_leaf:  # leaf node
                sz = str(decisionTree.data)
                dcNodes[iSplit].append(['leaf', sz, szParent, bBranch])
                return sz
            else:
                szCol = 'feature %s' % decisionTree.feature_i
                if columnsDict != None:
                    szCol = columnsDict[szCol]
                if decisionTree.split_type == "continuous":
                    decision = '%s <= %s?' % (szCol, decisionTree.threshold)
                else:
                    decision = '%s in %s?' % (szCol, decisionTree.threshold)

                toString(iSplit + 1, decisionTree.left_node, True, decision, indent + '\t\t')
                toString(iSplit + 1, decisionTree.right_node, False, decision, indent + '\t\t')
                dcNodes[iSplit].append([iSplit + 1, decision, szParent, bBranch])

        toString(0, decisionTree, None)

        lsDot = ['digraph Tree {',
                 'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
                 'edge [fontname=helvetica] ;'
                 ]
        i_node = 0
        dcParent = {}
        for nSplit in range(len(dcNodes)):
            lsY = dcNodes[nSplit]
            for lsX in lsY:
                iSplit, decision, szParent, bBranch = lsX
                if type(iSplit) == int:
                    szSplit = '%d-%s' % (iSplit, decision)
                    dcParent[szSplit] = i_node
                    lsDot.append('%d [label=<%s>, fillcolor="#e5813900"] ;' % (i_node,
                                                                               decision.replace('<=', '&le;').replace(
                                                                                   '?', '')))
                else:
                    lsDot.append('%d [label=<class %s>, fillcolor="#e5813900"] ;' % (i_node,
                                                                                     decision))
                if szParent != 'null':
                    if bBranch:
                        szAngle = '45'
                        szHeadLabel = 'True'
                    else:
                        szAngle = '-45'
                        szHeadLabel = 'False'
                    szSplit = '%d-%s' % (nSplit, szParent)
                    p_node = dcParent[szSplit]
                    if nSplit == 1:
                        lsDot.append('%d -> %d [labeldistance=2.5, labelangle=%s, headlabel="%s"] ;' % (p_node,
                                                                                                        i_node, szAngle,
                                                                                                        szHeadLabel))
                    else:
                        lsDot.append('%d -> %d ;' % (p_node, i_node))
                i_node += 1
        lsDot.append('}')
        dot_data = '\n'.join(lsDot)
        return dot_data

if __name__ == "__main__":
    resource_path = "C:\\Users\\morni\\Desktop\\github\\spark-random-forest-parse\\src\\main\\resources"

    file_name = resource_path + "\\json_model\\"
    parseSpark = ParseFromJson(file_name)

    tree_list = parseSpark.treeNodeList

    count = 0
    for tree in tree_list:
        dot_text = parseSpark.dotgraph(tree)
        import pydotplus
        graph = pydotplus.graph_from_dot_data(dot_text)
        print("=============>>>>>>>>>>>>>>>>%d"%count)
        graph.write_png(resource_path + "\\plot\\spark_tree_%d.png"%(count))
        count += 1

    testfilename = resource_path + "\\data\\iris.json"
    cols2index = {"petal length (cm)": 0, "petal width (cm)": 1, "sepal length (cm)": 2, "sepal width (cm)": 3}
    test_data = parseSpark.read_data(testfilename, cols2index)

    for dict in test_data:
        pred1 = parseSpark.predict_pro_one(dict)
        print(dict, pred1)