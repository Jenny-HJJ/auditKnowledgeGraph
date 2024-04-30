# -*- coding:utf-8 -*-
# !/usr/bin/env python
# @Time       :2022/9/5 下午3:12
# @AUTHOR     :jiajia huang
# @File       : labelSenClean.py
'''对标注命名实体的句子进一步统计各类命名实体出现次数，筛选出语料，并构建训练集、开发集和测试集。
'''
import os
from collections import defaultdict
import random
from util import  writeDict, writeSens

def getFeature():
    features2tag = {}
    removed_feature = set(['银行','会计','预算','收入','贷款','法律','法规','检查','分配','价格','中心','货币','目录','查看','规划',
                           '，内容不完整','、扣缴义务人编造虚假计税依据','、扣缴义务人逃避、拒绝或者以其他方式阻挠税务机关检查',
                           '、扩大范围收费','、与与建设单位或者施工单位串通，弄虚作假', '保障性安居','报不准确','本或股本', '本养老保险', '变动情况',
                           '不劳而获', '不真实、不', '财务业'])
    with open('./data/word_dict_new.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = line.strip().split('\t')
            if len(item) > 1 and item[0] not in removed_feature :
                features2tag[item[0]] = item[1].replace("\n", '')
    return features2tag


def getFeatureLableFrequency(features2tag):
    features2frequency = defaultdict(lambda : 0)
    removed_feature = set(['银行','审计署','会计','预算','收入','贷款','法律','法规','检查','分配','价格','中心','货币','目录','查看','规划'])
    tag2frequency = defaultdict(lambda : 0)
    dirs = os.listdir('./data/entity_frequency')
    for file in dirs:
        filename = './data/entity_frequency/' + file
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                item = line.strip().split('\t')
                if item[0] in features2tag.keys() and item[0] not in removed_feature:
                    tag = features2tag[item[0]]
                    features2frequency[item[0]] += int(item[1])
                # else:
                #     tag = 'audit laws'
                    tag2frequency[tag] += 1

    return features2frequency, tag2frequency

def sort_sentences(input_sens):
    all_sentences = []
    for sen, updated_sen in input_sens:
        tag_count = 0
        for each in updated_sen:
            tag = each.split(" ")[1].strip()
            if tag.split('-')[0] == 'B':
                tag_count += 1
        all_sentences.append((sen, updated_sen, tag_count))
    sorted_all_sentences = sorted(all_sentences, key=lambda x: x[2], reverse=True)
    return sorted_all_sentences

def getLabeledSens(features2tag):

    total_sents = []
    total_num_sents = 0
    dirs = os.listdir('./data/cut_data')
    all_sens = []
    for file in dirs:
        filename = './data/cut_data/' + file
        with open(filename, 'r', encoding='utf-8') as f:
            wordList = []
            labeledSen = []
            for line in f.readlines():
                if len(line.strip()) > 0:
                    tmp = line.strip().split(' ')
                    word = tmp[0]
                    if word not in [',','〿', '▿','丿','皿','＿'] :
                        #updated_sen =word + ' ' + ''.join([each.capitalize() for each in tmp[1:]])
                        updated_sen = word + ' ' + ''.join([each for each in tmp[1:]])
                        labeledSen.append(updated_sen)
                        wordList.append(word)
                else:
                    total_num_sents += 1
                    sen = ('').join(wordList)
                    all_sens.append((sen, labeledSen))
                    labeledSen = []
                    wordList = []
    print('原始数据中共有' + str(total_num_sents) + '  个句子。')
    writeRawSen(all_sens, './data/corpus/rawSentence_all.txt')

    #ramdon all the sentence and select some ones.
    #random.shuffle(all_sens)
    sorted_all_sentences = sort_sentences(all_sens) # 所有的句子按照其出现的 NE 的数量将序排列
    acctual_feature2frequency = defaultdict(lambda : 0)
    all_entities = features2tag.keys()
    for each in sorted_all_sentences:
        tmp_exit_feature = []
        for entity in all_entities:
            if each[0].find(entity) != -1: #记录每个句子中出现哪些实体
                tmp_exit_feature.append(entity)
        isAdd = False
        for entity in tmp_exit_feature:
            if acctual_feature2frequency[entity] < 50: # 若所有出现的实体已出现超过50次，则不添加 该句子
                isAdd = True
        if isAdd == True:
            total_sents.append(each[1])  # add BIO label sentences
            for entity in tmp_exit_feature:
                acctual_feature2frequency[entity] += 1

    writeDict(acctual_feature2frequency, './data/final_feature2frequency_50_all.txt')
    print('筛选后，共有' + str(len(total_sents)) + '  个句子。')
    return  total_sents


def writeRawSen(sents, filename):
    f = open(filename, 'w', encoding='utf-8')
    for sen in sents:
        f.write(sen[0] + '\n')
    f.close()



def writeSens(sents, filename):
    total_len = 0
    if len(sents) > 0:
        f = open(filename, 'w', encoding='utf-8')
        for sen in sents:
            for each in sen:
                f.write(each + '\n')
                total_len += 1
            f.write('\n')
        f.close()
    print('平均文本长度为: ', total_len/len(sents))



def countTagFre(sents):
    tag2frequency = defaultdict(lambda: 0)
    for sen in sents:
        for each in sen:
            tmp = each.split('-')
            if len(tmp) >= 2 :
                try:
                    if tmp[0].split()[1] == 'B':
                        tag = tmp[1]
                        tag2frequency[tag] += 1

                except Exception:
                    pass

    writeDict(tag2frequency, './data/final_tag2frequency_all.txt')


def buildSents(total_sents):
    #随机化所有句子
    NUM_sents = len(total_sents)
    random.shuffle(total_sents)
    countTagFre(total_sents)
    print('共有  '+ str(NUM_sents) + " 标注句子。")
    NUM_train = int(NUM_sents * 0.7)
    NUM_dev = int(NUM_sents * 0.2)
    print('共有  '+ str(NUM_train) + " 标注句子作为训练语料。")
    print('共有  ' + str(NUM_dev) + " 标注句子作为开发语料。")
    print('共有  ' + str( NUM_sents - NUM_train- NUM_dev) + " 标注句子作为测试语料。")
    train_sents = total_sents[:NUM_train]
    dev_sents = total_sents[NUM_train+1: NUM_train+NUM_dev]
    test_sents = total_sents[NUM_train+ NUM_dev+1 : ]
    writeSens(total_sents, './data/corpus/audit_50_update/audit.all')
    writeSens(train_sents, './data/corpus/audit_50_update/audit.train')
    writeSens(dev_sents, './data/corpus/audit_50_update/audit.dev')
    writeSens(test_sents, './data/corpus/audit_50_update/audit.test')


# def tokenAnalysis():
#     acctual_feature2frequency = defaultdict(lambda : 0)
#     all_sents = []
#     with open(filename, 'r', encoding='utf-8') as f:
#         wordList = []
#         for line in f.readlines():
#             if len(line.strip()) > 0:
#                 tmp = line.strip().split(' ')
#                 word = tmp[0]
#                 wordList.append(word)
#             else:
#                 sen = ('').join(wordList)
#                 wordList = []
#                 for entity in features2tag.keys():
#                     if sen.find(entity) != -1: #记录每个句子中出现哪些实体
#                         acctual_frequres2frequency[entity] += 1




if __name__ == "__main__":
    features2tag = getFeature()
    # features2frequency, tag2frequency = getFeatureLableFrequency(features2tag)
    # writeDict(features2frequency, './data/features2frequency.txt')
    # writeDict(tag2frequency, './data/tag2frequency.txt')
    total_sents = getLabeledSens(features2tag)
    buildSents(total_sents)

