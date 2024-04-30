# -*- coding:utf-8 -*-
# !/usr/bin/env python
# @Time       :2022/8/28 下午4:04
# @AUTHOR     :jiajia huang
# @File       : bioTextBuilding.py
'''
word_dict.txt 作为词典
基于外部词典对数据进行标注  BIO方式
启迪设计集团股份有限公司 INT
北京光环新网科技股份有限公司 INT
周口市综合投资有限公司 INT
上海汉得信息技术股份有限公司 INT
湖南湘江新区投资集团有限公司 INT
融信福建投资集团有限公司 INT
湖南尔康制药股份有限公司 INT
厦门灿坤实业股份有限公司 INT
中融国证钢铁行业指数分级证券投资基金 BON
华中证空天一体军工指数证券投资基金 BON
富国新兴成长量化精选混合型证券投资基金 BON
江西省政府一般债券 BON
占位词 NONE


result:
鹏 B-INT
华 I-INT
基 I-INT
金 I-INT
管 I-INT
理 I-INT
有 I-INT
限 I-INT
公 I-INT
司 I-INT
申 O
请 O
， O
本 B-INT
所 I-INT


'''
#将features_dict中的特征词和tag存入字典  特征词为key，tag为value
import re
import os
import codecs
from collections import  defaultdict
import chardet


def get_encoding(file):
    # 二进制方式读取，获取字节数据，检测类型
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']

dict={}
features_len_tuple = []
with open('./data/word_dict_new.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        item = line.strip().split('\t')
        if len(item) > 1 and item[0] not in features_len_tuple:
            features_len_tuple.append((item[0],len(item[0])))
            dict[item[0]]=item[1].replace("\n",'')
        else:
            with open('./data/error.txt', 'a', encoding='utf-8') as f:
                f.write(line + "\n")
    features_len_tuple.sort(key= lambda k:k[1], reverse= True)
    features_list = [each[0] for each in features_len_tuple]
    #print(features_list)
tag2feature = defaultdict(lambda :[])
for k,v in dict.items():
    tag2feature[v].append(k)
for (k, v) in tag2feature.items():
    print(k  +  '+++++ ' +  str(len(v)))


def findLawEntity(sen):
    matches = re.findall(r'《(.*?》)', sen)
    if len(matches) > 0:
        for each in matches:
            entity = each.replace('》','')
            features_list.append(entity)
            dict[entity] = 'audit laws'
            tag2feature['audit laws'].append(entity)
            new_feature.add(entity)


'''
根据字典中的word和tag进行自动标注，用字典中的key作为关键词去未标注的文本中匹配，匹配到之后即标注上value中的tag
'''
dirs = os.listdir('./data/nerForLLM/')
#dirs = os.listdir('/home/hjj/PycharmProjects/AuditKnowledgeGraph/venv/src/Clawer/会计法规/')
new_feature = set([])
for file in dirs:
    file_input = './data/nerForLLM/' + file
    file_output = './data/nerForLLM/nerForLLM_output.txt'
    # file_input = '/home/hjj/PycharmProjects/AuditKnowledgeGraph/venv/src/Clawer/会计法规/' + file
    # file_output = './data/会计法规/' + file

    file_output_count = './data/entity_frequency/' + file
    index_log = 0
    entity_count = defaultdict(lambda: 0)
    NUM_sentences = 0

    k = 0
    encode = get_encoding(file_input)
    count = 0
    
    with open(file_input, 'r', encoding = encode) as f:
        try:

            if file_input.index('会计法规') > 0:
                sen = file.replace('.txt', '')
                all_texts = sen
                findLawEntity(sen)

        except ValueError:
            #print(file_input + ' ' + encode)

            all_texts = ''
        try:
            for line in f.readlines():
                line = line.replace('\n', '').replace('〿','').replace('_百度百科','').strip()
                findLawEntity(line)
                if len(line) >= 15 and all_texts != line:
                    all_texts = all_texts + ' ' + line
        except UnicodeDecodeError:
                line = ''
                print('unable open the file: ', file_input)
                continue
        # try:
        #     all_texts = ''.join([line for line in f.readlines()]).replace('\n', '').replace('〿','').replace('_百度百科','')
        # except :
        #     print('unable open the file: ', file_input)

        result = re.sub('[！。？]', ' \x01', all_texts).split()
        for sen in result:
            sen = sen.strip().replace('\ufeff', '').replace('\x01', '').replace('"', '').replace('^', '').replace(' ', '')
            if len(sen) >= 30 and len(sen) < 128 :
                word_list = list(sen)
                tag_list = ["O" for i in range(len(word_list))]
                has_entity = False

                for keyword in features_list:
                    # print(keyword)
                    while 1:
                        index_start_tag = sen.find(keyword, index_log)
                        # 当前关键词查找不到就将index_log=0,跳出循环进入下一个关键词
                        if index_start_tag == -1:
                            index_log = 0
                            break
                        index_log = index_start_tag + 1
                        # print(keyword, ":", index_start_tag)
                        # 只对未标注过的数据进行标注，防止出现嵌套标注
                        for i in range(index_start_tag, index_start_tag + len(keyword)):
                            if index_start_tag == i:
                                if tag_list[i] == 'O':
                                    tag_list[i] = "B-" + dict[keyword].replace("\n", '')  # 首字
                                    has_entity = True
                            else:
                                if tag_list[i] == 'O':
                                    tag_list[i] = "I-" + dict[keyword].replace("\n", '')  # 非首字
                                    has_entity = True
                        entity_count[keyword] += 1
                if has_entity:
                    NUM_sentences += 1
                    with open(file_output, 'a', encoding='utf-8') as output_f:
                        for w, t in zip(word_list, tag_list):
                            # print(w + " " + t)
                            if w != '	' and w != ' ':
                                output_f.write(w + " " + t + '\n')
                                # output_f.write(w + " "+t)
                        output_f.write('\n')
                        # k += 1
                        # if k % 1000 == 0:
                        #     print(''.join(word_list))
    # count += 1
    # if count%1000 == 0:
    #     print('has finished ', str(count))
    print(file.replace('.txt', '') + '    共有' + str(NUM_sentences) + "  个标注句子。")


#     with open(file_output_count, 'w', encoding='utf-8') as output_f:
#         for k,v in entity_count.items():
#             output_f.write(k + "\t" + str(v) + '\n')
# for each in new_feature:
#     print(each)







