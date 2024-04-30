'''根据已有命名实体识别模型，从海量数据中重新识别所有命名实体，并标注识别出的命名实体是在已有label中还是新识别出来的
若是新识别的实体，则给出该实体对应的原文集合，以判断该实体是否是漏报实体'''

import csv
from  predict import Predictor
import codecs


def  loadLabeledNE():
    NE2Labeldict={}
    with open('./data/audit_raw/word_dict_new.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            item = line.strip().split('\t')
            if len(item) > 1 :
                NE2Labeldict[item[0]]=item[1].replace("\n",'')
            else:
                with open('./data/error.txt', 'a', encoding='utf-8') as f:
                    f.write(line + "\n")
    f.close()
    print('number of NER labeled is  ', len(NE2Labeldict))
    return NE2Labeldict



def writeCSV(NEResult, filename):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['entityName', 'NELabel', 'text'])
        for entity, value in NEResult.items():
            labels = ' '.join([each for each in  value['label']])
            sents = ' '.join([each for each in  value['sentence']])
            writer.writerow([entity, labels, sents])
    print("保存文件成功，处理结束")


import os
if __name__ == "__main__":
    #data_name = "dgre"
    data_name = "audit_50_old_bert"
    predictor = Predictor(data_name)

    labeled_NE = loadLabeledNE()
    labeled_NE_keys = labeled_NE.keys()
    filepath = '/home/hjj/PycharmProjects/AuditKnowledgeGraph/venv/src/NameEntityRecogntion/data/raw/raw2/'
    #with open( filepath + '1保险案例选编-270.txt', "r") as fp:
    dirs = os.listdir(filepath)
    sentences = ''
    for file in dirs:
        with open( filepath + file, "r") as fp:
            sentences += fp.read().split("\n")
    # with open( "./data/audit_raw/rawSentence.txt", "r") as fp:
    #     sentences = fp.read().split("\n")
    NEResult  = {}
    for text in sentences:
        ner_result = predictor.ner_predict(text)
        #print(ner_result)
        for label,NElist in ner_result.items():
            for entity in NElist:
                entity = entity[0]
                if entity not in labeled_NE_keys:
                    if entity not in NEResult:
                        NEResult[entity] = {'label': set([label]), 'sentence': [text]}
                    else:
                        NEResult[entity]['label'].add(label)
                        if len(NEResult[entity]['sentence']) <= 5:
                            NEResult[entity]['sentence'].append(text)

    for entity, value in NEResult.items():
        print(entity + '   ' ,end= '  '),
        print(' '.join([each for each in  value['label']]),end= '  '),
        print(value['sentence'])

    print(' number of labeled entity is  ', len(NEResult))
    writeCSV(NEResult, './data/audit_raw/new_labeling_entities.csv')









