# -*- coding: utf-8 -*-

import json
import codecs
import os

class ProcessAuditData:
    def __init__(self):
        current_path = os.getcwd()
        print('current path of this file is:', current_path)
        self.data_path = current_path + "/data/audit_50/"
        self.train_file = self.data_path + "ori_data/audit.all"

    def get_ner_data(self):
        res = []
        labels = set([])
        with codecs.open(self.train_file, 'r', encoding="utf-8", errors="replace") as fp:
            sen_id = 0
            wordList = []
            labelList = []
            for line in fp.readlines():
                if len(line.strip()) > 0:
                    try:
                        tmp = line.strip().split(' ')
                        wordList.append(tmp[0])
                        labelList.append(tmp[1])
                        labels.add(tmp[1].replace('I-','').replace('B-',''))
                    except IndexError:
                        print(tmp)
                else:
                    tmp ={}
                    tmp["id"] = str(sen_id)
                    tmp['text'] = [i for i in wordList]
                    tmp["labels"] = [i for i in labelList]
                    labelList = []
                    wordList = []
                    sen_id += 1
                    res.append(tmp)


        print(print('共有  '+ str(len(res)) + " 标注句子。"))
        train_num = int(len(res) * 0.85)
        dev_num = int(len(res)  * 0.1)
        train_data = res[:train_num]
        dev_data = res[train_num: train_num+ dev_num]
        test_data = res[train_num+ dev_num +1 : ]

        print('共有  '+ str(len(train_data)) + " 标注句子作为训练语料。")
        print('共有  ' + str(len(dev_data)) + " 标注句子作为开发语料。")
        print('共有  ' + str( len(res) - train_num- len(dev_data)) + " 标注句子作为测试语料。")
        with open(self.data_path + "ner_data/train.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in train_data]))

        with open(self.data_path + "ner_data/dev.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in dev_data]))

        with open(self.data_path + "ner_data/test.txt", "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in test_data]))


    # 这里标签一般从数据中处理得到，这里我们自定义
        labels = [each for each in labels if each not in ["O"]]
        with open(self.data_path + "ner_data/labels.txt", "w") as fp:
            fp.write("\n".join(labels))



if __name__ == "__main__":

    processAuditData = ProcessAuditData()
    processAuditData.get_ner_data()
