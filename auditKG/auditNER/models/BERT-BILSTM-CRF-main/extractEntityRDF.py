
# -*- coding:utf-8 -*-
'''根据命名实体抽取模型从审计案例文档中抽取对应的rdf，包括
（1）：按照模板（一）、（二）等格式抽取RDF
（2）：抽取命名实体及其类型，构建RDF
'''
from  predict import Predictor
from util2 import load_tupleAsDict, writeTriple

def extractEntityfromDoc(corpus, tag_dict, predictor):
    doc_extracted_result = []
    object_entity_tag =  ['publicworkAudit', 'fiscalAudit', 'financialAudit', 'customsAudit',
                          'companyAudit','revenueAudit','socialInsuranceAudit','environmentalAudit', 'economicResponsibilityAudit']
    relation_dict =   {
        "PER":"None",
        "auditLaws":"law_of_audit",
        "accountantSubject":"item_of_audit",
        "auditProblem":'fraud_of_audit',
        "auditORG":"audited_of_org",
        "auditMethod":'method_of_audit',
        "ORG":'org_of_audit',
        "auditRisk":'risk_of_audit',
        "auditAchievement":"achievement_of_audit",
        "industry":"included_domain"
    }
    i = 1
    for doc in corpus:
        result = predictor.ner_predict(doc)
        if i%10000 == 0:
            print('finish :', str(i))
            print('doc：  ', doc)
            print(result.items())
            print('====================')

        entity_set = {
            "PER":set([]),
            "auditLaws":set([]),
            "accountantSubject":set([]),
            "auditProblem":set([]),
            "auditORG":set([]),
            "auditMethod":set([]),
            "ORG":set([]),
            "auditRisk":set([]),
            "auditAchievement":set([]),
            "industry":set([])}
        object_entity = set([])
        all_triple = []
        for label, NElist in result.items():
            for each in NElist:
                if len(each[0]) >= 2 :
                    all_triple.append((tag_dict[label],'instance_of', each[0]))
                    if label in object_entity_tag:
                        object_entity.add(each[0])
                    else:
                        entity_set[label].add(each[0])
                        flag = True
        if len(entity_set['auditProblem']) > 0 and len(entity_set['auditLaws']) > 0 :
            for each1 in entity_set['auditProblem']:
                for each2 in entity_set['auditLaws']:
                    triple = (each1, relation_dict['auditLaws'], each2)
                    all_triple.append(triple)

        for oe in object_entity:
            for tag, entity_list in entity_set.items():
                if len(entity_list) > 0 and tag != 'auditLaws':
                    relation = relation_dict[tag]
                    for e in entity_list:
                        a_triple = (oe, relation, e)
                        all_triple.append(a_triple)

        if len(all_triple) > 0 :
            for each in all_triple:
                # print('doc:', doc)
                # print('rdf is: ', '\t'.join(each))
                # print('+++++++++++++++++++++++++++')
                doc_extracted_result.append(('\t'.join(each), doc))

        # if len(object_entity) > 0 and flag == True:
        #     tmp_triple = []
        #     for oe in object_entity:
        #         for tag, entity_list in entity_set.items():
        #             if len(entity_list) > 0:
        #                 relation = relation_dict[tag]
        #                 for e in entity_list:
        #                     a_triple = (oe, relation, e)
        #                     tmp_triple.append(a_triple)
        #
        #     if len(tmp_triple) > 0:
        #         for each in tmp_triple:
        #             print('doc:', doc)
        #             print('rdf is: ', '\t'.join(each))
        #             doc_extracted_result.append((doc, '\t'.join(each)))
        i += 1
    return doc_extracted_result




def extractEntityasTriple(corpus, tag_dict, predictor):

    triple_result = set([])
    object_entity_tag =  ['publicworkaudit', 'fiscalaudit', 'financialaudit', 'customsaudit','companyaudit',
                         'revenueaudit','socialinsuranceaudit','environmentalaudit']
    relation_dict = {
                       "auditlaws":"law_of_audit",
                       "accountantsubject":"item_of_audit",
                       "auditproblem":'fraud_of_audit',
                       "auditorg":"audited_of_org",
                       "auditmethod":'method_of_audit',
                       "ORG":'org_of_audit',
                       "auditrisk":'risk_of_audit',
                       "auditachievement":"achievement_of_audit",
                       "industry":"included_domain"
                       }

    count = 0

    for doc in corpus:
        result = predictor.ner_predict(doc)

        # print('doc：  ', doc)
        # print(result['entities'])
        # print('====================')
        flag = False
        entity_set = {
                          "audittarget":[],
                          "auditlaws":[],
                          "accountantsubject":[],
                          "auditproblem":[],
                          "auditorg":[],
                          "auditmethod":[],
                          "ORG":[],
                          "auditrisk":[],
                          "auditachievement":[],
                          "industry":[]}

        object_entity = set([])
        for label, NElist in result.items():
            for each in NElist:
                try:
                    triple_result.add((tag_dict[label],'instance_of', each[0]))
                except KeyError:
                    print(label + ' not in tag_dict')
                    break
                if label in object_entity_tag:
                    object_entity.add(each[0])
                else:
                    entity_set[label].append(each[0])
                    flag = True

        if len(object_entity) > 0 and flag ==True:
            for oe in object_entity:
                for tag, entity_list in entity_set.items():
                    if len(entity_list) > 0:
                        relation = relation_dict[tag]
                        for e in entity_list:
                            a_triple = (oe, relation, e)
                            triple_result.add(a_triple)
        count+= 1
        if count%100 == 0:
            print('has finished :', count)
            print(doc)
            for label, NElist in result.items():
                print(label + '  ' )
                print(NElist)
    return triple_result


def load_alL_corpus(filename):
    result_corpus = []
    f = open(filename, 'r', encoding='utf-8')
    lines = f.readlines()

    for sen in lines:
        sen = sen.strip().replace('\n','')
        if len(sen) >=12 and len(sen) <128:
            result_corpus.append(sen)
    f.close()
    print('total senences in the corpus is : ',len(result_corpus))
    return result_corpus


def load_alL_corpus1(filename):
    result_corpus = []
    f = open(filename, 'r', encoding='utf-8')
    lines = f.readlines()

    for line in lines:
        sens = line.strip().replace('\n','').split('。')
        for  sen in sens:
            if len(sen) >=50 and len(sen) <128:
                result_corpus.append(sen)
    f.close()
    print('total senences in the corpus is : ',len(result_corpus))
    return result_corpus

import os
if __name__ == '__main__':
    fatherDirectory = '/home/hjj/PycharmProjects/BertNER/data/'
    #fatherDirectory = '/root/autodl-tmp/BertNER/data'
    tag_dict = load_tupleAsDict(fatherDirectory + '/tag_dict')
    all_corpus = []
    filepath = '/home/hjj/PycharmProjects/AuditKnowledgeGraph/venv/src/NameEntityRecogntion/data/raw/raw2/'
    #with open( filepath + '1保险案例选编-270.txt', "r") as fp:
    dirs = os.listdir(filepath)
    # for file in dirs:
    #     all_corpus += load_alL_corpus1(filepath + file)
    all_corpus = load_alL_corpus1(fatherDirectory + '/corpus/法规条款.txt' )
    data_name = "audit_100"
    predictor = Predictor(data_name)
    #triple_result = extractEntityasTriple(all_corpus, tag_dict, predictor)
    doc_extracted_result = extractEntityfromDoc(all_corpus, tag_dict, predictor)
    print('共抽取三元组文档：  ', len(doc_extracted_result))
    #writeTriple(triple_result, fatherDirectory + 'result/审计词典_文本三元组_all.txt')
    writeTriple(doc_extracted_result, fatherDirectory + 'result1/LLforLaw.txt')
