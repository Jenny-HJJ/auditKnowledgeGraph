import codecs
import csv

def load_dict(filename):
    entities = set([])
    f_src = open(filename, "rb")
    lines = f_src.readlines()
    for word in lines:
        entities.add(word.strip().decode('utf-8'))
    return entities



def load_file(filename):
    corpus = []
    f_src = codecs.open(filename, 'r', 'utf-8').readlines()
    for  line in f_src:
        tmp1= line.strip().replace(" ", '')
        tmp1= tmp1.strip().replace("\t", '')
        if len(tmp1) >= 5:
            corpus.append(tmp1)
    return corpus


def load_tupleAsDict(filename):

    dict_result  = {}
    f_src = codecs.open(filename, 'r', 'utf-8').readlines()
    for  line in f_src:
        tmp = line.strip().replace(" ", '').split('\t')
        entity1 = tmp[0]
        entity2 = tmp[1]
        if entity2 not in dict_result:
            dict_result[entity1] = entity2
    return  dict_result

def load_tuple(filename):
    corpus = []
    f_src = codecs.open(filename, 'r', 'utf-8').readlines()
    for  line in f_src:
        tmp = line.strip().split(' ')
        entity = tmp[0]
        doc = ''.join(tmp[1:])
        corpus.append((entity, doc))
    return  corpus


def load_triple(filename):
    corpus = []
    f_src = codecs.open(filename, 'r', 'utf-8').readlines()
    for  line in f_src:
        tmp = line.strip().replace(" ", '').split('\t')
        if len(tmp) >= 3:
            entity = tmp[0]
            relation = tmp[1]
            doc = ''.join(tmp[2:])
            corpus.append((entity,relation, doc))
    return  corpus


def writeTriple(triples, filename):
    ef = open(filename, "w")
    for each in triples:
        for i in range(len(each) -1) :
            ef.write(each[i] + "\t" )
        ef.write(each[i+1] + '\n')
    ef.close()


def writeTuple(triples, filename):
    ef = open(filename, "w")
    for each in triples:
        ef.write(each[0] + "\t" + each[1] + "\n" )
    ef.close()


def writeList(list, filename):
    ef = open(filename, "w")
    for each in list:
        ef.write(each+ "\n" )
    ef.close()



def writeDict(dict, filename):
    ef = open(filename, "w")
    for k,v in dict.items():
        ef.write(k + "\t" + str(v) + "\n")
    ef.close()



def writeCSV(result, header, filename):
    f = open(filename, 'w')
    writer = csv.writer(f)
    writer.writerow(header)
    for row in result:
        writer.writerow(row)
    f.close()
