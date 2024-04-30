
# coding:utf-8
'''
   function：从w2v model 中获取部分 关键词的 vector
   author: jiajia huang
'''

def read_file(read_filename):
    f = open(read_filename, 'r', encoding='utf-8')
    data_list2 = f.readlines()
    f.close()
    results = []

    for line in data_list2:
        line = line.strip()
        if len(line) > 0:
            results.append(line)
    return results


def writeDict(wdict, filename):
    if len(wdict) > 0:
        f = open(filename, 'w', encoding='utf-8')
        for k, v in wdict.items():
            if type(v) == "list":
                v_str = " ".join(v)
            else:
                v_str = str(v)
            f.write(k + "\t" + v_str + "\n")
        f.close()
    else:
        print("It is error dict.")



def writeSens(sents, filename):
    if len(sents) > 0:
        f = open(filename, 'w', encoding='utf-8')
        for sen in sents:
            for each in sen:
                f.write(each + '\n')
            f.write('\n')
        f.close()