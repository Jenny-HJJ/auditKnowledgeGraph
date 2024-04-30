（1）使用审计领域文本继续预训练bert模型
（2）使用PCA对该模型进行降为
（3）测试与使用该模型：




 sens = ['保险收入', '财务报表日后资本公积转增资本',
'长期资产的减值准备', '长期银行借款',
 '加工贸易电子化手册','加工贸易业务流程',
'违规提取个人住房公积金账号内存储余额','违规提取个人住房公积金账号' ]
 test_model(sens)




def test_model(sens):
    path = sys.path[0]
    bert_dir =  path + '/fineTuningBert/'
    #bert_dir =  path + '/bert-base-chinese/'
    model = SentenceTransformer(bert_dir)
    #Compute embedding for both lists
    embeddings1 = model.encode(sens, convert_to_tensor=True)
    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings1)
    #Output the pairs with their score
    print(cosine_scores)
