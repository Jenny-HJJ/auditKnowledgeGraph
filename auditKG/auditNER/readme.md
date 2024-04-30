# BERT-BILSTM-CRF
使用BERT-BILSTM-CRF进行审计命名实体识别。 



# 依赖

```python
scikit-learn==1.1.3 
scipy==1.10.1 
seqeval==1.2.2
transformers==4.27.4
pytorch-crf==0.7.2
```

# 目录结构

```
#### ---models/

​	---bioTextLabeling.py 使用BIO格式标注原始sentence
​    ---labelSenClean.py  对标注的句子进行过滤和筛选，获得最终NER训练语料

##### ---models/—BERT-BILSTM-CRF-main

​    --checkpoint：模型和配置保存位置
​    --model_hub：预训练模型  （去huggingface下载）
​    ----chinese-bert-wwm-ext:
​         --------vocab.txt
​         --------pytorch_model.bin
​          --------config.json
​    --data：存放数据
​    ----audit_100  训练语料
​    --------audit.all：原始的数据
​    --------ner_data：处理之后的数据
​    ------------labels.txt：标签
​    ------------train.txt：训练数据
​    ------------dev.txt：测试数据
​    --config.py：配置
​    --model.py：模型
​    --process.py：处理ori数据得到ner数据
​    --predict.py：加载训练好的模型进行预测
​    --main.py：训练和测试
```



```python
1、去https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main下载相关文件到chinese-bert-wwm-ext下。
2、在process.py里面定义将ori_data里面的数据处理得到ner_data下的数据，ner_data下数据样本是这样的：
--labels.txt
--train.txt/dev.txt
{"id": "AT0001", "text":"如 果 财 务 报 表 未 作 出 充 分 披 露 ， 注 册 会 计 师 应 当 按 照 《 中 国 注 册 会 计 师 审 计 准 则 第 1 5 0 2 号 在 审 计 报 告 中 发 表 非 无 保 留 意 见 》 的 规 定 ， 恰 当 发 表 保 留 意 见 或 否 定 意 见", "label": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-PER", "I-PER", "I-PER", "I-PER", "I-PER", "O", "O", "O", "O", "O", "B-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "I-auditLaws", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
一行一条样本，格式为BIO。



3、在config.py里面定义一些参数，比如：
--max_seq_len：句子最大长度，GPU显存不够则调小。
--epochs：训练的epoch数
--train_batch_size：训练的batchsize大小，GPU显存不够则调小。
--dev_batch_size：验证的batchsize大小，GPU显存不够则调小。
--save_step：多少step保存模型
其余的可保持不变。


4、在main.py里面修改data_name为数据集名称。需要注意的是名称和data下的数据集名称保持一致。最后运行：python main.py

5、在predict.py修改data_name并加入预测数据，最后运行：python predict.py
```
### 实验结果

基准模型的准确率、召回率和 F-值

| Model           | Precision | Recall | F1 score |
| --------------- | --------- | ------ | -------- |
| BERT_CRF        | 0.956     | 0.954  | 0.955    |
| BERT_BiLSTM_CRF | 0.965     | 0.964  | 0.964    |
