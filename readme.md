# 审计知识图谱

## 一、 研究动机

大数据时代审计对象所产生的数据量日益庞大，进而对现有审计数据基础和审计分析方法提出了新要求。国家审计署相关领导也指出应推进以大数据为核心的审计信息化建设，构建大数据审计工作模式，积极开展审计大数据的综合利用[3]。人工智能领域专家认为，有效利用大数据价值的主要任务不是获取越来越多的数据，而是从数据中挖掘知识，对知识进行有效的组织关联，并用其解决实际问题。当前审计方法主要是直接利用现有文本/数据挖掘算法，鲜有研究深入考虑审计领域专业性对人工智能方法的挑战。

本文认为构建领域知识库有必要先设计出一个科学合理、能够得到领域专家和从业人员广泛认可的领域知识表示结构，以尽可能全面准确地囊括领域共性知识。

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml12668\wps13.png)

​                                                                                       审计结构框架示意图





![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml12668\wps14.png)

​                                                  图2    基于**信息抽取的审计知识库构建路线图

## 二、 目录结构

--- rawdataset/

​	----/审计法规语料/   

​		------  .txt  法律法规数据原始形式

​		------  .xlsx 基于规则抽取出的三元组长文档形式

​	-----/中国会计视野/   中国会计视野网中获得的数据形式

​	-----审计大辞典.txt   审计大辞典数据样例

​	-----审计常见问题定性与处理处罚参考.txt  数据样例

​	------# 审计知识图谱原始语料抽取与汇总.md  readme

---/auditKG/

​	------/auditNER/  审计命名实体识别

​	------/finalauditKGRDF/ 最终构建的审计三元组

​		------/rdf_all.csv  所有auditKG三元组

​		------/本体层.csv  审计本体结构

​		------/所有实体.txt  人工标注的实体

​		------/所有_关系统计.txt  auditKG 中出现的关系类型统计

​		------/tag_dict  实体类别中英文对照

---/LLM benchmark/  基于auditKG构建的大模型评测语料库

---/audit_bert_model/  利用审计领域文本继续预训练Bert





## 三、审计知识图谱概要

### 1、基于结构化和半结构化信息抽取的审计三元组汇总

|          | **层级** | **实体数量** | **RDF数量** | **关系数量** | **关系示例**                            |                             |
| -------- | -------- | ------------ | ----------- | ------------ | --------------------------------------- | --------------------------- |
| 本体层   |          |              | 875         | 899          | 1                                       | subconcept_of               |
| 实例层   | 实例层   | 实例         | 58478       | 62908        | 2                                       | instance_of, item_of        |
|          | 其他属性 | 152951       | 116459      | 1237         | published_department_of, published_date |                             |
| 关键词层 |          |              | 9040        | 21906        | 1                                       | keyowrd_of                  |
| 案例层   |          |              | 2543        | 3811         | 10                                      | law_of_audit, item_of_audit |
| 文档层   | 短文档   | 20165        | 14870       | 2            |                                         | definition_of, item_of_doc  |
|          | 长文档   | ——           | 33365       |              |                                         |                             |
| 合计     |          | 204339       | 220853      | 19*          | instance_of, item_of_audit              |                             |



###  2、基于命名实体识别的三元组抽取结果统计

##### 通过NER扩充后的三元组数量

| **扩充的知识层** | **抽取方式** | **实体数量** | **三元组数量** | **关系数量** | **关系示例**                |
| ---------------- | ------------ | ------------ | -------------- | ------------ | --------------------------- |
| 实例层/案例层    | 命名实体抽取 | 22580        | 42260          | 12           | instance_of, fraud_of_audit |
| 关键词层         | 关键词抽取   | 15061        | 66962          | 1            | keyword_of                  |

##### 表3 审计知识库中常用关系定义

| 应用层次 | 关系名称              | 关系描述                                  | 常用标签词                                                   |
| -------- | --------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| 本体层   | subconcept_of         | 概念与子概念的关系                        |   |
| 实例层   | instance_of           | 描述概念和实体之间的关系                  | 实例                                                         |
|          | characteristic_of     | 特点                                      | 特点、特征、性质                                             |
|          | item_of               | 描述实例与下级概念实例之间的关系          | 包括、包括                                                   |
|          | same_name_of          | 描述实体的同义词、其他别名、英文名等      | 同义词、外文名、英文名、英文缩写、英语简写、简称             |
|          | included_domain       | 描述实体与其所属行业之间的关系            | 所属行业、类别等                                             |
|          | member_of             | 描述机构类型实体与其成员之间关系          | 总经理、负责人等                                             |
| 文档层   | description_of        | 描述审计案例具体内容                      | 案例描述                                                     |
|          | definition_of         | 描述实体与其定义之间的关系                | 定义， 概述、概念、是指，是，即、本质、含义、解释            |
|          | item_of_doc           | 描述实体与文档内容之间的关系              | 第一章、第十二条                                             |
| 关键词层 | keyword_of            | 描述实体长文档与关键词之间的关系          | 关键词                                                       |
| 案例层   | auditor_of            | 描述审计类型或事项与审计人员之间的关系    | ——                                                           |
|          | fraud_of_audit        | 描述审计类型或事项与审计疑点之间的关系    | 审计疑点、 常见问题、舞弊、常见错弊                          |
|          | item_of_audit         | 描述审计类型和具体审计事项之间关系        | 审计事项、审计内容、内容、对象、主要内容， 类别、 普遍关键审计事项、审计范围 |
|          | law_of_audit          | 描述审计类型或事项与法律依据之间的关系    | 审计依据、处罚依据、定性依据、适用法规                       |
|          | method_of_audit       | 描述审计类型或事项和审计方法之间的关系    | 审计方法， 常用审计方法                                      |
|          | risk_of_audit         | 描述审计类型或事项与存在的风险之间的关 系 | 审计风险                                                     |
|          | org_of_audit          | 描述审计类型或事项与审计单位之间的关系    | 审计主体                                                     |
|          | achievement_of_ audit | 描述审计类型或事项与审计成果之间的关系    |                                                              |
|          | traget_of_audit       | 描述审计类型或事项与审计风险之间的关系    | ——                                                           |
|          | audited_of_org        | 描述审计类型或事项与被审计单位之间的关 系 | ——                                                           |
|          | included_domain       | 描述审计类型或事项与所属行业之间的关系    | ——                                                           |

### 3、审计命名实体实例

| 序号 | 实体类型                            | 实体实例                                                     |
| :--- | ----------------------------------- | ------------------------------------------------------------ |
| 1    | accountant subject（会计）          | 在加工物资，贷款项目项目进度表，营业保证金                   |
| 2    | audit achievement（审计成果）       | 专项审计报告，非标准意见审计报告，审计师意见                 |
| 3    | audit laws（审计依据）              | 互联网信息服务管理办法，现金管理条例，中华人民共和国行政许可法办法 |
| 4    | audit method（审计方法）            | 基本养老保险实缴总额分析，投入资本内部控制系统测试           |
| 5    | audit org（审计机构）               | 浙江审计厅，农业与资源保护审计司，审计署驻上海特派办         |
| 6    | audit problem（审计疑点）           | 非经营性资产转为经营性资产，擅自审批建设项目，公积金贷款超过限定年限 |
| 7    | audit risk（审计风险）              | 错报风险，评估风险，重大错误风险                             |
| 8    | audit target（审计目标）            | 合规范、真实性、合法性                                       |
| 9    | PER（审计人员/人名）                | 独立会计师，注册审计师，张磊                                 |
| 0    | company audit（企业审计）           | 国有资本经营，固定资产折旧审计，石油石化企业销售资质审计     |
| 11   | customs audit（海关审价）           | 进料加工贸易，自动进口许可证，出境通关                       |
| 12   | environmental audit（资源环境审计） | 地下水污染防控审计，重点流域水污染治理效果审计，环境统计信息管理系统安全性审计 |
| 13   | financial audit（金融审计）         | 流动资金贷款审计，商业银行投资租赁公司审计，保险中介及代理审计 |
| 14   | fiscal audit（财政审计）            | 部门决算结账事项审计，政府采购信息公开情况，行政事业性及基金性收入 |
| 15   | industry（行业）                    | 中介行业，交通运输行业，汽车制造业                           |
| 16   | ORG（机构/被审计单位）              | 国家开发投资公司，中国南方航空集团公司，城乡信用社           |
| 17   | publicwork audit（公共工程审计）    | 预算收支真实性审计，建设项目总投资和其他财务收支，施工合同签订管理审计 |
| 18   | revenue audit（税收审计）           | 教育费附加征收，非税收费项目，非工资性补贴                   |
| 19   | social insurance audit（民生审计）  | 基本养老保险基金预算，住房公积金归集，失业保险基金审计       |

#### 命名实体人工标注结果![](H:\auditKG\auditKG\finalauditKGRDF\本体与实体识别图.png)

# 4、系统展示



![image-20240430144630751](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240430144630751.png)



### （1）命名实体识别

输入一段示例文本，点击提交：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml12668\wps3.jpg)

​               图10  实体识别输入文本截图

即可查看识别结果：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml12668\wps4.jpg) 

​                图11  实体识别结果截图



### （2）实体查询功能展示

输入实例，例如“审计成果”，点击查询：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml12668\wps5.jpg)

​                图13  实体查询输入文本截图

即可展示与审计成果相关的关系及实体：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml12668\wps6.jpg) 

​                 图14  实体查询结果截图

向下翻页，还有以列表形势展现的审计成果相关信息：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml12668\wps7.jpg) 

​                 图15  实体查询结果截图

### （3）关系查询功能展示

输入想查询的实体以及关系，例如想要查询“审计成果”的子概念：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml12668\wps8.jpg) 

​                图17  关系查询输入文本截图

点击“Search”，即可展示审计成果及其所有子概念的关系图和关系列表：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml12668\wps9.jpg) 

​                 图18  关系查询结果截图



### （4）审计本体知识结构展示

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml12668\wps10.jpg)

​               图21  审计本体知识结构树展示截图

![](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240430145000138.png)              

图22  资产类展示截图



### （5)审计知识问答功能展示

输入想要查询的问题，例如“审计的子概念”，“按劳分配的定义”，即可看到回答：

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml12668\wps11.jpg)![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml12668\wps12.jpg)

​                  图30  输入问题效果截图