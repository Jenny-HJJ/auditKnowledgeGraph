import os
import json
import torch
import numpy as np

from collections import namedtuple
from model import BertNer
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer


def get_args(args_path, args_name=None):
    with open(args_path, "r") as fp:
        args_dict = json.load(fp)
    # 注意args不可被修改了
    args = namedtuple(args_name, args_dict.keys())(*args_dict.values())
    return args


class Predictor:
    def __init__(self, data_name):
        self.data_name = data_name
        self.ner_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "ner_args.json"), "ner_args")
        self.ner_id2label = {int(k): v for k, v in self.ner_args.id2label.items()}
        self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_dir)
        self.max_seq_len = self.ner_args.max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_model = BertNer(self.ner_args)
        self.ner_model.load_state_dict(torch.load(os.path.join(self.ner_args.output_dir, "pytorch_model_ner_final.bin"),
                                                  map_location=torch.device('cpu')))
        self.ner_model.to(self.device)
        self.data_name = data_name

    def ner_tokenizer(self, text):
        # print("文本长度需要小于：{}".format(self.max_seq_len))
        text = text[:self.max_seq_len - 2]
        text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = torch.tensor(np.array([input_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask

    def ner_predict(self, text):
        input_ids, attention_mask = self.ner_tokenizer(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.ner_model(input_ids, attention_mask)
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])
        logits = output.logits.squeeze()
        logits = logits[1:length - 1].detach().cpu().numpy()
        logits = [self.ner_id2label[i] for i in logits]
        entities = get_entities(logits)
        result = {}
        for ent in entities:
            ent_name = ent[0]
            ent_start = ent[1]
            ent_end = ent[2]
            if ent_name not in result:
                result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
            else:
                result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
        return result


if __name__ == "__main__":
    #data_name = "dgre"
    data_name = "audit_50"
    predictor = Predictor(data_name)
    if data_name == "dgre":
        texts = [
            "492号汽车故障报告故障现象一辆车用户用水清洗发动机后，在正常行驶时突然产生铛铛异响，自行熄火",
            "故障现象：空调制冷效果差。",
            "原因分析：1、遥控器失效或数据丢失;2、ISU模块功能失效或工作不良;3、系统信号有干扰导致。处理方法、体会：1、检查该车发现，两把遥控器都不能工作，两把遥控器同时出现故障的可能几乎是不存在的，由此可以排除遥控器本身的故障。2、检查ISU的功能，受其控制的部分全部工作正常，排除了ISU系统出现故障的可能。3、怀疑是遥控器数据丢失，用诊断仪对系统进行重新匹配，发现遥控器匹配不能正常进行。此时拔掉ISU模块上的电源插头，使系统强制恢复出厂设置，再插上插头，发现系统恢复，可以进行遥控操作。但当车辆发动在熄火后，遥控又再次失效。4、查看线路图发现，在点火开关处安装有一钥匙行程开关，当钥匙插入在点火开关内，处于ON位时，该开关接通，向ISU发送一个信号，此时遥控器不能进行控制工作。当钥匙处于OFF位时，开关断开，遥控器恢复工作，可以对门锁进行控制。如果此开关出现故障，也会导致遥控器不能正常工作。同时该行程开关也控制天窗的自动回位功能。测试天窗发现不能自动回位。确认该开关出现故障",
            "原因分析：1、发动机点火系统不良;2、发动机系统油压不足;3、喷嘴故障;4、发动机缸压不足;5、水温传感器故障。",
        ]
    elif data_name == "audit_50":
        texts = [
            "违规向拖欠土地出让金的用地单位发放国有土地使用证定性依据：(l)《国务院关于加强国有土地资产管理的通知》",
            "排污者未按规定缴纳排污费定性依据：(1)《排污费征收使用管理条例》第二条：直接向环境排放污染物的单位和个体工商户（以下简称排污者），应当依照本条例的规定缴纳排污费。",
            "侵占、截留、挤占、挪用环保专项资金定性依据：(1)《排污费征收使用管理条例》第五条：排污费应当全部专项用于环境污染防治，任何单位和个人不得截留、挤占或者挪作他用。",
            ")《财政违法行为处罚处分条例》第三条：财政收入执收单位及其工作人员有下列违反国家财政收入管理规定的行为之一的，责令改正，补收应当收取的财政收入，限期退还违法所得。",
            "虚列投资支出表现形式：虚列材料采购成本、虚列建筑工程和安装工程成本、虚列设备投资成本、隐匿建设项目资金结余。",
            "私分国有资产表现形式：国家机关、国有公司、企业事业单位、人民团体，违反国家规定，以单位名义将国有资产集体私分给个人，数额较大行为"

        ]
    for text in texts:
        ner_result = predictor.ner_predict(text)
        print("文本>>>>>：", text)
        print("实体>>>>>：", ner_result)
        print("="*100)


