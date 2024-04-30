import os
import json
import torch
import numpy as np

from config import NerConfig
from model import BertNer
from data_loader import NerDataset
from  utils import  get_logger
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer


class Trainer:
    def __init__(self,
                 output_dir=None,
                 model=None,
                 train_loader=None,
                 save_step=500,
                 dev_loader=None,
                 test_loader=None,
                 optimizer=None,
                 schedule=None,
                 epochs=1,
                 device="gpu",
                 id2label=None):
        self.output_dir = output_dir
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.schedule = schedule
        self.id2label = id2label
        self.save_step = save_step
        self.total_step = len(self.train_loader) * self.epochs

    def train(self):
        logger = get_logger(self.output_dir )
        global_step = 1
        logger.info(self.id2label)
        logger.info("start training")
        tmp_loss = []
        for epoch in range(1, self.epochs + 1):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for key, value in batch_data.items():
                    batch_data[key] = value.to(self.device)
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                labels = batch_data["labels"]
                output = self.model(input_ids, attention_mask, labels)
                loss = output.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.schedule.step()

                tmp_loss.append(loss.cpu().detach().numpy())
                #print(f"【train】{epoch}/{self.epochs} {global_step}/{self.total_step} loss:{loss.item()}")
                if global_step % self.save_step == 0:
                    logger.info("epoch:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        epoch, global_step ,self.total_step , np.mean(tmp_loss)))
                    #torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_ner.bin"))
                tmp_loss = []
                global_step += 1
            #torch.save(self.model.state_dict(), os.path.join(self.output_dir + '/stepcheckpoint/', "pytorch_model_ner_tmp_best.bin"))
            best = self.evaluate ("dev" , logger)
            if best:
                torch.save(self.model.state_dict(), os.path.join(self.output_dir + '/stepcheckpoint/', "pytorch_model_ner_tmp_best.bin"))
                logger.info('"best model saved, the test detail fscore in below lines."')
            self.evaluate("test", logger)
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_ner_final.bin"))



    def evaluate(self, name, logger):
        #self.model.load_state_dict(torch.load(os.path.join(loadmodelpath, "pytorch_model_ner_tmp_best.bin")))
        self.model.eval()
        self.model.to(self.device)
        preds = []
        trues = []
        if name == 'dev':
            file_loader = self.dev_loader
        else:
            file_loader = self.test_loader
        for step, batch_data in enumerate(tqdm(file_loader)):

            for key, value in batch_data.items():
                batch_data[key] = value.to(self.device)
                input_ids = batch_data["input_ids"].to(self.device)
                attention_mask = batch_data["attention_mask"].to(self.device)
                labels = batch_data["labels"].to(self.device)
                output = self.model(input_ids, attention_mask, labels)
                logits = output.logits[0].to(self.device)
                attention_mask = attention_mask.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                logits  = logits.detach().cpu().numpy()
                batch_size = input_ids.size(0)

                for i in range(batch_size):
                    length = sum(attention_mask[i])
                    logit = logits[i][1:length]
                    label = labels[i][1:length] # is array
                    label = [self.id2label[i] for i in label]
                    logit = [self.id2label[i] for i in logit]

                    preds.append(logit)
                    trues.append(label)
                    # sub_preds = []
                    # sub_trues = []
                    # for i in range(len(label)):
                    #     if label[i] != "O":
                    #         sub_trues.append(label[i])
                    #         sub_preds.append(logit[i])
                    # preds.append(sub_preds)
                    # trues.append(sub_trues)

        report = classification_report(trues, preds)
        f1 = f1_score(trues, preds,  average='weighted')

        logger.info(report)
        if name == "dev":
            best_test_f1 = self.model.best_dev_f1
            if f1 > best_test_f1:
                self.model.best_dev_f1 = f1
                logger.info("new best dev f1 score:{:>.5f}".format(f1))
            return f1 > best_test_f1
        elif name == "test":
            best_test_f1 = self.model.best_test_f1
            if f1 > best_test_f1:
                self.model.best_test_f1 = f1
                logger.info("new best test f1 score:{:>.5f}".format(f1))
            return f1 > best_test_f1

    #
    # def test(self):
    #     self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "pytorch_model_ner.bin")))
    #     self.model.eval()
    #     preds = []
    #     trues = []
    #     for step, batch_data in enumerate(tqdm(self.test_loader)):
    #         for key, value in batch_data.items():
    #             batch_data[key] = value.to(self.device)
    #         input_ids = batch_data["input_ids"]
    #         attention_mask = batch_data["attention_mask"]
    #         labels = batch_data["labels"]
    #         output = self.model(input_ids, attention_mask, labels)
    #         logits = output.logits[0]
    #         attention_mask = attention_mask.detach().cpu().numpy()
    #         labels = labels.detach().cpu().numpy()
    #
    #         batch_size = input_ids.size(0)
    #         for i in range(batch_size):
    #             length = sum(attention_mask[i])
    #             logit = logits[i][1:length].numpy()
    #             label = labels[i][1:length] # is array
    #             label = [self.id2label[i] for i in label]
    #             logit = [self.id2label[i] for i in logit]
    #
    #             preds.append(logit)
    #             trues.append(label)
    #
    #     report = classification_report(trues, preds)
    #     f1score = f1_score(trues, preds,  average='weighted')
    #     return report, f1score


def build_optimizer_and_scheduler(args, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module' or space[0] == "bert":
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.bert_learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.bert_learning_rate},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_learning_rate},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.crf_learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def main(data_name):
    args = NerConfig(data_name)

    with open(os.path.join(args.output_dir, "ner_args.json"), "w") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device= torch.device( "cpu")
    with open(os.path.join(args.data_path, "train.txt"), "r") as fp:
        train_data = fp.read().split("\n")
    train_data = [json.loads(d) for d in train_data]

    with open(os.path.join(args.data_path, "dev.txt"), "r") as fp:
        dev_data = fp.read().split("\n")
    dev_data = [json.loads(d) for d in dev_data]

    with open(os.path.join(args.data_path, "test.txt"), "r") as fp:
        test_data = fp.read().split("\n")
    test_data = [json.loads(d) for d in test_data]

    train_dataset = NerDataset(train_data, args, tokenizer)
    dev_dataset = NerDataset(dev_data, args, tokenizer)
    test_dataset = NerDataset(test_data, args, tokenizer)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=1)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.dev_batch_size, num_workers=1)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size= 32, num_workers=1)
    model = BertNer(args)
    #
    # for name,_ in model.named_parameters():
    #   print(name)

    model.to(device)
    t_toal = len(train_loader) * args.epochs
    optimizer, schedule = build_optimizer_and_scheduler(args, model, t_toal)

    train = Trainer(
        output_dir=args.output_dir,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        schedule=schedule,
        epochs=args.epochs,
        device=device,
        id2label=args.id2label
    )

    train.train()


if __name__ == "__main__":
    data_name = ["audit_50", "audit_150"]
    for each in data_name :
        main(each)
