#pip install ratsnlp
import ratsnlp.nlpbook
import torch
from ratsnlp import nlpbook
from ratsnlp.nlpbook.qa import QATrainArguments, KorQuADV1Corpus, QADataset, QATask
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, BertForQuestionAnswering
import torch
from ratsnlp.nlpbook.qa import QATrainArguments

args = QATrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_name="korquad-v1",
    
    #절대경로 사용 불가
    downstream_model_dir="",
    max_seq_length=128,
    max_query_length=32,
    doc_stride=64,
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    epochs=3,
    tpu_cores=0 if torch.cuda.is_available() else 8,
    seed=7,
)

from ratsnlp import nlpbook
nlpbook.set_seed(args)
nlpbook.set_logger(args)

nlpbook.download_downstream_dataset(args)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)

corpus = KorQuADV1Corpus()
train_dataset = QADataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)

val_dataset = QADataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="val",
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=SequentialSampler(val_dataset),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)

pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
)

model = BertForQuestionAnswering.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
)

task = QATask(model, args)
trainer = nlpbook.get_trainer(args)

#epoch 당 약 1
trainer.fit(
    task,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader
)
