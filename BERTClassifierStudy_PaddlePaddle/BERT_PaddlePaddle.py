import numpy as np
import paddle
from functools import partial
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from alive_progress import alive_bar

train_data, dev_data, test_data = load_dataset("chnsenticorp", splits=["train", "dev", "test"])
model = BertForSequenceClassification.from_pretrained("bert-wwm-chinese", num_classes=len(train_data.label_list))
tokenizer = BertTokenizer.from_pretrained("bert-wwm-chinese")

def convert_example(example, tokenizer):
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=512, pad_to_max_seq_len=True)
    return tuple([np.array(x, dtype="int64") for x in [encoded_inputs["input_ids"], encoded_inputs["token_type_ids"],
                                                      [example["label"]]]])

# Data pre-process.
train_data = train_data.map(partial(convert_example, tokenizer=tokenizer))
batch_sampler = paddle.io.BatchSampler(dataset=train_data, batch_size=8, shuffle=True)
train_data_loader = paddle.io.DataLoader(dataset=train_data, batch_sampler=batch_sampler, return_list=True)

optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
criterion = paddle.nn.loss.CrossEntropyLoss()

with alive_bar(total=len(train_data_loader), force_tty=True) as bar:
    for input_ids, token_type_ids, labels in train_data_loader():
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        probs = paddle.nn.functional.softmax(logits, axis=1)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        bar()