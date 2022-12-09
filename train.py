from datasets import load_dataset
from transformers import AutoFeatureExtractor
import evaluate
import numpy as np
import ast
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer


model_name = "facebook/wav2vec2-base"
# model_name = 'mnazari/wav2vec2-assyrian'

data = load_dataset("csv", data_files={'train': "train.csv", 'valid': "valid.csv", 'test': "test.csv"})

id2label = {0: 'non-velarized', 1: 'velarized'}
label2id = {'non-velarized': 0, 'velarized': 1}

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

def preprocess_function(examples):
    audio_arrays = [ast.literal_eval(x)["array"][0] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=16000, 
        truncation=True
    )

    return inputs
    
encoded_data = data.map(preprocess_function, remove_columns=["audio", 'path'], batched=True)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_scr = f1.compute(predictions=preds, references=labels)
    prec_scr = precision.compute(predictions=preds, references=labels)
    recall_scr = recall.compute(predictions=preds, references=labels)
    return {'accuracy': acc['accuracy'], 'f1': f1_scr['f1'], 'precision': prec_scr['precision'], 'recall': recall_scr['recall']}

model = AutoModelForAudioClassification.from_pretrained(
    model_name, num_labels=2, label2id=label2id, id2label=id2label
)

training_args = TrainingArguments(
    output_dir="velarization",
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
    gradient_checkpointing=True,
    fp16=True,
    #optim="adafactor",
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    remove_unused_columns = False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_data["train"],
    eval_dataset=encoded_data["valid"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# Testing
preds = trainer.predict(encoded_data['test'])
print('----------------------------------------')
print(preds[2])
print('--------------------')
incorrect = [path for pred, label, path in zip(preds[0], preds[1], data['test']['path']) if (pred[0] < pred[1] and label == 0) or (pred[0] > pred[1] and label == 1)]
print(incorrect)
print('----------------------------------------')

trainer.train()

# Testing
preds = trainer.predict(encoded_data['test'])
print('----------------------------------------')
print(preds[2])
print('--------------------')
incorrect = [path for pred, label, path in zip(preds[0], preds[1], data['test']['path']) if (pred[0] < pred[1] and label == 0) or (pred[0] > pred[1] and label == 1)]
print(incorrect)
print('----------------------------------------')

'''
trainer.model.save_pretrained('velarization')

model = AutoModelForAudioClassification.from_pretrained('velarization')

test_args = TrainingArguments(
    output_dir = 'ransom',
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = 16,   
    dataloader_drop_last = False    
)

trainer = Trainer(
    model = model, 
    args = test_args, 
    compute_metrics = compute_metrics
)

test_results = trainer.predict(encoded_data['test'])
'''