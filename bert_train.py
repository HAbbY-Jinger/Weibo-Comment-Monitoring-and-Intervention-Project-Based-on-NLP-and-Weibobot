import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import torch

label_map = {'0': 0,#positive
            '1': 1,#negative
            '2': 2#middle
}
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('./bert-base-uncased', num_labels=len(label_map))

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

def generate_data(train_test_data_path):

    excel_file_path = train_test_data_path  # 替换为Excel文件路径
    data_df1 = pd.read_csv(excel_file_path, encoding='gbk')  # GBK编码
    data = []  # 创建一个空列表，用于存储数据
    for index, row in data_df1.iterrows():
        # 确保label和text都是字符串，如果是NaN则替换为空字符串
        label = row['label']
        text = str(row['评论内容']) if pd.notna(row['评论内容']) else ''

        # 如果标签是浮点数，则转换为字符串并去掉小数点
        if pd.notna(label):
            label = str(label)  # 转换为整数字符串
        else:
            label = ''

        # 确保label在label_map中
        if label in label_map:
            data.append({"text": text.strip(), "label": label_map[label]})
        else:
            print(f"Warning: Unrecognized label '{label}' at index {index}")

    data_df = pd.DataFrame(data)
    print(data_df)
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    return dataset

def train_model(dataset):
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train()


def evaluate_model(evaluate_file_path):
    # 读取新的Excel文件路径
    test_excel_file_path = evaluate_file_path
    test_data_df = pd.read_excel(test_excel_file_path)  # 读取Excel文件
    test_texts = test_data_df['评论内容'].tolist()  # 获取评论内容列的文本

    # 预处理测试数据
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

    # 进行预测
    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 不需要计算梯度
        outputs = model(**test_encodings)  # 获取模型输出
        predictions = torch.argmax(outputs.logits, dim=-1).numpy()  # 获取预测的标签索引

    # 映射预测结果到标签
    predicted_labels = [list(label_map.keys())[list(label_map.values()).index(pred)] for pred in predictions]

    # 输出结果到DataFrame
    output_df = pd.DataFrame({'评论内容': test_texts, '预测标签': predicted_labels})

    # 计算每个标签的出现次数
    label_counts = output_df['预测标签'].value_counts()  # 统计每个标签的出现次数

    # 输出结果到Excel文件
    output_df.to_excel('预测结果.xlsx', index=False)  # 保存预测结果
    print(output_df)  # 打印所有评论和对应的预测标签

    # 输出出现次数最多的标签
    most_frequent_label = label_counts.idxmax()  # 获取出现次数最多的标签
    most_frequent_count = label_counts.max()  # 获取出现次数
    print(f'出现次数最多的标签: {most_frequent_label}, 次数: {most_frequent_count}')  # 打印结果


def main():
    train_test_data_path = "merge.csv"
    dataset = generate_data(train_test_data_path)

    # 导入bert 模型，先下载了github上的L=12，H=768，又下载了config.json,pytorch_model.bin
    # 将它们放到文件夹bert-base-uncased里
    # 升级accelerate
    # pip install accelerate>=0.26.0

    train_model(dataset)
    model.save_pretrained('./sentiment_model')
    tokenizer.save_pretrained('./sentiment_model')

    #evaluate_file_path=''

if __name__=="__main__":
    main()