
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import os

label_map = {'0': 0,#positive
            '1': 1,#negative
            '2': 2#middle
}
model = AutoModelForSequenceClassification.from_pretrained('./sentiment_model')
tokenizer = AutoTokenizer.from_pretrained('./sentiment_model')
def evaluate_model(folder,comments_group_list, model, tokenizer):
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_evaluate=[]
    for i in range(len(comments_group_list)):
        test_data_df = pd.read_csv(comments_group_list[i])  # 读取csv文件
        text = test_data_df['text'].tolist()  # 获取微博的文本
        weibo_id = test_data_df['weibo_id'].tolist()  # 获取微博的id
        test_texts = test_data_df['评论内容'].tolist()  # 获取评论内容列的文本
        like_counts = test_data_df['点赞数'].tolist()  # 获取点赞数列
        reply_counts = test_data_df['评论量'].tolist()  # 获取评论量列

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
        # 输出格式：text comment  weibo_id  label
        output_df = pd.DataFrame({
            'text': text,
            '评论内容': test_texts,
            'weibo_id': weibo_id,
            'label': predicted_labels,
            '点赞数': like_counts,
            '评论量': reply_counts
        })

        # 计算每个标签的加权出现次数
        output_df['weights'] = output_df['点赞数'] + output_df['评论量']  # 计算每条评论的权重
        label_weighted_counts = output_df.groupby('label')['weights'].sum()  # 按标签加权统计
        #print(label_weighted_counts.dtypes)  # 打印数据类型
        #print(label_weighted_counts)  # 打印数据的内容

        # 输出出现次数最多的标签
        most_frequent_label = label_weighted_counts.idxmax()  # 获取出现次数最多的标签
        most_frequent_count = label_weighted_counts.max()  # 获取出现次数
        print(f'#{text[0]}#话题情感权重最大的标签: {most_frequent_label}, 总次数: {most_frequent_count}')  # 打印结果
        # 如果情感极性为负，则输出文件
        if int(most_frequent_label) == 1:

            #print(output_df)  # 打印所有评论和对应的预测标签
            file_path = os.path.join(folder, f'{text[0]}预测结果.csv')

            # 保存预测结果为 CSV 文件到指定路径
            output_df.to_csv(file_path, index=False, encoding='utf-8')

            # print(output_df)  # 打印所有评论和对应的预测标签

            print(f'#{text[0]}#话题有网暴风险，评论情感分析结果已送入{file_path}')
            output_evaluate.append(file_path)

    return output_evaluate
