import torch
import random
import pandas as pd
import os
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from dataset.preprocess import data_clean

# 加载训练后的模型
tokenizer = AutoTokenizer.from_pretrained("./result/model_files")
model = AutoModelForSeq2SeqLM.from_pretrained("./result/model_files")

# 设置设备为 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 将模型输出的字符‘_’转换为换行符（模型定义如此）
def postprocess(text):
    return text.replace("_", "\n")

# 以下三个函数均为对回复内容的限制，不允许回复内容只有语气标点
def only_include(text):
    return bool(re.match('^[!！。？?]+$', text))

# 不允许回复内容太长和太短
def length_check(text):
    return len(text) > 32 or len(text) < 4

# 不允许回复出现无关内容，
def content_check(text):
    return '转发微博' in text

# 预测函数，即模型输出。给定一个微博文本，返回回复内容
def answer_fn(text, sample=True, top_p=0.8):
    '''
    sample: 是否抽样。生成任务, 可以设置为True;
    top_p: 0-1之间, 生成的内容越多样.
    '''
    origin = text
    # text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device)
    if not sample:  # 不进行采样
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128, num_beams=4, length_penalty=0.6)
    else:  # 采样（生成）
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128, do_sample=True, top_p=top_p)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    res = postprocess(out_text[0])
    # res = out_text[0]
    return answer_fn(origin, sample=True, top_p=top_p) if content_check(res) or length_check(res) or only_include(res) else res

# 预测前，需预处理输入的微博文本
def process_test_weibo(path):
    print(path)
    df = pd.read_csv(path)  # 或者其他加载方式

    #df = pd.read_csv(path, sep='\t')
    text = df['text'].tolist()
    weibo_id = df['weibo_id'].tolist()

    preprocess_weibo = []
    for weibo in tqdm(text):
        pre_weibo, _ = data_clean(weibo, max_len=512, is_control_length=True, is_format=False)
        preprocess_weibo.append('评论以下微博：_' + pre_weibo)

    return weibo_id, preprocess_weibo

# 预测入口，给定测试文件路径输出预测结果
def predict(folder,path_list):
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_comments_generation=[]

    for i in range(len(path_list)):
        weibo_id, weibo = process_test_weibo(path_list[i])
        print(weibo[0][8:])
        output=os.path.join(folder, f'#{weibo[0][8:]}#话题生成评论.csv')

        pred_list = []
        # 生成回复时尽量避免雷同（与前两条回复需不同）
        tmp_txt_up1 = ''
        tmp_txt_up2 = ''
        for text in tqdm(weibo):
            for retry_i in range(20):
                result = answer_fn(text, sample=True, top_p=0.8)
                if result != tmp_txt_up1 and result != tmp_txt_up2:
                    break
            tmp_txt_up2 = tmp_txt_up1
            tmp_txt_up1 = result
            pred_list.append(result)

        pd.DataFrame({'weibo_id': weibo_id, 'comment': pred_list}).to_csv(output, sep='\t', index=False)
        output_comments_generation.append(output)
    return output_comments_generation
'''def main():
    folder='./T5_output'
    list=['./bert_output\\小s谈与许雅均开放性关系  预测结果.csv', './bert_output\\央妈镜头的祝绪丹  预测结果.csv', './bert_output\\一起接43位志愿军烈士回国  预测结果.csv']
    weibo_bot_release = predict(folder, list)

if __name__=="__main__":
    main()'''

