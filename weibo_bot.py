import requests
import pandas as pd
import weibo


def send_comment(path_list):
    url = "https://api.weibo.com/2/comments/create.json"
    access_token = '2.00Ux4vgI07lHyO6e68f8bc17HTTR7D'  # 获取到的access_token
    rip = '223.80.110.76'

    # 读取CSV文件，使用制表符分隔
    for i in range(len(path_list)):
        df = pd.read_csv(path_list[i], sep='\t', encoding='utf-8')

        # 假设CSV文件的列名分别为 'weibo_id' 和 'comment'
        weibo_ids = df['weibo_id'].tolist()  # 获取微博ID列表
        comments = df['comment'].tolist()  # 获取评论内容列表

        # 遍历每条微博ID及对应评论
        for weibo_id, comment in zip(weibo_ids, comments):
            params = {
                'access_token': access_token,
                'id': weibo_ids,
                'comment': comment,
                'rip': rip
            }
            weibo.Client
            requests.post(url=url, data=params)

'''def main():
    path=['./T5_output\\#李蠕蠕模仿再见爱人李蠕蠕模仿麦林#话题生成评论.csv']
    send_comment(path)

if __name__=="__main__":
    main()
'''
