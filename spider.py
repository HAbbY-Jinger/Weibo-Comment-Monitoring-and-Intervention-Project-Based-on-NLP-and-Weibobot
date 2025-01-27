import requests
import csv
import pandas as pd
import os

def update_url(url, next_value):
    # 使用 f-string 替换 count=10 为 {next_value}
    updated_url = url.replace("count=10", f"{next_value}")
    return updated_url


def get_next(url, title, mid, comments_num, headers,csv_write,next='count=10', ):
    url = f'{url}'
    #print(url)
    #url = f'https://weibo.com/ajax/statuses/buildComments?is_reload=1&id=5094014382506022&is_show_bulletin=2&is_mix=0&{next}&uid=2656274875&fetch_level=0&locale=zh-CN'

    response = requests.get(url=url, headers=headers)
    json_data = response.json()

    data_list = json_data['data']
    max_id = json_data['max_id']
    for data in data_list:
        text_raw = data['text_raw']
        like_counts = data['like_counts']
        total_number = data['total_number']
        #print(title, text_raw,  like_counts, total_number, mid)
        comments_num+=1
        csv_write.writerow([title, text_raw,  like_counts, total_number, mid])
        # 输出格式：text comment 点赞量 评论量 weibo_id
    if comments_num>=10:
        return
    max_str = 'max_id=' + str(max_id)
    get_next(url,title,mid,comments_num,headers,csv_write,max_str)

def get_URL_mid_xls(folder,path):

    if not os.path.exists(folder):
        os.makedirs(folder)

    df = pd.read_excel(path)  # XLS 格式
    output_files = []

    # 遍历 DataFrame 中的每一行
    for index, row in df.iterrows():
        comments_num = 0
        title = row['title']  # 获取话题名
        url = row['URL']  # 获取URL
        mid = row['mid']  # 获取用户id
        url = update_url(url, "{next}")
        #print(url)

        # 发送GET请求来获取页面内容
        headers = {
            'cookie':'SINAGLOBAL=7506806151048.772.1701950322425; SCF=AsY_YNglQNMh3zfefCm3DABgNjsE8zkq_jQVUsxxqHyc1OHr0UgNXNtRgbvywuzl9X0YwviUO9GQjZM3ItaQR_A.; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5kWA-wNr7vU-MlOdABChBL5JpX5KMhUgL.FoM4Soe0Sh-fSoB2dJLoI0YLxK-L1hqLBoMLxK-L122LBK5LxK-LBoMLBK-LxKML1-2L1hBLxK.L1h5LB.zLxKnLB.-L1K2LxKML12zL1KMt; UOR=passport.weibo.com,weibo.com,cn.bing.com; ALF=1737428740; SUB=_2A25KY_JUDeRhGeFH7VES9CvJzTiIHXVpAQucrDV8PUJbkNANLW3hkW1NetWLSANap7pQXSSaies7Yz75VVBSyLMe; PC_TOKEN=94832f48a8; XSRF-TOKEN=Jw8mML9NwBZztsbtx58rV-fn; _s_tentry=weibo.com; Apache=9813180866871.729.1734836814037; ULV=1734836814063:13:4:2:9813180866871.729.1734836814037:1734236572909; WBPSESS=QMb2mNDtCtqG35B6uxA_cjGtbWVcTdQXX1Na0QOMfZft3rT6Rwdg2aZXOsQGqaifo_nqFX6ewgmQsoFVuHXT3MMFfY7ihEPWXsfCnmwVpYQ_t341EmFbrLAOoPPs0n1_jRQ64HKaLc0fxF9hJ5ivTA==',
            'referer': 'https://weibo.com/6444633552/P4vkUCI6I',  # Referer URL
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'
        }

        # 发起请求并解析网页内容
        response = requests.get(url, headers=headers)

        # 写入指定文件夹
        file_path = os.path.join(folder, f'{title}.csv')
        # 打开CSV文件并写入数据
        f = open(file_path, mode='a', encoding='utf-8-sig', newline='')
        output_files.append(file_path)
        csv_write = csv.writer(f)
        csv_write.writerow(['text', '评论内容', '点赞数', '评论量', 'weibo_id'])  # 读入格式：text comment 点赞量 评论量 weibo_id
        get_next(url, title, mid, comments_num, headers,csv_write,'count=10')
    return output_files

'''def main():
    folder='./train_for_bert'
    path = 'URL_mid 2.xls'

    get_URL_mid_xls(folder, path)#爬取语料库,返回语料库列表

if __name__=='__main__':
    main()'''









