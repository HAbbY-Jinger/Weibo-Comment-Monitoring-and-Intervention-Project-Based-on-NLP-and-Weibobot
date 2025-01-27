from bs4 import BeautifulSoup
import requests
import os
import json
import pandas as pd
import imageio
import seaborn as sns
import matplotlib.pyplot as plt
import jieba
import collections # 词频统计库

# 提取某一期的弹幕
def get_danmu(num1, num2, page):
    url = 'https://bullet-ali.hitv.com/bullet/tx/2025/01/25/{}/{}/{}.json'
    danmuurl = url.format(num1, num2, page)
    res = requests.get(danmuurl)
    res.encoding = 'utf-8'
    jd = json.loads(res.text)
    details = []
    for i in range(len(jd['data']['items'])):
        result = {}
        result['content'] = jd['data']['items'][i]['content']
        result['time'] = jd['data']['items'][i]['time']
        try:
            result['v2_up_count'] = jd['data']['items'][i]['v2_up_count']
        except:
            result['v2_up_count'] = ''
        details.append(result)

    return details


def count_danmu():
    danmu_total = []
    num1 = f'162912'
    num2 = f'22343716'
    page = 10
    for i in range(page):
        danmu_total.extend(get_danmu(num1, num2, i))

    return danmu_total


def main():
    danmu_end = []
    # 爬前四集，所以设置了循环4次
    danmu_end.extend(count_danmu())

    df = pd.DataFrame(danmu_end)
    print(len(df))
    # 将输出格式修改为CSV文件
    df.to_csv(f"芒果TV弹幕-再见爱人第13期-前10分钟弹幕-{len(df)}条弹幕.csv", index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
