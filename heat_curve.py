import requests
from datetime import datetime, timedelta
from urllib.parse import quote
import json
import schedule
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

# 保存数据的文件路径
DATA_FILE = 'weibo_hot_search_data.json'
THRESHOLD_FILE = 'weibo_hot_search_threshold.xls'

def hot_search():
    url = 'https://weibo.com/ajax/side/hotSearch'
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()['data']

def save_data(data):
    # 读取现有数据
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    
    # 添加新数据
    existing_data.append({
        'timestamp': datetime.now().isoformat(),
        'data': data
    })
    
    # 保存数据
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

def process_data():
    data = hot_search()
    if not data:
        print('获取微博热搜榜失败')
        return
    
    # 保存数据
    save_data(data)
    
    # 读取最近一小时的数据
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print('读取数据文件失败')
        return
    
    if len(all_data) < 2:
        print('数据不足，无法计算斜率')
        return
    
    # 获取最近一小时的数据
    current_time = datetime.now()
    one_hour_ago = current_time - timedelta(hours=1)
    recent_data = [entry for entry in all_data if datetime.fromisoformat(entry['timestamp']) >= one_hour_ago]
    
    if len(recent_data) < 2:
        print('最近一小时数据不足，无法绘制折线图')
        return
    
    # 初始化话题热度数据
    topic_data = defaultdict(list)
    
    for entry in recent_data:
        timestamp = datetime.fromisoformat(entry['timestamp'])
        for rs in entry['data']['realtime']:
            if 'num' in rs:
                title = rs['word']
                num_value = rs['num']
                topic_data[title].append((timestamp, num_value))
    
    # 检测热度阈值和斜率
    with open(THRESHOLD_FILE, 'a', encoding='utf-8', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for topic, data_points in topic_data.items():
            timestamps, num_values = zip(*data_points)
            
            # 计算斜率
            if len(timestamps) > 1:
                slope = (num_values[-1] - num_values[0]) / ((timestamps[-1] - timestamps[0]).total_seconds() / 3600)
                if num_values[-1] >= 1.9e6 or slope >= 3.5e6:
                    print(f'话题 "{topic}" 热度达到1.9兆或斜率达到3.5兆/小时，开始提取评论')
                    csvwriter.writerow([topic, num_values[-1], slope, f'https://s.weibo.com/weibo?q={quote(topic)}&Refer=top'])

                    # 绘制折线图
            plt.figure(figsize=(12, 8))
            plt.plot(timestamps, num_values, label=topic)
            plt.xlabel('时间')
            plt.ylabel('热度')
            plt.title(f'微博热搜话题 "{topic}" 热度变化')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

                    # 指定支持中文字符的字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            # 显示每个话题的图表
            plt.show()

                    # 输出文字信息
                    # print(f'话题: {topic}, 热度: {num_values[-1]}, 斜率: {slope:.2f}, 链接: https://s.weibo.com/weibo?q={quote(topic)}&Refer=top')


            


def do_get_curve():
    # 每3分钟爬取一次数据并处理
    schedule.every(3).minutes.do(process_data)
    
    # 初始运行一次
    process_data()
    
    while True:
        schedule.run_pending()
        time.sleep(1)
def main():
    do_get_curve()
if __name__ == '__main__':
    main()