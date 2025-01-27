#（本文首发在“程序员coding”公众号）
import requests
import pandas as pd

# episodes_danmu_DataFrame是存放一集所有弹幕的DataFrame
episodes_danmu_DataFrame = pd.DataFrame()

# 填写腾讯视频的参数，video_code是腾讯视频的编号，num是获取弹幕的次数，step是步进参数
video_code = "i410051a8yd"
num = 10000  # 设置一个较大的请求次数，程序会自动判断，当没有弹幕了会自动退出循环
step = 30000

# 循环num次获取弹幕
for i in range(num):
    url = f'https://dm.video.qq.com/barrage/segment/{video_code}/t/v1/{i * 30000}/{i * 30000 + step}'
    response = requests.get(url=url).json()
    if (len(response["barrage_list"])) > 0:
        # temp_danmu_DataFrame是存放本次弹幕的DataFrame
        temp_danmu_DataFrame = pd.json_normalize(response['barrage_list'], errors='ignore')
        episodes_danmu_DataFrame = pd.concat([episodes_danmu_DataFrame, temp_danmu_DataFrame])
        print("第", i + 1, "次请求弹幕,请求地址为：", url, "获取到：", temp_danmu_DataFrame.shape[0],
              "条弹幕，这一集总弹幕已获取到", episodes_danmu_DataFrame.shape[0], "条。")
    else:
        break

print("总共获取到", episodes_danmu_DataFrame.shape[0], "条弹幕")
# 查看 DataFrame 的行数和列数。
rows = episodes_danmu_DataFrame.shape
print("请求得到的表格行数与列数：", rows)

# 将 DataFrame 保存为 csv 文件
# 选择保存的列
episodes_danmu_DataFrame = episodes_danmu_DataFrame.loc[:, ['content']]
episodes_danmu_DataFrame.to_csv(f"腾讯视频弹幕-心动的信号第10期下-{episodes_danmu_DataFrame.shape[0]}条弹幕.csv", mode='w',
                                encoding="utf-8", errors='ignore', index=False)
print("弹幕保存完成！")

