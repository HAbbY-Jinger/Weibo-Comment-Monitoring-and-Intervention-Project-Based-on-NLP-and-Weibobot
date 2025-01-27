from bert_sentiment_analysize import model as bert_model
from bert_sentiment_analysize import tokenizer as bert_tokenizer
from bert_sentiment_analysize import evaluate_model
from spider import get_URL_mid_xls
from heat_curve import *
import warnings
warnings.filterwarnings("ignore", message="Glyph .* missing from current font")
from pre_cpu import *
from weibo_bot import *


def main():
    '''do_get_curve() #爬取热度曲线+阈值判断
    #path=THRESHOLD_FILE'''
    path = 'URL_mid.xls'
    folder_spider_output = './spider_output'
    folder_bert_output = './bert_output'
    folder_T5_output = './T5_output'
    output_files_list = get_URL_mid_xls(folder_spider_output, path)#爬取语料库,返回语料库列表
    print(output_files_list)
    output_files_list_by_bert_list = evaluate_model(folder_bert_output, output_files_list, bert_model, bert_tokenizer)
    print(output_files_list_by_bert_list)
    weibo_bot_release = predict(folder_T5_output, output_files_list_by_bert_list)
    print(weibo_bot_release)
    send_comment(weibo_bot_release)

if __name__ == '__main__':
    main()