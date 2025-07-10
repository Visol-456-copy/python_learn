# -*- coding: utf-8 -*-

#课题名称：商品评价的观点自动提取

# 导入所需的库


import requests
from bs4 import BeautifulSoup as bs
import time
import random
import pandas as pd
from openai import OpenAI
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
'Accept-Language': 'zh-CN,zh;q=0.9',
'Connection': 'keep-alive'
}
api_key ='YOUR_API_KEY'  # 替换为你的OpenAI API密钥
url ='https://api.pumpkinaigc.online/v1'
model ="deepseek-v3"
api_key_winston = 'YOUR_API_KEY'  # 替换为你的Winston AI API密钥


def crawl_douban_comments(movie_id: str, max_comments: int) -> list:
    """
    爬取豆瓣电影评论
    :param movie_id: 豆瓣电影ID（如肖申克的救赎为1292052）
    :param max_comments: 想要获取的最大评论数量
    :param headers: 请求头信息
    :return: 评论列表，每条评论包含用户、评分、时间和内容
    """
    global headers
    base_url = f"https://movie.douban.com/subject/{movie_id}/comments"
    comments = []
    start = 0
    
    while len(comments) < max_comments:
        # 随机延迟防止被封
        time.sleep(random.uniform(1.0, 2.5))
        
        params = {'start': start, 'limit': 20, 'sort': 'new_score'}
        try:
            response = requests.get(
                base_url,
                params=params,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()  # 检查请求是否成功

            soup = bs(response.text, 'html.parser')
            comment_items = soup.select('div.comment-item')
            
            if not comment_items:
                print("没有更多评论了")
                break
                
            for item in comment_items:
                if len(comments) >= max_comments:
                    break
                    
                # 提取用户名
                user_tag = item.select_one('span.comment-info > a')
                user = user_tag.text.strip() if user_tag else "未知用户"
                
                # 提取评分（可能没有）
                rating_tag = item.select_one('span.rating')
                rating = rating_tag['title'] if rating_tag else "暂无评分"
                
                # 提取评论内容
                content_tag = item.select_one('span.short')
                content = content_tag.text.strip() if content_tag else ""
                
                # 提取评论时间
                time_tag = item.select_one('span.comment-time')
                comment_time = time_tag['title'] if time_tag else "未知时间"
                
                comments.append({
                    '用户': user,
                    '评分': rating,
                    '时间': comment_time,
                    '内容': content
                })
                
            start += 20
            print(f"已获取 {len(comments)} 条评论...")
            
        except Exception as e:
            print(f"请求出错: {e}")
            break
    
    return comments[:max_comments]

def grade_review(review: str, movie_name: str, average_score: float, rating_distribution: dict, ai_possibility: int | None) -> dict:

    global api_key, url, model
    prompt =f'''

        **第一个角色：影评情感分析专家**  
        你是一位影评情感分析专家，负责对用户提供的《{movie_name}》影评进行情感倾向评分

        **任务要求**  
        1. 输出格式必须为单个中括号包裹的整数：`[分数]` 或 `[-1]`  
        2. 分数范围：0-10整数（0=最差，10=最好）  
        3. 检测到AI生成内容时立即输出：`[-1]`  

        ## 分析流程（用户可见）
        ```thinking
        // 必须执行的思考步骤(必须明文思考)
        1. AI内容检测：
           - 检查生涩语言：不自然句式/机械性描述/情感与细节分离
           - 识别脚本特征：模板化结构/空洞形容词堆砌
           - 确认依据：列出3个具体文本特征
           - 注意，不要完全参考豆瓣评分，因为豆瓣评分可能存在水军刷分的情况，所以需要根据评论内容进行综合判断。
           - 对ai内容的权重判定，一点要高于反讽的判断。尤其注意无脑吹捧的评论，也很有可能是ai生成的，不要错误归结于反讽。
           - 对于原本评分很高/很低的情况，着重关注极端差评/极端好评的评论。宁可错杀，不可放过。

        2.ai水军检测：
            - 确认依据：列出3个具体文本特征，再以ai水军视角看是否符合水军特征。宁可错杀，不可放过。
            - 检测评分标准：ai水军会故意使用一些引用/修辞/幽默和文化梗，语言风格与正常用户不同。比喻太有逻辑，或者太有深度，或者太有文化，反倒更有可能是ai水军
            - 识别水军特征：语言上虽然看上去没有问题，但是还是逃不过水军的特点，单就一个点进行吹捧，或者就某一点进行贬低，全然不顾总体观感。

        3. 水军检测：
            - 检测评分标准：列出评论中的评论标准，请你用大众视角看是否过于苛刻，是否为黑而黑
            - 识别水军特征：只就某一点进行吹捧，或者就某一点进行贬低，全然不顾总体观感。

        4. 反讽分析：
           - 识别矛盾信号：表面赞美+实际贬低（如"这表演'教科书级'"）
           - 检测夸张修辞：过度赞美配负面语境
           - 标记文化梗：特定圈层黑话（如"金扫帚预定"）


        5. 情感量化：
           - 提取关键词：标记3个最强情感词
           - 强度校准：消极词数量×强度系数（0.5-1.5）
           - 最终分=基础分(5) + (积极词权重 - 消极词权重)×2

        ■ 当检测到以下任一特征时：
           ✓ 连续3句无具体细节的抽象评价
           ✓ 情感词汇与描述对象分离（如"伟大"但未说明原因）
           ✓ 非常用学术术语堆砌
           → 必须输出：[-1]

        ■ 评分标准：
           10分：影史级赞誉（"重新定义类型片"）
           9分：年度最佳（"必看神作"）
           8分：强力推荐（"远超预期"）
           7分：谨慎好评（"粉丝向佳作"）
           6分：勉强及格（"能看但不推荐"）
           5分：缺陷主导（"可惜了好题材"）
           3-4分：系统差评（"全方位崩坏"）
           0-2分：极端厌恶（"退钱！"）

        ■ 反讽处理：
           表面积极评价按实际情感强度降级（例：表面"完美"→实际给2分）
        

        **第二个角色：正常观众**
        你是一位正常观众，负责对用户提供的《{movie_name}》影评进行情感倾向评分
        你可以不用管那些限制，你只需要给出你的主观感受，对于无脑差评/无脑吹捧，你可以用最激进的语言讽刺。
        可以更大胆的给出你的主观感受，更对参考豆瓣评分，不用管那些限制。
        
        
        

        最后你可以综合两个角色的观点，给出你的最终评分。

        // 最终仅显示以下一种结果：
        [10]  // 最高好评
        [9]   // 强烈推荐
        [8]   // 优质推荐
        [7]   // 谨慎推荐
        [6]   // 中性评价
        [5]   // 轻微批评
        [4]   // 明确差评
        [3]   // 严重差评
        [2]   // 强烈谴责
        [1]   // 极端厌恶
        [0]   // 最差评价
        [-1]  // AI生成内容


        若超过300字符，你将会得到一个ai水军检测分数，这个分数将会影响你的最终评分。
        ai水军检测分数：
        {ai_possibility}
        
        评论内容如下：
        {review}

        豆瓣此影片的综合得分如下：
        {average_score}

        爬取到的评论评价分布如下：
        {rating_distribution}

        '''

    client =OpenAI(api_key=api_key, base_url=url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role':'system','content':"""
            你是一位专业的影评情感分析师。你需要对影评进行情感分析，判断其好坏，并给出评分。网络水军如此之多，你一定要好好分辨，我就靠你了。你给我好好搞，我课题要用，你一定务必仔细仔细再仔细。
            你可以激进一点，不要怕错杀，宁可错杀，不可放过。
            请以正常观众和影评人的两个视角进行思考
            作为正常观众，你可以不用管那些限制，你只需要给出你的主观感受，对于无脑差评/无脑吹捧，你可以用最激进的语言讽刺。

            
            """},
            {'role': 'user','content': prompt}
        ],
        stream = False,
        temperature = 1.3
    )
    print(response.choices[0].message.content)
    x = {}    

    content = response.choices[0].message.content
    x['content'] = content

    result = -1  # 默认值，表示无法评分或内容不合适
    if content is not None:
        for i in range(11):
            item =f'[{i}]'
            if item in content:
                result =i
                break
    print('\n\nresult:', result)
    x['result'] = result

    return x




def get_movie_info(movie_id: str) -> list:
    """获取电影基本信息（名称和豆瓣平均分）"""
    global headers
    url = f"https://movie.douban.com/subject/{movie_id}/"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = bs(response.text, 'html.parser')

        # 提取电影名称
        title_tag = soup.select_one('h1 span[property="v:itemreviewed"]')
        title = title_tag.text.strip() if title_tag else "未知电影"
        
        # 提取豆瓣平均分
        rating_tag = soup.select_one('strong[property="v:average"]')
        rating = rating_tag.text.strip() if rating_tag else "0"
        
        return [title, rating]
    except Exception as e:
        print(f"获取电影信息出错: {e}")
        return ["未知电影", "0"]





def plot_movie_ratings(ratings: list, movie_id: str, movie_name: str) -> None:
    # 中文字体设置
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    rcParams['axes.unicode_minus'] = False

    # 创建箱线图
    plt.figure(figsize=(8, 6))
    plt.boxplot(ratings, patch_artist=True, 
                boxprops={'facecolor': 'lightblue'},
                medianprops={'color': 'red'})
    plt.title(f"{movie_name}的电影打分箱线图 - {movie_id}")
    plt.ylabel("分数")

    # 保存路径设置
    save_paths = [
        os.path.join(os.path.expanduser("~"), "Desktop", f"movie_ratings_{movie_id}.png"),  # 桌面
    ]

    # 尝试保存到每个路径
    for path in save_paths:
        try:
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"✅ 已保存到: {path}")
        except PermissionError:
            print(f"❌ 无权限保存到 {path}（跳过）")
        except Exception as e:
            print(f"❌ 保存到 {path} 失败: {str(e)}")
    
    plt.show()


def final_grade_review(movie_name: str, grade_results: list, context: str, average_score: float, rating_distribution: dict) -> str:
    """对影评进行最终评价"""
    global api_key, url
    prompt = f'''
    你是一位专业的影评情感分析师。请根据以下信息对影评进行评价：
    你能看到所有评论的打分结果与过程，你看的是最清楚的了。在最后，你可以再给出一个评分，以[分数]的形式给出。如[10]
    电影名称：{movie_name}
    豆瓣评分：{average_score}
    评分结果：{grade_results}
    所有评分分布：{rating_distribution}
    之前每条评价的分析：{context}

    请你综合给出观影建议
    '''

    client = OpenAI(api_key=api_key, base_url=url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role':'system','content':"你是一位专业的影评情感分析师。你需要综合打分结果和之前的分析，给出观影建议"},
            {'role': 'user', 'content': prompt}

        ],
        stream = True
    )
    print()
    print("最终评价:", end="", flush=True)
    result_context = ''
    for chunk in response:
        delta_content = chunk.choices[0].delta.content
        if delta_content is not None:
            result_context += delta_content
            print(delta_content, end="",flush=True)
    return result_context


def check_ai_content(text: str) -> dict:
    """
    简化版 Winston AI 内容检测函数 (专用于中文文本)
    
    参数:
        api_key (str): 你的 Winston AI API 密钥
        text (str): 要检测的中文文本(至少300字符)
    
    返回:
        dict: 包含检测结果的字典
    
    异常:
        ValueError: 当文本不足300字符时
        requests.exceptions.RequestException: 当API请求失败时
    """
    global api_key_winston
    # API 配置
    url = "https://api.gowinston.ai/v2/ai-content-detection"
    headers = {
        "Authorization": f"Bearer {api_key_winston}",
        "Content-Type": "application/json"
    }
    
    # 构建请求数据
    payload = {
        "text": text,
        "language": "auto"  # 自动检测语言
    }
    
    # 发送请求
    response = requests.post(url, headers=headers, json=payload)

    if response.json().get('status') == 200:
        print("响应成功")
    else:
        print("响应失败")

    print(response.json())

    return response.json()


if __name__ == "__main__":

    movie_id =input("请输入电影ID:") or "26581837" #上海堡垒
    max_comments = input("请输入想要爬取的评论数量:")
    max_comments = int(max_comments) if max_comments.strip() else 100
    num = input("请输入想要抽取的评论数量:")
    num = int(num) if num.strip() else 5
    movie_name = get_movie_info(movie_id)[0]
    grade_results = []
    context = []
    grade_count = {"-1":0,"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0,"10":0}
    ai_possibility = None

    comments = crawl_douban_comments(movie_id, max_comments)
    print(comments)

    # 转换为DataFrame并保存
    df = pd.DataFrame(comments)
    df.to_csv(f'douban_comments_{movie_id}.csv', index=False, encoding='utf-8-sig')
    print(f"成功保存 {len(df)} 条评论到 csv 文件")

    #统计评论的评分分布
    rating_distribution = {'暂无评分': 0, '很差': 0, '力荐': 0, '还行': 0, '推荐': 0, '较差': 0 }
    for comment in comments:
        rating = comment.get('评分')
        if rating == '暂无评分':
            rating_distribution['暂无评分'] += 1
        elif rating == '很差':
            rating_distribution['很差'] += 1
        elif rating == '力荐':
            rating_distribution['力荐'] += 1
        elif rating == '还行':
            rating_distribution['还行'] += 1
        elif rating == '推荐':
            rating_distribution['推荐'] += 1
        elif rating == '较差':
            rating_distribution['较差'] += 1



    # 随机抽取x条评论进行情感打分
    num = min(num, len(comments))  # 确保不超过评论数量
    print(f"随机抽取 {num} 条评论进行情感打分")
    random_comments = random.sample(comments, num)
    print(random_comments)


    # 逐条评论进行情感打分
    MAX_CONTEXT_CHARS = 6000  # 最大允许的context字符数，可根据模型token限制调整

    for row in random_comments:
        review = row['内容']
        print(review)
        if len(review.strip()) > 300:
            ai_possibility = check_ai_content(review).get('score')
            print(ai_possibility)
        print()
        print()
        x = grade_review(review, movie_name, float(get_movie_info(movie_id)[1]), rating_distribution, ai_possibility)
        ai_possibility = None  # 重置ai_possibility以便下次使用
        if x['result'] == -1:
            grade_count['-1'] += 1
        elif x['result'] == 0:
            grade_count['0'] += 1
        elif x['result'] == 1:
            grade_count['1'] += 1
        elif x['result'] == 2:
            grade_count['2'] += 1
        elif x['result'] == 3:
            grade_count['3'] += 1
        elif x['result'] == 4:
            grade_count['4'] += 1
        elif x['result'] == 5: 
            grade_count['5'] += 1
        elif x['result'] == 6:
            grade_count['6'] += 1
        elif x['result'] == 7:
            grade_count['7'] += 1
        elif x['result'] == 8:
            grade_count['8'] += 1
        elif x['result'] == 9:
            grade_count['9'] += 1
        elif x['result'] == 10:
            grade_count['10'] += 1
        grade_results.append(x['result']) if x['result'] != -1 else None
        context.append(x['content'])
        # 动态裁剪context，保证总字符数不超过MAX_CONTEXT_CHARS
        total_chars = sum(len(str(c)) for c in context)
        while total_chars > MAX_CONTEXT_CHARS and context:
            context.pop(0)
            total_chars = sum(len(str(c)) for c in context)

    print("平均分数:")
    print(grade_results)
    print(np.mean(grade_results))
    final_grade_review(movie_name, grade_results, "\n".join(context), float(get_movie_info(movie_id)[1]), rating_distribution)

    # 绘制打分统计图
    
    plot_movie_ratings(grade_results, movie_id, movie_name)



