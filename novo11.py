import transformers
import torch
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import os
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
from PIL import Image
from opencc import OpenCC

# 初始化简繁转换器
cc = OpenCC('t2s')


def extract_news_elements(model_id, title, content):
    """
    从新闻文本中提取关键要素并保存到 Excel 文件
    :param model_id: 模型路径
    :param title: 新闻标题
    :param content: 新闻正文
    :return: 解析后的新闻要素字典
    """
    # 加载模型和分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    deepseek_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 视觉要素提示
    system_prompt = """
    作为新闻视觉分析师，需从新闻内容中提取以下7大要素，以JSON格式输出：
    要素规范（必填）：
    1. 核心事件（名词短语，一定要和标题紧密相关，3-5个）
    2. 关键动作（动宾结构，2-3个）
    3. 显著实体（具体名称，具体的人名和物品名和机构名一定要直接提取出来，不需要在后面注明是人名还是机构名）
    4. 场景特征（环境/时间/视觉元素）
    5. 情感内容(丰富)
    6. 视觉隐喻（象征性元素）
    7. 数据统计信息（新闻中的数据和统计信息）
    输出必须使用如下格式：
    核心事件：[...] \n 关键动作：[...] \n 显著实体：[...] \n 场景特征：[...] \n 情感内容：[...] \n 视觉隐喻：[...]\n 数据统计信息：[...]
    """

    input_text = f"标题：{title}\n正文：{content}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": input_text}]

    # 生成与解析
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(
        deepseek_model.device)
    outputs = deepseek_model.generate(input_ids, max_new_tokens=2048)
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    def parse_elements(text):
        elements = {
            "核心事件": [], "关键动作": [], "显著实体": [],
            "场景特征": [], "情感倾向": [], "视觉隐喻": [], "数据统计信息": []
        }
        try:
            json_block = text.split("```json")[-1].split("```")[0].strip()
            parsed_json = json.loads(json_block)
            converted_parsed = {cc.convert(k): v for k, v in parsed_json.items()}
            return {k: converted_parsed.get(cc.convert(k), []) for k in elements}
        except:
            return elements

    parsed = parse_elements(raw_output)
    print(parsed)

    # 保存到 Excel
    excel_data = {
        "标题": [title], "正文": [content],
        "核心事件": [parsed.get("核心事件", [])],
        "关键动作": [parsed.get("关键动作", [])],
        "显著实体": [parsed.get("显著实体", [])],
        "场景特征": [parsed.get("场景特征", [])],
        "情感倾向": [parsed.get("情感倾向", [])],
        "视觉隐喻": [parsed.get("视觉隐喻", [])],
        "数据统计信息": [parsed.get("数据统计信息", [])]
    }
    excel_path = "news_analysis.xlsx"
    df = pd.DataFrame(excel_data)
    df.to_excel(excel_path, index=False, engine='openpyxl')

    # 清理 DeepSeek 模型占用的 GPU 内存
    del deepseek_model
    torch.cuda.empty_cache()

    return parsed


def preprocess_image_features(model_name, image_dir, cache_dir):
    """
    预处理图像特征并进行缓存
    :param model_name: 模型名称
    :param image_dir: 图像文件夹路径
    :param cache_dir: 缓存文件夹路径
    :return: 图像特征字典
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = load_from_name(model_name, device=device, download_root='./')
    clip_model.to(device)

    cache_file = os.path.join(cache_dir, "image_features.npy")
    if os.path.exists(cache_file):
        print(f"从缓存加载图片嵌入: {cache_file}")
        return np.load(cache_file, allow_pickle=True).item()

    print(f"🛠 预处理图片并生成缓存: {cache_file}")
    image_features = {}
    for img_name in tqdm(os.listdir(image_dir), desc="预处理图片"):
        try:
            image = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
            inputs = clip_preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = clip_model.encode_image(inputs).cpu().numpy()
            image_features[img_name] = feat / np.linalg.norm(feat)
        except Exception as e:
            print(f"跳过无效图片 {img_name}: {str(e)}")
            continue
    np.save(cache_file, image_features)
    print(f"缓存已保存: {cache_file}")
    return image_features


def encode_text(text, text_feature_cache, model_name):
    """
    对文本进行编码
    :param text: 待编码的文本
    :param text_feature_cache: 文本特征缓存字典
    :param model_name: 模型名称
    :return: 编码后的文本特征向量
    """
    if not text.strip():
        return np.zeros((1, 512))
    cache_key = hash(text)
    if cache_key in text_feature_cache:
        return text_feature_cache[cache_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = load_from_name(model_name, device=device, download_root='./')
    clip_model.to(device)

    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens).cpu().numpy()
    normalized_feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    text_feature_cache[cache_key] = normalized_feat
    return normalized_feat


def calculate_text_feature(news_row, fixed_weights, title, content, word_threshold, model_name):
    """
    计算文本特征向量
    :param news_row: 新闻要素数据行
    :param fixed_weights: 固定权重字典
    :param content: 新闻正文
    :param word_threshold: 正文长度阈值
    :param model_name: 模型名称
    :return: 文本特征向量
    """
    text_feature_cache = {}
    content_length = len(content)
    if content_length < word_threshold:
        # 正文长度小于阈值，直接用正文匹配图片

        text_feat = encode_text(title+content, text_feature_cache, model_name)
        print('正在匹配' + title+content)
    else:
        # 正常流程：使用固定权重计算加权文本特征
        weighted_feat = np.zeros((1, 512))
        converted_news_row = {cc.convert(k): v for k, v in news_row.items()}
        for field, weight in fixed_weights.items():
            text = str(converted_news_row.get(cc.convert(field), ""))
            weighted_feat += weight * encode_text(text, text_feature_cache, model_name)
        text_feat = weighted_feat / np.linalg.norm(weighted_feat)  # 归一化特征向量
    return text_feat


def calculate_similarities(text_feat, image_features):
    """
    计算文本与图像的相似度
    :param text_feat: 文本特征向量
    :param image_features: 图像特征字典
    :return: 按相似度降序排列的图像列表
    """
    similarities = []
    for img_name, img_feat in image_features.items():
        sim = text_feat @ img_feat.T
        similarities.append((img_name, sim[0][0]))
    similarities.sort(key=lambda x: -x[1])  # 按相似度降序排列
    return similarities


def main():

    # 模型配置
    model_id = "/home/zjm/llm/DeepSeek-R1-Distill-Llama-8B-hf"
    title = """
    "“書生侘寂—施永華七十個展”9月14日在臺中市港區藝術中心舉辦開幕式。
    """
    content = """
"“書生侘寂—施永華七十個展”9月14日在臺中市港區藝術中心舉辦開幕式。展覽共分為七個主題，包含“自書自話”呈現施永華以白話詩記錄日常，打破僅限古典詩詞的書法框架，將書法與現代生活結合；“舊作品回顧”呈現他書法風格的演變；“臨帖”展出他臨摹古代書帖的作品；“新詩書寫”書寫七位包括故友和他崇拜的詩人，展現詩的意境；“紅樓夢”嘗試以小說或日記的形式完成書法創作；“對聯”展示不同筆調書寫的對聯作品；“實驗創作”展示在筆法、布局、墨色、裝裱和拼貼上的實驗創作，探索書法新表現。圖為施永華與作品《痴》。  香港中通社圖片

"""

    # 步骤 1：生成新闻要素 Excel
    parsed = extract_news_elements(model_id, title, content)

    # 配置参数
    model_name = "ViT-B-16"
    cache_dir = "CNA_cache"
    image_dir = "/home/zjm/CNA_images"
    output_dir = "result"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 定义 Excel 列名
    columns = ["标题", "正文", "核心事件", "关键动作",
               "显著实体", "场景特征", "情感倾向", "视觉隐喻", "数据统计信息"]

    # 加载新闻要素数据
    news_df = pd.read_excel("news_analysis.xlsx", sheet_name='Sheet1', nrows=1)
    if len(news_df) == 0:
        raise ValueError("Excel 文件中无数据")
    news_row = news_df[columns].iloc[0]

    # 固定权重配置
    fixed_weights = {
        "标题": 0.3,  # 标题通常包含核心信息，权重较高
        "核心事件": 0.25,  # 核心事件是新闻的核心语义单元
        "关键动作": 0.05,  # 关键动作描述事件动态
        "显著实体": 0.25,  # 显著实体（人名、机构名）是重要匹配线索
        "场景特征": 0.08,  # 场景特征（时间、地点）辅助定位
        "情感倾向": 0.03,  # 情感倾向对视觉氛围有影响，但权重较低
        "视觉隐喻": 0.02,  # 视觉隐喻较抽象，权重最低
        "数据统计信息": 0.02,  # 数据统计作为补充信息
    }
    # 确保权重和为 1
    total_weight = sum(fixed_weights.values())
    fixed_weights = {k: v / total_weight for k, v in fixed_weights.items()}
    print("使用固定权重：", fixed_weights)

    # 步骤 2：加载图像特征
    image_features = preprocess_image_features(model_name, image_dir, cache_dir)

    # 步骤 3：计算文本特征
    word_threshold = 250
    text_feat = calculate_text_feature(news_row, fixed_weights, title, content, word_threshold, model_name)

    # 步骤 4：计算 Top 匹配图像
    similarities = calculate_similarities(text_feat, image_features)

    print("\nTop10 匹配图像:")
    for i, (img_name, sim) in enumerate(similarities[:100], 1):
        print(f"{i}. {img_name} 相似度: {sim:.4f}")


if __name__ == "__main__":
    main()
