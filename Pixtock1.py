import clip
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
from tqdm import tqdm

# 配置参数
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"

IMAGE_FOLDER = "E:\\Cursor\\Pixtock\\backend\\Flickr8K\\Images"
CACHE_DIR = "cache"
IMAGE_EMBED_CACHE = os.path.join(CACHE_DIR, "image_embeddings.npz")

# 自动创建缓存目录
os.makedirs(CACHE_DIR, exist_ok=True)

# 初始化CLIP模型（使用clip库）
model, preprocess = clip.load(MODEL_NAME, device=device)

def _get_image_modification_time():
    """获取图片库最新修改时间"""
    if not os.path.isdir(IMAGE_FOLDER):
        return 0
    return max(os.path.getmtime(os.path.join(root, f))
               for root, _, files in os.walk(IMAGE_FOLDER)
               for f in files)

def precompute_image_embeddings():
    """预处理并缓存图片嵌入（带进度条和自动更新）"""
    cache_valid = os.path.exists(IMAGE_EMBED_CACHE) and \
                  os.path.getmtime(IMAGE_EMBED_CACHE) >= _get_image_modification_time()

    if cache_valid:
        print("加载最新的图片嵌入缓存...")
        with np.load(IMAGE_EMBED_CACHE) as data:
            return data['filenames'].tolist(), data['embeddings']

    print("开始重新计算图片嵌入...")
    image_files = [f for f in os.listdir(IMAGE_FOLDER)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    num_images = len(image_files)
    embeddings = np.zeros((num_images, 512), dtype=np.float32)

    # 初始化进度条
    with tqdm(total=num_images, desc="生成嵌入", unit="image", ncols=80) as pbar:
        with torch.no_grad():
            for i, image_file in enumerate(image_files):
                try:
                    image_path = os.path.join(IMAGE_FOLDER, image_file)
                    image = Image.open(image_path).convert("RGB")
                    inputs = preprocess(image).unsqueeze(0).to(device)  # 使用clip的preprocess
                    embedding = model.encode_image(inputs).cpu().numpy().flatten()  # 使用clip的encode_image
                    embeddings[i] = embedding

                    # 进度条更新（成功案例）
                    pbar.set_postfix({"当前文件": image_file[:20] + "..."}, refresh=False)
                    pbar.update(1)

                except Exception as e:
                    print(f"\n跳过损坏文件 {image_file}: {str(e)}")
                    # 进度条更新（失败案例）
                    pbar.update(1)
                    pbar.set_postfix({"跳过文件": image_file[:20] + "..."}, refresh=False)

    # 保存带进度提示的缓存
    np.savez_compressed(
        IMAGE_EMBED_CACHE,
        filenames=np.array(image_files),
        embeddings=embeddings
    )
    print(f"\n保存图片嵌入缓存：{IMAGE_EMBED_CACHE} ({num_images}张)")
    return image_files, embeddings

# 加载图片嵌入（带进度条）
image_files, image_embeddings = precompute_image_embeddings()

def get_image_similarities(search_query, top_k=10):
    text_inputs = clip.tokenize([search_query]).to(device)  # 使用clip的tokenize
    with torch.no_grad():
        text_embedding = model.encode_text(text_inputs).cpu().numpy()

    similarities = cosine_similarity(text_embedding, image_embeddings)[0]
    indices = np.argsort(-similarities)[:top_k]
    return [(os.path.join(IMAGE_FOLDER, image_files[idx]), similarities[idx])
            for idx in indices]

if __name__ == "__main__":
    print(f"[缓存状态] 图片嵌入: {IMAGE_EMBED_CACHE}")

    search_query = input("\n请输入搜索词: ").strip()
    top_k = 20

    print(f"\n正在搜索 '{search_query}'...")
    recommendations = get_image_similarities(search_query, top_k)

    print(f"\n找到{len(recommendations)}张相关图片，相似度排序：")
    for rank, (path, score) in enumerate(recommendations, 1):
        print(f"{rank}. {os.path.basename(path):<30} 相似度: {score:.4f}")