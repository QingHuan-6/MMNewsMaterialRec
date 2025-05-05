import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
from tqdm import tqdm
from cn_clip.clip import load_from_name, tokenize

# 配置参数
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16"

IMAGE_FOLDER = "./CNA_images"  # 图片库路径
CACHE_DIR = "CNA_cache"
IMAGE_EMBED_CACHE = os.path.join(CACHE_DIR, "CNA_image_embeddings.npz")

# 自动创建缓存目录
os.makedirs(CACHE_DIR, exist_ok=True)

# 初始化 cn_clip 模型
model, preprocess = load_from_name(MODEL_NAME, device=device)
model.eval()


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

    with tqdm(total=num_images, desc="生成嵌入", unit="image", ncols=80) as pbar:
        with torch.no_grad():
            for i, image_file in enumerate(image_files):
                try:
                    image_path = os.path.join(IMAGE_FOLDER, image_file)
                    image = Image.open(image_path).convert("RGB")
                    inputs = preprocess(image).unsqueeze(0).to(device)
                    embedding = model.encode_image(inputs).cpu().numpy().flatten()
                    embeddings[i] = embedding
                    pbar.set_postfix({"当前文件": image_file[:20] + "..."}, refresh=False)
                    pbar.update(1)
                except Exception as e:
                    print(f"\n跳过损坏文件 {image_file}: {str(e)}")
                    pbar.update(1)
                    pbar.set_postfix({"跳过文件": image_file[:20] + "..."}, refresh=False)

    np.savez_compressed(
        IMAGE_EMBED_CACHE,
        filenames=np.array(image_files),
        embeddings=embeddings
    )
    print(f"\n保存图片嵌入缓存：{IMAGE_EMBED_CACHE} ({num_images}张)")
    return image_files, embeddings


# 加载图片嵌入
image_files, image_embeddings = precompute_image_embeddings()


def get_image_similarities_by_image(query_image_path, top_k=10):
    """通过图片路径获取相似图片"""
    try:
        # 加载查询图片并生成嵌入
        query_image = Image.open(query_image_path).convert("RGB")
        query_inputs = preprocess(query_image).unsqueeze(0).to(device)

        with torch.no_grad():
            query_embedding = model.encode_image(query_inputs).cpu().numpy()

        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding, image_embeddings)[0]
        indices = np.argsort(-similarities)[:top_k]  # 降序排序

        return [(image_files[idx], similarities[idx]) for idx in indices]

    except Exception as e:
        print(f"处理查询图片时出错: {e}")
        return []


if __name__ == "__main__":
    print(f"[缓存状态] 图片嵌入: {IMAGE_EMBED_CACHE}")

    # 指定查询图片路径（需存在于IMAGE_FOLDER或其他路径）
    query_image_path = "./CNA_images/2024091218440600.jpg"  # 替换为实际图片路径
    top_k = 20

    print(f"\n正在搜索与 '{os.path.basename(query_image_path)}' 相似的图片...")
    recommendations = get_image_similarities_by_image(query_image_path, top_k)

    print(f"\n找到{len(recommendations)}张相似图片，相似度排序：")
    for rank, (image_name, score) in enumerate(recommendations, 1):
        print(f"{rank}. {image_name:<30} 相似度: {score:.4f}")