import clip
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import time  # 新增时间测量模块
from tqdm import tqdm

# 配置参数
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"
# 初始化CLIP模型
model, preprocess = clip.load(MODEL_NAME, device=device)

DATA_FOLDER = r'E:\archive\1.Flickr8k\flickr8k_allimgs\Images'

CACHE_DIR = "cache"
IMAGE_EMBED_CACHE = os.path.join(CACHE_DIR, "image_embeddings.npz")
EMBEDDING_DIM = 512
# 自动创建缓存目录
os.makedirs(CACHE_DIR, exist_ok=True)


class Material:
    def __init__(self, image_path, embedding):
        self.image_path = image_path
        self.embedding = embedding


def _get_image_modification_time():
    """获取图片库最新修改时间"""
    max_mtime = 0
    for root, _, files in os.walk(DATA_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            mtime = os.path.getmtime(file_path)
            if mtime > max_mtime:
                max_mtime = mtime
    return max_mtime


def precompute_image_embeddings():
    """预处理并缓存图片嵌入（带进度条和自动更新）"""
    cache_valid = os.path.exists(IMAGE_EMBED_CACHE) and \
                  os.path.getmtime(IMAGE_EMBED_CACHE) >= _get_image_modification_time()

    if cache_valid:
        print("加载最新的图片嵌入缓存...")
        with np.load(IMAGE_EMBED_CACHE) as data:
            return data['filenames'].tolist(), data['embeddings']  # 统一键名

    print("开始重新计算图片嵌入...")
    image_files = [
        f for f in os.listdir(DATA_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]
    num_images = len(image_files)
    embeddings = np.zeros((num_images, EMBEDDING_DIM), dtype=np.float32)

    with tqdm(total=num_images, desc="生成嵌入", unit="image", ncols=80) as pbar:
        with torch.no_grad():
            for i, image_file in enumerate(image_files):
                try:
                    image_path = os.path.join(DATA_FOLDER, image_file)
                    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                    embedding = model.encode_image(image).cpu().numpy().flatten()
                    embeddings[i] = embedding
                    pbar.set_postfix({"当前文件": image_file[:20] + "..."}, refresh=False)
                    pbar.update(1)
                except Exception as e:
                    print(f"\n跳过损坏文件 {image_file}: {str(e)}")
                    pbar.update(1)
                    pbar.set_postfix({"跳过文件": image_file[:20] + "..."}, refresh=False)

    np.savez_compressed(
        IMAGE_EMBED_CACHE,
        image_paths=np.array(image_files),  # 统一键名
        embeddings=embeddings
    )
    print(f"\n保存图片嵌入缓存：{IMAGE_EMBED_CACHE} ({num_images}张)")
    return image_files, embeddings


def load_materials():
    """加载图片素材（带缓存管理）"""
    image_files, embeddings = precompute_image_embeddings()
    return [Material(os.path.join(DATA_FOLDER, f), e) for f, e in zip(image_files, embeddings)]


def recommend_similar(focused_image, all_materials, top_k=10):
    """基于CLIP的相似性推荐（带时间测量）"""
    start_time = time.perf_counter()  # 高精度时间测量

    target_idx = next((i for i, m in enumerate(all_materials) if m.image_path == focused_image), None)
    if target_idx is None:
        elapsed_time = time.perf_counter() - start_time
        print(f"[搜索耗时] {elapsed_time:.4f} 秒（未找到目标图片）")
        return []

    target_embedding = all_materials[target_idx].embedding
    all_embeddings = np.array([m.embedding for m in all_materials])

    # 优化相似度计算（矩阵运算替代sklearn）
    target_embedding_normalized = target_embedding / np.linalg.norm(target_embedding)
    all_embeddings_normalized = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    similarities = np.dot(all_embeddings_normalized, target_embedding_normalized)

    sorted_indices = np.argsort(-similarities)[:top_k + 1]
    results = [
                  (all_materials[idx], similarities[idx])
                  for idx in sorted_indices
                  if idx != target_idx
              ][:top_k]

    elapsed_time = time.perf_counter() - start_time
    print(f"[搜索耗时] {elapsed_time:.4f} 秒 ({len(all_materials)}张图片中搜索)")
    return results


# 使用示例
if __name__ == "__main__":
    FOCUSED_IMAGE = r'E:\archive\1.Flickr8k\flickr8k_allimgs\Images\10815824_2997e03d76.jpg'

    # 加载素材（自动管理缓存）
    all_materials = load_materials()

    # 执行推荐（带时间测量）
    recommendations = recommend_similar(FOCUSED_IMAGE, all_materials, top_k=20)

    # 输出结果
    print(f"\n基于 [{os.path.basename(FOCUSED_IMAGE)}] 的推荐结果：")
    for rank, (material, sim) in enumerate(recommendations, 1):
        print(f"第 {rank} 名: {os.path.basename(material.image_path):<30} 相似度: {sim:.4f}")