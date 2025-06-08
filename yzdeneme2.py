import os
import time
import tempfile
import psutil
import requests
import io
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import streamlit as st
from stable_baselines3 import PPO  # Ajanı yüklemek için

# ---------- COCO örnek görsel URL'leri -------------
COCO_SAMPLE_URLS = [
    "https://images.cocodataset.org/val2017/000000000139.jpg",
    "https://images.cocodataset.org/val2017/000000000632.jpg",
    "http://farm9.staticflickr.com/8488/8248903344_de9d38205c_z.jpg"
]

@st.cache_data
def download_images():
    images = []
    image_bytes_list = []
    for url in COCO_SAMPLE_URLS:
        response = requests.get(url, verify=False)
        image_bytes = response.content
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        images.append(img)
        image_bytes_list.append(len(image_bytes))
    return images, image_bytes_list

def resize_image_keep_aspect(img, max_size=(256, 256)):
    img_copy = img.copy()
    img_copy.thumbnail(max_size, Image.LANCZOS)
    return img_copy

def compress_huffman(img_pil):
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img_pil.save(tmp.name, format="PNG", optimize=True)
        comp_path = tmp.name
    t1 = time.time()
    comp_img = Image.open(comp_path).convert("RGB")
    decode_time = time.time() - t1
    orig_bytes = img_pil.tobytes()
    comp_size = os.path.getsize(comp_path)
    return {
        "Method": "Huffman (PNG)",
        "CompressionRatio": len(orig_bytes) / comp_size,
        "TimeEncode": t1 - t0,
        "TimeDecode": decode_time,
        "MemoryEncode_MB": psutil.Process().memory_info().rss / (1024 * 1024),
        "MemoryDecode_MB": psutil.Process().memory_info().rss / (1024 * 1024),
        "PSNR": psnr(np.array(img_pil), np.array(comp_img)),
        "SSIM": ssim(np.array(img_pil), np.array(comp_img), channel_axis=2),
        "CompressedImage": comp_img,
        "CompressedSizeBytes": comp_size
    }

def compress_jpeg(img_pil, quality=75, isRL= False):
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img_pil.save(tmp.name, format="JPEG", quality=quality, optimize=True)
        comp_path = tmp.name
    t1 = time.time()
    comp_img = Image.open(comp_path).convert("RGB")
    decode_time = time.time() - t1
    orig_bytes = img_pil.tobytes()
    comp_size = os.path.getsize(comp_path)
    if isRL:
        return {
            "Method": f"RL Agent (q={quality})",
            "CompressionRatio": len(orig_bytes) / comp_size,
            "TimeEncode": t1 - t0,
            "TimeDecode": decode_time,
            "MemoryEncode_MB": psutil.Process().memory_info().rss / (1024 * 1024),
            "MemoryDecode_MB": psutil.Process().memory_info().rss / (1024 * 1024),
            "PSNR": psnr(np.array(img_pil), np.array(comp_img)),
            "SSIM": ssim(np.array(img_pil), np.array(comp_img), channel_axis=2),
            "CompressedImage": comp_img,
            "CompressedSizeBytes": comp_size
        }
    else:
        return {
            "Method": f"JPEG (q={quality})",
            "CompressionRatio": len(orig_bytes) / comp_size,
            "TimeEncode": t1 - t0,
            "TimeDecode": decode_time,
            "MemoryEncode_MB": psutil.Process().memory_info().rss / (1024 * 1024),
            "MemoryDecode_MB": psutil.Process().memory_info().rss / (1024 * 1024),
            "PSNR": psnr(np.array(img_pil), np.array(comp_img)),
            "SSIM": ssim(np.array(img_pil), np.array(comp_img), channel_axis=2),
            "CompressedImage": comp_img,
            "CompressedSizeBytes": comp_size
        }

# ---------------- RL MODELİ YÜKLE ----------------
agent = PPO.load("jpeg_rl_agent_optimized3.zip")

def preprocess_image(img_pil):
    img_resized = img_pil.resize((64, 64))  # Ajan eğitimi bu boyutta yapıldıysa
    img_array = np.array(img_resized) / 255.0
    obs = img_array.transpose(2, 0, 1)  # (3, 64, 64)
    return obs

def compress_jpeg_rl(img_pil):
    obs = preprocess_image(img_pil)
    action, _ = agent.predict(obs, deterministic=True)
    action = int(action)  # 0–18
    quality = 5 + 5 * action  # 5, 10, ..., 95
    return compress_jpeg(img_pil, quality=quality, isRL=True)

def format_bytes(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

# ---------------- Streamlit Arayüzü ----------------
def main():
    st.title("📦 Görsel Sıkıştırma Yöntemleri Karşılaştırması (RL Destekli)")

    images, original_sizes = download_images()

    for idx, (img, orig_size) in enumerate(zip(images, original_sizes)):
        st.header(f"📷 Görsel {idx+1}")

        st.markdown("**Orijinal Görsel**")
        st.image(img, caption=f"Orijinal Görsel - {format_bytes(orig_size)}", use_container_width=True)

        results = []
        results.append(compress_huffman(img))
        results.append(compress_jpeg(img, quality=75))
        results.append(compress_jpeg_rl(img))
        
        st.markdown("### 🔧 Sıkıştırılmış Görseller")

        cols = st.columns(len(results))
        for col, res in zip(cols, results):
            col.markdown(f"**{res['Method']}**")
            comp_img_resized = resize_image_keep_aspect(res["CompressedImage"])
            col.image(comp_img_resized, use_container_width=True)
            col.markdown(f"`{format_bytes(res['CompressedSizeBytes'])}`")

        st.markdown("### 📊 Karşılaştırma Tablosu")
        for r in results:
            r.pop("CompressedImage", None)
            r.pop("CompressedSizeBytes", None)
        st.table(results)

if __name__ == "__main__":
    main()
