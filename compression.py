# ============================== #
# GÖRSEL SIKIŞTIRMA ARACI        #
# RL destekli JPEG kalite ayarı  #
# ============================== #

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
from stable_baselines3 import PPO  # PPO tabanlı RL ajanını yüklemek için

# ------------------------------------------------------------------------------
# COCO örnek görsellerinin URL listesi
# Bunlar test amaçlı kullanılacak, HUFFMAN/JPEG/RL destekli JPEG sıkıştırmalarına tabii tutulacak.
# ------------------------------------------------------------------------------
COCO_SAMPLE_URLS = [
    "https://images.cocodataset.org/val2017/000000000139.jpg",
    "https://images.cocodataset.org/val2017/000000000632.jpg",
    "http://farm9.staticflickr.com/8488/8248903344_de9d38205c_z.jpg"
]

# ------------------------------------------------------------------------------
# COCO görsellerini indir ve bellekte tut
# - Cache kullanımı tekrar indirmenin önüne geçer (Streamlit cache_data)
# - RGB formatına dönüştürülür
# ------------------------------------------------------------------------------
@st.cache_data
def download_images():
    images = []
    image_bytes_list = []
    for url in COCO_SAMPLE_URLS:
        response = requests.get(url, verify=False)  # Güvenlik sertifikası yoksa bile al
        image_bytes = response.content
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # RGB: 3 kanal
        images.append(img)
        image_bytes_list.append(len(image_bytes))  # Orijinal boyutu byte cinsinden sakla
    return images, image_bytes_list

# ------------------------------------------------------------------------------
# Görseli orantılı şekilde yeniden boyutlandır
# - Maksimum boyut 256x256 olacak şekilde orantıyı koruyarak küçült
# ------------------------------------------------------------------------------
def resize_image_keep_aspect(img, max_size=(256, 256)):
    img_copy = img.copy()
    img_copy.thumbnail(max_size, Image.LANCZOS)  # LANCZOS = yüksek kaliteli küçültme
    return img_copy

# ------------------------------------------------------------------------------
# PNG formatında Huffman tabanlı sıkıştırma uygula
# - PNG lossless (kayıpsız) sıkıştırma yapar
# - Bellek, zaman, kalite ölçümleri döner
# ------------------------------------------------------------------------------
def compress_huffman(img_pil):
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img_pil.save(tmp.name, format="PNG", optimize=True)
        comp_path = tmp.name
    t1 = time.time()

    comp_img = Image.open(comp_path).convert("RGB")  # Sıkıştırılmış görseli geri yükle
    decode_time = time.time() - t1  # Açma (decoding) süresi

    orig_bytes = img_pil.tobytes()  # Ham RGB verisi
    comp_size = os.path.getsize(comp_path)  # PNG dosya boyutu

    return {
        "Method": "Huffman (PNG)",
        "CompressionRatio": len(orig_bytes) / comp_size,
        "TimeEncode": t1 - t0,
        "TimeDecode": decode_time,
        "MemoryEncode_MB": psutil.Process().memory_info().rss / (1024 * 1024),
        "MemoryDecode_MB": psutil.Process().memory_info().rss / (1024 * 1024),
        "PSNR": psnr(np.array(img_pil), np.array(comp_img)),  # Görsel kalite metriği
        "SSIM": ssim(np.array(img_pil), np.array(comp_img), channel_axis=2),  # Yapısal benzerlik
        "CompressedImage": comp_img,
        "CompressedSizeBytes": comp_size
    }

# ------------------------------------------------------------------------------
# JPEG sıkıştırma işlemi (RL ajanı ile veya manuel kalite ile (default 75))
# - isRL=True olduğunda ajana ait etiketleme yapılır
# ------------------------------------------------------------------------------
def compress_jpeg(img_pil, quality=75, isRL=False):
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img_pil.save(tmp.name, format="JPEG", quality=quality, optimize=True)
        comp_path = tmp.name
    t1 = time.time()

    comp_img = Image.open(comp_path).convert("RGB")
    decode_time = time.time() - t1

    orig_bytes = img_pil.tobytes()
    comp_size = os.path.getsize(comp_path)

    return {
        "Method": f"{'RL Agent' if isRL else 'JPEG'} (q={quality})",
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

# ------------------------------------------------------------------------------
# Eğittiğim RL modelini yükler
# - Bu ajan, JPEG kalite seviyesini tahmin etmek için eğitildi
# ------------------------------------------------------------------------------
agent = PPO.load("jpeg_rl_agent_optimized3.zip")

# ------------------------------------------------------------------------------
# RL ajanın kullanacağı formatta gözlem üret (ön işlem)
# - Görsel 64x64'e küçültülür ve normalleştirilir
# - RGB kanalları (CHW) şeklinde döner
# ------------------------------------------------------------------------------
def preprocess_image(img_pil):
    img_resized = img_pil.resize((64, 64))  # Eğitim boyutu
    img_array = np.array(img_resized) / 255.0  # Normalize et (0-1)
    obs = img_array.transpose(2, 0, 1)  # Kanal önce olacak şekilde transpoze et (CHW)
    return obs

# ------------------------------------------------------------------------------
# RL ajanı ile JPEG sıkıştırma
# - Görsel girdisine göre uygun kalite seviyesini tahmin eder
# - Tahmin edilen kalite ile sıkıştırmayı uygular
# ------------------------------------------------------------------------------
def compress_jpeg_rl(img_pil):
    obs = preprocess_image(img_pil)
    action, _ = agent.predict(obs, deterministic=True)  # Kalite seviyesini tahmin et
    quality = 5 + 5 * int(action)  # 0–18 -> 5, 10, 15, ..., 95
    return compress_jpeg(img_pil, quality=quality, isRL=True)

# ------------------------------------------------------------------------------
# Byte değerlerini insan tarafından okunabilir hale getir (KB, MB, ...)
# ------------------------------------------------------------------------------
def format_bytes(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

# ------------------------------------------------------------------------------
# Streamlit arayüzünün ana uygulama fonksiyonu
# - Tüm görseller için sıkıştırmaları uygular ve karşılaştırmalı olarak sunar
# ------------------------------------------------------------------------------
def main():
    st.title("📦 Görsel Sıkıştırma Yöntemleri Karşılaştırması (RL Destekli)")

    images, original_sizes = download_images()

    for idx, (img, orig_size) in enumerate(zip(images, original_sizes)):
        st.header(f"📷 Görsel {idx+1}")
        st.markdown("**Orijinal Görsel**")
        st.image(img, caption=f"Orijinal Görsel - {format_bytes(orig_size)}", use_container_width=True)

        # Sıkıştırma yöntemleri uygulanır
        results = [
            compress_huffman(img),
            compress_jpeg(img, quality=75),
            compress_jpeg_rl(img)
        ]

        # Sıkıştırılmış görselleri göster
        st.markdown("### 🔧 Sıkıştırılmış Görseller")
        cols = st.columns(len(results))
        for col, res in zip(cols, results):
            col.markdown(f"**{res['Method']}**")
            comp_img_resized = resize_image_keep_aspect(res["CompressedImage"])
            col.image(comp_img_resized, use_container_width=True)
            col.markdown(f"`{format_bytes(res['CompressedSizeBytes'])}`")

        # Sayısal karşılaştırma tablosu
        st.markdown("### 📊 Karşılaştırma Tablosu")
        for r in results:
            r.pop("CompressedImage", None)
            r.pop("CompressedSizeBytes", None)
        st.table(results)

# ------------------------------------------------------------------------------
# Uygulama başlatma
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
