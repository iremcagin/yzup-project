# === KÜTÜPHANELER ===
import os
import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import tempfile
import warnings
import glob

warnings.filterwarnings("ignore", category=UserWarning, module='gymnasium')

# === ÖZEL GYM ORTAMI TANIMI ===

class JPEGCompressionEnv(gym.Env):
    def __init__(self, images):
        super().__init__()
        self.images = images         # Eğitimde kullanılacak resim listesi
        self.index = 0               # Sıradaki görselin indeksini tutar
        self.action_space = gym.spaces.Discrete(19)  # 19 farklı kalite seviyesi: 5,10,...,95
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Yeni bir resim ile ortamı sıfırla
        super().reset(seed=seed)
        self.current_img = self.images[self.index % len(self.images)]
        self.index += 1

        img_resized = self.current_img.resize((64, 64), Image.LANCZOS)  # Görseli 64x64'e küçült
        obs = np.array(img_resized, dtype=np.float32) / 255.0           # Normalize et [0, 1]
        self.obs = np.transpose(obs, (2, 0, 1))                          # (H,W,C) → (C,H,W)
        return self.obs, {}

    def step(self, action):
        # Ajanın verdiği aksiyona göre JPEG kalitesi belirlenir
        quality = 5 + 5 * int(action)  # 5'ten başlayarak 5'er artan 19 değer

        # Görseli hazırla
        orig_img = self.current_img.resize((64, 64), Image.LANCZOS)

        # Geçici dosyada resmi JPEG olarak sıkıştır
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            orig_img.save(tmp.name, format="JPEG", quality=quality, optimize=True)
            comp_path = tmp.name

        # Sıkıştırılmış resmi tekrar oku ve yeniden boyutlandır
        comp_img = Image.open(comp_path).convert("RGB").resize((64, 64), Image.LANCZOS)
        comp_array = np.array(comp_img)
        orig_array = np.array(orig_img)

        # Kalite metrikleri hesaplanır
        psnr_value = psnr(orig_array, comp_array)
        ssim_value = ssim(orig_array, comp_array, channel_axis=2)

        # Dosya boyutu KB cinsinden alınır
        size_kb = os.path.getsize(comp_path) / 1024.0

        # Ödül fonksiyonu: kalite - boyut cezası
        reward = (0.5 * psnr_value + 0.5 * ssim_value * 100) - 5.0 * np.log1p(size_kb)

        # Geçici dosyayı sil
        os.remove(comp_path)

        # RL ortamının step fonksiyonu dönüşü
        terminated = True     # Tek adımlı bir ortam — her adımda episode biter
        truncated = False
        info = {
            "quality": quality,
            "psnr": psnr_value,
            "ssim": ssim_value,
            "compressed_size_kb": size_kb,
            "reward": reward
        }

        return self.obs, reward, terminated, truncated, info

# === GÖRSELLERİ YÜKLEME ===

def load_sample_images(folder_path):
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    images = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Resim yüklenemedi: {path} — {e}")

    return images

# === ANA EĞİTİM BLOĞU ===

if __name__ == "__main__":
    print("Görseller yükleniyor...")
    images = load_sample_images("./val2017")  # COCO validation set klasörü

    print("Ortam başlatılıyor...")
    env = DummyVecEnv([lambda: Monitor(JPEGCompressionEnv(images))])  # Ortamı sarmalla
    check_env(JPEGCompressionEnv(images), warn=True)  # Ortamın doğru yapılandığını doğrula

    print("Model eğitiliyor...")

    # PPO ajanı tanımlanıyor
    model = PPO(
        "CnnPolicy",           # Görseller için CNN tabanlı politika
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./ppo_jpeg_tensorboard/",  # TensorBoard için loglar
        policy_kwargs={"normalize_images": False}   # Girişler zaten normalize
    )

    # Değerlendirme ortamı ve callback
    eval_env = DummyVecEnv([lambda: JPEGCompressionEnv(images)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",   # En iyi model buraya kaydedilir
        log_path="./logs/",
        eval_freq=1000,                     # Her 1000 adımda bir değerlendirme yapılır
        deterministic=True,
        render=False
    )

    # Öğrenme süreci başlatılıyor
    model.learn(total_timesteps=100000, callback=eval_callback)

    # Model kaydediliyor
    print("Model kaydediliyor...")
    model.save("jpeg_rl_agent_optimized3")
    print("Model eğitildi ve kaydedildi.")
