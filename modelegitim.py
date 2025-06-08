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
from io import BytesIO
import requests
import warnings
import glob


warnings.filterwarnings("ignore", category=UserWarning, module='gymnasium')

# ---------- Ortam Tanımı ----------

class JPEGCompressionEnv(gym.Env):
    def __init__(self, images):
        super().__init__()
        self.images = images
        self.index = 0
        self.action_space = gym.spaces.Discrete(19)  # 5,10,...,95
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_img = self.images[self.index % len(self.images)]
        self.index += 1
        img_resized = self.current_img.resize((64, 64), Image.LANCZOS)
        obs = np.array(img_resized, dtype=np.float32) / 255.0
        self.obs = np.transpose(obs, (2, 0, 1))  # CHW
        return self.obs, {}


    def step(self, action):
        # Kalite aralığını 5–95 arasında sınırla
        quality = 5 + 5 * int(action)
    
        # Orijinal resmi uygun boyuta getir
        orig_img = self.current_img.resize((64, 64), Image.LANCZOS)
    
        # Geçici olarak sıkıştırılmış resmi oluştur
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            orig_img.save(tmp.name, format="JPEG", quality=quality, optimize=True)
            comp_path = tmp.name
    
        # Sıkıştırılmış görseli yükle ve orijinal boyutta karşılaştır
        comp_img = Image.open(comp_path).convert("RGB").resize((64, 64), Image.LANCZOS)
        comp_array = np.array(comp_img)
        orig_array = np.array(orig_img)
    
        # Kalite metrikleri
        psnr_value = psnr(orig_array, comp_array)
        ssim_value = ssim(orig_array, comp_array, channel_axis=2)
    
        # Dosya boyutu (KB)
        size_kb = os.path.getsize(comp_path) / 1024.0
    
    
        # Reward hesaplama: kalite odaklı - boyut cezası
        reward = (0.5 * psnr_value + 0.5 * ssim_value * 100) - 5.0 * np.log1p(size_kb)
        
        
        # Temizlik
        os.remove(comp_path)
    
        # RL step çıktıları
        terminated = True
        truncated = False
        info = {
            "quality": quality,
            "psnr": psnr_value,
            "ssim": ssim_value,
            "compressed_size_kb": size_kb,
            "reward": reward
        }
    
        return self.obs, reward, terminated, truncated, info


# ----------- Görselleri Yükleme Fonksiyonu -----------

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



# ----------- Eğitim -----------

if __name__ == "__main__":
    print("Görseller yükleniyor...")
    images = load_sample_images("./val2017")

    print("Ortam başlatılıyor...")
    env = DummyVecEnv([lambda: Monitor(JPEGCompressionEnv(images))])
    check_env(JPEGCompressionEnv(images), warn=True)

    print("Model eğitiliyor...")

    model = PPO(
        "CnnPolicy",
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
        tensorboard_log="./ppo_jpeg_tensorboard/",
        policy_kwargs={"normalize_images": False}
    )


    eval_env = DummyVecEnv([lambda: JPEGCompressionEnv(images)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=100000, callback=eval_callback)

    print("Model kaydediliyor...")
    model.save("jpeg_rl_agent_optimized3")
    print("Model eğitildi ve kaydedildi.")
