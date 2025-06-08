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
    """
    JPEG sıkıştırma kalitesi üzerine RL ajanı eğitmek için özel bir Gym ortamı.
    Her adımda bir görsel seçilir ve ajan JPEG kalitesini belirler.
    Amaç hem kaliteyi yüksek tutmak hem de dosya boyutunu azaltmaktır.
    """
    def __init__(self, images):
        super().__init__()
        self.images = images  # Eğitimde kullanılacak görsel listesi
        self.index = 0        # Sıradaki görselin indeksini tutar

        # Eylem uzayı: 19 tane ayrık kalite değeri (5, 10, ..., 95)
        self.action_space = gym.spaces.Discrete(19)

        # Gözlem uzayı: Normalize edilmiş 64x64 RGB görsel (kanal, yükseklik, genişlik)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Ortam her başlatıldığında yeni bir görsel yüklenir.
        Görsel 64x64'e küçültülür ve normalize edilir.
        """
        super().reset(seed=seed)
        self.current_img = self.images[self.index % len(self.images)]
        self.index += 1

        # Görsel yeniden boyutlandırılır ve normalize edilir
        img_resized = self.current_img.resize((64, 64), Image.LANCZOS)
        obs = np.array(img_resized, dtype=np.float32) / 255.0
        self.obs = np.transpose(obs, (2, 0, 1))  # (H, W, C) → (C, H, W)

        return self.obs, {}

    def step(self, action):
        """
        Ajan bir kalite seviyesi seçer. Seçilen kaliteye göre JPEG sıkıştırması yapılır.
        Ödül, kalite metrikleri ve dosya boyutuna göre hesaplanır.
        """
        # Ajanın verdiği eyleme karşılık gelen JPEG kalite değeri
        quality = 5 + 5 * int(action)

        # Orijinal görselin boyutlandırılmış hali
        orig_img = self.current_img.resize((64, 64), Image.LANCZOS)

        # Görsel geçici olarak sıkıştırılır
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            orig_img.save(tmp.name, format="JPEG", quality=quality, optimize=True)
            comp_path = tmp.name

        # Sıkıştırılmış görsel okunur ve aynı boyuta yeniden boyutlandırılır
        comp_img = Image.open(comp_path).convert("RGB").resize((64, 64), Image.LANCZOS)
        comp_array = np.array(comp_img)
        orig_array = np.array(orig_img)

        # Kalite metrikleri hesaplanır
        psnr_value = psnr(orig_array, comp_array)
        ssim_value = ssim(orig_array, comp_array, channel_axis=2)

        # Dosya boyutu kilobayt olarak hesaplanır
        size_kb = os.path.getsize(comp_path) / 1024.0

        # Ödül fonksiyonu: kalite artı yapısal benzerlik eksi boyut cezası
        reward = (0.5 * psnr_value + 0.5 * ssim_value * 100) - 5.0 * np.log1p(size_kb)

        # Geçici dosya silinir
        os.remove(comp_path)

        # Ortam tek adımda sonlanır (terminated=True)
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

# === GÖRSELLERİ YÜKLEME ===

def load_sample_images(folder_path):
    """
    Belirtilen klasördeki tüm .jpg uzantılı dosyaları yükler ve liste olarak döner.
    RGB formatına dönüştürülür.
    """
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
    # Ortam RL ile çalışmaya uygun hale getirilir (vektörleştirilir ve loglanır)
    env = DummyVecEnv([lambda: Monitor(JPEGCompressionEnv(images))])
    check_env(JPEGCompressionEnv(images), warn=True)  # Ortam yapısı doğrulanır

    print("Model eğitiliyor...")

    # PPO ajanı tanımlanır
    model = PPO(
        "CnnPolicy",           # Görsel girdiler için CNN tabanlı politika ağı
        env,                   # Ortam
        verbose=1,             # Eğitim sürecinde detaylı çıktı alınır
        learning_rate=3e-4,    # Öğrenme hızı
        n_steps=2048,          # Her rollout sırasında alınacak adım sayısı
        batch_size=64,         # Her güncellemede kullanılacak örnek sayısı
        n_epochs=10,           # Her güncellemede yapılacak epoch sayısı
        gamma=0.99,            # Gelecekteki ödüllerin iskonto oranı
        gae_lambda=0.95,       # GAE parametresi
        clip_range=0.2,        # PPO clipping aralığı
        ent_coef=0.01,         # Entropi cezası katsayısı
        tensorboard_log="./ppo_jpeg_tensorboard/",  # TensorBoard log klasörü
        policy_kwargs={"normalize_images": False}   # Girdiler zaten normalize edildiği için tekrar yapılmaz
    )

    # Değerlendirme ortamı ve otomatik model kaydetme callback'i
    eval_env = DummyVecEnv([lambda: JPEGCompressionEnv(images)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",  # En iyi model buraya kaydedilir
        log_path="./logs/",                # Log kayıt klasörü
        eval_freq=1000,                    # Her 1000 adımda bir değerlendirme yapılır
        deterministic=True,
        render=False
    )

    # Eğitim süreci başlatılır
    model.learn(total_timesteps=100000, callback=eval_callback)

    # Model kaydedilir
    print("Model kaydediliyor...")
    model.save("jpeg_rl_agent_optimized3")
    print("Model eğitildi ve kaydedildi.")
