import os
from custom_policy import CustomPPO
from wrappers import make_base_env  # [新增] 必須引入這行來建立環境

# Game Settings
GAME = "SuperMarioWorld-Snes"
STATE = "YoshiIsland1"

# Training Settings
BASE_CHUNK  = 8192
TRAIN_CHUNK = BASE_CHUNK * 32
TOTAL_STEPS = TRAIN_CHUNK * 160
N_ENVS = 16

# Evaluation & Recording Settings
EVAL_EPISODES = 3
EVAL_MAX_STEPS = 18000
RECORD_STEPS = 1200

# Directories
LOG_DIR = "./runs_smw"
VIDEO_DIR       = os.path.join(LOG_DIR, "videos")
CKPT_DIR        = os.path.join(LOG_DIR, "checkpoints")
TENSORBOARD_LOG = os.path.join(LOG_DIR, "tb")

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(CKPT_DIR,  exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# ================= 設定區 =================
# 請確保這些變數有被定義 (這裡沿用你原本的變數名稱)
# GAME = "SuperMarioWorld-Snes"
# STATE = "Level1" 
# CKPT_DIR = "./"
# RECORD_STEPS = 2000
PSVD_DIR = "./runs_smw/preserved/"

target_numbers = list(range(90, 105))
# target_numbers = [124, 137, 147, 151, 179]

# ================= 執行迴圈 =================
for num in target_numbers:
    model_path = os.path.join(CKPT_DIR, f"Run_{num}.zip")
    
    if not os.path.exists(model_path):
        # print(f"⚠️ 找不到檔案: {model_path}，跳過。")
        continue
    
    # print(f"\n[{num}] 正在載入模型: {model_path} ...")
    
    env = None
    try:
        model = CustomPPO.load(model_path, device="auto")
        env = make_base_env(game=GAME, state=STATE)
        
        obs, info = env.reset()
        final_score = 0
        final_coins = 0 # [新增] 初始化金幣紀錄
        
        for step in range(RECORD_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 從 info 中讀取當前數值 
            final_score = info.get("score", final_score)
            final_coins = info.get("coins", final_coins)
            
            if terminated or truncated:
                break
        
        # 修改後的印出格式
        print(f"[{num}] coins: {final_coins} | score: {final_score}")
        
    except Exception as e:
        print(f"❌ 發生錯誤 (Model: {num}): {e}")
    finally:
        if env is not None:
            env.close()

print("\n所有測試結束。")