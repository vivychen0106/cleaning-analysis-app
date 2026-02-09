# 手把手、零基礎可用的範例程式（含「無 Streamlit 環境」保底方案）
# --------------------------------------------------
# 說明：
# 原本版本使用 Streamlit 建立圖形化介面。
# 若你遇到錯誤：ModuleNotFoundError: No module named 'streamlit'
# 代表「目前執行環境沒有安裝 streamlit，且無法即時安裝」。
#
# 為了讓【程式一定能跑】、【科展不中斷】，本檔案已改為：
# ✅ 自動偵測是否有 streamlit
#   - 有 → 使用「左右上傳照片」的圖形介面（原本功能）
#   - 沒有 → 自動切換為「命令列（CLI）版本」
#              仍可完成：裁切、亮度校正、洗淨力百分比分析
#
# 教學上你只要記一句話：
# 👉「有 Streamlit 就用介面，沒有也能算數據」
# --------------------------------------------------

import sys
import numpy as np
from PIL import Image

# ===== 嘗試載入第三方套件 =====
HAS_STREAMLIT = True
try:
    import streamlit as st
except ModuleNotFoundError:
    HAS_STREAMLIT = False

try:
    import cv2
except ModuleNotFoundError:
    raise ModuleNotFoundError("需要安裝 opencv-python，否則無法進行影像分析")

try:
    from skimage import exposure
except ModuleNotFoundError:
    raise ModuleNotFoundError("需要安裝 scikit-image，否則無法進行亮度校正")

# --------------------------------------------------
# 共用工具函式（GUI / CLI 兩邊都會用）
# --------------------------------------------------

def load_image_from_path(path: str) -> np.ndarray:
    """從檔案路徑讀取圖片並轉為 RGB numpy array"""
    return np.array(Image.open(path).convert("RGB"))


def analyze_cleaning(before_img: np.ndarray, after_img: np.ndarray,
                     x: int, y: int, w: int, h: int) -> float:
    """
    核心分析函式（可視為『洗淨力計算引擎』）
    回傳：洗淨差異百分比（0~100）
    """
    before_crop = before_img[y:y+h, x:x+w]
    after_crop = after_img[y:y+h, x:x+w]

    before_gray = cv2.cvtColor(before_crop, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_crop, cv2.COLOR_RGB2GRAY)

    # 亮度校正
    after_matched = exposure.match_histograms(after_gray, before_gray)

    diff = cv2.absdiff(before_gray, after_matched.astype(np.uint8))
    diff_percent = float(np.mean(diff) / 255 * 100)
    return diff_percent

# --------------------------------------------------
# 一、Streamlit 介面版本（原本設計，環境支援才啟用）
# --------------------------------------------------

if HAS_STREAMLIT:
    st.set_page_config(page_title="抹布洗淨力影像分析", layout="wide")

    st.title("🧼 抹布清洗前後洗淨力影像分析")
    st.write("請分別上傳清洗前與清洗後的照片，完成裁切與校正後即可得到洗淨力差異百分比。")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("清洗前")
        before_file = st.file_uploader("上傳清洗前照片", type=["jpg", "png", "jpeg"], key="before")

    with col2:
        st.subheader("清洗後")
        after_file = st.file_uploader("上傳清洗後照片", type=["jpg", "png", "jpeg"], key="after")

    if before_file and after_file:
        before_img = np.array(Image.open(before_file).convert("RGB"))
        after_img = np.array(Image.open(after_file).convert("RGB"))

        st.divider()
        st.subheader("① 調整裁切位置（請讓兩張圖留下『同一塊抹布』）")

        h_img, w_img, _ = before_img.shape
        x = st.slider("裁切 X 起點", 0, w_img - 50, 0)
        y = st.slider("裁切 Y 起點", 0, h_img - 50, 0)
        cw = st.slider("裁切寬度", 50, w_img - x, w_img - x)
        ch = st.slider("裁切高度", 50, h_img - y, h_img - y)

        diff_percent = analyze_cleaning(before_img, after_img, x, y, cw, ch)

        st.divider()
        st.subheader("② 洗淨力分析結果")
        st.success(f"📊 洗淨差異百分比：約 {diff_percent:.2f} %")

        st.markdown("""
        ### 🔍 結果說明（科展可直接使用）
        - 百分比越高，代表清洗前後顏色改變越明顯
        - 數值來自影像像素差異的平均值
        - 已透過裁切與亮度校正降低拍照誤差
        - 適合比較不同清潔方式、清潔劑、洗滌次數
        """)
    else:
        st.info("請同時上傳清洗前與清洗後的照片")

# --------------------------------------------------
# 二、CLI 命令列版本（沒有 Streamlit 時自動啟用）
# --------------------------------------------------

else:
    print("\n【CLI 模式】目前環境未安裝 streamlit，已切換為命令列分析模式")
    print("用法：")
    print("python app.py before.jpg after.jpg x y width height")
    print("範例：")
    print("python app.py before.jpg after.jpg 50 60 200 200\n")

    if len(sys.argv) != 7:
        print("❌ 參數數量錯誤，請依照格式輸入")
        sys.exit(1)

    _, before_path, after_path, x, y, w, h = sys.argv
    x, y, w, h = map(int, (x, y, w, h))

    before_img = load_image_from_path(before_path)
    after_img = load_image_from_path(after_path)

    diff_percent = analyze_cleaning(before_img, after_img, x, y, w, h)

    print(f"📊 洗淨差異百分比：約 {diff_percent:.2f} %")

# --------------------------------------------------
# 三、內建簡易測試（確保核心演算法正常）
# --------------------------------------------------

def _test_analyze_cleaning_basic():
    """測試：兩張完全相同的圖片，差異應接近 0%"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = analyze_cleaning(img, img, 0, 0, 100, 100)
    assert result < 0.01, "相同圖片差異不應大於 0%"


def _test_analyze_cleaning_difference():
    """測試：明顯亮度差異，百分比應大於 0"""
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    result = analyze_cleaning(img1, img2, 0, 0, 100, 100)
    assert result > 10, "亮度差異應產生明顯百分比"


if __name__ == "__main__":
    # 只在 CLI 執行時跑測試（Streamlit 不跑）
    if not HAS_STREAMLIT:
        _test_analyze_cleaning_basic()
        _test_analyze_cleaning_difference()
        print("✅ 內建測試通過")
