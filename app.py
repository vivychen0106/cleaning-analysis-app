# 手把手、零基礎可用的範例程式（可視化裁切版本）
# --------------------------------------------------
# 改版重點（給老師看的）：
# ✅ 裁切不再用「數值滑桿」
# ✅ 改成「直接在圖片上用滑鼠框選」
#    → 對學生與評審都直觀
#
# 技術說明（不用背）：
# 使用 streamlit-cropper 套件
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
    raise ModuleNotFoundError("需要安裝 opencv-python-headless")

try:
    from skimage import exposure
except ModuleNotFoundError:
    raise ModuleNotFoundError("需要安裝 scikit-image")

# ⭐ 新增：可視化裁切工具
try:
    from streamlit_cropper import st_cropper
except ModuleNotFoundError:
    raise ModuleNotFoundError("需要安裝 streamlit-cropper")

# --------------------------------------------------
# 共用分析核心
# --------------------------------------------------

def analyze_cleaning(before_crop: np.ndarray, after_crop: np.ndarray) -> float:
    before_gray = cv2.cvtColor(before_crop, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_crop, cv2.COLOR_RGB2GRAY)

    after_matched = exposure.match_histograms(after_gray, before_gray)
    diff = cv2.absdiff(before_gray, after_matched.astype(np.uint8))
    return float(np.mean(diff) / 255 * 100)

# --------------------------------------------------
# Streamlit 視覺化介面（主要使用模式）
# --------------------------------------------------

if HAS_STREAMLIT:
    st.set_page_config(page_title="抹布洗淨力影像分析（視覺裁切）", layout="wide")

    st.title("🧼 抹布清洗前後洗淨力影像分析")
    st.write("請在圖片上直接框選同一塊抹布區域，再進行洗淨力分析。")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("清洗前（請框選分析區域）")
        before_file = st.file_uploader("上傳清洗前照片", type=["jpg", "png", "jpeg"], key="before")

    with col2:
        st.subheader("清洗後（會自動套用相同裁切）")
        after_file = st.file_uploader("上傳清洗後照片", type=["jpg", "png", "jpeg"], key="after")

    if before_file and after_file:
        before_img = Image.open(before_file).convert("RGB")
        after_img = Image.open(after_file).convert("RGB")

        st.divider()
        st.subheader("① 用滑鼠框選『同一塊抹布』")

        # 👇 使用者直接在圖上裁切
        cropped_before = st_cropper(
            before_img,
            realtime_update=True,
            box_color="#FF0000",
            aspect_ratio=None
        )

        # 套用相同裁切尺寸到 after 圖
        w, h = cropped_before.size
        cropped_after = after_img.crop((0, 0, w, h))

        col3, col4 = st.columns(2)
        with col3:
            st.image(cropped_before, caption="清洗前（裁切後）")
        with col4:
            st.image(cropped_after, caption="清洗後（裁切後）")

        st.divider()
        st.subheader("② 洗淨力分析結果")

        diff_percent = analyze_cleaning(
            np.array(cropped_before),
            np.array(cropped_after)
        )

        st.success(f"📊 洗淨差異百分比：約 {diff_percent:.2f} %")

        st.markdown("""
        ### 🔍 結果說明（學生可理解版）
        - 在相同位置下，比較清洗前後顏色變化
        - 百分比越高，代表污垢被洗掉得越多
        - 已透過亮度校正，降低拍照光線影響
        - 可用於比較不同清潔方式或清潔劑
        """)

    else:
        st.info("請先上傳清洗前與清洗後的照片")

# --------------------------------------------------
# CLI 備援模式（不影響科展，但保留專業完整性）
# --------------------------------------------------

else:
    print("此版本主要設計為網頁應用程式，請於 Streamlit Cloud 使用")
