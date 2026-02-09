# 手把手、零基礎可用的範例程式（單框同步裁切、完全穩定版）
# --------------------------------------------------
# 改版重點：
# ✅ 只使用一個框選工具（清洗前）
# ✅ 清洗後自動套用相同裁切範圍
# ✅ 完全檢查 st_cropper 回傳值，避免 None、浮點數、tuple 格式錯誤
# ✅ 座標限制在圖片範圍內
# ✅ 避免 cv2 尺寸不一致或 PIL 裁切錯誤
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

try:
    from streamlit_cropper import st_cropper
except ModuleNotFoundError:
    raise ModuleNotFoundError("需要安裝 streamlit-cropper")

# --------------------------------------------------
# 分析核心
# --------------------------------------------------

def analyze_cleaning(before_crop: np.ndarray, after_crop: np.ndarray) -> float:
    before_gray = cv2.cvtColor(before_crop, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_crop, cv2.COLOR_RGB2GRAY)

    if before_gray.shape != after_gray.shape:
        after_gray = cv2.resize(after_gray, (before_gray.shape[1], before_gray.shape[0]))

    after_matched = exposure.match_histograms(after_gray, before_gray)
    diff = cv2.absdiff(before_gray, after_matched.astype(np.uint8))
    return float(np.mean(diff) / 255 * 100)

# --------------------------------------------------
# Streamlit 視覺化介面
# --------------------------------------------------

if HAS_STREAMLIT:
    st.set_page_config(page_title="抹布洗淨力影像分析（單框穩定版）", layout="wide")

    st.title("🧼 抹布清洗前後洗淨力影像分析")
    st.write("請在清洗前圖片上框選分析區域，清洗後將自動套用相同區域")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("清洗前（可拖動框選）")
        before_file = st.file_uploader("上傳清洗前照片", type=["jpg", "png", "jpeg"], key="before")

    with col2:
        st.subheader("清洗後（自動套用相同裁切範圍）")
        after_file = st.file_uploader("上傳清洗後照片", type=["jpg", "png", "jpeg"], key="after")

    if before_file and after_file:
        before_img = Image.open(before_file).convert("RGB")
        after_img = Image.open(after_file).convert("RGB")

        st.divider()
        st.subheader("① 在清洗前圖片上選擇分析區域")

        cropped_before, box_coords = st_cropper(
            before_img,
            realtime_update=True,
            box_color="#FF0000",
            aspect_ratio=None,
            return_type='both',
            key="single_crop"
        )

        # 安全處理座標
        if not box_coords or len(box_coords) != 4:
            # 如果框未拉或格式錯誤，使用整張圖片
            x0, y0, x1, y1 = 0, 0, before_img.width, before_img.height
        else:
            try:
                # 嘗試轉整數
                x0, y0, x1, y1 = [int(round(c)) for c in box_coords]
            except Exception:
                x0, y0, x1, y1 = 0, 0, before_img.width, before_img.height

            # 限制範圍在圖片內
            x0 = max(0, min(x0, after_img.width))
            x1 = max(0, min(x1, after_img.width))
            y0 = max(0, min(y0, after_img.height))
            y1 = max(0, min(y1, after_img.height))

        cropped_after = after_img.crop((x0, y0, x1, y1))

        col3, col4 = st.columns(2)
        with col3:
            st.image(cropped_before, caption="清洗前（裁切後）")
        with col4:
            st.image(cropped_after, caption="清洗後（裁切後，自動套用框）")

        st.divider()
        st.subheader("② 洗淨力分析結果")

        diff_percent = analyze_cleaning(np.array(cropped_before), np.array(cropped_after))

        st.success(f"📊 洗淨差異百分比：約 {diff_percent:.2f} %")

        st.markdown("""
        ### 🔍 結果說明（學生可理解版）
        - 清洗前框選的區域，自動套用到清洗後
        - 百分比越高，代表污垢被洗掉得越多
        - 已透過亮度校正，降低拍照光線影響
        - 可用於比較不同清潔方式或清潔劑
        """)

    else:
        st.info("請先上傳清洗前與清洗後的照片")

else:
    print("此版本主要設計為網頁應用程式，請於 Streamlit Cloud 使用")
