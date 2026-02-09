# 零基礎科展範例（清洗後框可微調，大小固定）
import numpy as np
from PIL import Image

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


def analyze_cleaning(before_crop: np.ndarray, after_crop: np.ndarray) -> float:
    before_gray = cv2.cvtColor(before_crop, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_crop, cv2.COLOR_RGB2GRAY)

    if before_gray.shape != after_gray.shape:
        after_gray = cv2.resize(after_gray, (before_gray.shape[1], before_gray.shape[0]))

    after_matched = exposure.match_histograms(after_gray, before_gray)
    diff = cv2.absdiff(before_gray, after_matched.astype(np.uint8))
    return float(np.mean(diff) / 255 * 100)


if HAS_STREAMLIT:
    st.set_page_config(page_title="抹布洗淨力分析", layout="wide")
    st.title("🧼 抹布清洗前後洗淨力影像分析")
    st.write("清洗前自由框選，清洗後框大小固定，可拖動位置")

    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("上傳清洗前照片", type=["jpg","png","jpeg"], key="before")
    with col2:
        after_file = st.file_uploader("上傳清洗後照片", type=["jpg","png","jpeg"], key="after")

    if before_file and after_file:
        before_img = Image.open(before_file).convert("RGB")
        after_img = Image.open(after_file).convert("RGB")

        st.subheader("① 清洗前框選")
        cropped_before, box_coords = st_cropper(
            before_img,
            realtime_update=True,
            box_color="#FF0000",
            aspect_ratio=None,
            return_type='both',
            key="before_crop"
        )

        # 安全處理座標
        if not box_coords or len(box_coords) != 4:
            x0, y0, x1, y1 = 0, 0, before_img.width, before_img.height
        else:
            try:
                x0, y0, x1, y1 = [int(round(float(c))) for c in box_coords]
            except Exception:
                x0, y0, x1, y1 = 0, 0, before_img.width, before_img.height

        width = x1 - x0
        height = y1 - y0

        st.subheader("② 清洗後框選（大小固定，可拖動位置）")
        # 清洗後框大小固定，使用 fixed_size
        cropped_after = st_cropper(
            after_img,
            realtime_update=True,
            box_color="#00AAFF",
            aspect_ratio=None,
            return_type='image',
            key="after_crop",
            fixed_size=(width, height)  # 固定大小
        )

        col3, col4 = st.columns(2)
        with col3:
            st.image(cropped_before, caption="清洗前（裁切後）")
        with col4:
            st.image(cropped_after, caption="清洗後（裁切後，可拖動框）")

        st.subheader("③ 洗淨力分析")
        diff_percent = analyze_cleaning(np.array(cropped_before), np.array(cropped_after))
        st.success(f"📊 洗淨差異百分比：約 {diff_percent:.2f} %")

        st.markdown("""
        ### 🔍 結果說明
        - 清洗前自由框選，清洗後框大小固定，可拖動位置
        - 百分比越高表示污垢被清除越多
        - 可比較不同清潔方式或清潔劑
        """)

    else:
        st.info("請先上傳清洗前與清洗後的照片")

else:
    print("此版本主要設計為網頁應用程式，請於 Streamlit Cloud 使用")
