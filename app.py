import io
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


SCALE_MM_PER_PX = 0.30

EDGE_EXCLUDE_RATIO = 0.08
L_DARK_THRESH = 18
A_BROWN_THRESH = 7
BLACKHAT_THRESH = 13
MIN_DAMAGE_AREA_PX = 30
MIN_VISUAL_DAMAGE_AREA_PX = 18
MIN_LINE_DAMAGE_AREA_PX = 24
MIN_STRONG_BROWN_DIFF = 6
MIN_STRONG_DARK_DIFF = 14
MAX_COMPONENT_AREA_RATIO = 0.08
MAX_TOTAL_DAMAGE_RATIO = 0.20


@dataclass
class AnalysisResult:
    annotated_bgr: np.ndarray
    destination: str
    priority: str
    reason: str
    details: dict


st.set_page_config(
    page_title="すだち出荷先判定",
    page_icon="🍋",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 980px; padding-top: 1.4rem;}
    .main-title {font-size: clamp(2rem, 7vw, 3.4rem); font-weight: 900; line-height: 1.05;}
    .lead {font-size: 1.05rem; color: #4b5563; margin-bottom: 1rem;}
    .result-band {
        border: 1px solid #d9e2d0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8fbf4;
    }
    .result-label {font-size: .9rem; color: #526047; font-weight: 700;}
    .result-value {font-size: clamp(1.8rem, 8vw, 3rem); font-weight: 900; line-height: 1.1;}
    .reason {font-size: clamp(1.05rem, 4vw, 1.35rem); font-weight: 700; line-height: 1.45;}
    div[data-testid="stMetricValue"] {font-size: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">すだち出荷先判定</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lead">スマートフォンで撮影した画像をアップロードすると、色・傷・形・サイズから出荷先を提案します。</div>',
    unsafe_allow_html=True,
)


def load_image(uploaded_file) -> np.ndarray | None:
    data = uploaded_file.read()
    pil_image = Image.open(io.BytesIO(data))
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    max_side = max(bgr.shape[:2])
    if max_side > 1400:
        scale = 1400 / max_side
        bgr = cv2.resize(
            bgr,
            (int(bgr.shape[1] * scale), int(bgr.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return bgr


def find_fruit_mask(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    sat_mask = (hsv[:, :, 1] > 35).astype(np.uint8) * 255
    non_white_mask = (gray < 245).astype(np.uint8) * 255
    candidate = cv2.bitwise_or(sat_mask, non_white_mask)

    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = img_bgr.shape[0] * img_bgr.shape[1] * 0.01
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None, None

    fruit = max(contours, key=cv2.contourArea)
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [fruit], -1, 255, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return mask, fruit


def estimate_stem_mask(mask_roi: np.ndarray, hsv_roi: np.ndarray, lab_roi: np.ndarray) -> np.ndarray:
    h, w = mask_roi.shape
    fruit = mask_roi > 0
    if not np.any(fruit):
        return np.zeros_like(mask_roi)

    top_band = np.zeros_like(mask_roi)
    top_band[: max(1, int(h * 0.28)), :] = 255
    l_roi = lab_roi[:, :, 0]
    s_roi = hsv_roi[:, :, 1]
    v_roi = hsv_roi[:, :, 2]

    dark_top = ((l_roi < np.percentile(l_roi[fruit], 35)) | (v_roi < 95)) & (s_roi > 25)
    stem = (dark_top & (top_band > 0) & fruit).astype(np.uint8) * 255

    contours, _ = cv2.findContours(stem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stem_mask = np.zeros_like(mask_roi)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 8:
            cv2.drawContours(stem_mask, [c], -1, 255, -1)

    stem_mask = cv2.dilate(stem_mask, np.ones((23, 23), np.uint8), iterations=1)
    return cv2.bitwise_and(stem_mask, mask_roi)


def detect_damage(mask_roi, hsv_roi, lab_roi, stem_mask):
    lab_roi = lab_roi.astype(np.float32)

    l_roi = lab_roi[:, :, 0]
    a_roi = lab_roi[:, :, 1]
    b_roi = lab_roi[:, :, 2]
    h_roi = hsv_roi[:, :, 0]
    s_roi = hsv_roi[:, :, 1]

    if mask_roi.sum() == 0:
        return np.zeros_like(mask_roi), [], 0.0

    dist = cv2.distanceTransform(mask_roi, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist <= 0:
        inner_mask = mask_roi.copy()
    else:
        inner_mask = ((dist > max_dist * EDGE_EXCLUDE_RATIO) & (mask_roi > 0)).astype(np.uint8) * 255

    eroded_mask = cv2.erode(inner_mask, np.ones((5, 5), np.uint8))
    if eroded_mask.sum() == 0:
        eroded_mask = inner_mask

    fruit_inner = eroded_mask > 0
    fruit_area = np.count_nonzero(fruit_inner)
    if fruit_area == 0:
        return np.zeros_like(mask_roi), [], 0.0

    mean_l = np.mean(l_roi[fruit_inner])
    mean_a = np.mean(a_roi[fruit_inner])
    mean_b = np.mean(b_roi[fruit_inner])
    mean_s = np.mean(s_roi[fruit_inner])

    dark_mask = l_roi < (mean_l - L_DARK_THRESH)
    brown_mask = a_roi > (mean_a + A_BROWN_THRESH)
    healthy_green = (h_roi >= 35) & (h_roi <= 85) & (s_roi > mean_s - 8)
    yellow_mura = (b_roi > mean_b + 8) & (a_roi < mean_a + 12) & (l_roi > mean_l - 18)

    spot_mask = (dark_mask & brown_mask & (~healthy_green) & (~yellow_mura) & fruit_inner).astype(np.uint8) * 255
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    l_u8 = np.clip(l_roi, 0, 255).astype(np.uint8)
    bh_h = cv2.morphologyEx(l_u8, cv2.MORPH_BLACKHAT, np.ones((3, 17), np.uint8))
    bh_v = cv2.morphologyEx(l_u8, cv2.MORPH_BLACKHAT, np.ones((17, 3), np.uint8))
    blackhat = np.maximum(bh_h, bh_v)

    scratch_mask = (
        (blackhat > BLACKHAT_THRESH)
        & (l_roi < mean_l - 12)
        & (a_roi > mean_a + 5)
        & fruit_inner
    ).astype(np.uint8) * 255
    scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    stem_inv = cv2.bitwise_not(stem_mask)
    candidate_mask = cv2.bitwise_or(cv2.bitwise_and(spot_mask, stem_inv), cv2.bitwise_and(scratch_mask, stem_inv))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask)
    final_mask = np.zeros_like(candidate_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w0 = stats[i, cv2.CC_STAT_WIDTH]
        h0 = stats[i, cv2.CC_STAT_HEIGHT]
        if area <= 0 or area / fruit_area > MAX_COMPONENT_AREA_RATIO or area < MIN_VISUAL_DAMAGE_AREA_PX:
            continue

        comp_mask = labels == i
        aspect = max(w0, h0) / (min(w0, h0) + 1e-5)
        extent = area / (w0 * h0 + 1e-5)
        dark_diff = mean_l - np.mean(l_roi[comp_mask])
        brown_diff = np.mean(a_roi[comp_mask]) - mean_a

        accepted = False
        if area >= MIN_DAMAGE_AREA_PX:
            accepted = extent > 0.18 and brown_diff >= MIN_STRONG_BROWN_DIFF and dark_diff >= MIN_STRONG_DARK_DIFF
        if not accepted:
            accepted = (
                area >= MIN_LINE_DAMAGE_AREA_PX
                and aspect >= 4.0
                and max(w0, h0) >= 24
                and brown_diff >= 4
                and dark_diff >= 10
            )
        if accepted:
            final_mask[comp_mask] = 255

    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    final_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    damages = []
    for c in final_contours:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        if area < MIN_VISUAL_DAMAGE_AREA_PX or area / fruit_area > MAX_COMPONENT_AREA_RATIO:
            continue
        damages.append(c)

    damage_area_px = float(np.sum([cv2.contourArea(c) for c in damages]))
    damage_ratio = damage_area_px / fruit_area if fruit_area > 0 else 0
    if damage_ratio > MAX_TOTAL_DAMAGE_RATIO:
        return np.zeros_like(candidate_mask), [], 0.0

    return final_mask, damages, damage_area_px


def classify_size(diameter_mm: float) -> str:
    if diameter_mm < 28:
        return "小玉"
    if diameter_mm <= 40:
        return "標準"
    return "大玉"


def classify_destination(
    damage_count: int,
    damage_ratio: float,
    h_val: float,
    s_val: float,
    color_mura: float,
    circularity: float,
) -> tuple[str, str, str, str, str]:
    yellow_strong = (h_val < 32 and s_val >= 45) or (32 <= h_val < 38 and s_val < 95)
    green_good = 38 <= h_val <= 78 and s_val >= 55
    color_mura_bad = color_mura >= 13
    damaged = damage_count > 0 or damage_ratio >= 0.003
    heavy_damage = damage_count >= 2 or damage_ratio >= 0.012

    if yellow_strong:
        return (
            "果汁向け",
            "高",
            "黄色みが強く、青果向けより果汁加工を優先したい状態です。",
            "黄色み強め",
            "要確認",
        )

    if heavy_damage:
        return (
            "果汁向け寄り",
            "高",
            "傷があり保存性に不安があるため、早めの果汁加工を優先します。",
            "緑系" if green_good else "要確認",
            "要確認",
        )

    if damaged:
        return (
            "早期出荷または果汁向け",
            "中",
            "軽い傷があります。保存性を考えると長期保管より早めの処理が安心です。",
            "緑系" if green_good else "要確認",
            "軽微",
        )

    if not green_good:
        return (
            "目視確認",
            "中",
            "平均色が青果向けの緑から外れています。黄色みや照明条件を確認してください。",
            "要確認",
            "なし",
        )

    if color_mura_bad:
        return (
            "青果向け",
            "中",
            "色ムラはありますが、色ムラだけでは果汁向けにせず青果向けとして確認します。",
            "緑系",
            "なし",
        )

    if circularity < 0.72:
        return (
            "青果向け確認",
            "低",
            "色と傷は大きな問題がなく、形だけ目視確認するとよい状態です。",
            "緑系",
            "なし",
        )

    return (
        "青果向け",
        "低",
        "緑色が保たれていて、目立つ傷も少ないため青果向け候補です。",
        "緑系",
        "なし",
    )


def analyze_sudachi(img_bgr: np.ndarray, scale: float) -> AnalysisResult | None:
    mask, fruit_contour = find_fruit_mask(img_bgr)
    if mask is None:
        return None

    x, y, w, h = cv2.boundingRect(fruit_contour)
    area_px = cv2.contourArea(fruit_contour)
    perimeter = cv2.arcLength(fruit_contour, True)
    circularity = 4 * np.pi * area_px / (perimeter**2) if perimeter > 0 else 0
    (_, _), radius_px = cv2.minEnclosingCircle(fruit_contour)
    diameter_mm = radius_px * 2 * scale

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    fruit_pixels = mask > 0
    h_val = float(cv2.mean(hsv[:, :, 0], mask=mask)[0])
    s_val = float(cv2.mean(hsv[:, :, 1], mask=mask)[0])
    color_mura = float(np.std(lab[fruit_pixels][:, 1]))

    mask_roi = mask[y : y + h, x : x + w]
    hsv_roi = hsv[y : y + h, x : x + w]
    lab_roi = lab[y : y + h, x : x + w]
    stem_mask = estimate_stem_mask(mask_roi, hsv_roi, lab_roi)
    _, damages, damage_area_px = detect_damage(mask_roi, hsv_roi, lab_roi, stem_mask)

    fruit_inner_area = max(1, np.count_nonzero(mask_roi))
    damage_ratio = damage_area_px / fruit_inner_area
    destination, priority, reason, color_label, damage_label = classify_destination(
        len(damages), damage_ratio, h_val, s_val, color_mura, circularity
    )
    color_mura_label = "多い" if color_mura >= 13 else "少ない"

    annotated = img_bgr.copy()
    cv2.drawContours(annotated, [fruit_contour], -1, (40, 150, 40), 3)
    for c in damages:
        dx, dy, dw, dh = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x + dx, y + dy), (x + dx + dw, y + dy + dh), (0, 0, 255), 3)

    details = {
        "傷数": len(damages),
        "傷面積": f"{damage_area_px:.0f} px2 ({damage_ratio * 100:.2f}%)",
        "H値": round(h_val, 1),
        "S値": round(s_val, 1),
        "色判定": color_label,
        "色ムラ": round(color_mura, 1),
        "色ムラ判定": color_mura_label,
        "円形度": round(circularity, 3),
        "サイズ区分": classify_size(diameter_mm),
    }

    return AnalysisResult(annotated, destination, priority, reason, details)


with st.sidebar:
    st.header("設定")
    scale = st.number_input(
        "スケール mm/px",
        min_value=0.05,
        max_value=2.00,
        value=SCALE_MM_PER_PX,
        step=0.05,
        help="定規と一緒に撮影した場合は、実測値に合わせるとサイズ区分が安定します。",
    )
    st.caption("判定は研究・確認用の補助です。最終判断は目視確認と出荷基準に合わせてください。")


uploaded_file = st.file_uploader(
    "画像をアップロード",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("白い紙や明るい単色背景に、すだちを1個置いて撮影した画像がおすすめです。")
    st.stop()

try:
    image_bgr = load_image(uploaded_file)
except Exception as exc:
    st.error(f"画像を読み込めませんでした: {exc}")
    st.stop()

with st.spinner("OpenCVで解析しています..."):
    result = analyze_sudachi(image_bgr, scale)

if result is None:
    st.error("すだちの輪郭を検出できませんでした。背景を明るくして、1個だけ大きめに写してください。")
    st.stop()

st.markdown(
    f"""
    <div class="result-band">
        <div class="result-label">出荷先提案</div>
        <div class="result-value">{result.destination}</div>
        <div class="result-label" style="margin-top:.9rem;">処理優先度</div>
        <div class="result-value">{result.priority}</div>
        <div class="result-label" style="margin-top:.9rem;">理由</div>
        <div class="reason">{result.reason}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.image(
    cv2.cvtColor(result.annotated_bgr, cv2.COLOR_BGR2RGB),
    caption="赤枠: 傷候補",
    use_container_width=True,
)

st.subheader("詳細")
detail_df = pd.DataFrame(
    [{"項目": key, "値": value} for key, value in result.details.items()]
)
st.dataframe(detail_df, hide_index=True, use_container_width=True)
