"""Calco: tattoo stencil generator (Streamlit + OpenCV)."""
import hashlib

import cv2
import streamlit as st

import stencil_lib as sl

st.set_page_config(page_title="Calco · Tattoo stencil generator", page_icon="🖋️", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pirata+One&display=swap');
.calco-title{font-family:'Pirata One',serif;font-size:4.4rem;line-height:.95;color:#7f5fe0;margin:0 0 .2rem 0}
.calco-sub{font-size:1.05rem;opacity:.8;max-width:60ch;margin:0 0 1rem 0}
div[data-testid="stMetricValue"]{color:#7f5fe0}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# State and recipes
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    contrast=1.6, brightness=0, method="Adaptive (Gaussian)", threshold=127, block=25, C=10,
    blur=1.5, clean=45, weight=0, shade=35,
    ink="Violet (classic stencil)", mirror=False, width_cm=15, marks=True, ruler=True, box=True,
    label="", background="Paper", use_yolo=False, yolo_crop=True, yolo_conf=0.25,
)
for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)

RECIPES = {
    "Clean drawing": dict(method="Otsu (automatic)", contrast=1.3, brightness=0, blur=0.75, clean=25, weight=0, shade=0),
    "Realistic photo": dict(method="Adaptive (Gaussian)", contrast=1.6, brightness=0, block=25, C=10, blur=1.5, clean=45, weight=0, shade=35),
    "Pencil sketch": dict(method="Adaptive (mean)", contrast=2.2, brightness=5, block=41, C=6, blur=1.0, clean=25, weight=1, shade=0),
    "Bold line": dict(method="Global inverse", contrast=1.4, brightness=-10, threshold=110, blur=1.0, clean=30, weight=1, shade=0),
}


def apply_recipe(name):
    st.session_state.update(RECIPES[name])


@st.cache_data(show_spinner=False)
def cached_sample():
    return sl.sample_image()


@st.cache_resource(show_spinner="Loading YOLO...")
def load_yolo():
    from ultralytics import YOLO  # optional: pip install ultralytics

    return YOLO("yolov8n.pt")


# ----------------------------------------------------------------------------
# Sidebar: controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 1 · Reference photo")
    uploaded = st.file_uploader(
        "Upload the client's image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help="If you upload nothing, a sample mandala is shown.",
    )

    st.markdown("### 2 · Light")
    st.slider("Contrast", 0.3, 4.0, step=0.05, key="contrast",
              help="Multiplies every pixel. More contrast separates lines from the paper.")
    st.slider("Brightness", -100, 100, step=1, key="brightness",
              help="Positive adds light, negative removes it. Values past 0 or 255 are clipped.")

    st.markdown("### 3 · Lines")
    st.selectbox("Threshold method", list(sl.METHODS), key="method",
                 help="Global and global inverse use a fixed cutoff. Adaptive methods look at each pixel's surroundings. Otsu picks the cutoff by itself.")
    method = sl.METHODS[st.session_state.method]
    if method in ("global", "global_inv"):
        st.slider("Threshold", 0, 255, key="threshold", help="Pixels darker than this become ink.")
    if method in ("adaptive_mean", "adaptive_gauss"):
        st.slider("Neighbourhood (px)", 3, 101, step=2, key="block", help="Size of the area each pixel is compared with.")
        st.slider("Sensitivity (C)", -10, 40, key="C", help="Higher = fewer lines, lower = more lines.")
    st.slider("Pre-smoothing", 0.0, 4.0, step=0.25, key="blur", help="Removes photo grain before looking for lines.")

    st.markdown("### 4 · Cleanup and shading")
    st.slider("Remove specks (px)", 0, 400, step=5, key="clean", help="Deletes ink blobs smaller than this.")
    st.slider("Line weight", -3, 4, key="weight", help="Positive thickens, negative thins.")
    st.slider("Dot shading", 0, 100, key="shade", help="Turns mid-tones into dots. 0 = no shading.")

    st.markdown("### 5 · Print sheet")
    st.selectbox("Ink colour", list(sl.INK_BGR), key="ink")
    st.checkbox("Mirror (flip horizontally)", key="mirror", help="Turn on if your transfer paper reverses the design.")
    st.slider("Design width (cm)", 4, 40, key="width_cm")
    st.checkbox("Registration marks", key="marks")
    st.checkbox("Ruler in cm", key="ruler")
    st.checkbox("Box with design size", key="box")
    st.text_input("Label (accents are removed)", key="label", max_chars=40, placeholder="Client, body area, date...")
    st.radio("Preview on", sl.PREVIEW_BACKGROUNDS, key="background", horizontal=True)

    with st.expander("Bonus: detect objects with YOLO"):
        st.checkbox("Use YOLO", key="use_yolo",
                    help="Detects people, animals and common objects (COCO classes). It does not recognise abstract tattoo designs.")
        st.checkbox("Crop to detections", key="yolo_crop")
        st.slider("Minimum confidence", 0.05, 0.9, step=0.05, key="yolo_conf")

# ----------------------------------------------------------------------------
# Input image
# ----------------------------------------------------------------------------
if uploaded is not None:
    img = sl.decode_image(uploaded.getvalue())
    if img is None:
        st.error("I could not read that image. Try a JPG or PNG.")
        st.stop()
    source_name = f"'{uploaded.name}'"
else:
    img = cached_sample()
    source_name = "sample mandala"

original_view = img
if st.session_state.use_yolo:
    try:
        detections = sl.detect_objects(img, load_yolo(), st.session_state.yolo_conf)
        if detections:
            original_view = sl.draw_detections(img, detections)
            if st.session_state.yolo_crop:
                x1, y1, x2, y2 = sl.union_box(detections, img.shape)
                img = img[y1:y2, x1:x2]
            st.sidebar.success(f"YOLO found {len(detections)} object(s).")
        else:
            st.sidebar.info("YOLO found no known objects in this image.")
    except ImportError:
        st.sidebar.warning("To use YOLO install the optional package: pip install ultralytics")
    except Exception as e:  # noqa: BLE001
        st.sidebar.warning(f"YOLO could not run: {e}")

# ----------------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------------
S = st.session_state
params = sl.Params(
    contrast=float(S.contrast), brightness=int(S.brightness), blur=float(S.blur), method=method,
    value=int(S.threshold), block=int(S.block), C=int(S.C), clean=int(S.clean), weight=int(S.weight),
    shade=int(S.shade), ink_bgr=sl.INK_BGR[S.ink],
)
result = sl.process(img, params)
sheet, px_cm = sl.build_sheet(
    result["rgba"], S.width_cm, params.ink_bgr,
    mirror=S.mirror, marks=S.marks, ruler=S.ruler, box=S.box, label=S.label,
)
view = sl.preview(sheet, S.background)
stats = result["stats"]

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.markdown('<p class="calco-title">Calco</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="calco-sub">Turn a reference photo into a clean line stencil with a transparent background, '
    "ready to print on hectograph or thermal paper.</p>",
    unsafe_allow_html=True,
)

st.caption("Quick recipes")
recipe_cols = st.columns(len(RECIPES))
for col, name in zip(recipe_cols, RECIPES):
    col.button(name, on_click=apply_recipe, args=(name,), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Reference")
    st.image(cv2.cvtColor(original_view, cv2.COLOR_BGR2RGB), use_container_width=True,
             caption=f"{source_name} · {img.shape[1]} × {img.shape[0]} px")
sheet_w_cm = sheet.shape[1] / px_cm
sheet_h_cm = sheet.shape[0] / px_cm
with right:
    st.subheader("Stencil")
    st.image(cv2.cvtColor(view, cv2.COLOR_BGR2RGB), use_container_width=True,
             caption=f"Sheet of {sheet_w_cm:.1f} × {sheet_h_cm:.1f} cm")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Ink on the sheet", f"{stats['coverage'] * 100:.1f}%")
m2.metric("Specks removed", stats["removed"])
used = stats["threshold"]
m3.metric("Threshold used", f"{used:.0f}" if used is not None else "adaptive",
          help="With Otsu, OpenCV computes this value by itself.")
m4.metric("Lost shadows / blown highlights",
          f"{stats['clipped_low'] * 100:.0f}% / {stats['clipped_high'] * 100:.0f}%",
          help="Values that contrast and brightness pushed below 0 or above 255. Blown paper highlights are harmless; lost shadows are not.")

with st.expander("See the process step by step", expanded=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.image(cv2.cvtColor(result["adjusted"], cv2.COLOR_BGR2RGB), caption="1 · Light: contrast and brightness", use_container_width=True)
    c2.image(result["binary"], caption="2 · Thresholding", use_container_width=True, clamp=True)
    c3.image(result["ink"], caption="3 · Clean ink (mask)", use_container_width=True, clamp=True)
    c4.image(result["rgba"][:, :, 3], caption="4 · Alpha channel", use_container_width=True, clamp=True)
    c5.image(cv2.cvtColor(sl.preview(result["rgba"], "Transparent"), cv2.COLOR_BGR2RGB), caption="5 · Transparent stencil", use_container_width=True)
    st.caption(
        "With normal thresholding the paper stays white; the ink mask is inverted so lines are 255, "
        "and that mask is merged with an ink colour as the transparency channel."
    )

# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------
st.markdown("#### Save or print")
signature = hashlib.md5(sheet.tobytes()).hexdigest()
st.caption(f"Exported at 300 ppi with the real size stored in the file: {sheet_w_cm:.1f} × {sheet_h_cm:.1f} cm.")
if st.button("Prepare PNG for printing", type="primary"):
    with st.spinner("Generating..."):
        st.session_state["export"] = dict(
            signature=signature,
            transparent=sl.export_png(sheet, px_cm),
            white=sl.export_png(sheet, px_cm, white_background=True),
        )
export = st.session_state.get("export")
if export:
    if export["signature"] != signature:
        st.info("You changed something after preparing the files. Press “Prepare PNG for printing” again to refresh them.")
    d1, d2 = st.columns(2)
    d1.download_button("Download transparent PNG", export["transparent"], "stencil_transparent.png", "image/png", use_container_width=True)
    d2.download_button("Download PNG with white background", export["white"], "stencil_white_background.png", "image/png", use_container_width=True)

st.info("Tip: test on stencil paper first and tune line weight and shading for your thermal machine. "
        "If the design comes out reversed after transfer, turn on “Mirror”.")
