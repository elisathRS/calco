# Calco · Tattoo stencil generator

Turns a reference photo into a clean line stencil with a **transparent background**, ready to print
on hectograph or thermal paper. Built with **Python, OpenCV and Streamlit**.

## Notebook concepts it applies

| Concept | Where you see it in the app | OpenCV functions |
|---|---|---|
| Image arithmetic and clipping | **Contrast** and **Brightness** sliders; lost-shadows / blown-highlights metric | `cv2.multiply`, `cv2.add`, `cv2.subtract`, `np.clip` |
| Thresholding | Method selector: global, global inverse, adaptive (mean and Gaussian) and Otsu | `cv2.threshold`, `cv2.adaptiveThreshold`, `THRESH_BINARY_INV`, `THRESH_OTSU` |
| Masks and alpha channels | The ink becomes a mask, then a transparency channel | `cv2.bitwise_not`, `cv2.split`, `cv2.merge` |
| Drawing and annotation | Registration marks, cm ruler, box with design size and a text label | `cv2.line`, `cv2.circle`, `cv2.rectangle`, `cv2.putText` |
| Bonus: YOLO | Detects objects in the reference and can crop to them | `ultralytics` (optional) |

It also uses `cv2.GaussianBlur`, `cv2.connectedComponentsWithStats` (speck removal),
`cv2.dilate` / `cv2.erode` (line weight), `cv2.flip` (mirror) and `cv2.findNonZero` +
`cv2.boundingRect` (box with the design's measurements).

## How the pipeline works

1. **Light:** multiply by the contrast, add or subtract the brightness, clip to 0-255.
2. **Thresholding:** separate lines from mid-tones. Normal methods leave the paper at 255.
3. **Ink mask:** `bitwise_not` makes the lines 255 (`global inverse` already returns them that way).
4. **Cleanup and shading:** remove specks, adjust line weight and turn mid-tones into dots.
5. **Alpha channel:** `merge([b, g, r, alpha])` joins the ink colour with the mask.
6. **Sheet:** optional mirror, then marks, ruler and measurements drawn with OpenCV drawing functions.
7. **Export:** PNG at 300 ppi with the real size stored in the file.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Publish with a public link (Streamlit Community Cloud, free)

1. Create an account on <https://github.com> and a new **public** repository (for example `calco-stencils`).
2. Upload these files to the repository: `app.py`, `stencil_lib.py`, `requirements.txt` and
   `.streamlit/config.toml` (dragging them into the GitHub web page works; keep the `.streamlit` folder).
3. Sign in at <https://share.streamlit.io> with your GitHub account and click **Create app**.
4. Pick the repository, the `main` branch and `app.py` as the main file, then click **Deploy**.
5. After 1-3 minutes you get a link like `https://<name>.streamlit.app`.
   Under **Share**, check the app is **public**, and open it in a private window to confirm it
   loads without logging in.

### Alternative: Hugging Face Spaces

Create a new Space with the **Streamlit** SDK, upload the same files and it publishes itself.

### Turning on YOLO (optional)

Replace `requirements.txt` with the contents of `requirements-yolo.txt`. The first run downloads the
`yolov8n.pt` model. The free tier has little memory, so if the app runs out of resources, remove YOLO:
everything else works without it.

## Structure

```
app.py            Streamlit interface
stencil_lib.py    all the OpenCV processing (testable without Streamlit)
requirements.txt  dependencies
.streamlit/       visual theme
```
