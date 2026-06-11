import sys
import os
# make parent folder importable (run this before importing ghostai)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import shutil
import cv2
import pytesseract
import numpy as np
import re

script_dir = os.path.dirname(__file__)

# CLI: allow --video, --live (camera), --max-frames
parser = argparse.ArgumentParser(description="Extract telemetry from video (or live).")
parser.add_argument("video", nargs="?", help="path to mp4 file (optional)")
parser.add_argument("--live", action="store_true", help="use camera (index 0) instead of a file")
parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
parser.add_argument("--dump-debug", action="store_true", help="save first few crop images to data/debug for inspection")
parser.add_argument("--no-display", action="store_true", help="disable cv2.imshow (faster, headless)")
parser.add_argument("--skip-frames", type=int, default=0, help="skip N frames between processed frames (0 = process every frame)")
parser.add_argument("--display-every", type=int, default=10, help="when displaying, show one frame every N processed frames")
args = parser.parse_args()

csv_path = 'data/my_laps.csv'
lap_data = []

# ensure tesseract available
if not shutil.which("tesseract"):
    print("Warning: tesseract not found. Install it with: brew install tesseract")
    # continue so you can still dump images and inspect crops

# helper: simple preprocess for OCR
def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

# decide what to open: camera, explicit file, or packaged MP4
if args.live:
    video_path = 0
elif args.video:
    video_path = args.video

else:
    def find_videos_in(path):
        try:
            return [os.path.join(path, f) for f in os.listdir(path)
                    if f.lower().endswith('.mp4')]
        except Exception:
            return []

    # search locations in order of likelihood
    pkg_dir = os.path.abspath(os.path.join(script_dir, '..'))    # project root
    cwd = os.getcwd()
    user_home = os.path.expanduser('~')
    search_paths = [
        pkg_dir,
        script_dir,
        cwd,
        os.path.join(user_home, 'Downloads'),
        os.path.join(user_home, 'Movies'),
        user_home
    ]

    candidates = []
    for p in search_paths:
        candidates.extend(find_videos_in(p))

    # deduplicate and keep absolute paths
    candidates = list(dict.fromkeys([os.path.abspath(p) for p in candidates]))

    if candidates:
        video_path = candidates[0]
        print(f"Using first found video: {os.path.basename(video_path)} (from {os.path.dirname(video_path)})")
    else:
        # fallback: common bundled name
        bundled = os.path.join(pkg_dir, 'THE CREW MOTORFEST Video Test 1.mp4')
        if os.path.isfile(bundled):
            video_path = bundled
            print(f"Using bundled sample video: {os.path.basename(video_path)}")
        else:
            # Interactive prompt so you can paste a path if you don't know commands
            try:
                print("No .mp4 found in project, Downloads, Movies, or home folders.")
                user_input = input("Enter full path to an .mp4 file (or press Enter to abort): ").strip()
            except Exception:
                user_input = ""

            if user_input:
                user_input = os.path.expanduser(user_input)
                if os.path.isfile(user_input):
                    video_path = os.path.abspath(user_input)
                    print(f"Using user-specified video: {video_path}")
                else:
                    print(f"Provided path does not exist or is not a file: {user_input}")
                    sys.exit(1)
            else:
                print("No video specified. Run with --live or pass /path/to/video.mp4")
                sys.exit(1)

# open capture (video_path may be int for camera)
cap = cv2.VideoCapture(video_path)

# immediate check and helpful message if OpenCV can't open the file
if not cap.isOpened():
    print(f"OpenCV: Couldn't open video at '{video_path}'. cwd={os.getcwd()}")
    print("Files in cwd:")
    for f in sorted(os.listdir('.')):
        print(" ", f)
    sys.exit(1)

frame_num = 0
failure_log_time = 0.0
failure_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame skipping to reduce CPU / avoid lag (process every skip_frames+1 frame)
        if args.skip_frames and (frame_num % (args.skip_frames + 1)) != 0:
            frame_num += 1
            continue

        # Replace these with the coordinates you pick from scripts/pick_coords.py
        # Format from pick_coords: (x1, y1) top-left and (x2, y2) bottom-right
        # Default crop boxes (adjusted from pick_coords)
        # sector region: narrow box around the left HUD numbers (3 numbers vertically)
        sector_tl = (15, 9)
        sector_br = (1256, 708)
        # lap-time region: top-center HUD lap time (mm:ss.s or seconds)
        lap_tl    = (480, 8)
        lap_br    = (640, 48)

        # use (y1:y2, x1:x2) indexing
        sector_region = frame[sector_tl[1]:sector_br[1], sector_tl[0]:sector_br[0]]
        lap_time_region = frame[lap_tl[1]:lap_br[1], lap_tl[0]:lap_br[0]]

        # write an annotated debug image showing the rectangles
        if args.dump_debug and frame_num < 10:
            ann = frame.copy()
            cv2.rectangle(ann, sector_tl, sector_br, (0,255,0), 2)
            cv2.rectangle(ann, lap_tl, lap_br, (0,0,255), 2)
            cv2.imwrite(f"data/debug/frame_{frame_num}_rects.png", ann)

        # debug dumps (first few frames)
        if args.dump_debug and frame_num < 10:
            os.makedirs('data/debug', exist_ok=True)
            cv2.imwrite(f"data/debug/frame_{frame_num}_full.png", frame)
            cv2.imwrite(f"data/debug/frame_{frame_num}_sector_raw.png", sector_region)
            cv2.imwrite(f"data/debug/frame_{frame_num}_laptime_raw.png", lap_time_region)

        # improved OCR helper
        def ocr_read(img, psm=7, whitelist="0123456789:. "):
            if img is None or img.size == 0:
                return ""
            # upscale to help OCR
            img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # denoise and threshold
            blur = cv2.bilateralFilter(gray, 9, 75, 75)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"
            try:
                return pytesseract.image_to_string(th, config=config)
            except Exception:
                return ""

        sector_proc = preprocess_for_ocr(sector_region)         # keep small debug proc too
        lap_time_proc = preprocess_for_ocr(lap_time_region)
        # use whitelist; sector area likely has multiple numbers -> psm 6 (assume block of text)
        sector_text = ocr_read(sector_region, psm=6, whitelist="0123456789. ")
        lap_time_text = ocr_read(lap_time_region, psm=7, whitelist="0123456789:.")

        if args.dump_debug and frame_num < 10:
            cv2.imwrite(f"data/debug/frame_{frame_num}_sector_proc.png", sector_proc)
            cv2.imwrite(f"data/debug/frame_{frame_num}_laptime_proc.png", lap_time_proc)
            print("Wrote debug images to data/debug — inspect crops and OCR text:")
            print("  sector_text:", repr(sector_text))
            print("  lap_time_text:", repr(lap_time_text))

        try:
            # normalize OCR text and extract numeric values with regex
            sector_text_clean = " ".join(sector_text.replace('\n', ' ').split())
            lap_time_text_clean = lap_time_text.strip()
            # find floats/ints in sector text (robust to garbage)
            nums = re.findall(r"\d+\.\d+|\d+", sector_text_clean)
            if len(nums) >= 3:
                sector1, sector2, sector3 = map(float, nums[:3])
            else:
                raise ValueError(f"Could not find 3 sector numbers in OCR output: {sector_text_clean!r}")
            # lap time may contain colon or decimal
            lap_nums = re.findall(r"\d+\.\d+|\d+:\d+\.\d+|\d+:\d+|\d+", lap_time_text_clean)
            if lap_nums:
                # prefer direct float, else convert mm:ss.s to seconds
                t = lap_nums[0]
                if ":" in t:
                    parts = t.split(":")
                    lap_time = float(parts[-1]) + 60.0 * float(parts[-2])
                else:
                    lap_time = float(t)
            else:
                raise ValueError(f"Could not parse lap time from OCR output: {lap_time_text_clean!r}")

            abs_region = frame[200:220, 50:70]
            esp_region = frame[220:240, 50:70]
            tcs_region = frame[240:260, 50:70]

            abs_text = pytesseract.image_to_string(abs_region)
            esp_text = pytesseract.image_to_string(esp_region)
            tcs_text = pytesseract.image_to_string(tcs_region)

            abs_val = 1 if "ON" in abs_text else 0
            esp_val = 1 if "ON" in esp_text else 0
            tcs_val = 1 if "ON" in tcs_text else 0
            lap_data.append([sector1, sector2, sector3, abs_val, esp_val, tcs_val, lap_time])
        except Exception as e:
            # throttle noisy OCR failures to 1 line/sec
            now = time.time()
            if now - failure_log_time > 1.0:
                print(f"Frame {frame_num}: Failed to parse telemetry. OCR output: sector='{sector_text}', lap_time='{lap_time_text}'. Error: {e}")
                failure_log_time = now
            failure_count += 1

        # allow quitting with 'q' when running with a display
        # display occasionally (or not at all) to avoid UI lag
        if not args.no_display:
            try:
                if frame_num % max(1, args.display_every) == 0:
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quit requested (q).")
                        break
            except Exception:
                # headless / no GUI available: ignore imshow errors
                pass

        frame_num += 1
        if args.max_frames and frame_num >= args.max_frames:
            print(f"Reached max frames ({args.max_frames}). Stopping.")
            break
except KeyboardInterrupt:
    print("Interrupted by user (Ctrl+C). Saving collected data and exiting.")
finally:
    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    # save CSV even if some frames failed
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    if lap_data:
        np.savetxt(csv_path, lap_data, delimiter=',', header='sector1,sector2,sector3,abs,esp,tcs,lap_time', comments='')
        print(f"Saved {len(lap_data)} lap rows to {csv_path}")
    else:
        print("No lap data extracted; CSV not written.")

# postponed import of GhostAI until after CSV is written:
from ghost_ai import GhostAI

if os.path.isfile(csv_path):
    # Train and evaluate GhostAI (only if we created a CSV)
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    if data.size and data.ndim == 2 and data.shape[0] >= 2:
        X = data[:, :6]
        y = data[:, 6]
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        ghost = GhostAI(mode="regression")
        ghost.learn(X_train, y_train)
        score, ok = ghost.evaluate(X_test, y_test)
        print("Model score:", score)
        print("Meets 93% threshold?", ok)

        # Feature importance feedback
        if hasattr(ghost.model, "feature_importances_"):
            print("Feature importances:", ghost.model.feature_importances_)
    else:
        print("CSV exists but doesn't contain enough rows to train the model.")

