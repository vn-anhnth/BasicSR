import os
import sys
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Fix Windows console UTF-8 encoding
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_ROOT = r"D:\IEEE\data\phD\data\50k_OCR\aaa_train_ne_version_5"
TEST_FILES = [
    "test_labels.txt",
    "test_brazil_labels.txt",
    "test_china_labels.txt",
    "test_vn_labels.txt",
]
SCALES = [2, 4]


def get_scaled_rel_path(rel_path, scale):
    """
    If 'degraded' is in a folder name, append _x{scale} to that folder name.
    Example:
      RodoSol-ALPR_20000/cars-br_5000_degraded/type1_249_lp.jpg
      -> RodoSol-ALPR_20000/cars-br_5000_degraded_x2/type1_249_lp.jpg
      vietnam_3173/degraded/Tgmt_0461.png
      -> vietnam_3173/degraded_x2/Tgmt_0461.png
    """
    norm = rel_path.replace("\\", "/")
    parts = norm.split("/")
    new_parts = []
    for p in parts[:-1]:  # directory parts only
        if "degraded" in p.lower():
            new_parts.append(f"{p}_x{scale}")
        else:
            new_parts.append(p)
    new_parts.append(parts[-1])  # filename unchanged
    return "/".join(new_parts)


def process_image(rel_path):
    """
    Read an original degraded image and generate downscaled x2 and x4 versions.
    Uses cv2.INTER_CUBIC, exactly matching create_multiscale_lmdb.py.
    """
    src_full = os.path.join(DATA_ROOT, rel_path)
    img = cv2.imread(src_full)
    if img is None:
        return rel_path, False

    h, w = img.shape[:2]

    for scale in SCALES:
        target_w = max(1, w // scale)
        target_h = max(1, h // scale)
        scaled_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        dst_rel = get_scaled_rel_path(rel_path, scale)
        dst_full = os.path.join(DATA_ROOT, dst_rel)
        os.makedirs(os.path.dirname(dst_full), exist_ok=True)
        cv2.imwrite(dst_full, scaled_img)

    return rel_path, True


def main():
    print(f"[*] Base dataset root: {DATA_ROOT}")

    # 1. Collect all unique degraded image relative paths
    all_degraded_images = set()
    test_file_lines = {}

    for fn in TEST_FILES:
        fp = os.path.join(DATA_ROOT, fn)
        if not os.path.exists(fp):
            print(f"[!] Warning: File {fp} does not exist!")
            continue

        with open(fp, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        test_file_lines[fn] = lines

        for line in lines:
            parts = line.split("\t")
            rel_path = parts[0]
            if "degraded" in rel_path.lower():
                all_degraded_images.add(rel_path)

    print(f"[*] Found {len(all_degraded_images):,} unique degraded images across all test files.")

    # 2. Generate x2 and x4 downscaled images in parallel
    print(f"[*] Generating x2 and x4 downscaled images (cv2.INTER_CUBIC)...")
    degraded_list = sorted(list(all_degraded_images))

    success_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process_image, degraded_list), total=len(degraded_list), desc="Downscaling images"))
        for _, ok in results:
            if ok:
                success_count += 1

    print(f"[+] Successfully downscaled {success_count:,}/{len(degraded_list):,} images for scales {SCALES}.")

    # 3. Generate label files
    # User requested: test_brazil_labels_sp.txt, ...
    # We will generate:
    #   - *_sp.txt (defaults to x2 degraded)
    #   - *_sp_x2.txt (pointing to degraded_x2)
    #   - *_sp_x4.txt (pointing to degraded_x4)
    print(f"[*] Generating scaled test label files (*_sp.txt, *_sp_x2.txt, *_sp_x4.txt)...")

    for fn, lines in test_file_lines.items():
        base_name = fn.replace(".txt", "")

        for scale in SCALES:
            out_filename_scale = f"{base_name}_sp_x{scale}.txt"
            out_path_scale = os.path.join(DATA_ROOT, out_filename_scale)

            scaled_lines = []
            for line in lines:
                parts = line.split("\t")
                rel_path = parts[0]
                label = parts[1] if len(parts) > 1 else ""

                if "degraded" in rel_path.lower():
                    new_rel = get_scaled_rel_path(rel_path, scale)
                else:
                    new_rel = rel_path

                scaled_lines.append(f"{new_rel}\t{label}\n")

            with open(out_path_scale, "w", encoding="utf-8") as f_out:
                f_out.writelines(scaled_lines)

            print(f"    [+] Created: {out_filename_scale} ({len(scaled_lines):,} samples)")

            # Create the default _sp.txt alias for x2
            if scale == 2:
                out_filename_default = f"{base_name}_sp.txt"
                out_path_default = os.path.join(DATA_ROOT, out_filename_default)
                with open(out_path_default, "w", encoding="utf-8") as f_out:
                    f_out.writelines(scaled_lines)
                print(f"    [+] Created: {out_filename_default} (default alias -> x2)")

    print(f"\n[✓] ALL DONE! Scaled test images and label files successfully created.")


if __name__ == "__main__":
    main()
