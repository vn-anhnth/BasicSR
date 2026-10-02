import os
import sys
import glob
import yaml
import csv
import argparse
import time
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Windows stdout encoding fix
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup paths
BASICSR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LITEALPR_ROOT = os.path.abspath(os.path.join(BASICSR_ROOT, "..", "LiteALPR"))

if BASICSR_ROOT not in sys.path:
    sys.path.insert(0, BASICSR_ROOT)
if LITEALPR_ROOT not in sys.path:
    sys.path.insert(0, LITEALPR_ROOT)

from basicsr.archs import build_network
from basicsr.archs.class_sr_classifier_arch import ClassSRClassifier
from basicsr.utils import img2tensor, tensor2img
from litealpr.pipeline import LiteALPR


def levenshtein_distance(s1, s2):
    """Compute standard Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_experiment_yml(exp_dir, rel_path):
    """Find the training YAML config for a given experiment."""
    if os.path.isdir(exp_dir):
        ymls = glob.glob(os.path.join(exp_dir, "*.yml"))
        if ymls:
            return ymls[0]

    exp_basename = os.path.basename(rel_path.strip().replace('\\', '/'))
    found = glob.glob(os.path.join(BASICSR_ROOT, "options", "train", "**", f"*{exp_basename}*.yml"), recursive=True)
    if found:
        return found[0]

    return None


def find_latest_checkpoint(exp_dir):
    """Find net_g_latest.pth or highest iteration checkpoint."""
    models_dir = os.path.join(exp_dir, "models")
    if not os.path.isdir(models_dir):
        return None

    latest_path = os.path.join(models_dir, "net_g_latest.pth")
    if os.path.exists(latest_path):
        return latest_path

    g_ckpts = glob.glob(os.path.join(models_dir, "net_g_*.pth"))
    if g_ckpts:
        def get_iter(p):
            base = os.path.splitext(os.path.basename(p))[0]
            parts = base.split('_')
            try:
                return int(parts[-1])
            except ValueError:
                return -1
        g_ckpts.sort(key=get_iter, reverse=True)
        return g_ckpts[0]

    return None


def load_sr_model(network_opt, checkpoint_path, device='cuda'):
    """Initialize and load SR model checkpoint."""
    model = build_network(network_opt).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'], strict=True)
    elif 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.eval()
    return model


def run_super_resolution(sr_model, img_bgr, scale, network_opt, device='cuda'):
    """Super-resolve an input image using the SR model."""
    h_lr, w_lr = img_bgr.shape[:2]
    num_in_ch = network_opt.get('num_in_ch', network_opt.get('in_nc', network_opt.get('in_chans', 3)))

    im_float = img_bgr.astype(np.float32) / 255.0

    if num_in_ch == 1:
        im_lq_y = cv2.cvtColor(im_float, cv2.COLOR_BGR2YCrCb)[:, :, 0:1]
        lq_tensor = img2tensor(im_lq_y, bgr2rgb=False, float32=True).unsqueeze(0).to(device)
    else:
        lq_tensor = img2tensor(im_float, bgr2rgb=True, float32=True).unsqueeze(0).to(device)

    # Handle window-based padding for Transformer SR
    net_type = network_opt.get('type', '')
    if net_type in ['SwinIR', 'DAT', 'ELAN']:
        pad_base = 16
    else:
        pad_base = scale

    mod_pad_h = (pad_base - h_lr % pad_base) % pad_base
    mod_pad_w = (pad_base - w_lr % pad_base) % pad_base
    if mod_pad_h != 0 or mod_pad_w != 0:
        lq_tensor_pad = torch.nn.functional.pad(lq_tensor, (0, mod_pad_w, 0, mod_pad_h), mode='replicate')
    else:
        lq_tensor_pad = lq_tensor

    with torch.no_grad():
        sr_tensor_pad = sr_model(lq_tensor_pad)

    sr_tensor = sr_tensor_pad[:, :, :h_lr * scale, :w_lr * scale]

    if num_in_ch == 1:
        im_sr_y = tensor2img(sr_tensor, rgb2bgr=False)
        im_lq_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        im_lq_crcb = cv2.resize(im_lq_ycrcb[:, :, 1:], (im_sr_y.shape[1], im_sr_y.shape[0]), interpolation=cv2.INTER_CUBIC)
        im_sr_ycrcb = np.dstack([im_sr_y, im_lq_crcb])
        im_sr = cv2.cvtColor(im_sr_ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        im_sr = tensor2img(sr_tensor)

    return im_sr


def evaluate_sr_ocr(
    sr_model,
    scale,
    network_opt,
    ocr_pipeline,
    test_data,
    data_root,
    quality_classifier=None,
    device='cuda'
):
    """Run SR + OCR and compute word Accuracy (Acc) and Character Error Rate (CER).

    If quality_classifier (ClassSR) is provided:
      - Class 0 (Clear): Passes directly to OCR (bypasses SR)
      - Class 1 (Degraded): Passes through SR before OCR
    """
    correct_words = 0
    total_words = len(test_data)
    total_edit_distance = 0
    total_chars = 0
    sr_routed_count = 0

    for rel_path, gt_label in tqdm(test_data, desc=f"Evaluating SR x{scale} + SVTRv26", leave=False):
        img_path = os.path.join(data_root, rel_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # ClassSR Routing logic
        need_sr = True
        if quality_classifier is not None:
            # Preprocess crop to (32, 96) for Classifier
            inp_cls = cv2.resize(img_bgr, (96, 32), interpolation=cv2.INTER_LINEAR)
            inp_cls = cv2.cvtColor(inp_cls, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t_cls = torch.from_numpy(inp_cls).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_cls = quality_classifier(t_cls)
                class_id = torch.argmax(pred_cls, dim=1).item()
            # 0: Clear, 1: Degraded
            need_sr = (class_id == 1)

        # Apply SR only if needed
        if sr_model is not None and need_sr:
            img_to_ocr = run_super_resolution(sr_model, img_bgr, scale, network_opt, device=device)
            sr_routed_count += 1
        else:
            img_to_ocr = img_bgr

        pred_text, _ = ocr_pipeline.recognize(img_to_ocr)

        # Normalize strings: strip whitespace and uppercase
        pred_clean = str(pred_text).strip().upper()
        gt_clean = str(gt_label).strip().upper()

        if pred_clean == gt_clean:
            correct_words += 1

        dist = levenshtein_distance(gt_clean, pred_clean)
        total_edit_distance += dist
        total_chars += max(len(gt_clean), 1)

    acc = (correct_words / total_words) * 100.0 if total_words > 0 else 0.0
    cer = (total_edit_distance / total_chars) if total_chars > 0 else 0.0
    if quality_classifier is not None:
        print(f"    [ClassSR Routing] Degraded routed to SR: {sr_routed_count:,}/{total_words:,} ({sr_routed_count/total_words*100:.1f}%) | Clear bypassed: {total_words - sr_routed_count:,}")
    return acc, cer


def load_test_labels(label_path, scale=None, max_samples=None):
    """
    Load test pairs (rel_path, label).
    If scale == 1: uses the original unscaled test files (e.g. test_labels.txt) directly!
    If scale in [2, 4]: automatically looks for scale-specific files (e.g. test_labels_sp_x2.txt).
    """
    actual_path = label_path
    if scale is not None and scale != 1 and os.path.exists(label_path):
        dir_name = os.path.dirname(label_path)
        base_name = os.path.basename(label_path)

        # Check if user passed an unscaled base file
        candidates = []
        if base_name.endswith(".txt"):
            stem = base_name[:-4]
            # If not already explicit scale
            if not stem.endswith(f"_x{scale}"):
                # e.g. test_labels_sp_x2.txt or test_brazil_labels_sp_x2.txt
                candidates.append(os.path.join(dir_name, f"{stem}_sp_x{scale}.txt"))
                candidates.append(os.path.join(dir_name, f"{stem}_x{scale}.txt"))
                if "_sp" not in stem:
                    candidates.append(os.path.join(dir_name, f"{stem}_sp.txt"))

        for cand in candidates:
            if os.path.exists(cand):
                actual_path = cand
                break

    data = []
    if not os.path.exists(actual_path):
        print(f"[!] Warning: Test label file not found: {actual_path}")
        return data, actual_path

    with open(actual_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                data.append((parts[0], parts[1]))

    if max_samples and max_samples < len(data):
        data = data[:max_samples]

    return data, actual_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate SR + SVTRv26 Recognition Accuracy and CER")
    parser.add_argument("--csv", type=str, default=os.path.join(BASICSR_ROOT, "options", "tonghop_x1.csv"), help="Path to tonghop_x1.csv")
    parser.add_argument("--test_labels", type=str, default=r"D:\IEEE\data\phD\data\50k_OCR\aaa_train_ne_version_5\test_labels.txt", help="Path to test_labels.txt")
    parser.add_argument("--countries", nargs="+", type=str, default=["all", "brazil", "china", "vn"], help="List of countries to evaluate (e.g. all brazil china vn)")
    parser.add_argument("--rec_weights", type=str, default=r"D:\IEEE\data\phD\LiteALPR\pretrained_models\rec\svtr26_tiny\best.pth", help="SVTR26 weight path")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test samples per country (e.g. 1000 for quick test)")
    parser.add_argument("--target_model", type=str, default=None, help="Target specific model name in tonghop_x1.csv (e.g. IMDN, CFSR, LPAWSRN-M)")
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 4], help="Scales to evaluate (e.g. 1 2 4)")
    parser.add_argument("--scale", type=int, default=None, choices=[1, 2, 4], help="Chỉ định chạy duy nhất 1 scale (ví dụ: --scale 1, --scale 2 hoặc --scale 4)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--skip_existing", action="store_true", help="Skip models/scales/countries that already have acc and cer values in CSV")
    parser.add_argument("--classifier", type=str, default=os.path.join(BASICSR_ROOT, "experiments", "pretrained_models", "class_sr_classifier_best.pth"), help="Path to ClassSR quality classifier weights")
    args = parser.parse_args()

    # Nếu truyền --scale (ví dụ: --scale 2 hoặc --scale 4) thì ghi đè danh sách args.scales
    if args.scale is not None:
        args.scales = [args.scale]

    # Pre-map available countries to base label files
    label_root = os.path.dirname(os.path.abspath(args.test_labels))
    country_map = {
        "all": os.path.join(label_root, "test_labels.txt"),
        # "brazil": os.path.join(label_root, "test_brazil_labels.txt"),
        # "china": os.path.join(label_root, "test_china_labels.txt"),
        # "vn": os.path.join(label_root, "test_vn_labels.txt")
    }

    # Filter countries based on user input
    selected_countries = [c.lower() for c in args.countries if c.lower() in country_map]
    if not selected_countries:
        selected_countries = ["all"]

    data_root = label_root
    print(f"[*] Base data directory: {data_root}")
    print(f"[*] Selected countries to evaluate: {selected_countries}")

    # 2. Initialize LiteALPR Recognition Pipeline
    print(f"[*] Initializing LiteALPR SVTRv26-tiny on {args.device}...")
    ocr_pipeline = LiteALPR(
        use_det=False,
        use_rec=True,
        rec_model_path=args.rec_weights,
        device=args.device
    )

    # 3. Initialize ClassSR Quality Classifier (if exists)
    quality_classifier = None
    if args.classifier and os.path.exists(args.classifier):
        print(f"[*] Loading ClassSR Quality Classifier from: {args.classifier}")
        quality_classifier = ClassSRClassifier(in_nc=3, num_classes=2).to(args.device)
        ckpt_cls = torch.load(args.classifier, map_location=args.device)
        if 'state_dict' in ckpt_cls:
            quality_classifier.load_state_dict(ckpt_cls['state_dict'])
        else:
            quality_classifier.load_state_dict(ckpt_cls)
        quality_classifier.eval()
        print(f"[+] ClassSR Quality Classifier loaded successfully!")
    else:
        print(f"[i] No ClassSR Classifier found at {args.classifier}. Proceeding with standard evaluation.")

    # 4. Read CSV
    if not os.path.exists(args.csv):
        print(f"[!] Error: CSV not found at {args.csv}")
        return

    with open(args.csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    col_idx = {name.strip(): i for i, name in enumerate(header)}
    print(f"[*] tonghop_x1.csv columns detected: {len(col_idx)} columns")

    # Process each model row
    updated_count = 0
    for r_idx in range(1, len(rows)):
        row = rows[r_idx]
        if not row or not any(row):
            continue

        model_name = row[col_idx['Model']].strip()
        if not model_name:
            continue

        if args.target_model and model_name.lower() != args.target_model.lower():
            continue

        print(f"\n======================================================================")
        print(f"[*] Processing Model: {model_name}")
        print(f"======================================================================")

        for scale in args.scales:
            where_col = f"Where (x{scale})"
            if where_col not in col_idx:
                continue

            exp_rel = row[col_idx[where_col]].strip()
            if not exp_rel:
                continue

            # Check if all selected countries are already evaluated for this scale
            if args.skip_existing:
                all_done = True
                for c in selected_countries:
                    c_acc_col = None
                    c_cer_col = None
                    for cand in [f"acc {c} (x{scale})", f"acc {c}"]:
                        if cand in col_idx:
                            c_acc_col = cand
                            break
                    for cand in [f"cer {c} (x{scale})", f"cer {c}"]:
                        if cand in col_idx:
                            c_cer_col = cand
                            break
                    if c_acc_col and c_cer_col:
                        if not (row[col_idx[c_acc_col]].strip() and row[col_idx[c_cer_col]].strip()):
                            all_done = False
                            break
                if all_done:
                    print(f"[i] Scale x{scale} already evaluated for all selected countries. Skipping...")
                    continue

            # Locate experiment path
            candidates = [
                os.path.join(BASICSR_ROOT, exp_rel),
                os.path.join(r"F:\license_plate_dataset\license_plate_sr", exp_rel),
                exp_rel
            ]
            exp_dir = None
            for cand in candidates:
                if os.path.exists(cand):
                    exp_dir = cand
                    break

            if not exp_dir:
                print(f"[!] Warning: Experiment folder not found for {model_name} (x{scale}): {exp_rel}")
                continue

            yml_path = find_experiment_yml(exp_dir, exp_rel)
            ckpt_path = find_latest_checkpoint(exp_dir)

            if not yml_path or not ckpt_path:
                print(f"[!] Warning: Missing yml ({yml_path}) or ckpt ({ckpt_path}) for {model_name} (x{scale})")
                continue

            print(f"[*] Scale x{scale}: Loading YAML -> {yml_path}")
            print(f"[*] Scale x{scale}: Loading Checkpoint -> {ckpt_path}")

            with open(yml_path, 'r', encoding='utf-8') as yf:
                opt = yaml.safe_load(yf)

            network_opt = opt.get('network_g', {})
            try:
                sr_model = load_sr_model(network_opt, ckpt_path, device=args.device)
            except Exception as e:
                print(f"[!] Error loading SR model {model_name}: {e}")
                continue

            # Evaluate each selected country
            for country in selected_countries:
                # Tìm cột acc và cer tương ứng (hỗ trợ cả tonghop.csv: 'acc all (x1)' lẫn tonghop_x1.csv: 'acc all')
                cand_acc = [f"acc {country} (x{scale})", f"acc {country}"]
                cand_cer = [f"cer {country} (x{scale})", f"cer {country}"]

                acc_col = None
                for c in cand_acc:
                    if c in col_idx:
                        acc_col = c
                        break

                cer_col = None
                for c in cand_cer:
                    if c in col_idx:
                        cer_col = c
                        break

                if not acc_col or not cer_col:
                    continue

                if args.skip_existing and row[col_idx[acc_col]].strip() and row[col_idx[cer_col]].strip():
                    print(f"[i] Scale x{scale} ({country.upper()}) already evaluated: Acc={row[col_idx[acc_col]]}%, CER={row[col_idx[cer_col]]}. Skipping...")
                    continue

                country_base_file = country_map[country]
                scale_test_data, actual_label_path = load_test_labels(country_base_file, scale=scale, max_samples=args.max_samples)
                print(f"[*] Scale x{scale} [{country.upper()}]: Using labels -> {os.path.basename(actual_label_path)} ({len(scale_test_data):,} samples)")

                t0 = time.time()
                acc, cer = evaluate_sr_ocr(
                    sr_model=sr_model,
                    scale=scale,
                    network_opt=network_opt,
                    ocr_pipeline=ocr_pipeline,
                    test_data=scale_test_data,
                    data_root=data_root,
                    quality_classifier=quality_classifier,
                    device=args.device
                )
                elapsed = time.time() - t0

                print(f"[✓] {model_name} (x{scale}) [{country.upper()}] -> Acc: {acc:.2f}%, CER: {cer:.4f} (Time: {elapsed:.1f}s)")

                # Update row
                row[col_idx[acc_col]] = f"{acc:.2f}"
                row[col_idx[cer_col]] = f"{cer:.4f}"
                updated_count += 1

                # Save CSV incrementally after each evaluation
                with open(args.csv, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                print(f"[+] Successfully updated {args.csv}")

            # Free SR model memory
            del sr_model
            torch.cuda.empty_cache()

    print(f"\n[DONE] Completed evaluation. Updated {updated_count} entries in {args.csv}.")


if __name__ == "__main__":
    main()


# 1. Chạy đo toàn bộ các model trong tonghop_x1.csv (bỏ qua những model đã có kết quả):
# powershell
# & "d:\IEEE\data\phD\OpenOCR\env\Scripts\python.exe" "d:\IEEE\data\phD\BasicSR\scripts\test_benchmark_ocr.py" --skip_existing
# & "d:\IEEE\data\phD\OpenOCR\env\Scripts\python.exe" "d:\IEEE\data\phD\BasicSR\scripts\test_benchmark_ocr.py" --scale 2

# 2. Chạy đo thử nghiệm cho một model cụ thể (Ví dụ: IMDN, CFSR, LPIMDN, AWSRN):
# powershell
# # Chạy cả scale x2 và x4 cho model IMDN:
# & "d:\IEEE\data\phD\OpenOCR\env\Scripts\python.exe" "d:\IEEE\data\phD\BasicSR\scripts\test_benchmark_ocr.py" --target_model IMDN
# # Chỉ chạy scale x2 cho IMDN:
# & "d:\IEEE\data\phD\OpenOCR\env\Scripts\python.exe" "d:\IEEE\data\phD\BasicSR\scripts\test_benchmark_ocr.py" --target_model IMDN --scales 2

# 3. Chạy test thử số lượng mẫu nhỏ trước (Ví dụ: 1000 ảnh):
# powershell
# & "d:\IEEE\data\phD\OpenOCR\env\Scripts\python.exe" "d:\IEEE\data\phD\BasicSR\scripts\test_benchmark_ocr.py" --target_model IMDN --max_samples 1000


