import os
import sys
import cv2
import lmdb
import torch
import glob
import yaml
import csv
import argparse
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import normalize

# BasicSR modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from basicsr.archs import build_network
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from basicsr.utils import img2tensor, tensor2img

try:
    import lpips
    has_lpips = True
except ImportError:
    has_lpips = False

# Console encoding Windows fix
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_experiment_yml(exp_dir, rel_path):
    """Find the training YAML config for a given experiment."""
    if os.path.isdir(exp_dir):
        ymls = glob.glob(os.path.join(exp_dir, "*.yml"))
        if ymls:
            return ymls[0]

    exp_basename = os.path.basename(rel_path.strip().replace('\\', '/'))
    found = glob.glob(os.path.join("options", "train", "**", f"*{exp_basename}*.yml"), recursive=True)
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


def evaluate_single_model(
    model_name,
    network_opt,
    checkpoint_path,
    lq_lmdb_path,
    gt_lmdb_path,
    scale=2,
    device='cuda',
    max_samples=None,
    save_sr_dir=None,
    loss_fn_vgg=None
):
    print("\n" + "=" * 75)
    print(f"[*] DANH GIA MO HINH: {model_name} (Scale x{scale})")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    LQ LMDB:    {lq_lmdb_path}")
    print(f"    GT LMDB:    {gt_lmdb_path}")
    print("=" * 75)

    if not os.path.exists(checkpoint_path):
        print(f"[!] ERROR: Khong tim thay checkpoint: {checkpoint_path}")
        return None

    # 1. Khoi tao kien truc mang tu network_opt
    try:
        model = build_network(network_opt).to(device)
    except Exception as e:
        print(f"[!] ERROR: Khong the khoi tao network_opt: {e}")
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'params_ema' in checkpoint:
            model.load_state_dict(checkpoint['params_ema'], strict=True)
            print("[+] Da load params_ema thanh cong.")
        elif 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'], strict=True)
            print("[+] Da load params thanh cong.")
        else:
            model.load_state_dict(checkpoint, strict=True)
            print("[+] Da load state_dict thanh cong.")
    except Exception as e:
        print(f"[!] ERROR khi load checkpoint: {e}")
        return None

    model.eval()

    # 2. Mo du lieu LMDB
    env_lq = lmdb.open(lq_lmdb_path, readonly=True, lock=False)
    env_gt = lmdb.open(gt_lmdb_path, readonly=True, lock=False)

    txn_lq = env_lq.begin()
    txn_gt = env_gt.begin()

    total_entries = txn_lq.stat()['entries']
    num_eval = min(total_entries, max_samples) if max_samples else total_entries
    print(f"[*] Tong so mau danh gia: {num_eval:,} / {total_entries:,}")

    if save_sr_dir:
        os.makedirs(save_sr_dir, exist_ok=True)

    psnr_list = []
    ssim_list = []
    lpips_list = []

    mean_lpips = [0.5, 0.5, 0.5]
    std_lpips = [0.5, 0.5, 0.5]

    with torch.no_grad():
        for idx in tqdm(range(num_eval), desc=f"Evaluating {model_name} x{scale}"):
            key = f"{idx:06d}".encode('ascii')

            lq_buf = txn_lq.get(key)
            gt_buf = txn_gt.get(key)

            if lq_buf is None or gt_buf is None:
                continue

            # Decode anh va chuan hoa ve [0, 1] float32 theo chuan BasicSR
            im_lq = cv2.imdecode(np.frombuffer(lq_buf, np.uint8), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            im_gt = cv2.imdecode(np.frombuffer(gt_buf, np.uint8), cv2.IMREAD_COLOR).astype(np.float32) / 255.0

            # Dam bao GT khop chuan boi so scale
            h_lq, w_lq = im_lq.shape[:2]
            im_gt = im_gt[:h_lq * scale, :w_lq * scale, :]

            # Xu ly kenh dau vao (RGB 3 kenh hoac Grayscale/Y 1 kenh nhu ECBSR Y)
            num_in_ch = network_opt.get('num_in_ch', network_opt.get('in_nc', network_opt.get('in_chans', 3)))
            if num_in_ch == 1:
                im_lq_y = cv2.cvtColor(im_lq, cv2.COLOR_BGR2YCrCb)[:, :, 0:1]
                lq_tensor = img2tensor(im_lq_y, bgr2rgb=False, float32=True).unsqueeze(0).to(device)
            else:
                lq_tensor = img2tensor(im_lq, bgr2rgb=True, float32=True).unsqueeze(0).to(device)

            # Xu ly padding neu can:
            # Cac kien truc Transformer (SwinIR, DAT, ELAN) yeu cau kich thuoc anh LR la boi so cua Window Size (toi da 16)
            net_type = network_opt.get('type', '')
            if net_type in ['SwinIR', 'DAT', 'ELAN']:
                pad_base = 16
            else:
                pad_base = scale

            mod_pad_h = (pad_base - h_lq % pad_base) % pad_base
            mod_pad_w = (pad_base - w_lq % pad_base) % pad_base
            if mod_pad_h != 0 or mod_pad_w != 0:
                lq_tensor_pad = torch.nn.functional.pad(lq_tensor, (0, mod_pad_w, 0, mod_pad_h), mode='replicate')
            else:
                lq_tensor_pad = lq_tensor

            sr_tensor_pad = model(lq_tensor_pad)

            # Crop ve dung kich thuoc SR goc truoc khi pad
            sr_tensor = sr_tensor_pad[:, :, :h_lq * scale, :w_lq * scale]

            # Chuyen ket qua ra uint8 numpy [0, 255]
            if num_in_ch == 1:
                im_sr_y = tensor2img(sr_tensor, rgb2bgr=False)
                im_lq_ycrcb = cv2.cvtColor((im_lq * 255.0).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
                im_lq_crcb = cv2.resize(im_lq_ycrcb[:, :, 1:], (im_sr_y.shape[1], im_sr_y.shape[0]), interpolation=cv2.INTER_CUBIC)
                im_sr_ycrcb = np.dstack([im_sr_y, im_lq_crcb])
                im_sr = cv2.cvtColor(im_sr_ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                im_sr = tensor2img(sr_tensor)

            im_gt_u8 = tensor2img(img2tensor(im_gt, bgr2rgb=True, float32=True))

            # 1. Tinh PSNR (kenh Y)
            cur_psnr = calculate_psnr(im_sr, im_gt_u8, crop_border=scale, test_y_channel=True)
            psnr_list.append(cur_psnr)

            # 2. Tinh SSIM (kenh Y)
            cur_ssim = calculate_ssim(im_sr, im_gt_u8, crop_border=scale, test_y_channel=True)
            ssim_list.append(cur_ssim)

            # 3. Tinh LPIPS
            if loss_fn_vgg is not None:
                gt_tensor = img2tensor([im_gt], bgr2rgb=True, float32=True)[0].unsqueeze(0).to(device)
                sr_t_rgb = img2tensor([im_sr], bgr2rgb=True, float32=True)[0].unsqueeze(0).to(device) / 255.0

                sr_norm = sr_t_rgb.clone()
                gt_norm = gt_tensor.clone()
                normalize(sr_norm[0], mean_lpips, std_lpips, inplace=True)
                normalize(gt_norm[0], mean_lpips, std_lpips, inplace=True)

                # VGG yeu cau kich thuoc toi thieu 32x32 de khong bi loi pooling (5 tang pooling 2x2: 32 -> 16 -> 8 -> 4 -> 2 -> 1)
                h_cur, w_cur = sr_norm.shape[2], sr_norm.shape[3]
                min_sz = 32
                if h_cur < min_sz or w_cur < min_sz:
                    pad_h = max(0, min_sz - h_cur)
                    pad_w = max(0, min_sz - w_cur)
                    sr_norm = torch.nn.functional.pad(sr_norm, (0, pad_w, 0, pad_h), mode='replicate')
                    gt_norm = torch.nn.functional.pad(gt_norm, (0, pad_w, 0, pad_h), mode='replicate')

                cur_lpips = loss_fn_vgg(sr_norm, gt_norm).item()
                lpips_list.append(cur_lpips)

            # Luu anh mau visualization neu co yeu cau
            if save_sr_dir and idx < 100:
                cv2.imwrite(os.path.join(save_sr_dir, f"{idx:06d}_SR.png"), im_sr)
                cv2.imwrite(os.path.join(save_sr_dir, f"{idx:06d}_GT.png"), (im_gt * 255.0).clip(0, 255).astype(np.uint8))
                cv2.imwrite(os.path.join(save_sr_dir, f"{idx:06d}_LR.png"), (im_lq * 255.0).clip(0, 255).astype(np.uint8))

    env_lq.close()
    env_gt.close()

    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    avg_lpips = float(np.mean(lpips_list)) if lpips_list else 0.0

    print(f"[*] Ket qua {model_name} (x{scale}): PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")

    return {
        'model_name': model_name,
        'scale': scale,
        'samples': len(psnr_list),
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips
    }


def save_csv_results(csv_path, rows, fieldnames):
    """Safely write rows to CSV."""
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PSNR, SSIM, LPIPS and update tonghop.csv")
    parser.add_argument("--csv", type=str, default="options/tonghop.csv", help="Path to tonghop.csv")
    parser.add_argument("--experiments_root", type=str, default=r"F:\license_plate_dataset\license_plate_sr\experiments",
                        help="Root folder of trained experiments")
    parser.add_argument("--lmdb_root", type=str, default=r"F:\license_plate_dataset\license_plate_sr\lmdb",
                        help="Root folder of LMDB datasets")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"],
                        help="Dataset split to evaluate: 'test' (test_HR.lmdb) or 'val' (val_HR.lmdb)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of test images to evaluate (e.g. 1000 for fast test, None for full test set)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to evaluate on ('cuda' or 'cpu')")
    parser.add_argument("--models", nargs="*", default=None,
                        help="List of model names to run benchmark on (e.g. --models ECBSR EDSR-M). If not specified, runs all.")
    parser.add_argument("--skip_existing", action="store_true", default=False,
                        help="Skip models that already have PSNR/SSIM filled in CSV")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'

    # Khoi tao LPIPS VGG
    loss_fn_vgg = None
    if has_lpips:
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        print("[+] Da khoi tao mang LPIPS (VGG) tren:", device)
    else:
        print("[!] Warning: chua cai dat thu vien 'lpips', chi so LPIPS se de trong.")

    # Xac dinh cac duong dan LMDB
    gt_lmdb = os.path.join(args.lmdb_root, f"{args.split}_HR.lmdb")
    lq_lmdb_x2 = os.path.join(args.lmdb_root, f"{args.split}_LR_x2.lmdb")
    lq_lmdb_x4 = os.path.join(args.lmdb_root, f"{args.split}_LR_x4.lmdb")

    if not os.path.exists(gt_lmdb):
        print(f"[!] ERROR: Khong tim thay GT LMDB: {gt_lmdb}")
        return

    # Doc tonghop.csv
    rows = []
    fieldnames = []
    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            if r.get('Model') and r.get('Model').strip():
                rows.append(r)

    # Chuan hoa cac cot metric can dien
    # Chu y: Trong CSV ban dau co the co dau space o 'ssim (x4) '
    psnr_col_x2 = 'psnr (x2)'
    psnr_col_x4 = 'psnr (x4)'
    ssim_col_x2 = 'ssim (x2)'
    ssim_col_x4 = 'ssim (x4) ' if 'ssim (x4) ' in fieldnames else 'ssim (x4)'
    lpips_col_x2 = 'lpips (x2)'
    lpips_col_x4 = 'lpips (x4)'

    print("=" * 80)
    print(f"[*] BENCHMARK PSNR, SSIM, LPIPS VA CAP NHAT TONGHOP.CSV")
    print(f"    CSV Path:          {args.csv}")
    print(f"    Split:             {args.split} ({gt_lmdb})")
    print(f"    Max Samples:       {args.max_samples if args.max_samples else 'ALL'}")
    print(f"    Device:            {device}")
    print(f"    Total Models:      {len(rows)}")
    print("=" * 80)

    for idx, row in enumerate(rows, 1):
        m_name = row['Model'].strip()
        if args.models and m_name not in args.models:
            continue

        w2 = row.get('Where (x2)', '').strip()
        w4 = row.get('Where (x4)', '').strip()

        print(f"\n[{idx:02d}/{len(rows):02d}] Model: {m_name}")

        # ----------------- SCALE X2 -----------------
        if w2:
            already_x2 = row.get(psnr_col_x2) and str(row.get(psnr_col_x2)).strip()
            if args.skip_existing and already_x2:
                print(f"  -> Bo qua x2 vi da co ket qua: {row.get(psnr_col_x2)}")
            else:
                exp_dir_x2 = os.path.join(args.experiments_root, os.path.basename(w2.replace('\\', '/')))
                yml_x2 = find_experiment_yml(exp_dir_x2, w2)
                ckpt_x2 = find_latest_checkpoint(exp_dir_x2)

                if yml_x2 and ckpt_x2:
                    with open(yml_x2, 'r', encoding='utf-8') as yf:
                        cfg_x2 = yaml.safe_load(yf)
                    res_x2 = evaluate_single_model(
                        model_name=f"{m_name}_x2",
                        network_opt=cfg_x2['network_g'],
                        checkpoint_path=ckpt_x2,
                        lq_lmdb_path=lq_lmdb_x2,
                        gt_lmdb_path=gt_lmdb,
                        scale=2,
                        device=device,
                        max_samples=args.max_samples,
                        loss_fn_vgg=loss_fn_vgg
                    )
                    if res_x2:
                        row[psnr_col_x2] = f"{res_x2['psnr']:.2f}"
                        row[ssim_col_x2] = f"{res_x2['ssim']:.4f}"
                        row[lpips_col_x2] = f"{res_x2['lpips']:.4f}"
                        save_csv_results(args.csv, rows, fieldnames)
                else:
                    print(f"  [!] Thieu yml hoac checkpoint x2 cho {m_name}")

        # ----------------- SCALE X4 -----------------
        if w4:
            already_x4 = row.get(psnr_col_x4) and str(row.get(psnr_col_x4)).strip()
            if args.skip_existing and already_x4:
                print(f"  -> Bo qua x4 vi da co ket qua: {row.get(psnr_col_x4)}")
            else:
                exp_dir_x4 = os.path.join(args.experiments_root, os.path.basename(w4.replace('\\', '/')))
                yml_x4 = find_experiment_yml(exp_dir_x4, w4)
                ckpt_x4 = find_latest_checkpoint(exp_dir_x4)

                if yml_x4 and ckpt_x4:
                    with open(yml_x4, 'r', encoding='utf-8') as yf:
                        cfg_x4 = yaml.safe_load(yf)
                    res_x4 = evaluate_single_model(
                        model_name=f"{m_name}_x4",
                        network_opt=cfg_x4['network_g'],
                        checkpoint_path=ckpt_x4,
                        lq_lmdb_path=lq_lmdb_x4,
                        gt_lmdb_path=gt_lmdb,
                        scale=4,
                        device=device,
                        max_samples=args.max_samples,
                        loss_fn_vgg=loss_fn_vgg
                    )
                    if res_x4:
                        row[psnr_col_x4] = f"{res_x4['psnr']:.2f}"
                        row[ssim_col_x4] = f"{res_x4['ssim']:.4f}"
                        row[lpips_col_x4] = f"{res_x4['lpips']:.4f}"
                        save_csv_results(args.csv, rows, fieldnames)
                else:
                    print(f"  [!] Thieu yml hoac checkpoint x4 cho {m_name}")

    print("\n" + "=" * 80)
    print(f"[+] HOAN TAT BENCHMARK VA CAP NHAT TOAN BO VAO: {args.csv}")
    print("=" * 80)


if __name__ == '__main__':
    main()

# command
# .venv_38_gpu\Scripts\python.exe scripts/test_benchmark_lpips.py --split test --skip_existing
