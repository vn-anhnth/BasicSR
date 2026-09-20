import os
import sys
import cv2
import lmdb
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import normalize

# BasicSR modules
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


def evaluate_single_model(
    model_name,
    network_opt,
    checkpoint_path,
    lq_lmdb_path,
    gt_lmdb_path,
    scale=2,
    device='cuda',
    max_samples=None,  # Để None nếu muốn test toàn bộ 24,487 ảnh
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
    model = build_network(network_opt).to(device)

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
        for idx in tqdm(range(num_eval), desc=f"Evaluating {model_name}"):
            key = f"{idx:06d}".encode('ascii')

            lq_buf = txn_lq.get(key)
            gt_buf = txn_gt.get(key)

            if lq_buf is None or gt_buf is None:
                continue

            # Decode ảnh và chuẩn hóa về [0, 1] float32 theo chuẩn của BasicSR
            im_lq = cv2.imdecode(np.frombuffer(lq_buf, np.uint8), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            im_gt = cv2.imdecode(np.frombuffer(gt_buf, np.uint8), cv2.IMREAD_COLOR).astype(np.float32) / 255.0

            # Đảm bảo GT khớp chuẩn bội số scale
            h_lq, w_lq = im_lq.shape[:2]
            im_gt = im_gt[:h_lq * scale, :w_lq * scale, :]

            # Xử lý kênh đầu vào (RGB 3 kênh hoặc Grayscale/Y 1 kênh như ECBSR Y)
            num_in_ch = network_opt.get('num_in_ch', 3)
            if num_in_ch == 1:
                # Chuyển BGR sang Y channel
                im_lq_y = cv2.cvtColor(im_lq, cv2.COLOR_BGR2YCrCb)[:, :, 0:1]
                lq_tensor = img2tensor(im_lq_y, bgr2rgb=False, float32=True).unsqueeze(0).to(device)
            else:
                lq_tensor = img2tensor(im_lq, bgr2rgb=True, float32=True).unsqueeze(0).to(device)

            sr_tensor = model(lq_tensor)

            # Chuyển kết quả ra uint8 numpy [0, 255]
            if num_in_ch == 1:
                # Tái tạo lại RGB/BGR từ Y của SR và CrCb của Bicubic LQ
                im_sr_y = tensor2img(sr_tensor, rgb2bgr=False)
                im_lq_ycrcb = cv2.cvtColor((im_lq * 255.0).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
                im_lq_crcb = cv2.resize(im_lq_ycrcb[:, :, 1:], (im_sr_y.shape[1], im_sr_y.shape[0]), interpolation=cv2.INTER_CUBIC)
                im_sr_ycrcb = np.dstack([im_sr_y, im_lq_crcb])
                im_sr = cv2.cvtColor(im_sr_ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                im_sr = tensor2img(sr_tensor)

            im_gt_u8 = tensor2img(img2tensor(im_gt, bgr2rgb=True, float32=True))

            # 1. Tính PSNR
            cur_psnr = calculate_psnr(im_sr, im_gt_u8, crop_border=scale, test_y_channel=False)
            psnr_list.append(cur_psnr)

            # 2. Tính SSIM
            cur_ssim = calculate_ssim(im_sr, im_gt_u8, crop_border=scale, test_y_channel=False)
            ssim_list.append(cur_ssim)

            # 3. Tính LPIPS
            if loss_fn_vgg is not None:
                gt_tensor = img2tensor([im_gt], bgr2rgb=True, float32=True)[0].unsqueeze(0).to(device)
                sr_t_rgb = img2tensor([im_sr], bgr2rgb=True, float32=True)[0].unsqueeze(0).to(device) / 255.0

                sr_norm = sr_t_rgb.clone()
                gt_norm = gt_tensor.clone()
                normalize(sr_norm[0], mean_lpips, std_lpips, inplace=True)
                normalize(gt_norm[0], mean_lpips, std_lpips, inplace=True)

                cur_lpips = loss_fn_vgg(sr_norm, gt_norm).item()
                lpips_list.append(cur_lpips)

            # Lưu ảnh kết quả nếu cần
            if save_sr_dir and idx < 100:
                cv2.imwrite(os.path.join(save_sr_dir, f"{idx:06d}_SR.png"), im_sr)
                cv2.imwrite(os.path.join(save_sr_dir, f"{idx:06d}_GT.png"), (im_gt * 255.0).clip(0, 255).astype(np.uint8))
                cv2.imwrite(os.path.join(save_sr_dir, f"{idx:06d}_LR.png"), (im_lq * 255.0).clip(0, 255).astype(np.uint8))

    env_lq.close()
    env_gt.close()

    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))
    avg_lpips = float(np.mean(lpips_list)) if lpips_list else 0.0

    return {
        'model_name': model_name,
        'scale': scale,
        'samples': len(psnr_list),
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Khởi tạo mô hình LPIPS dùng chung
    loss_fn_vgg = None
    if has_lpips:
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        print("[+] Da khoi tao mang LPIPS (VGG) tren GPU.")

    # Đường dẫn gốc LMDB tập Test trên ổ F
    lmdb_root = r"F:\license_plate_dataset\license_plate_sr\lmdb"
    gt_lmdb = os.path.join(lmdb_root, "test_HR.lmdb")

    # =========================================================================
    # DANH SACH CAC MO HINH CAN TEST BENCHMARK
    # Bạn chỉ cần thêm / bớt các model vào danh sách này
    # =========================================================================
    model_list = [
        # 1. EDSR-L x2
        {
            'name': 'EDSR_Lx2',
            'scale': 2,
            'lq_lmdb': os.path.join(lmdb_root, 'test_LR_x2.lmdb'),
            'checkpoint': r"D:\IEEE\data\phD\BasicSR\experiments\204_EDSR_Lx2_LicensePlate_pretrained\models\net_g_60000.pth",
            'save_dir': r"D:\IEEE\data\phD\BasicSR\results\EDSR_Lx2_Visuals",
            'network': {
                'type': 'EDSR',
                'num_in_ch': 3,
                'num_out_ch': 3,
                'num_feat': 256,
                'num_block': 32,
                'upscale': 2,
                'res_scale': 0.1,
                'img_range': 255.0,
                'rgb_mean': [0.4488, 0.4371, 0.4040]
            }
        },
        # 2. EDSR-M x2 (Ví dụ khi bạn train xong)
        # {
        #     'name': 'EDSR_Mx2',
        #     'scale': 2,
        #     'lq_lmdb': os.path.join(lmdb_root, 'test_LR_x2.lmdb'),
        #     'checkpoint': r"D:\IEEE\data\phD\BasicSR\experiments\201_EDSR_Mx2_LicensePlate_pretrained\models\net_g_40000.pth",
        #     'save_dir': r"D:\IEEE\data\phD\BasicSR\results\EDSR_Mx2_Visuals",
        #     'network': {
        #         'type': 'EDSR',
        #         'num_in_ch': 3,
        #         'num_out_ch': 3,
        #         'num_feat': 64,
        #         'num_block': 16,
        #         'upscale': 2,
        #         'res_scale': 1.0,
        #         'img_range': 255.0,
        #         'rgb_mean': [0.4488, 0.4371, 0.4040]
        #     }
        # },
        # 3. ECBSR x2 (Ví dụ khi bạn train xong)
        # {
        #     'name': 'ECBSR_x2_m4c16',
        #     'scale': 2,
        #     'lq_lmdb': os.path.join(lmdb_root, 'test_LR_x2.lmdb'),
        #     'checkpoint': r"D:\IEEE\data\phD\BasicSR\experiments\101_train_ECBSR_x2_m4c16_prelu\models\net_g_20000.pth",
        #     'save_dir': r"D:\IEEE\data\phD\BasicSR\results\ECBSR_x2_Visuals",
        #     'network': {
        #         'type': 'ECBSR',
        #         'num_in_ch': 1,
        #         'num_out_ch': 1,
        #         'num_block': 4,
        #         'num_channel': 16,
        #         'with_idt': False,
        #         'act_type': 'prelu',
        #         'scale': 2
        #     }
        # },
        # 4. EDSR-L x4 (Ví dụ khi bạn train xong)
        # {
        #     'name': 'EDSR_Lx4',
        #     'scale': 4,
        #     'lq_lmdb': os.path.join(lmdb_root, 'test_LR_x4.lmdb'),
        #     'checkpoint': r"D:\IEEE\data\phD\BasicSR\experiments\206_EDSR_Lx4_LicensePlate_pretrained\models\net_g_76000.pth",
        #     'save_dir': r"D:\IEEE\data\phD\BasicSR\results\EDSR_Lx4_Visuals",
        #     'network': {
        #         'type': 'EDSR',
        #         'num_in_ch': 3,
        #         'num_out_ch': 3,
        #         'num_feat': 256,
        #         'num_block': 32,
        #         'upscale': 4,
        #         'res_scale': 0.1,
        #         'img_range': 255.0,
        #         'rgb_mean': [0.4488, 0.4371, 0.4040]
        #     }
        # },
    ]

    # Số lượng mẫu test: Để None để test toàn bộ 24,487 ảnh, hoặc số cụ thể (ví dụ 1000)
    MAX_SAMPLES = None

    results = []
    for item in model_list:
        res = evaluate_single_model(
            model_name=item['name'],
            network_opt=item['network'],
            checkpoint_path=item['checkpoint'],
            lq_lmdb_path=item['lq_lmdb'],
            gt_lmdb_path=gt_lmdb,
            scale=item['scale'],
            device=device,
            max_samples=MAX_SAMPLES,
            save_sr_dir=item.get('save_dir'),
            loss_fn_vgg=loss_fn_vgg
        )
        if res is not None:
            results.append(res)

    # In Bảng tổng hợp Benchmark so sánh đẹp mắt
    print("\n" + "=" * 80)
    print("                 BANG TONG HOP BENCHMARK TREN TAP TEST")
    print("=" * 80)
    print(f"{'Model Name':<25} | {'Scale':<6} | {'Samples':<8} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<8} | {'LPIPS ↓':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['model_name']:<25} | x{r['scale']:<5} | {r['samples']:<8} | {r['psnr']:<12.4f} | {r['ssim']:<8.4f} | {r['lpips']:<8.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
