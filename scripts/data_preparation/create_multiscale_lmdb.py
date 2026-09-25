import os
import sys
import cv2
import lmdb
from tqdm import tqdm

# Fix encoding console Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def make_lmdb_multiscale(sr_base_dir, split_name, meta_file, output_root, scales=[2, 4], save_original_lr=True):
    """
    Tạo trọn bộ các gói LMDB:
      - split_name_HR.lmdb: Lưu ảnh gốc sắc nét HR (Ground Truth)
      - split_name_LR.lmdb: Lưu ảnh gốc mờ LR (kích thước nguyên bản 1:1)
      - split_name_LR_x2.lmdb: Lưu ảnh LR thu nhỏ 2x (w//2, h//2)
      - split_name_LR_x4.lmdb: Lưu ảnh LR thu nhỏ 4x (w//4, h//4)
    Đảm bảo đầy đủ 100% cho mọi bài toán: Deblur/Restoration 1:1 và Super-Resolution x2, x4.
    """
    print(f"\n==================================================================")
    print(f"[*] TAO LMDB CHO TAP: {split_name.upper()} (Scales: {scales}, Original LR: {save_original_lr})")
    print(f"==================================================================")

    if not os.path.exists(meta_file):
        print(f"[!] Khong tim thay meta file: {meta_file}")
        return

    pairs = []
    root_dir = os.path.join(sr_base_dir, split_name)

    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 2:
                lq_rel, gt_rel = parts[0], parts[1]
                lq_full = os.path.join(root_dir, lq_rel)
                gt_full = os.path.join(root_dir, gt_rel)
                if os.path.exists(lq_full) and os.path.exists(gt_full):
                    pairs.append((lq_full, gt_full))

    total = len(pairs)
    print(f"Tong so cap anh hop le: {total:,}")

    os.makedirs(output_root, exist_ok=True)
    gt_lmdb_path = os.path.join(output_root, f"{split_name}_HR.lmdb")
    map_size = 1 * 1024 * 1024 * 1024  # 1GB

    # Open env HR
    env_gt = lmdb.open(gt_lmdb_path, map_size=map_size)
    txn_gt = env_gt.begin(write=True)
    meta_gt_lines = []

    # Open env original LR (1:1)
    env_orig_lq = None
    txn_orig_lq = None
    orig_lq_path = os.path.join(output_root, f"{split_name}_LR.lmdb")
    meta_orig_lq_lines = []
    if save_original_lr:
        env_orig_lq = lmdb.open(orig_lq_path, map_size=map_size)
        txn_orig_lq = env_orig_lq.begin(write=True)

    # Open envs cho tung scale LR
    env_lq = {}
    txn_lq = {}
    lq_paths = {}
    meta_lq_lines = {scale: [] for scale in scales}

    for scale in scales:
        path = os.path.join(output_root, f"{split_name}_LR_x{scale}.lmdb")
        lq_paths[scale] = path
        env = lmdb.open(path, map_size=map_size)
        env_lq[scale] = env
        txn_lq[scale] = env.begin(write=True)

    commit_interval = 2000

    for idx, (lq_path, gt_path) in enumerate(tqdm(pairs, desc=f"Processing {split_name}")):
        key_str = f"{idx:06d}"
        key_byte = key_str.encode("ascii")

        # Đọc ảnh HR và LR
        im_gt = cv2.imread(gt_path)
        im_lq = cv2.imread(lq_path)

        if im_gt is None or im_lq is None:
            continue

        h_gt, w_gt, c_gt = im_gt.shape
        h_lq, w_lq, c_lq = im_lq.shape

        # 1. Lưu HR vào HR.lmdb
        _, gt_buf = cv2.imencode('.png', im_gt)
        gt_bytes = gt_buf.tobytes()
        txn_gt.put(key_byte, gt_bytes)
        meta_gt_lines.append(f"{key_str}.png ({h_gt},{w_gt},{c_gt}) 1\n")

        # 2. Lưu LR nguyên bản vào LR.lmdb (1:1)
        if save_original_lr:
            _, orig_lr_buf = cv2.imencode('.png', im_lq)
            orig_lr_bytes = orig_lr_buf.tobytes()
            txn_orig_lq.put(key_byte, orig_lr_bytes)
            meta_orig_lq_lines.append(f"{key_str}.png ({h_lq},{w_lq},{c_lq}) 1\n")

        # 3. Sinh ảnh LR thu nhỏ cho từng scale (x2, x4)
        for scale in scales:
            target_w_lr = max(1, w_gt // scale)
            target_h_lr = max(1, h_gt // scale)

            im_lr_scaled = cv2.resize(im_lq, (target_w_lr, target_h_lr), interpolation=cv2.INTER_CUBIC)
            h_lr_sc, w_lr_sc, c_lr_sc = im_lr_scaled.shape

            _, lr_buf = cv2.imencode('.png', im_lr_scaled)
            lr_bytes = lr_buf.tobytes()

            txn_lq[scale].put(key_byte, lr_bytes)
            meta_lq_lines[scale].append(f"{key_str}.png ({h_lr_sc},{w_lr_sc},{c_lr_sc}) 1\n")

        # Periodic commit
        if (idx + 1) % commit_interval == 0:
            txn_gt.commit()
            txn_gt = env_gt.begin(write=True)
            if save_original_lr:
                txn_orig_lq.commit()
                txn_orig_lq = env_orig_lq.begin(write=True)
            for scale in scales:
                txn_lq[scale].commit()
                txn_lq[scale] = env_lq[scale].begin(write=True)

    # Final commit & close
    txn_gt.commit()
    env_gt.close()

    if save_original_lr:
        txn_orig_lq.commit()
        env_orig_lq.close()

    for scale in scales:
        txn_lq[scale].commit()
        env_lq[scale].close()

    # Ghi file meta_info.txt vào từng LMDB directory
    with open(os.path.join(gt_lmdb_path, "meta_info.txt"), "w", encoding="utf-8") as f:
        f.writelines(meta_gt_lines)

    if save_original_lr:
        with open(os.path.join(orig_lq_path, "meta_info.txt"), "w", encoding="utf-8") as f:
            f.writelines(meta_orig_lq_lines)

    for scale in scales:
        with open(os.path.join(lq_paths[scale], "meta_info.txt"), "w", encoding="utf-8") as f:
            f.writelines(meta_lq_lines[scale])

    print(f"[OK] Da hoan thanh {split_name}:")
    print(f"     HR:       {gt_lmdb_path}")
    if save_original_lr:
        print(f"     LR (1:1): {orig_lq_path}")
    for scale in scales:
        print(f"     LR x{scale}:   {lq_paths[scale]}")


def main():
    sr_base_dir = r"D:\IEEE\data\phD\data\50k_OCR\aaa_train_ne_version_5\license_plate_sr"
    meta_dir = r"D:\IEEE\data\phD\BasicSR\basicsr\data\meta_info"
    # Đường dẫn xuất ra ổ F theo yêu cầu
    output_root = r"F:\license_plate_dataset\license_plate_sr\lmdb"

    train_meta = os.path.join(meta_dir, "meta_info_license_plate_sr_train.txt")
    val_meta = os.path.join(meta_dir, "meta_info_license_plate_sr_val.txt")
    test_meta = os.path.join(meta_dir, "meta_info_license_plate_sr_test.txt")

    # Tạo train (24,468 cặp): HR, LR, LR_x2, LR_x4
    make_lmdb_multiscale(sr_base_dir, "train", train_meta, output_root, scales=[2, 4], save_original_lr=True)

    # Tạo val (2,517 cặp): HR, LR, LR_x2, LR_x4
    make_lmdb_multiscale(sr_base_dir, "val", val_meta, output_root, scales=[2, 4], save_original_lr=True)

    # Tạo test (24,487 cặp): HR, LR, LR_x2, LR_x4
    make_lmdb_multiscale(sr_base_dir, "test", test_meta, output_root, scales=[2, 4], save_original_lr=True)

    print("\n[V] TRON BO DU LIEU LMDB (HR, LR 1:1, LR x2, LR x4) DA HOAN TAT XUAT VAO F:\\license_plate_dataset\\license_plate_sr\\lmdb !")


if __name__ == "__main__":
    main()
