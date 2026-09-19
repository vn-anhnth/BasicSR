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


def make_lmdb_pair(sr_base_dir, split_name, meta_file, output_root):
    """
    Tạo cặp LMDB:
      - output_root / split_name_LR.lmdb
      - output_root / split_name_HR.lmdb
    Mỗi ảnh dùng unique key dạng 000000, 000001,... để đảm bảo 100% khớp nhau giữa LR và HR.
    """
    print(f"\n==================================================================")
    print(f"[*] DANG TAO LMDB CHO TAP: {split_name.upper()}")
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
    lq_lmdb_path = os.path.join(output_root, f"{split_name}_LR.lmdb")
    gt_lmdb_path = os.path.join(output_root, f"{split_name}_HR.lmdb")

    # Tổng dung lượng ảnh thực tế chỉ ~150MB, đặt map_size 1GB (1 * 1024 * 1024 * 1024)
    map_size = 1 * 1024 * 1024 * 1024

    env_lq = lmdb.open(lq_lmdb_path, map_size=map_size)
    env_gt = lmdb.open(gt_lmdb_path, map_size=map_size)

    txn_lq = env_lq.begin(write=True)
    txn_gt = env_gt.begin(write=True)

    meta_lq_lines = []
    meta_gt_lines = []

    commit_interval = 2000

    for idx, (lq_path, gt_path) in enumerate(tqdm(pairs, desc=f"Writing {split_name} LMDB")):
        # Unique key 6 chu so (vd: 000001, 000002...)
        key_str = f"{idx:06d}"
        key_byte = key_str.encode("ascii")

        # Doc truc tiep raw bytes
        with open(lq_path, "rb") as f_lq:
            lq_bytes = f_lq.read()
        with open(gt_path, "rb") as f_gt:
            gt_bytes = f_gt.read()

        # Doc shape de tao meta_info.txt chuan cua BasicSR
        # (chi can doc header bang cv2 de lay h, w, c)
        im_lq = cv2.imread(lq_path)
        im_gt = cv2.imread(gt_path)

        if im_lq is None or im_gt is None:
            continue

        h_lq, w_lq, c_lq = im_lq.shape
        h_gt, w_gt, c_gt = im_gt.shape

        txn_lq.put(key_byte, lq_bytes)
        txn_gt.put(key_byte, gt_bytes)

        meta_lq_lines.append(f"{key_str}.png ({h_lq},{w_lq},{c_lq}) 1\n")
        meta_gt_lines.append(f"{key_str}.png ({h_gt},{w_gt},{c_gt}) 1\n")

        if (idx + 1) % commit_interval == 0:
            txn_lq.commit()
            txn_gt.commit()
            txn_lq = env_lq.begin(write=True)
            txn_gt = env_gt.begin(write=True)

    # Commit cuoi cung
    txn_lq.commit()
    txn_gt.commit()
    env_lq.close()
    env_gt.close()

    # Ghi meta_info.txt vao tung folder lmdb (yeu cau bat buoc cua BasicSR)
    with open(os.path.join(lq_lmdb_path, "meta_info.txt"), "w", encoding="utf-8") as f:
        f.writelines(meta_lq_lines)
    with open(os.path.join(gt_lmdb_path, "meta_info.txt"), "w", encoding="utf-8") as f:
        f.writelines(meta_gt_lines)

    print(f"[OK] Da hoan thanh tao LMDB cho {split_name}:")
    print(f"     LR: {lq_lmdb_path}")
    print(f"     HR: {gt_lmdb_path}")


def main():
    sr_base_dir = r"D:\IEEE\data\phD\data\50k_OCR\aaa_train_ne_version_5\license_plate_sr"
    meta_dir = r"D:\IEEE\data\phD\BasicSR\basicsr\data\meta_info"
    output_root = os.path.join(sr_base_dir, "lmdb")

    train_meta = os.path.join(meta_dir, "meta_info_license_plate_sr_train.txt")
    val_meta = os.path.join(meta_dir, "meta_info_license_plate_sr_val.txt")

    # 1. Tao LMDB cho tap Train (24,468 cap anh)
    make_lmdb_pair(sr_base_dir, "train", train_meta, output_root)

    # 2. Tao LMDB cho tap Val (2,517 cap anh)
    make_lmdb_pair(sr_base_dir, "val", val_meta, output_root)

    print("\n[V] TAT CA DU LIEU 50K ANH DA DUOC DONG GOI LMDB THANH CONG!")


if __name__ == "__main__":
    main()
