import os
import sys
import glob
import yaml
import csv
import argparse
import torch
from torch.utils.flop_counter import FlopCounterMode

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from basicsr.archs import build_network

# Windows encoding safety
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def format_params(param_count):
    """Format parameter count into M with 2 decimal places."""
    m_val = param_count / 1e6
    return f"{m_val:.2f}M"


def format_flops(flop_count):
    """Format FLOPs into G (GigaFLOPs) with 2 decimal places."""
    g_val = flop_count / 1e9
    return f"{g_val:.2f}G"


def format_size(size_bytes):
    """Format file size in MB with 2 decimal places."""
    mb_val = size_bytes / (1024 ** 2)
    return f"{mb_val:.2f}MB"


def find_experiment_yml(exp_dir, rel_path):
    """Find the training YAML config for a given experiment."""
    # Check inside experiment directory first
    if os.path.isdir(exp_dir):
        ymls = glob.glob(os.path.join(exp_dir, "*.yml"))
        if ymls:
            return ymls[0]

    # Fallback to options/train/ in workspace
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

    # Priority 1: net_g_latest.pth
    latest_path = os.path.join(models_dir, "net_g_latest.pth")
    if os.path.exists(latest_path):
        return latest_path

    # Priority 2: net_g_*.pth with largest iter
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

    # Priority 3: any .pth
    all_ckpts = glob.glob(os.path.join(models_dir, "*.pth"))
    if all_ckpts:
        return all_ckpts[0]

    return None


def calculate_metrics_for_model(exp_base_dir, exp_rel_path, scale=2, lr_h=24, lr_w=24, device='cpu'):
    """
    Calculate Params, FLOPs, and Model Size (latest checkpoint) for an experiment.
    """
    exp_dir = os.path.join(exp_base_dir, os.path.basename(exp_rel_path.strip().replace('\\', '/')))
    
    yml_path = find_experiment_yml(exp_dir, exp_rel_path)
    if not yml_path or not os.path.exists(yml_path):
        print(f"  [!] Cannot find YAML config for {exp_rel_path}")
        return None, None, None

    with open(yml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    net_opt = cfg.get('network_g')
    if not net_opt:
        print(f"  [!] 'network_g' missing in {yml_path}")
        return None, None, None

    # 1. Build network and compute Params
    try:
        model = build_network(net_opt).to(device)
        model.eval()
    except Exception as e:
        print(f"  [!] Failed to build network for {yml_path}: {e}")
        return None, None, None

    total_params = sum(p.numel() for p in model.parameters())

    # 2. Compute FLOPs
    # Determine input channels
    in_ch = net_opt.get('num_in_ch', net_opt.get('in_nc', net_opt.get('in_chans', 3)))
    dummy_input = torch.randn(1, in_ch, lr_h, lr_w, device=device)

    try:
        with torch.no_grad():
            with FlopCounterMode(display=False) as fmode:
                _ = model(dummy_input)
                total_flops = fmode.get_total_flops()
    except Exception as e:
        print(f"  [!] Error computing FLOPs: {e}")
        total_flops = 0

    # 3. Model size on disk (latest checkpoint)
    ckpt_path = find_latest_checkpoint(exp_dir)
    if ckpt_path and os.path.exists(ckpt_path):
        size_bytes = os.path.getsize(ckpt_path)
        size_str = format_size(size_bytes)
    else:
        # Fallback estimation if checkpoint not yet generated: total_params * 4 bytes (fp32)
        est_bytes = total_params * 4
        size_str = f"~{format_size(est_bytes)} (est)"

    return format_params(total_params), format_flops(total_flops), size_str


def main():
    parser = argparse.ArgumentParser(description="Calculate Params, FLOPs, and Model Size for BasicSR models in tonghop.csv")
    parser.add_argument("--csv", type=str, default="options/tonghop.csv", help="Path to tonghop.csv")
    parser.add_argument("--experiments_root", type=str, default=r"F:\license_plate_dataset\license_plate_sr\experiments",
                        help="Root folder of trained experiments")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output path for updated CSV (defaults to overwriting input CSV)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to build model on ('cpu' or 'cuda')")
    parser.add_argument("--lr_h_x2", type=int, default=24, help="Input height for scale x2 (default: 24)")
    parser.add_argument("--lr_w_x2", type=int, default=24, help="Input width for scale x2 (default: 24)")
    parser.add_argument("--lr_h_x4", type=int, default=12, help="Input height for scale x4 (default: 12)")
    parser.add_argument("--lr_w_x4", type=int, default=12, help="Input width for scale x4 (default: 12)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[!] Error: CSV file not found: {args.csv}")
        return

    print("=" * 80)
    print(f"[*] BENCHMARK PARAMS, FLOPS, MODEL SIZE CHO CAC MO HINH BASICSR")
    print(f"    CSV Input:         {args.csv}")
    print(f"    Experiments Root:  {args.experiments_root}")
    print(f"    Input Size x2:     [1, C, {args.lr_h_x2}, {args.lr_w_x2}]")
    print(f"    Input Size x4:     [1, C, {args.lr_h_x4}, {args.lr_w_x4}]")
    print(f"    Device:            {args.device}")
    print("=" * 80)

    rows = []
    fieldnames = []
    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"[*] Tong so mo hinh can xu ly: {len(rows)}\n")

    for idx, row in enumerate(rows, 1):
        m_name = row.get('Model', f'Model_{idx}')
        w2 = row.get('Where (x2)', '').strip()
        w4 = row.get('Where (x4)', '').strip()

        print(f"[{idx:02d}/{len(rows):02d}] Dang xu ly model: {m_name}")

        # Compute x2
        if w2:
            p2, f2, s2 = calculate_metrics_for_model(
                args.experiments_root, w2, scale=2,
                lr_h=args.lr_h_x2, lr_w=args.lr_w_x2, device=args.device
            )
            if p2 is not None:
                row['Params (x2)'] = p2
                row['FLOPS (x2)'] = f2
                row['Model size (x2)'] = s2
                print(f"    -> Scale x2: Params={p2}, FLOPS={f2}, Size={s2}")

        # Compute x4
        if w4:
            p4, f4, s4 = calculate_metrics_for_model(
                args.experiments_root, w4, scale=4,
                lr_h=args.lr_h_x4, lr_w=args.lr_w_x4, device=args.device
            )
            if p4 is not None:
                row['Params (x4)'] = p4
                row['FLOPS (x4)'] = f4
                row['Model size (x4)'] = s4
                print(f"    -> Scale x4: Params={p4}, FLOPS={f4}, Size={s4}")

    out_csv = args.output_csv if args.output_csv else args.csv
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 80)
    print(f"[+] HOAN TAT! Da cap nhat ket qua vao: {out_csv}")
    print("=" * 80)


if __name__ == '__main__':
    main()
