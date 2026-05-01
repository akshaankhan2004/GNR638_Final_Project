#!/usr/bin/env python3
"""
GNR638 Project – Map Reconstruction + VQA
==========================================

Expected test_dir layout:
    patches/        patch_0.png, patch_1.png, …   (patch_0 is top-left anchor)
    test.csv        columns: id, question, option_1, option_2, option_3, option_4, option_5

Writes:
    ./submission.csv
"""

import argparse
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ─────────────────────────────────────────────────────────────────────────────
# setup.bash downloads to: $HOME/models/qwen25vl-3b
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path.home() / "models" / "qwen25vl-3b"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GNR638 Map Reconstruction + VQA")
parser.add_argument("--test_dir", required=True,
                    help="Absolute path to the test directory")
args = parser.parse_args()

TEST_DIR    = Path(args.test_dir)
PATCHES_DIR = TEST_DIR / "patches"
TEST_CSV    = TEST_DIR / "test.csv"
OUTPUT_CSV  = Path("submission.csv")

print("=" * 65)
print("  GNR638 Project – Map Reconstruction + VQA")
print("=" * 65)
print(f"  CWD        : {Path.cwd()}")
print(f"  test_dir   : {TEST_DIR}")
print(f"  patches/   : {PATCHES_DIR}")
print(f"  test.csv   : {TEST_CSV}")
print(f"  model      : {MODEL_PATH}")
print(f"  output     : {OUTPUT_CSV.resolve()}")
print(f"  CUDA       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM       : "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print("=" * 65)

# ── Hard assertions (fail fast with a clear message) ─────────────────────────
assert PATCHES_DIR.exists(), \
    f"ERROR: patches/ directory not found at {PATCHES_DIR}"
assert TEST_CSV.exists(), \
    f"ERROR: test.csv not found at {TEST_CSV}"
assert MODEL_PATH.exists(), \
    (f"ERROR: model weights not found at {MODEL_PATH}\n"
     f"       Did setup.bash complete successfully?")

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_start = time.time()

# =============================================================================
# PART 1 – MAP STITCHING
# GPU-accelerated Normalised Cross-Correlation (NCC) on gradient images.
# Logic mirrors the v9 notebook exactly.
# =============================================================================

# ── Helpers ───────────────────────────────────────────────────────────────────
def rot90(img, k):
    """Rotate numpy image by k*90 degrees counter-clockwise."""
    return np.rot90(img, k % 4).copy()


def load_patches(d: Path) -> dict:
    """Load all patch_N.png files sorted by N."""
    files = sorted(d.glob("patch_*.png"),
                   key=lambda f: int(f.stem.split("_")[1]))
    out = {}
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            out[f.stem] = img
        else:
            print(f"  WARNING: could not read {f}")
    return out


# ── Load patches ──────────────────────────────────────────────────────────────
print("\n[1/6] Loading patches ...")
patches     = load_patches(PATCHES_DIR)
patch_names = list(patches.keys())
N           = len(patches)
ph, pw      = patches["patch_0"].shape[:2]
print(f"  {N} patches  size={pw}x{ph}")

# ── Algorithm parameters (same as notebook) ───────────────────────────────────
MAX_OVERLAP_SEARCH      = min(pw // 2, 64)
CONF_THRESHOLD          = 0.2
CONF_THRESHOLD_INTERIOR = 0.15
MIN_GAP                 = 0.02
COL_EXT_THRESHOLD       = 0.3
COL_EXT_MIN_AGREE       = 0.75

print(f"  MAX_OVERLAP={MAX_OVERLAP_SEARCH}  CONF={CONF_THRESHOLD}  "
      f"MIN_GAP={MIN_GAP}  COL_EXT_AGREE={COL_EXT_MIN_AGREE}")

# ── Gradient cache (all patches × 4 rotations) ────────────────────────────────
print("\n[2/6] Building gradient cache on GPU ...")


def make_gradient_tensor(img: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return torch.from_numpy(np.sqrt(gx**2 + gy**2)).to(device)


grad_cache = {}   # (name, k) -> Tensor
idx_to_key = []   # linear index -> (name, k)
key_to_idx = {}   # (name, k)  -> linear index

for name in patch_names:
    for k in range(4):
        key = (name, k)
        t   = make_gradient_tensor(rot90(patches[name], k))
        grad_cache[key] = t
        key_to_idx[key] = len(idx_to_key)
        idx_to_key.append(key)

all_grads = torch.stack([grad_cache[k] for k in idx_to_key], dim=0)
print(f"  all_grads {tuple(all_grads.shape)}  "
      f"({all_grads.element_size()*all_grads.nelement()/1e6:.1f} MB)  "
      f"t={time.time()-t_start:.1f}s")


# ── GPU batch-NCC on edge strips ──────────────────────────────────────────────
def batch_ncc_edge(ref: torch.Tensor,
                   cands: torch.Tensor,
                   direction: str,
                   max_ov: int = None) -> tuple:
    """
    Compare right/bottom edge strip of `ref` against left/top strips of
    each candidate in `cands`.  Returns (best_scores, best_overlaps).
    """
    if max_ov is None:
        max_ov = MAX_OVERLAP_SEARCH
    M            = cands.shape[0]
    score_matrix = torch.full((M, max_ov), -1.0, device=device)

    for ov in range(1, max_ov + 1):
        if direction == 'h':
            ref_strip   = ref[:, -ov:]
            cand_strips = cands[:, :, :ov]
        else:
            ref_strip   = ref[-ov:, :]
            cand_strips = cands[:, :ov, :]
        r_flat = ref_strip.reshape(-1).float()
        c_flat = cand_strips.reshape(M, -1).float()
        r_c    = r_flat - r_flat.mean()
        c_c    = c_flat - c_flat.mean(dim=1, keepdim=True)
        r_std  = r_c.std() + 1e-8
        c_std  = c_c.std(dim=1) + 1e-8
        ncc    = (c_c * r_c.unsqueeze(0)).mean(dim=1) / (r_std * c_std)
        score_matrix[:, ov - 1] = ncc

    best_scores, best_ov_idx = score_matrix.max(dim=1)
    return (best_scores.cpu().numpy(),
            (best_ov_idx + 1).cpu().numpy().astype(int))


def get_cand_tensors(remaining: set) -> tuple:
    cand_keys = [(name, k) for name in remaining for k in range(4)]
    cand_idx  = [key_to_idx[key] for key in cand_keys]
    return cand_keys, all_grads[cand_idx]


def gref(name: str) -> torch.Tensor:
    """Gradient tensor for an already-placed patch."""
    return grad_cache[(name, placed_rot[name])]


def find_best_match(ref_tensor, remaining, direction, threshold):
    if not remaining:
        return None
    cand_keys, cand_tens = get_cand_tensors(remaining)
    best_scores, best_ovs = batch_ncc_edge(ref_tensor, cand_tens, direction)
    order  = np.argsort(-best_scores)
    top_sc = best_scores[order[0]]
    if top_sc < threshold:
        return None
    # Reject if margin over second-best is too small (ambiguous match)
    if (len(order) > 1
            and (top_sc - best_scores[order[1]] < MIN_GAP)
            and top_sc < 0.6):
        return None
    name, k = cand_keys[order[0]]
    return name, k, float(top_sc), int(best_ovs[order[0]])


def find_best_dual_match(ref_left, ref_above, remaining, threshold):
    if not remaining:
        return None
    cand_keys, cand_tens = get_cand_tensors(remaining)
    scores_l = np.zeros(len(cand_keys))
    ovs_l    = np.ones(len(cand_keys), dtype=int)
    scores_a = np.zeros(len(cand_keys))
    ovs_a    = np.ones(len(cand_keys), dtype=int)
    if ref_left  is not None:
        scores_l, ovs_l = batch_ncc_edge(ref_left,  cand_tens, 'h')
    if ref_above is not None:
        scores_a, ovs_a = batch_ncc_edge(ref_above, cand_tens, 'v')
    n_c      = (1 if ref_left is not None else 0) + \
               (1 if ref_above is not None else 0)
    combined = (scores_l + scores_a) / max(n_c, 1)
    order    = np.argsort(-combined)
    top_sc   = combined[order[0]]
    if top_sc < threshold:
        return None
    if (len(order) > 1
            and (top_sc - combined[order[1]] < MIN_GAP)
            and top_sc < 0.5):
        return None
    name, k = cand_keys[order[0]]
    return name, k, float(top_sc), int(ovs_l[order[0]]), int(ovs_a[order[0]])


# ── Step 1+2: Sequential stitch from patch_0 ─────────────────────────────────
print("\n[3/6] Sequential stitching ...")

grid        = {(0, 0): "patch_0"}   # (col, row) -> patch_name
placed_rot  = {"patch_0": 0}        # patch_name -> rotation k
placed_ov_h = {(0, 0): 0}          # (col, row)  -> horizontal overlap px
placed_ov_v = {(0, 0): 0}          # (col, row)  -> vertical overlap px
remaining   = set(patch_names) - {"patch_0"}

# Grow row 0 rightward
col = 0
while True:
    m = find_best_match(gref(grid[(col, 0)]), remaining, 'h', CONF_THRESHOLD)
    if m is None:
        break
    nn, nk, sc, ov = m
    col += 1
    grid[(col, 0)] = nn;  placed_rot[nn] = nk
    placed_ov_h[(col, 0)] = ov;  placed_ov_v[(col, 0)] = 0
    remaining.discard(nn)
    print(f"  ({col},0)={nn}  k={nk}  ncc={sc:.3f}  ov={ov}px")
ROW_LEN = col + 1

# Grow col 0 downward
row = 0
while True:
    m = find_best_match(gref(grid[(0, row)]), remaining, 'v', CONF_THRESHOLD)
    if m is None:
        break
    nn, nk, sc, ov = m
    row += 1
    grid[(0, row)] = nn;  placed_rot[nn] = nk
    placed_ov_h[(0, row)] = 0;  placed_ov_v[(0, row)] = ov
    remaining.discard(nn)
    print(f"  (0,{row})={nn}  k={nk}  ncc={sc:.3f}  ov={ov}px")
COL_LEN = row + 1

# Fill interior
for r in range(1, COL_LEN):
    for c in range(1, ROW_LEN):
        if (c, r) in grid or not remaining:
            continue
        ref_l = gref(grid[(c-1, r)]) if (c-1, r) in grid else None
        ref_a = gref(grid[(c, r-1)]) if (c, r-1) in grid else None
        m = find_best_dual_match(ref_l, ref_a, remaining,
                                 CONF_THRESHOLD_INTERIOR)
        if m is None:
            m = find_best_dual_match(ref_l, ref_a, remaining, 0.05)
        if m is None:
            continue
        name, k, sc, ov_h, ov_v = m
        grid[(c, r)] = name;  placed_rot[name] = k
        placed_ov_h[(c, r)] = ov_h;  placed_ov_v[(c, r)] = ov_v
        remaining.discard(name)

    if r % 3 == 0 or r == COL_LEN - 1:
        n_placed = sum(1 for (_, rr) in grid if rr == r)
        print(f"  Row {r}: {n_placed}/{ROW_LEN} placed  "
              f"pool={len(remaining)}  t={time.time()-t_start:.1f}s")

print(f"  After stitch: {ROW_LEN}×{COL_LEN}={ROW_LEN*COL_LEN}  "
      f"placed={len(grid)}  remaining={len(remaining)}")

# ── Threshold search if grid dimensions don't cover all N patches ─────────────
if ROW_LEN * COL_LEN != N and remaining:
    print("  Grid size mismatch – scanning thresholds ...")
    for thresh in [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]:
        # Probe with a throw-away grid
        g2   = {(0, 0): "patch_0"}
        r2   = {"patch_0": 0}
        rem2 = set(patch_names) - {"patch_0"}
        c = 0
        while True:
            m = find_best_match(
                grad_cache[(g2[(c, 0)], r2[g2[(c, 0)]])], rem2, 'h', thresh)
            if m is None: break
            nn, nk, sc, ov = m; c += 1
            g2[(c, 0)] = nn; r2[nn] = nk; rem2.discard(nn)
        rl = c + 1; rw = 0
        while True:
            m = find_best_match(
                grad_cache[(g2[(0, rw)], r2[g2[(0, rw)]])], rem2, 'v', thresh)
            if m is None: break
            nn, nk, sc, ov = m; rw += 1
            g2[(0, rw)] = nn; r2[nn] = nk; rem2.discard(nn)
        cl = rw + 1
        print(f"    thresh={thresh}: {rl}×{cl}={rl*cl}  (need {N})")
        if rl * cl == N:
            # Rebuild properly with this threshold
            grid        = {(0, 0): "patch_0"}
            placed_rot  = {"patch_0": 0}
            placed_ov_h = {(0, 0): 0}
            placed_ov_v = {(0, 0): 0}
            remaining   = set(patch_names) - {"patch_0"}
            c = 0
            while True:
                m = find_best_match(gref(grid[(c, 0)]), remaining, 'h', thresh)
                if m is None: break
                nn, nk, sc, ov = m; c += 1
                grid[(c, 0)] = nn; placed_rot[nn] = nk
                placed_ov_h[(c, 0)] = ov; placed_ov_v[(c, 0)] = 0
                remaining.discard(nn)
            ROW_LEN = c + 1; rw = 0
            while True:
                m = find_best_match(gref(grid[(0, rw)]), remaining, 'v', thresh)
                if m is None: break
                nn, nk, sc, ov = m; rw += 1
                grid[(0, rw)] = nn; placed_rot[nn] = nk
                placed_ov_h[(0, rw)] = 0; placed_ov_v[(0, rw)] = ov
                remaining.discard(nn)
            COL_LEN = rw + 1
            for r in range(1, COL_LEN):
                for c2 in range(1, ROW_LEN):
                    if (c2, r) in grid or not remaining: continue
                    ref_l = gref(grid[(c2-1, r)]) if (c2-1, r) in grid else None
                    ref_a = gref(grid[(c2, r-1)]) if (c2, r-1) in grid else None
                    m = find_best_dual_match(ref_l, ref_a, remaining,
                                            thresh * 0.7)
                    if m is None: continue
                    name, k, sc, ov_h, ov_v = m
                    grid[(c2, r)] = name; placed_rot[name] = k
                    placed_ov_h[(c2, r)] = ov_h; placed_ov_v[(c2, r)] = ov_v
                    remaining.discard(name)
            print(f"    Rebuilt with thresh={thresh}: "
                  f"{ROW_LEN}×{COL_LEN}  remaining={len(remaining)}")
            break

# ── Column-by-column right extension ─────────────────────────────────────────
if remaining:
    print(f"  Column extension for {len(remaining)} patches ...")
    min_agree = max(1, int(COL_EXT_MIN_AGREE * COL_LEN))
    while remaining:
        new_col    = ROW_LEN
        row_cands  = {}
        used_names = set()
        for r in range(COL_LEN):
            if (new_col - 1, r) not in grid: continue
            pool = remaining - used_names
            if not pool: continue
            m = find_best_match(gref(grid[(new_col-1, r)]),
                                pool, 'h', COL_EXT_THRESHOLD)
            if m:
                nn, nk, sc, ov = m
                row_cands[r] = (nn, nk, sc, ov)
                used_names.add(nn)
        print(f"    Col {new_col}: "
              f"{len(row_cands)}/{COL_LEN} rows matched")
        if len(row_cands) < min_agree:
            print(f"    Need {min_agree} rows – stopping column extension.")
            break
        for r, (name, k, sc, ov) in row_cands.items():
            grid[(new_col, r)] = name;  placed_rot[name] = k
            placed_ov_h[(new_col, r)] = ov;  placed_ov_v[(new_col, r)] = 0
            remaining.discard(name)
        ROW_LEN += 1
        if not remaining: break

# ── Offline image-processing fallback (Step 5 from notebook) ──────────────────
if remaining:
    print(f"\n  Offline fallback for {len(remaining)} unplaced patches ...")

    # Build a partial canvas for visual-context descriptors
    h_ovs5 = [placed_ov_h.get((c, r), 0) for (c, r) in grid
               if c > 0 and placed_ov_h.get((c, r), 0) > 0]
    v_ovs5 = [placed_ov_v.get((c, r), 0) for (c, r) in grid
               if r > 0 and placed_ov_v.get((c, r), 0) > 0]
    _moh  = int(np.median(h_ovs5)) if h_ovs5 else 0
    _mov  = int(np.median(v_ovs5)) if v_ovs5 else 0
    _sx   = max(1, pw - _moh);  _sy = max(1, ph - _mov)
    _mc   = max(c for (c, _) in grid);  _mr = max(r for (_, r) in grid)
    _cw   = _sx * _mc + pw;  _ch = _sy * _mr + ph
    partial = np.zeros((_ch, _cw, 3), dtype=np.uint8)
    for (_pc, _pr), _pn in grid.items():
        _pk = placed_rot.get(_pn, 0)
        _pi = rot90(patches[_pn], _pk)
        _cx = _pc * _sx + (_moh if _pc > 0 else 0)
        _cy = _pr * _sy + (_mov if _pr > 0 else 0)
        _ph2, _pw2 = _pi.shape[:2]
        if 0 <= _cx < _cw and 0 <= _cy < _ch:
            cx1 = min(_cx + _pw2, _cw)
            cy1 = min(_cy + _ph2, _ch)
            
            partial[_cy:cy1, _cx:cx1] = _pi[:cy1 - _cy, :cx1 - _cx]

    _HIST_BINS  = 16
    _GRAD_BINS  = 8
    _RESIZE_DIM = 64

    def _compute_descriptor(img_bgr: np.ndarray) -> np.ndarray:
        """58-D float32: HSV histograms + gradient orientation + brightness."""
        s   = cv2.resize(img_bgr, (_RESIZE_DIM, _RESIZE_DIM),
                         interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV).astype(np.float32)
        parts = []
        for chi in range(3):
            h, _ = np.histogram(hsv[:, :, chi].ravel(),
                                bins=_HIST_BINS, range=(0, 256))
            h = h.astype(np.float32)
            parts.append(h / (h.sum() + 1e-8))
        gray = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag  = np.sqrt(gx**2 + gy**2).ravel()
        ang  = (np.arctan2(gy, gx).ravel() + np.pi) / (2 * np.pi)
        gh, _ = np.histogram(ang, bins=_GRAD_BINS, range=(0, 1), weights=mag)
        gh    = gh.astype(np.float32)
        parts.append(gh / (gh.sum() + 1e-8))
        v = hsv[:, :, 2].ravel() / 255.0
        parts.append(np.array([v.mean(), v.std()], dtype=np.float32))
        return np.concatenate(parts)

    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a);  nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8: return 0.0
        return float(np.dot(a, b) / (na * nb))

    # NCC score matrix over all empty cells × all remaining candidates
    empty_cells_5        = list(
        {(c, r) for c in range(ROW_LEN) for r in range(COL_LEN)}
        - set(grid.keys()))
    remain_list_5        = list(remaining)
    cand_keys_5, cand_tens_5 = get_cand_tensors(remaining)
    ncc_mat = {i: {} for i in range(len(cand_keys_5))}

    for (tc, tr) in empty_cells_5:
        s_l = np.zeros(len(cand_keys_5))
        s_a = np.zeros(len(cand_keys_5))
        nc  = 0
        if (tc-1, tr) in grid:
            s_l, _ = batch_ncc_edge(gref(grid[(tc-1, tr)]),
                                    cand_tens_5, 'h');  nc += 1
        if (tc, tr-1) in grid:
            s_a, _ = batch_ncc_edge(gref(grid[(tc, tr-1)]),
                                    cand_tens_5, 'v');  nc += 1
        combo = (s_l + s_a) / max(nc, 1)
        for i in range(len(cand_keys_5)):
            ncc_mat[i][(tc, tr)] = float(combo[i])

    all_ncc = [v for d in ncc_mat.values() for v in d.values()]
    ncc_lo  = min(all_ncc);  ncc_hi = max(all_ncc)
    ncc_rng = max(ncc_hi - ncc_lo, 1e-8)

    # Per-patch descriptor (best rotation chosen by NCC average)
    patch_desc = {}
    for name in remain_list_5:
        best_k = 0;  best_avg = -1e9
        for ki in range(4):
            idxs = [i for i, (nm, kk) in enumerate(cand_keys_5)
                    if nm == name and kk == ki]
            if idxs:
                avg = float(np.mean([ncc_mat[idxs[0]].get(c, 0.)
                                     for c in empty_cells_5]))
                if avg > best_avg: best_avg = avg;  best_k = ki
        patch_desc[name] = (_compute_descriptor(rot90(patches[name], best_k)),
                            best_k)

    # Per-cell context descriptor from partial canvas
    cell_desc = {}
    for (tc, tr) in empty_cells_5:
        cx0 = max(0, (tc-1)*_sx);  cy0 = max(0, (tr-1)*_sy)
        cx1 = min(_cw, (tc+2)*_sx+pw);  cy1 = min(_ch, (tr+2)*_sy+ph)
        crop = partial[cy0:cy1, cx0:cx1]
        if crop.size == 0: crop = np.zeros((ph, pw, 3), dtype=np.uint8)
        cell_desc[(tc, tr)] = _compute_descriptor(crop)

    all_sims = [_cosine_sim(patch_desc[n][0], cell_desc[c])
                for n in remain_list_5 for c in empty_cells_5]
    sim_lo  = min(all_sims) if all_sims else 0.
    sim_hi  = max(all_sims) if all_sims else 1.
    sim_rng = max(sim_hi - sim_lo, 1e-8)

    # Combined score and greedy assignment
    scored_5 = []
    for i, (name, k) in enumerate(cand_keys_5):
        dp, best_k = patch_desc[name]
        for (tc, tr) in empty_cells_5:
            ncc_n = (ncc_mat[i].get((tc, tr), 0.) - ncc_lo) / ncc_rng
            if k == best_k:
                sim_n = (_cosine_sim(dp, cell_desc[(tc, tr)]) - sim_lo) / sim_rng
                score = 0.65 * ncc_n + 0.35 * sim_n
            else:
                score = ncc_n
            scored_5.append((score, name, k, tc, tr))

    scored_5.sort(key=lambda x: -x[0])
    used_p5, used_c5 = set(), set()
    for score, name, k, tc, tr in scored_5:
        if name in used_p5 or (tc, tr) in used_c5: continue
        grid[(tc, tr)] = name;  placed_rot[name] = k
        placed_ov_h[(tc, tr)] = 0;  placed_ov_v[(tc, tr)] = 0
        remaining.discard(name); used_p5.add(name); used_c5.add((tc, tr))
        print(f"  NCC+Desc: {name} -> ({tc},{tr}) k={k} score={score:.3f}")
        if not remaining: break

    # Absolute last resort – should never fire
    empty_final = [(c, r) for c in range(ROW_LEN) for r in range(COL_LEN)
                   if (c, r) not in grid]
    for name in list(remaining):
        if not empty_final: break
        tc, tr = empty_final.pop(0)
        grid[(tc, tr)] = name;  placed_rot[name] = 0
        placed_ov_h[(tc, tr)] = 0;  placed_ov_v[(tc, tr)] = 0
        remaining.discard(name)
        print(f"  Last resort: {name} -> ({tc},{tr})")

    print(f"  Fallback done: {len(grid)}/{N} placed  "
          f"{len(remaining)} unresolved")

# ── Step 6: Render stitched map ───────────────────────────────────────────────
print("\n[4/6] Rendering stitched map ...")

h_ovs = [placed_ov_h.get((c, r), 0) for (c, r) in grid
         if c > 0 and placed_ov_h.get((c, r), 0) > 0]
v_ovs = [placed_ov_v.get((c, r), 0) for (c, r) in grid
         if r > 0 and placed_ov_v.get((c, r), 0) > 0]

med_ov_h = int(np.median(h_ovs)) if h_ovs else 0
med_ov_v = int(np.median(v_ovs)) if v_ovs else 0
step_x   = max(1, pw - med_ov_h)
step_y   = max(1, ph - med_ov_v)
max_col  = max(c for (c, r) in grid)
max_row  = max(r for (c, r) in grid)
cw = step_x * max_col + pw
ch = step_y * max_row + ph

print(f"  Overlap h={med_ov_h}px v={med_ov_v}px | "
      f"step={step_x}×{step_y} | canvas={cw}×{ch}")

canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
for (col, row), name in grid.items():
    k          = placed_rot.get(name, 0)
    img        = rot90(patches[name], k)
    crop_x     = med_ov_h if col > 0 else 0
    crop_y     = med_ov_v if row > 0 else 0
    patch_crop = img[crop_y:, crop_x:]
    ph_c, pw_c = patch_crop.shape[:2]
    cx0 = col * step_x + crop_x
    cy0 = row * step_y + crop_y
    cx1 = min(cx0 + pw_c, cw)
    cy1 = min(cy0 + ph_c, ch)
    if 0 <= cx0 < cw and 0 <= cy0 < ch:
        canvas[cy0:cy1, cx0:cx1] = patch_crop[:cy1-cy0, :cx1-cx0]

MAP_PATH = Path("reconstructed_map.png")
cv2.imwrite(str(MAP_PATH), canvas)

black_px = int((canvas.sum(axis=2) == 0).sum())
total_px = cw * ch
print(f"  Saved → {MAP_PATH}  shape={canvas.shape}")
print(f"  Black pixels: {black_px}/{total_px} ({100*black_px/total_px:.2f}%)")

# Print grid layout (same as notebook)
ARROWS = "↑→↓←"
print(f"\n  Grid layout ({max_col+1}×{max_row+1}):")
for r in range(max_row + 1):
    row_str = ""
    for c in range(max_col + 1):
        nm = grid.get((c, r))
        if nm:
            idx = nm.split("_")[1]
            k   = placed_rot.get(nm, 0)
            row_str += f"{idx:>4}{ARROWS[k]}"
        else:
            row_str += "  -- "
    print(f"  {row_str}")

print(f"\n  Stitching done  t={time.time()-t_start:.1f}s")

# Free GPU memory before loading the large VLM
del all_grads, grad_cache
torch.cuda.empty_cache()

# =============================================================================
# PART 2 – VQA with Qwen2.5-VL-3B-Instruct
# Model loaded from disk (no internet needed at this point).
# Architecture: Qwen2_5_VLForConditionalGeneration (same as Qwen3-VL-4B Kaggle)
# =============================================================================
print("\n[5/6] Loading Qwen2.5-VL from disk (no internet) ...")

processor = AutoProcessor.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    trust_remote_code=True,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)
model.eval()
print(f"  Model ready.  device_map={model.hf_device_map}")

# Resize reconstructed map: longest side → 1024 px
map_pil_full = Image.open(str(MAP_PATH)).convert("RGB")
MAX_SIDE     = 1024
w, h_img     = map_pil_full.size
scale        = MAX_SIDE / max(w, h_img)
map_pil      = map_pil_full.resize(
    (int(w * scale), int(h_img * scale)), Image.LANCZOS)
print(f"  Map resized: {w}×{h_img} → {map_pil.size}")

df = pd.read_csv(str(TEST_CSV))
print(f"  {len(df)} questions  |  columns={df.columns.tolist()}")


def build_prompt(row) -> str:
    return (
        "Look at this map carefully and answer the following "
        "multiple-choice question.\n"
        f"Question: {row['question']}\n"
        "Options:\n"
        f"1) {row['option_1']}\n"
        f"2) {row['option_2']}\n"
        f"3) {row['option_3']}\n"
        f"4) {row['option_4']}\n"
        "5) None of the above\n"
        "Reply with only a single digit (1, 2, 3, 4, or 5). "
        "Do not explain. Do not write anything else."
    )


def query_model(image: Image.Image, prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ],
    }]
    text   = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,   # cap to avoid OOM
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=16, do_sample=False)

    trimmed = [out[len(inp):]
               for inp, out in zip(inputs.input_ids, generated_ids)]
    output  = processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()
    torch.cuda.empty_cache()
    return output


def parse_answer(raw: str) -> int:
    """Extract first digit 1-5 from model output; default to 5."""
    for ch in raw:
        if ch in "12345":
            return int(ch)
    return 5


# ── Inference loop ─────────────────────────────────────────────────────────────
print("\n[6/6] VQA inference ...")
results = []
for idx, row in df.iterrows():
    qid    = row["id"]
    prompt = build_prompt(row)
    raw    = query_model(map_pil, prompt)
    answer = parse_answer(raw)
    results.append({"id": qid, "answer": answer})
    print(f"  [{idx+1}/{len(df)}]  id={qid}  raw='{raw}'  → {answer}")

# ── Save submission.csv in CWD ─────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(str(OUTPUT_CSV), index=False)

print(f"\n  Saved {len(results)} rows → {OUTPUT_CSV.resolve()}")
print(results_df.to_string(index=False))
print(f"\nTotal runtime: {time.time()-t_start:.1f}s  "
      f"({(time.time()-t_start)/60:.1f} min)")
