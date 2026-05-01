# GNR638 Project – Map Reconstruction + VQA

## Files in this repo

```
├── inference.py       ← main script (grader runs this)
├── requirements.txt   ← all Python dependencies
├── setup.bash         ← environment + model download (grader runs this first)
└── README.md
```

---

## Exact grader flow

```bash
cd ./your_directory          # grader cds into your cloned repo
bash setup.bash              # creates env, installs deps, downloads model
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
python grading_script.py --submission_file submission.csv
conda remove --name gnr_project_env --all -y
```

`submission.csv` is written to **the same directory** (`your_directory/submission.csv`).

---

## Before submitting — REQUIRED CHANGES

### 1. Edit `setup.bash` line 10 — set your GitHub repo URL
```bash
# Change this line in setup.bash:
REPO_URL="https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
```

### 2. Push ALL files to GitHub (make repo PUBLIC)
```bash
git init
git add inference.py requirements.txt setup.bash README.md
git commit -m "GNR638 project submission"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

⚠️ Make repo **public** before **03 May 2026 11:00 AM** (evaluation start).

### 3. Create the submission zip

Format: `project_<num>_<roll1>_<roll2>.zip`

The zip contains **only setup.bash**:
```bash
# Example: project 1, rolls 22m2162 and 22m2152
mkdir project_1_22m2162_22m2152
cp setup.bash project_1_22m2162_22m2152/
zip project_1_22m2162_22m2152.zip project_1_22m2162_22m2152/setup.bash
```

---

## What setup.bash does

| Step | Action |
|------|--------|
| 1 | Creates conda env `gnr_project_env` with Python 3.11 |
| 2 | Installs PyTorch 2.4.1 (CUDA 12.1 wheels — works on CUDA 12.6) |
| 3 | Installs all requirements from `requirements.txt` |
| 4 | Downloads `Qwen/Qwen2.5-VL-3B-Instruct` (~7 GB, no HF token) to `~/models/qwen25vl-3b/` |

---

## Model

| Item | Value |
|------|-------|
| Model ID | `Qwen/Qwen2.5-VL-3B-Instruct` |
| HF class | `Qwen2_5_VLForConditionalGeneration` |
| Availability | Public — **no Hugging Face token required** |
| Download size | ~7 GB |
| Local path | `~/models/qwen25vl-3b/` |
| VRAM at inference | ~8 GB (L40s has 48 GB) |

---

## Test directory structure expected

```
<test_dir>/
├── patches/
│   ├── patch_0.png     ← always top-left anchor
│   ├── patch_1.png
│   └── …
├── test.csv            ← id, question, option_1, option_2, option_3, option_4
└── submission.csv      ← dummy file (grader provides it; we overwrite in CWD)
```

---

## Local testing

```bash
# From inside your cloned repo dir:
bash setup.bash
conda activate gnr_project_env
python inference.py --test_dir /absolute/path/to/sample_test_dir
cat submission.csv
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `AssertionError: Model not found` | `setup.bash` didn't finish — re-run it |
| `ModuleNotFoundError` | Ensure `conda activate gnr_project_env` succeeded |
| `AssertionError: patches/ not found` | Check `--test_dir` points to the right folder |
| CUDA OOM | Already guarded: `max_pixels=256*28*28` in query_model |
