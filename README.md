# ViTrans — Máy Dịch Việt (MarianNMT)

Hệ thống dịch thuật **Anh · Trung · Hàn · Nhật ↔ Tiếng Việt** dựa trên [MarianMT](https://huggingface.co/Helsinki-NLP) với inference nhanh qua CTranslate2.

- **Nhẹ**: Model INT8 ~80–120 MB/cặp, chạy tốt trên máy 4 GB RAM
- **Nhanh**: CTranslate2 nhanh hơn HuggingFace 3–4×, hỗ trợ batch dịch
- **Đầy đủ**: Web UI (Gradio), CLI, Python API
- **Huấn luyện**: Script train từ đầu + fine-tune từ model pretrained

---

## Mục lục

1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Cài đặt nhanh](#cài-đặt-nhanh)
3. [Các cặp ngôn ngữ](#các-cặp-ngôn-ngữ)
4. [Sử dụng — Web Interface](#web-interface)
5. [Sử dụng — CLI](#cli)
6. [Sử dụng — Python API](#python-api)
7. [Download & Quản lý Models](#quản-lý-models)
8. [Fine-Tune Model](#fine-tune-model)
9. [Train từ Đầu](#train-từ-đầu)
10. [Đánh giá (Evaluation)](#đánh-giá)
11. [Cấu trúc dự án](#cấu-trúc-dự-án)

---

## Yêu cầu hệ thống

| | Tối thiểu | Khuyến nghị |
|---|---|---|
| **RAM** | 2 GB | 4 GB |
| **CPU** | 2 nhân | 4 nhân |
| **Python** | 3.9 | 3.10+ |
| **Disk** | 1 GB | 3 GB |
| **GPU** | Không cần | CUDA 11.8+ (tùy chọn) |

---

## Cài đặt nhanh

```bash
# 1. Clone/vào thư mục dự án
cd MarianNMT

# 2. Chạy script setup tự động (tạo venv, cài gói, tải model EN↔VI)
bash scripts/setup.sh

# --- Hoặc cài thủ công ---

# 3. Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# 4. Cài dependencies
pip install -r requirements.txt

# 5. Tải model EN↔VI (model nhỏ nhất, ~300 MB)
python scripts/download_models.py

# 6. Chuyển sang CTranslate2 INT8 (nhanh hơn 3-4x, RAM ít hơn 50%)
python scripts/download_models.py --convert-ct2
```

---

## Các cặp ngôn ngữ

| Hướng | Model | Phương pháp | Chất lượng |
|---|---|---|---|
| 🇬🇧 EN → 🇻🇳 VI | `Helsinki-NLP/opus-mt-en-vi` | Trực tiếp | ⭐⭐⭐⭐ |
| 🇻🇳 VI → 🇬🇧 EN | `Helsinki-NLP/opus-mt-vi-en` | Trực tiếp | ⭐⭐⭐⭐ |
| 🇨🇳 ZH → 🇻🇳 VI | `opus-mt-zh-en` → `opus-mt-en-vi` | Pivot (EN) | ⭐⭐⭐ |
| 🇻🇳 VI → 🇨🇳 ZH | `opus-mt-vi-en` → `opus-mt-en-zh` | Pivot (EN) | ⭐⭐⭐ |
| 🇯🇵 JA → 🇻🇳 VI | `opus-mt-jap-en` → `opus-mt-en-vi` | Pivot (EN) | ⭐⭐⭐ |
| 🇻🇳 VI → 🇯🇵 JA | `opus-mt-vi-en` → `opus-mt-en-jap` | Pivot (EN) | ⭐⭐⭐ |
| 🇰🇷 KO → 🇻🇳 VI | `opus-mt-ko-en` → `opus-mt-en-vi` | Pivot (EN) | ⭐⭐⭐ |
| 🇻🇳 VI → 🇰🇷 KO | `opus-mt-vi-en` → `opus-mt-en-ko` | Pivot (EN) | ⭐⭐⭐ |

> **Pivot translation**: ZH/JA/KO ↔ VI không có model trực tiếp nên dịch qua tiếng Anh (ZH→EN→VI). Bạn có thể cải thiện bằng cách fine-tune model ZH→EN trên corpus liên quan.

---

## Web Interface

```bash
# Khởi động giao diện web
python interface/app.py

# Tùy chọn
python interface/app.py --port 7860 --host 0.0.0.0
python interface/app.py --share          # Tạo link public qua Gradio
python interface/app.py --warmup         # Pre-load EN↔VI (phản hồi nhanh hơn)
```

Mở trình duyệt tại: **http://localhost:7860**

**Tính năng:**
- Dịch văn bản với nút ⇄ hoán đổi ngôn ngữ
- Chọn beam size (chất lượng ↑ hoặc tốc độ ↑)
- Upload file `.txt` và tải về bản dịch
- Các ví dụ có sẵn để thử nhanh

---

## CLI

### Dịch văn bản

```bash
# Dịch trực tiếp
python interface/cli.py translate --src en --tgt vi --text "Hello, how are you?"

# Dịch từ file
python interface/cli.py translate --src zh --tgt vi \
    --file input.txt --output output.txt

# Dịch từ stdin
echo "안녕하세요" | python interface/cli.py translate --src ko --tgt vi

# Tùy chỉnh beam size và batch size
python interface/cli.py translate --src ja --tgt vi \
    --file large_file.txt --output out.txt --beam 4 --batch-size 64
```

### Chế độ tương tác (REPL)

```bash
python interface/cli.py interactive --src en --tgt vi

# Trong REPL:
# :swap       — hoán đổi ngôn ngữ
# :src <code> — đổi ngôn ngữ nguồn (en/vi/zh/ja/ko)
# :tgt <code> — đổi ngôn ngữ đích
# :quit       — thoát
```

### Quản lý models

```bash
# Xem trạng thái tất cả models
python interface/cli.py models list

# Tải model
python interface/cli.py models download --pair en-vi
python interface/cli.py models download --all              # tất cả
python interface/cli.py models download --all --convert-ct2  # + chuyển CT2

# Chuyển model sang CTranslate2
python interface/cli.py models convert --pair en-vi --quantization int8
```

### Đánh giá

```bash
# Đánh giá với FLORES-200 benchmark
python interface/cli.py evaluate --src en --tgt vi --flores

# Đánh giá với file tùy chỉnh
python interface/cli.py evaluate --src en --tgt vi \
    --src-file test_en.txt --ref-file test_vi.txt
```

---

## Python API

```python
from src.translate_engine import TranslationEngine

# Khởi tạo engine (tối đa 3 models trong RAM cùng lúc)
engine = TranslationEngine(device="cpu", max_loaded_models=3)

# Dịch một câu
result = engine.translate("Hello, world!", src="en", tgt="vi")
print(result)  # → "Xin chào, thế giới!"

# Dịch nhiều câu (batch — hiệu quả hơn)
texts = ["Good morning", "Thank you", "How are you?"]
results = engine.translate_batch(texts, src="en", tgt="vi", beam_size=4)

# Dịch ZH → VI (qua pivot EN tự động)
vi_text = engine.translate("你好，很高兴认识你。", src="zh", tgt="vi")

# Pre-load models để request đầu tiên nhanh hơn
engine.warmup(["en-vi", "vi-en"])
```

---

## Quản lý Models

### Tải models

```bash
# Chỉ EN↔VI (nhẹ nhất, ~600 MB)
python scripts/download_models.py

# Tất cả (EN, ZH, JA, KO ↔ VI, ~3 GB)
python scripts/download_models.py --all

# Tải và chuyển CT2 INT8 ngay
python scripts/download_models.py --all --convert-ct2

# Xem trạng thái
python scripts/download_models.py --status
```

### So sánh backend

| Backend | RAM/model | Tốc độ | Ghi chú |
|---|---|---|---|
| HuggingFace (fp32) | ~300 MB | 1× | Dùng để train/fine-tune |
| CTranslate2 float16 | ~150 MB | 2× | Cần GPU fp16 |
| CTranslate2 int8 | ~80 MB | 3–4× | **Khuyến nghị** cho CPU |

---

## Train Model Riêng (Multilingual)

Đây là cách train **một model MarianMT duy nhất** xử lý tất cả 8 hướng dịch, với auto-detect ngôn ngữ.

### Cơ chế hoạt động

```
Input: >>vi<< Hello, world!   →   Model   →   Xin chào, thế giới!
Input: >>en<< Xin chào        →   Model   →   Hello
Input: >>vi<< 你好世界          →   Model   →   Xin chào thế giới
```

- **Language prefix token** (`>>vi<<`, `>>en<<`, ...) cho model biết ngôn ngữ đích
- **Shared SentencePiece BPE** tokenizer 50K vocab — chung cho cả 5 ngôn ngữ
- **Mixed batches** từ tất cả cặp ngôn ngữ — model học song song
- **Temperature sampling** cân bằng dữ liệu giữa các ngôn ngữ

### Chạy training

```bash
# Training đầy đủ (tải data OPUS + CCMatrix tự động)
python src/multilingual_train.py

# Hoặc qua CLI
python interface/cli.py train-custom

# Tùy chỉnh số lượng data
python interface/cli.py train-custom \
    --en-vi 500000 --zh-vi 200000 --ja-vi 200000 --ko-vi 200000 \
    --epochs 10 --batch-size 16 --vocab-size 50000

# Tiếp tục từ checkpoint
python interface/cli.py train-custom --resume models/custom_vi/checkpoint-5000

# Sau khi train xong, convert sang CT2 INT8 (3-4x nhanh hơn)
python interface/cli.py convert-custom
# hoặc tự động:
python interface/cli.py train-custom --convert-ct2
```

### Thông tin model sau khi train

- **File**: `models/custom_vi/` (HuggingFace format)
- **CT2**: `models/ct2_custom_vi/` (INT8, ~200-300 MB)
- **Tự động dùng**: Khi `models/custom_vi/` tồn tại, engine tự ưu tiên model này

### Auto-detect ngôn ngữ

```bash
# CLI — tự nhận diện
python interface/cli.py translate --src auto --tgt vi --text "Hello"
python interface/cli.py translate --src auto --tgt vi --text "안녕하세요"
python interface/cli.py detect "今日はいい天気ですね"   # → ja

# Python API
from src.translate_engine import detect_language, TranslationEngine

print(detect_language("你好世界"))     # → zh
print(detect_language("안녕하세요"))  # → ko

engine = TranslationEngine()
result = engine.translate("Hello!", src="auto", tgt="vi")
```

Web interface: chọn **"🔍 Tự động nhận diện"** ở dropdown ngôn ngữ nguồn.

### Dữ liệu cần thiết

| Cặp | Nguồn | Số câu tối thiểu |
|---|---|---|
| EN-VI | OPUS-100 (tự tải) | 200K |
| ZH-VI | CCMatrix (tự tải) | 100K |
| JA-VI | CCMatrix (tự tải) | 100K |
| KO-VI | CCMatrix (tự tải) | 100K |

> Nếu CCMatrix không có cặp ZH/JA/KO-VI, script sẽ cảnh báo — bạn có thể cung cấp file TSV riêng.

---

## Fine-Tune Model

Fine-tune giúp cải thiện chất lượng dịch cho **domain cụ thể** (y tế, pháp lý, game...) hoặc phương ngữ tiếng Việt.

### Chuẩn bị dữ liệu

Dữ liệu cần ở dạng **song ngữ (parallel corpus)** — mỗi dòng là một cặp câu tương đương.

**Format TSV** (`data/my_data.tsv`):
```
Hello	Xin chào
Good morning	Chào buổi sáng
Thank you	Cảm ơn
```

**Format 2 file riêng**:
```
# data/en.txt       data/vi.txt
Hello               Xin chào
Good morning        Chào buổi sáng
```

### Chạy fine-tune

```bash
# Fine-tune EN→VI với data tùy chỉnh
python src/finetune.py \
    --pair en-vi \
    --train-tsv data/my_en_vi.tsv \
    --epochs 3 \
    --batch-size 8

# Fine-tune với OPUS-100 corpus (tải tự động)
python src/finetune.py --pair en-vi --use-opus --epochs 3

# Fine-tune với parallel files
python src/finetune.py \
    --pair en-vi \
    --train-src data/en.txt \
    --train-tgt data/vi.txt \
    --epochs 5 --lr 3e-5

# Fine-tune model ZH→EN (cải thiện ZH→VI gián tiếp)
python src/finetune.py \
    --pair zh-en \
    --base-model Helsinki-NLP/opus-mt-zh-en \
    --use-opus --epochs 3

# CLI alternative
python interface/cli.py finetune --pair en-vi --train-tsv data/my.tsv
```

### Tùy chọn fine-tune

| Option | Mặc định | Mô tả |
|---|---|---|
| `--pair` | `en-vi` | Cặp ngôn ngữ |
| `--epochs` | 3 | Số epoch |
| `--batch-size` | 8 | Batch size trên mỗi thiết bị |
| `--lr` | 5e-5 | Learning rate |
| `--max-samples` | 100,000 | Số câu tối đa |
| `--fp16` | False | Mixed precision (cần GPU) |
| `--output-dir` | `models/ft_<pair>` | Thư mục lưu |

### Sau khi fine-tune

```bash
# Tự động dùng fine-tuned model khi có trong models/ft_<pair>/
python interface/app.py

# Chuyển fine-tuned model sang CT2 INT8
python interface/cli.py models convert --pair en-vi
```

---

## Train từ Đầu

Dùng khi không có model pretrained cho cặp ngôn ngữ, hoặc cần model chuyên biệt.

> ⚠️ Cần ít nhất **200,000+ câu** và nhiều giờ training trên CPU.

```bash
python src/train.py \
    --pair en-vi \
    --use-opus \
    --epochs 10 \
    --batch-size 16 \
    --max-samples 500000 \
    --vocab-size 32000

# Với GPU
python src/train.py --pair en-vi --use-opus --fp16
```

---

## Đánh giá

```bash
# FLORES-200 benchmark (chuẩn quốc tế)
python interface/cli.py evaluate --src en --tgt vi --flores

# Với file riêng
python interface/cli.py evaluate \
    --src en --tgt vi \
    --src-file test_en.txt \
    --ref-file test_vi.txt

# Python API
from src.evaluate import evaluate_flores
from src.translate_engine import TranslationEngine

engine = TranslationEngine()
results = evaluate_flores(engine, src_lang="en", tgt_lang="vi")
print(results)
# → {"bleu": 30.5, "chrf": 55.2, "ter": 62.1, "n_sentences": 1012}
```

---

## Cấu trúc dự án

```
MarianNMT/
├── src/
│   ├── config.py           # Đăng ký cặp ngôn ngữ & models
│   ├── translate_engine.py # Engine chính (CT2 + HF fallback, LRU cache)
│   ├── data_utils.py       # Download OPUS, load TSV, build HF datasets
│   ├── model_utils.py      # Download HF, convert CT2, list status
│   ├── finetune.py         # Fine-tuning với Seq2SeqTrainer
│   ├── train.py            # Train từ đầu
│   └── evaluate.py         # BLEU, chrF, TER
│
├── interface/
│   ├── app.py              # Gradio web interface
│   └── cli.py              # CLI (click-based)
│
├── configs/
│   ├── base.yaml           # Config chung
│   ├── finetune_en_vi.yaml # Config fine-tune EN↔VI
│   ├── finetune_zh_vi.yaml # Config fine-tune ZH→VI
│   ├── finetune_ja_vi.yaml # Config fine-tune JA→VI
│   └── finetune_ko_vi.yaml # Config fine-tune KO→VI
│
├── scripts/
│   ├── setup.sh            # Cài đặt tự động
│   ├── download_models.py  # Tải & convert models
│   └── download_data.py    # Tải corpus OPUS/FLORES
│
├── data/
│   └── sample/
│       ├── en_vi.tsv       # Dữ liệu mẫu
│       └── README.md
│
├── models/                 # Models sẽ được tải về đây
│   # hf_en-vi/            # HuggingFace format
│   # ct2_en-vi/           # CTranslate2 INT8
│   # ft_en-vi/            # Fine-tuned model
│
└── requirements.txt
```

---

## Lưu ý & Tips

### Tối ưu RAM (4 GB)

```python
# Giới hạn số models load đồng thời (LRU cache)
engine = TranslationEngine(max_loaded_models=2)  # chỉ giữ 2 models trong RAM

# Dùng CTranslate2 INT8 (ít RAM nhất)
python scripts/download_models.py --convert-ct2 --quantization int8
```

### Tăng tốc độ

```bash
# Tăng batch size khi dịch file
python interface/cli.py translate --src en --tgt vi \
    --file large.txt --batch-size 64 --beam 2
```

### Cải thiện chất lượng ZH/JA/KO → VI

1. **Fine-tune model pivot**: Fine-tune `ZH→EN` trên corpus liên quan, kết quả ZH→VI cải thiện gián tiếp
2. **Chuẩn bị corpus ZH-VI**: Nếu có corpus ZH-VI, bạn có thể train model trực tiếp
3. **Post-processing**: Thêm lớp post-processing cho tiếng Việt (dấu thanh, chính tả)

---

## Giấy phép

- Code: MIT License
- Models Helsinki-NLP: CC-BY 4.0
- OPUS corpus: phụ thuộc nguồn gốc (xem [opus.nlpl.eu](http://opus.nlpl.eu))
