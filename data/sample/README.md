# Sample Data

Place your parallel corpus files here.

## Format 1: TSV (Tab-Separated Values)

One sentence pair per line: `source_text<TAB>target_text`

```
Hello, how are you?	Xin chào, bạn khỏe không?
The weather is nice today.	Hôm nay thời tiết đẹp.
```

## Format 2: Parallel Text Files

Two aligned files, one sentence per line:

**en.txt:**
```
Hello, how are you?
The weather is nice today.
```

**vi.txt:**
```
Xin chào, bạn khỏe không?
Hôm nay thời tiết đẹp.
```

## Usage

```bash
# Fine-tune with TSV
python src/finetune.py --pair en-vi --train-tsv data/sample/en_vi.tsv

# Fine-tune with parallel files
python src/finetune.py --pair en-vi --train-src data/sample/en.txt --train-tgt data/sample/vi.txt

# Download OPUS-100 corpus automatically
python src/finetune.py --pair en-vi --use-opus
```
