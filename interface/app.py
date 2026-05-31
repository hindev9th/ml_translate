"""
Gradio web interface for the Vietnamese translation system.
Supports EN, ZH, JA, KO ↔ VI (bidirectional).
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import LANG_NAMES, LANG_FLAGS, LANGUAGE_PAIRS
from src.translate_engine import TranslationEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Translations engine (singleton, lazy-loaded)
# ---------------------------------------------------------------------------

_engine: TranslationEngine = None


def get_engine() -> TranslationEngine:
    global _engine
    if _engine is None:
        _engine = TranslationEngine(device="cpu", max_loaded_models=3)
    return _engine


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------

SUPPORTED_LANGS = ["auto", "en", "vi", "zh", "ja", "ko"]
SRC_LANGS = SUPPORTED_LANGS           # source can be auto
TGT_LANGS = ["en", "vi", "zh", "ja", "ko"]  # target must be explicit

def lang_label(code: str) -> str:
    if code == "auto":
        return "🔍 Tự động nhận diện"
    flag = LANG_FLAGS.get(code, "")
    name = LANG_NAMES.get(code, code.upper())
    return f"{flag} {name}"


def code_from_label(label: str) -> str:
    if "Tự động" in label or "auto" in label:
        return "auto"
    for code in TGT_LANGS:
        if LANG_NAMES[code] in label or code in label:
            return code
    return "en"


def valid_pair(src_code: str, tgt_code: str) -> bool:
    return f"{src_code}-{tgt_code}" in LANGUAGE_PAIRS


# ---------------------------------------------------------------------------
# Core translate function
# ---------------------------------------------------------------------------

def translate_text(text: str, src_label: str, tgt_label: str, beam_size: int) -> tuple:
    """Called by Gradio; returns (translation, status_message)."""
    if not text.strip():
        return "", "⚠️ Vui lòng nhập văn bản cần dịch."

    src = code_from_label(src_label)
    tgt = code_from_label(tgt_label)

    if src == tgt:
        return text, "⚠️ Ngôn ngữ nguồn và đích giống nhau."

    if not valid_pair(src, tgt):
        return "", f"❌ Cặp ngôn ngữ {src}→{tgt} chưa được hỗ trợ."

    try:
        from src.translate_engine import detect_language, CUSTOM_MODEL_DIR
        import os

        engine = get_engine()

        # Resolve auto-detect
        detected_src = src
        if src == "auto":
            detected_src = detect_language(text.strip())

        result = engine.translate(text.strip(), src=detected_src, tgt=tgt, beam_size=beam_size)

        # Build status message
        if os.path.isdir(CUSTOM_MODEL_DIR):
            mode = "custom multilingual model"
        else:
            cfg_pair = LANGUAGE_PAIRS.get(f"{detected_src}-{tgt}")
            mode = "pivot (EN)" if (cfg_pair and cfg_pair.use_pivot) else "direct"

        detect_info = f" [detected: {lang_label(detected_src)}]" if src == "auto" else ""
        status = f"✅ {lang_label(detected_src)} → {lang_label(tgt)} [{mode}]{detect_info}"
        return result, status
    except RuntimeError as e:
        if "No model found" in str(e):
            return "", (
                f"❌ Chưa tải model cho {src}→{tgt}.\n"
                "Chạy: python scripts/download_models.py"
            )
        raise e
    except Exception as e:
        logger.exception(e)
        return "", f"❌ Lỗi: {e}"


def translate_file(file_obj, src_label: str, tgt_label: str, beam_size: int):
    """Translate a plain-text file line by line."""
    if file_obj is None:
        return None, "⚠️ Chưa chọn file."

    src = code_from_label(src_label)
    tgt = code_from_label(tgt_label)

    try:
        with open(file_obj.name, encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f]

        engine = get_engine()
        non_empty = [l for l in lines if l.strip()]
        translations = engine.translate_batch(non_empty, src=src, tgt=tgt, beam_size=beam_size)

        # Re-insert empty lines
        result_lines = []
        ti = 0
        for l in lines:
            if l.strip():
                result_lines.append(translations[ti])
                ti += 1
            else:
                result_lines.append("")

        out_path = file_obj.name + f".{tgt}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(result_lines))

        return out_path, f"✅ Đã dịch {len(non_empty)} dòng."
    except Exception as e:
        logger.exception(e)
        return None, f"❌ Lỗi: {e}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLES = [
    ["Hello, how are you today?", "🇬🇧 English", "🇻🇳 Tiếng Việt"],
    ["今日はとても良い天気ですね。", "🇯🇵 日本語", "🇻🇳 Tiếng Việt"],
    ["안녕하세요, 반갑습니다.", "🇰🇷 한국어", "🇻🇳 Tiếng Việt"],
    ["你好，很高兴认识你。", "🇨🇳 中文", "🇻🇳 Tiếng Việt"],
    ["Xin chào, tôi đến từ Việt Nam.", "🇻🇳 Tiếng Việt", "🇬🇧 English"],
    ["Việt Nam là một đất nước xinh đẹp.", "🇻🇳 Tiếng Việt", "🇯🇵 日本語"],
]

CSS = """
:root {
    --primary: #dc2626;
    --secondary: #facc15;
    --bg: #0f172a;
    --surface: #1e293b;
    --surface2: #334155;
    --text: #f1f5f9;
    --muted: #94a3b8;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

.header-box {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--surface2);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    text-align: center;
}

.header-box h1 {
    color: var(--primary);
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}

.header-box p {
    color: var(--muted);
    margin: 0;
    font-size: 1rem;
}

.translate-panel {
    background: var(--surface);
    border: 1px solid var(--surface2);
    border-radius: 12px;
    padding: 20px;
}

.status-bar {
    font-size: 0.85rem;
    color: var(--muted);
    padding: 8px 12px;
    background: var(--surface2);
    border-radius: 8px;
    margin-top: 8px;
}

label, .label-wrap {
    color: var(--text) !important;
    font-weight: 600 !important;
}

textarea {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}

button.primary {
    background: var(--primary) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}

.swap-btn {
    font-size: 1.4rem !important;
    background: var(--surface2) !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    color: var(--secondary) !important;
}

.gr-examples {
    background: var(--surface) !important;
    border-radius: 12px !important;
}

footer { display: none !important; }
"""


def build_app():
    import gradio as gr

    lang_choices = [lang_label(c) for c in SUPPORTED_LANGS]

    with gr.Blocks(css=CSS, title="ViTrans — Máy Dịch Việt") as demo:

        # ---- Header ----
        gr.HTML("""
        <div class="header-box">
            <h1>🇻🇳 ViTrans</h1>
            <p>Dịch thuật Anh · Trung · Hàn · Nhật ↔ Tiếng Việt | Powered by MarianNMT</p>
        </div>
        """)

        with gr.Tabs():

            # ================================================================
            # Tab 1: Text translation
            # ================================================================
            with gr.Tab("📝 Dịch Văn Bản"):

                with gr.Row():
                    src_lang = gr.Dropdown(
                        choices=[lang_label(c) for c in SRC_LANGS],
                        value=lang_label("auto"),
                        label="Ngôn ngữ nguồn",
                        scale=4,
                    )
                    swap_btn = gr.Button("⇄", elem_classes=["swap-btn"], scale=1, min_width=60)
                    tgt_lang = gr.Dropdown(
                        choices=[lang_label(c) for c in TGT_LANGS],
                        value=lang_label("vi"),
                        label="Ngôn ngữ đích",
                        scale=4,
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        src_text = gr.Textbox(
                            label="Văn bản gốc",
                            placeholder="Nhập văn bản cần dịch...",
                            lines=8,
                            max_lines=20,
                        )
                        char_count = gr.Markdown("0 ký tự", elem_classes=["status-bar"])

                    with gr.Column(scale=1):
                        tgt_text = gr.Textbox(
                            label="Bản dịch",
                            placeholder="Bản dịch sẽ hiện ở đây...",
                            lines=8,
                            max_lines=20,
                            interactive=False,
                        )
                        status_msg = gr.Markdown("", elem_classes=["status-bar"])

                with gr.Row():
                    beam_slider = gr.Slider(
                        minimum=1, maximum=8, value=4, step=1,
                        label="Beam size (chất lượng ↑ = tốc độ ↓)",
                        scale=3,
                    )
                    translate_btn = gr.Button(
                        "🔁 Dịch", variant="primary", scale=2, size="lg"
                    )
                    clear_btn = gr.Button("🗑️ Xóa", scale=1)

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[src_text, src_lang, tgt_lang],
                    label="Ví dụ nhanh",
                )

                # ---- Events ----
                def update_char_count(text):
                    return f"{len(text):,} ký tự"

                def do_swap(src_label, tgt_label, src_t, tgt_t):
                    return tgt_label, src_label, tgt_t, src_t

                src_text.change(update_char_count, inputs=src_text, outputs=char_count)

                translate_btn.click(
                    fn=translate_text,
                    inputs=[src_text, src_lang, tgt_lang, beam_slider],
                    outputs=[tgt_text, status_msg],
                )

                src_text.submit(
                    fn=translate_text,
                    inputs=[src_text, src_lang, tgt_lang, beam_slider],
                    outputs=[tgt_text, status_msg],
                )

                swap_btn.click(
                    fn=do_swap,
                    inputs=[src_lang, tgt_lang, src_text, tgt_text],
                    outputs=[src_lang, tgt_lang, src_text, tgt_text],
                )

                clear_btn.click(
                    fn=lambda: ("", "", ""),
                    outputs=[src_text, tgt_text, status_msg],
                )

            # ================================================================
            # Tab 2: File translation
            # ================================================================
            with gr.Tab("📄 Dịch File"):
                gr.Markdown("### Dịch file văn bản (.txt, .csv) theo từng dòng")

                with gr.Row():
                    file_src_lang = gr.Dropdown(
                        choices=lang_choices, value=lang_label("en"),
                        label="Ngôn ngữ nguồn", scale=4
                    )
                    file_tgt_lang = gr.Dropdown(
                        choices=lang_choices, value=lang_label("vi"),
                        label="Ngôn ngữ đích", scale=4
                    )
                    file_beam = gr.Slider(1, 8, value=4, step=1, label="Beam", scale=2)

                file_upload = gr.File(label="Tải file lên (.txt)", file_types=[".txt"])
                file_translate_btn = gr.Button("🔁 Dịch File", variant="primary")
                file_download = gr.File(label="Tải xuống bản dịch", visible=False)
                file_status = gr.Markdown("")

                def handle_file_translation(f, sl, tl, beam):
                    out_path, status = translate_file(f, sl, tl, beam)
                    if out_path:
                        return gr.update(value=out_path, visible=True), status
                    return gr.update(visible=False), status

                file_translate_btn.click(
                    fn=handle_file_translation,
                    inputs=[file_upload, file_src_lang, file_tgt_lang, file_beam],
                    outputs=[file_download, file_status],
                )

            # ================================================================
            # Tab 3: Model info
            # ================================================================
            with gr.Tab("ℹ️ Thông Tin Model"):
                gr.Markdown("""
### Các cặp ngôn ngữ hỗ trợ

| Hướng dịch | Model | Phương pháp |
|---|---|---|
| 🇬🇧 EN → 🇻🇳 VI | `Helsinki-NLP/opus-mt-en-vi` | Trực tiếp |
| 🇻🇳 VI → 🇬🇧 EN | `Helsinki-NLP/opus-mt-vi-en` | Trực tiếp |
| 🇨🇳 ZH → 🇻🇳 VI | `opus-mt-zh-en` + `opus-mt-en-vi` | Pivot (EN) |
| 🇻🇳 VI → 🇨🇳 ZH | `opus-mt-vi-en` + `opus-mt-en-zh` | Pivot (EN) |
| 🇯🇵 JA → 🇻🇳 VI | `opus-mt-jap-en` + `opus-mt-en-vi` | Pivot (EN) |
| 🇻🇳 VI → 🇯🇵 JA | `opus-mt-vi-en` + `opus-mt-en-jap` | Pivot (EN) |
| 🇰🇷 KO → 🇻🇳 VI | `opus-mt-ko-en` + `opus-mt-en-vi` | Pivot (EN) |
| 🇻🇳 VI → 🇰🇷 KO | `opus-mt-vi-en` + `opus-mt-en-ko` | Pivot (EN) |

### Yêu cầu hệ thống

- **RAM tối thiểu:** 2 GB (CPU inference với INT8 quantization)
- **RAM khuyến nghị:** 4 GB
- **GPU:** Tùy chọn — tự động dùng CUDA nếu có

### Cài đặt model

```bash
# Tải tất cả model (khuyến nghị)
python scripts/download_models.py --all

# Chuyển sang CTranslate2 INT8 (nhanh hơn 3-4x, nhẹ hơn 50%)
python scripts/download_models.py --convert-ct2
```
                """)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--warmup", action="store_true", help="Pre-load EN↔VI models")
    args = parser.parse_args()

    if args.warmup:
        logger.info("Warming up EN↔VI models ...")
        get_engine().warmup(["en-vi", "vi-en"])

    demo = build_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
