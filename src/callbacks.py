"""
Custom training callbacks.

TranslationSampleCallback:
  - Chạy sau mỗi evaluation step
  - Dịch một tập câu mẫu cố định bằng model hiện tại
  - Ghi kết quả dạng bảng Markdown vào TensorBoard tab "Text"
  - Giúp theo dõi chất lượng dịch qua từng checkpoint bằng mắt thường
"""

import os
import logging
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl

logger = logging.getLogger(__name__)

# Câu mẫu cố định để so sánh qua các epoch
# Format: (src_text, reference_vi, target_lang_tag)
DEFAULT_SAMPLES = [
    # EN → VI
    ("Hello, how are you?",                "Xin chào, bạn có khỏe không?",    ">>vi<<"),
    ("The weather is nice today.",          "Hôm nay thời tiết đẹp.",           ">>vi<<"),
    ("I love learning new languages.",      "Tôi thích học ngôn ngữ mới.",      ">>vi<<"),
    ("Thank you very much for your help.",  "Cảm ơn bạn rất nhiều vì sự giúp đỡ.", ">>vi<<"),

    # ZH → VI
    ("你好，很高兴认识你。",                 "Xin chào, rất vui được gặp bạn.",  ">>vi<<"),
    ("今天天气很好。",                       "Hôm nay thời tiết rất đẹp.",       ">>vi<<"),

    # JA → VI
    ("今日はとても良い天気ですね。",          "Hôm nay thời tiết rất đẹp.",       ">>vi<<"),
    ("ありがとうございます。",               "Cảm ơn bạn rất nhiều.",            ">>vi<<"),

    # KO → VI
    ("안녕하세요, 잘 지내세요?",             "Xin chào, bạn có khỏe không?",    ">>vi<<"),
    ("감사합니다.",                          "Cảm ơn bạn.",                      ">>vi<<"),

    # VI → EN
    ("Xin chào, tôi đến từ Việt Nam.",      "Hello, I am from Vietnam.",        ">>en<<"),
    ("Hôm nay trời đẹp lắm.",               "The weather is very nice today.",  ">>en<<"),
]


class TranslationSampleCallback(TrainerCallback):
    """
    Ghi bảng dịch mẫu vào TensorBoard sau mỗi eval step.

    TensorBoard → tab "Text" → "translations/samples"
    Mỗi row: | Nguồn | Tham chiếu (ref) | Bản dịch model |
    """

    def __init__(
        self,
        tokenizer,
        sample_pairs: list = None,
        beam_size: int = 4,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.sample_pairs = sample_pairs or DEFAULT_SAMPLES
        self.beam_size = beam_size
        self.max_length = max_length
        self._writer = None

    def _get_writer(self, args):
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.environ.get(
                "TENSORBOARD_LOGGING_DIR",
                os.path.join(args.output_dir, "runs"),
            )
            self._writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard writer: {log_dir}")
        return self._writer

    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if model is None:
            return

        writer = self._get_writer(args)
        device = next(model.parameters()).device
        model.eval()

        header = (
            "| # | Nguồn | Tham chiếu | Bản dịch (model) | OK? |\n"
            "|---|---|---|---|---|\n"
        )
        rows = []

        for i, (src, ref, tgt_tag) in enumerate(self.sample_pairs, 1):
            tagged = f"{tgt_tag} {src}"
            try:
                inputs = self.tokenizer(
                    tagged,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                ).to(device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        num_beams=self.beam_size,
                        max_length=self.max_length,
                    )
                translation = self.tokenizer.decode(out[0], skip_special_tokens=True)
            except Exception as e:
                translation = f"[ERROR: {e}]"

            # Đánh dấu chất lượng sơ bộ
            if not translation or translation == src:
                flag = "❌"  # rỗng hoặc copy nguyên văn
            elif len(set(translation.split()) & set(src.split())) / max(len(src.split()), 1) > 0.7:
                flag = "⚠️"  # quá nhiều từ giống nguồn (không dịch được)
            elif len(translation.split()) < 2:
                flag = "⚠️"  # quá ngắn
            else:
                flag = "✅"
            rows.append(
                f"| {i} | {src} | {ref} | {translation} | {flag} |"
            )

        table = header + "\n".join(rows)
        writer.add_text("translations/samples", table, global_step=state.global_step)
        writer.flush()
        logger.info(f"[TensorBoard] Logged {len(rows)} sample translations at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        if self._writer:
            self._writer.close()
            self._writer = None
