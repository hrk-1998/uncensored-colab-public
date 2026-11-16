"""Local chat UI. The Colab notebook launches this after the model download."""

import re
from threading import Lock

import gradio as gr
from llama_cpp import Llama

EFFORTS = {
    "کم": {"max_tokens": 384, "temperature": 0.35, "think": False},
    "معمولی": {"max_tokens": 800, "temperature": 0.6, "think": False},
    "زیاد": {"max_tokens": 1500, "temperature": 0.7, "think": True},
    "حداکثر": {"max_tokens": 2600, "temperature": 0.85, "think": True},
}

DEFAULT_SYSTEM = (
    "تو یک دستیار محلی و مستقیم هستی. به زبان کاربر جواب بده، کوتاه و مشخص، بدون موعظه. "
    "محتوای جنسی مربوط به افراد زیر ۱۸ سال نساز. "
    "برای حمله خشونت‌آمیز، مواد منفجره یا کلاهبرداری راهنمای اجرایی نده. "
    "برای بقیه موضوع‌ها، از جمله سؤال‌های حساس و بزرگسالان، واضح جواب بده. "
    "اگر نتیجه جستجو داده شد، فقط بخش مرتبط را استفاده کن و منبع را بنویس."
)

_llm = None
_lock = Lock()


def load_model(model_path: str) -> str:
    global _llm
    _llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=8192,
        n_batch=512,
        verbose=False,
    )
    return "مدل آماده است."


def _search(query: str, n: int) -> str:
    from ddgs import DDGS

    hits = DDGS().text(query[:240], max_results=int(n)) or []
    lines = []
    for i, hit in enumerate(hits, 1):
        title = hit.get("title") or ""
        body = hit.get("body") or ""
        url = hit.get("href") or hit.get("url") or ""
        lines.append(f"[{i}] {title}\n{body}\n{url}")
    return "\n\n".join(lines) if lines else "نتیجه‌ای پیدا نشد."


def _strip_think(text: str, show: bool) -> str:
    if show:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def reply(message, history, effort, deep, n_results, system, show_think):
    text = (message or "").strip()
    if not text:
        return history or [], ""
    if _llm is None:
        raise gr.Error("مدل هنوز بارگذاری نشده.")
    cfg = EFFORTS.get(effort, EFFORTS["معمولی"])
    user = text
    if deep:
        try:
            found = _search(text, n_results)
        except Exception as exc:
            found = f"جستجو ناموفق بود: {exc}"
        user = f"نتایج DeepSearch:\n{found}\n\nسؤال کاربر:\n{text}"
    user += "\n/think" if cfg["think"] else "\n/no_think"
    messages = [{"role": "system", "content": (system or DEFAULT_SYSTEM).strip()}]
    for turn in history or []:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": str(turn.get("content") or "")})
    messages.append({"role": "user", "content": user})
    with _lock:
        out = _llm.create_chat_completion(
            messages=messages,
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            top_p=0.9,
            repeat_penalty=1.05,
        )
    answer = _strip_think(out["choices"][0]["message"]["content"] or "", show_think)
    updated = list(history or []) + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": answer or "پاسخی تولید نشد."},
    ]
    return updated, ""


def build(model_path: str):
    status = load_model(model_path)
    with gr.Blocks(title="چت محلی") as demo:
        gr.Markdown(
            "## چت محلی\n"
            "مدل روی همین کولب اجرا می‌شود. **ایفورت** طول و عمق جواب را عوض می‌کند. "
            "**DeepSearch** قبل از جواب در وب جستجو می‌کند."
        )
        gr.Markdown(status)
        try:
            chat = gr.Chatbot(height=480, type="messages", label="گفتگو")
        except TypeError:
            chat = gr.Chatbot(height=480, label="گفتگو")
        message = gr.Textbox(label="پیام", lines=3, autofocus=True)
        with gr.Row():
            effort = gr.Radio(list(EFFORTS), value="معمولی", label="ایفورت")
            deep = gr.Checkbox(value=False, label="DeepSearch")
            n_results = gr.Slider(3, 8, value=5, step=1, label="تعداد نتیجه جستجو")
            show_think = gr.Checkbox(value=False, label="نمایش فکر مدل")
        system = gr.Textbox(label="دستور سیستم", value=DEFAULT_SYSTEM, lines=4)
        with gr.Row():
            send = gr.Button("ارسال", variant="primary")
            clear = gr.Button("پاک کردن")
        send.click(
            reply,
            [message, chat, effort, deep, n_results, system, show_think],
            [chat, message],
        )
        message.submit(
            reply,
            [message, chat, effort, deep, n_results, system, show_think],
            [chat, message],
        )
        clear.click(lambda: ([], ""), outputs=[chat, message])
    return demo
