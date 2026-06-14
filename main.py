import base64
import gradio as gr
import requests

from openai import OpenAI

def to_history_display(messages):
    history_display = []
    for m in messages:
        if m["role"] == "user":
            history_display.append({"role": "user", "content": "(音声入力)"})
        else:
            history_display.append(m)
    return history_display

def process_audio(audio_path, history):
    messages = history.copy()
    if not audio_path:
        history_display = to_history_display(messages)
        return "speech.wav", messages, history_display, None
    with open(audio_path, "rb") as f:
        b64_audio = base64.b64encode(f.read()).decode("utf-8")

    client = OpenAI(base_url="http://localhost:8800/v1")
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": b64_audio, "format": "wav"}
            }
        ]
    })
    chat_res = client.chat.completions.create(
        model="gemma-4-E4B-it.gguf",
        messages=[
            {
                "role": "system",
                "content": "100 tokens以下で簡潔に応答して",
            }
        ] + messages,
        extra_body={"enable_thinking": False},
    )

    reply_text = chat_res.choices[0].message.content
    messages.append({"role": "assistant", "content": reply_text})

    payload = {
        "model": "irodori-tts",
        "input": reply_text,
        "voice": "none",
        "response_format": "wav",
        "irodori": {
            "num_steps": 8,
            "t_schedule_mode": "sway",
        },
    }

    res = requests.post(
        "http://localhost:8088/v1/audio/speech",
        json=payload
    )

    with open("speech.wav", "wb") as f:
        f.write(res.content)

    history_display = to_history_display(messages)

    return "speech.wav", messages, history_display, None

with gr.Blocks() as demo:
    gr.Markdown("音声チャットデモ")

    state = gr.State([])
    chatbot = gr.Chatbot()
    mic = gr.Audio(sources=["microphone"], type="filepath", label="マイク入力")
    out_audio = gr.Audio(label="AI 音声出力", autoplay=True)

    mic.change(process_audio, inputs=[mic, state], outputs=[out_audio, state, chatbot, mic])

if __name__ == "__main__":
    demo.launch()
