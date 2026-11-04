#!/usr/bin/env python3
# v14 メディア追加フェーズ第2便：通話開始時の着信系SE（se-call）の生成スクリプト。
# 姉妹作・ボイプラ編 v4.67 の se-call（1.05秒）と同じ作法・同じ尺で、本作の合成生成のみ。
# ベル音2打（1318Hz）を 0.5秒間隔で並べた短い呼び出し音。仕様は v13 の効果音と同じ：
# 44.1kHz／モノラル／MP3 128kbps、ピーク 0.6 正規化。
import numpy as np, subprocess, os, wave as wavmod

SR = 44100
BASE = os.path.dirname(os.path.abspath(__file__))
SE_OUT = os.path.join(BASE, "..", "assets", "se")

def tone(freq_a, freq_b, dur, kind="blip", gain=0.5):
    n = int(dur * SR); t = np.arange(n) / SR
    fr = freq_a + (freq_b - freq_a) * (t / dur)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    if kind == "bell":
        wave = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
        env = np.exp(-t / (dur * 0.45)) * np.minimum(1.0, t / 0.005)
    else:
        wave = np.sin(ph) + 0.25 * np.sin(2 * ph)
        env = np.exp(-t / (dur * 0.4)) * np.minimum(1.0, t / 0.004)
    return gain * wave * env

def save_se(samples, path):
    x = samples.astype(np.float64)
    pk = np.max(np.abs(x))
    if pk > 0: x = x * (0.6 / pk)
    wav_path = "/tmp/_gen_chuang_se_call.wav"
    with wavmod.open(wav_path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "128k", path], check=True)

# 着信：ベルの2打（0.52秒＋0.53秒＝1.05秒）
call = np.concatenate([tone(1318, 1318, 0.52, kind="bell"), tone(1318, 1318, 0.53, kind="bell") * 0.92])
save_se(call, os.path.join(SE_OUT, "se-call.mp3"))
print("done")
