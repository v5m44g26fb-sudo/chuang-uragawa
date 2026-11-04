#!/usr/bin/env python3
# v13 メディア追加フェーズ第1便：場面BGM 2曲＋効果音3種の生成スクリプト。
# すべてシンセサイズ生成のみのオリジナル音源（著作権のある楽曲は使っていない）。
# BGMの仕様は姉妹作・日プ編の生成BGMにそろえる：24.000秒ループ／44.1kHz／モノラル／
# MP3 128kbps、ピーク約 -5.6dB（0.52）。ループの継ぎ目は「1周分のイベント列を
# 周期的に敷き詰めてから [0,24) を切り出す」方式でサンプル単位で連続させる。
# 効果音は短い合成音（送信／受信／確定）。長さは姉妹作の実測（send 0.14／recv 0.33／
# done 0.62秒）に寄せた。
import numpy as np, subprocess, os, math, wave as wavmod

SR = 44100
LOOP = 24.0
BASE = os.path.dirname(os.path.abspath(__file__))
MUSIC_OUT = os.path.join(BASE, "..", "assets", "music")
SE_OUT = os.path.join(BASE, "..", "assets", "se")

def f(midi): return 440.0 * 2 ** ((midi - 69) / 12.0)

def render(events, total=LOOP):
    """events: (start_in_loop, dur, freq, gain, kind) の列。周期敷き詰めして [0,total) を返す。"""
    buf_len = int((total + 12) * SR)
    buf = np.zeros(buf_len, dtype=np.float64)
    off = 6.0
    for (s, d, fr, g, kind) in events:
        for shift in (-LOOP, 0.0, LOOP):
            start = s + shift
            if start + d < -off or start > total + 6: continue
            i0 = max(0, int((start + off) * SR)); i1 = min(buf_len, int((start + off + d) * SR))
            if i1 <= i0: continue
            tabs = np.arange(i0, i1) / SR - off
            t = tabs - start
            ph = 2 * np.pi * fr * tabs
            if kind == "pad":
                wave = np.sin(ph) + 0.45 * np.sin(2 * ph) + 0.18 * np.sin(3 * ph)
                a, r = 1.3, 2.1
                env = np.minimum(0.5 - 0.5 * np.cos(np.pi * np.clip(t / a, 0, 1)),
                                 0.5 + 0.5 * np.cos(np.pi * np.clip((t - (d - r)) / r, 0, 1)))
            elif kind == "pluck":
                wave = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.10 * np.sin(3 * ph)
                env = np.exp(-t / 0.38) * np.minimum(1.0, t / 0.006)
            elif kind == "sparkle":
                wave = np.sin(ph) + 0.20 * np.sin(2 * ph)
                env = np.exp(-t / 0.85) * np.minimum(1.0, t / 0.008)
            elif kind == "bass":
                wave = np.sin(ph) + 0.22 * np.sin(2 * ph)
                env = np.minimum(1.0, t / 0.03) * np.exp(-t / (d * 0.9)) * 0.9 + np.minimum(1.0, t / 0.03) * 0.1 * np.exp(-t / d)
            else:
                wave = np.sin(ph); env = np.exp(-t / 0.3)
            buf[i0:i1] += g * wave * env
    a = int(off * SR); b = a + int(total * SR)
    return buf[a:b]

def vel(seed):
    return 0.85 + 0.3 * ((math.sin(seed * 12.9898) * 43758.5453) % 1.0)

def build(chords, arp_step, arp_pattern, bass_notes, melody, sparkle, pad_gain=0.16, arp_gain=0.20):
    ev = []
    for i, ch in enumerate(chords):
        s0 = i * 6.0
        for m in ch:
            ev.append((s0, 6.0, f(m), pad_gain / len(ch) * 2.2, "pad"))
        tones = ch + [ch[0] + 12]
        for j, dt in enumerate(arp_pattern):
            m = tones[j % len(tones)]
            ev.append((s0 + dt, 1.2, f(m), arp_gain * vel(i * 31 + j), "pluck"))
        for (dt, m, d) in bass_notes[i]:
            ev.append((s0 + dt, d, f(m), 0.30, "bass"))
    for (t, m, d) in melody:
        ev.append((t, d, f(m), 0.16, "pluck"))
    for (t, m, d) in sparkle:
        ev.append((t, d, f(m), 0.10, "sparkle"))
    return ev

def save_mp3(samples, path, peak=0.52):
    x = samples.astype(np.float64)
    pk = np.max(np.abs(x))
    if pk > 0: x = x * (peak / pk)
    wav_path = "/tmp/_gen_chuang.wav"
    with wavmod.open(wav_path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "128k", path], check=True)

# ---- BGM 1: 三亜の風（日常・昼の場面）Cmaj7-Am7-Fmaj7-G6、明るく穏やか ----
sanya = build(
    chords=[[60, 64, 67, 71], [57, 60, 64, 67], [53, 57, 60, 64], [55, 59, 62, 64]],
    arp_step=0.75,
    arp_pattern=[0.0, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25],
    bass_notes=[[(0.0, 36, 2.8), (3.0, 43, 2.8)], [(0.0, 33, 2.8), (3.0, 40, 2.8)],
                [(0.0, 29, 2.8), (3.0, 36, 2.8)], [(0.0, 31, 2.8), (3.0, 38, 2.8)]],
    melody=[(1.5, 76, 1.0), (4.5, 74, 1.0), (7.5, 72, 1.0), (10.5, 76, 1.0),
            (13.5, 79, 1.2), (16.5, 76, 1.0), (19.5, 74, 1.0), (22.5, 71, 1.0)],
    sparkle=[(5.6, 88, 0.8), (17.6, 86, 0.8)],
    pad_gain=0.15, arp_gain=0.17)
save_mp3(render(sanya), os.path.join(MUSIC_OUT, "bgm-sanya.mp3"))
save_mp3(render(sanya), os.path.join(BASE, "bgm-sanya.mp3"))

# ---- BGM 2: 合宿の夜（夜・静かな場面）Am7-Fmaj7-Cmaj7-Em7、疎らで静か ----
yoru = build(
    chords=[[57, 60, 64, 67], [53, 57, 60, 64], [48, 52, 55, 59], [52, 55, 59, 62]],
    arp_step=1.5,
    arp_pattern=[0.0, 1.5, 3.0, 4.5],
    bass_notes=[[(0.0, 33, 4.0)], [(0.0, 29, 4.0)], [(0.0, 24, 4.0)], [(0.0, 28, 4.0)]],
    melody=[(3.0, 69, 1.6), (9.0, 67, 1.6), (15.0, 64, 1.6), (21.0, 67, 1.6)],
    sparkle=[(11.6, 81, 1.0), (23.6, 79, 1.0)],
    pad_gain=0.13, arp_gain=0.11)
save_mp3(render(yoru), os.path.join(MUSIC_OUT, "bgm-yoru.mp3"))
save_mp3(render(yoru), os.path.join(BASE, "bgm-yoru.mp3"))

# ---- 効果音 ----
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
    wav_path = "/tmp/_gen_chuang_se.wav"
    with wavmod.open(wav_path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "128k", path], check=True)

# 送信：短い上昇ブリップ 0.15秒
save_se(tone(660, 990, 0.15), os.path.join(SE_OUT, "se-send.mp3"))
# 受信：2音の柔らかいチャイム 0.35秒
recv = np.concatenate([tone(1046, 1046, 0.18), tone(1568, 1568, 0.17) * 0.85])
save_se(recv, os.path.join(SE_OUT, "se-recv.mp3"))
# 確定：2音のベルの確定音 0.65秒
done = np.concatenate([tone(880, 880, 0.22, kind="bell"), tone(1318, 1318, 0.43, kind="bell") * 0.9])
save_se(done, os.path.join(SE_OUT, "se-done.mp3"))
print("done")
