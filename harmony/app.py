import streamlit as st
from spleeter.separator import Separator 
import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from pydub import AudioSegment
import os


# FFmpegのパスをエスケープして設定
os.environ['PATH'] += os.pathsep + "C:\\Users\\kotaf\\ffmpeg\\ffmpeg-7.0.2-essentials_build\\bin"


# ボーカルと伴奏を抽出する関数
def extract_vocals_and_accompaniment(input_file, output_dir):
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(input_file, output_dir, codec='wav')
    
    # ボーカルと伴奏のパスを返す
    vocal_file_path = os.path.join(output_dir, "uploaded_audio", "vocals.wav")
    accompaniment_file_path = os.path.join(output_dir, "uploaded_audio", "accompaniment.wav")
    
    return vocal_file_path, accompaniment_file_path

# Streamlitアプリの構成
st.title("ボーカルと伴奏の抽出")

# ファイルアップロードウィジェット
uploaded_file = st.file_uploader("音声ファイルをアップロードしてください", type=["wav"])

# 出力ディレクトリを設定
output_dir = "./output"

# ボタンが押されたときに処理を実行
if uploaded_file is not None:
    if st.button("ボーカルと伴奏を抽出"):
        # アップロードされたファイルを保存
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ボーカルと伴奏を抽出して保存
        vocal_file_path, accompaniment_file_path = extract_vocals_and_accompaniment("uploaded_audio.wav", output_dir)

        # 抽出されたボーカルと伴奏のファイルをダウンロードできるリンクを表示
        if os.path.exists(vocal_file_path) and os.path.exists(accompaniment_file_path):
            st.success("ボーカルと伴奏が正常に抽出されました！")
            
            # ボーカルファイルのダウンロードリンク
            st.audio(vocal_file_path, format='audio/wav')
            
            # 伴奏ファイルのダウンロードリンク
            st.audio(accompaniment_file_path, format='audio/wav')
            
        else:
            st.error("ファイルが見つかりません。")  


# 遷移確率行列
transition_probabilities_same_pitch = {
    "major_third": {"major_third": 0.7, "minor_third": 0.15, "perfect_fourth": 0.1, "none": 0.05},
    "minor_third": {"major_third": 0.15, "minor_third": 0.7, "perfect_fourth": 0.1, "none": 0.05},
    "perfect_fourth": {"major_third": 0.15, "minor_third": 0.15, "perfect_fourth": 0.6, "none": 0.1},
    "none": {"major_third": 0.4, "minor_third": 0.4, "perfect_fourth": 0.15, "none": 0.05}
}

transition_probabilities_different_pitch = {
    "major_third": {"major_third": 0.35, "minor_third": 0.35, "perfect_fourth": 0.25, "none": 0.05},
    "minor_third": {"major_third": 0.35, "minor_third": 0.35, "perfect_fourth": 0.25, "none": 0.05},
    "perfect_fourth": {"major_third": 0.35, "minor_third": 0.35, "perfect_fourth": 0.25, "none": 0.05},
    "none": {"major_third": 0.3, "minor_third": 0.3, "perfect_fourth": 0.3, "none": 0.1}
}

# ファイルアップロードウィジェット
st.title("ハモリ生成")

uploaded_vocal_file = st.file_uploader("ボーカルトラックをアップロードしてください (WAVファイル)", type="wav")
uploaded_accompaniment_file = st.file_uploader("伴奏トラックをアップロードしてください (WAVファイル)", type="wav")

harmony_direction = st.selectbox("ハモリの方向を選択してください", ["up", "down"])

# ボタンが押されたときに処理を実行
if st.button("ハモリを生成") and uploaded_vocal_file is not None and uploaded_accompaniment_file is not None:
    vocal_path = "uploaded_vocal.wav"
    accompaniment_path = "uploaded_accompaniment.wav"

    # アップロードされたファイルを保存
    with open(vocal_path, "wb") as f:
        f.write(uploaded_vocal_file.getbuffer())
    with open(accompaniment_path, "wb") as f:
        f.write(uploaded_accompaniment_file.getbuffer())

    # ハモリ生成関数
    def load_and_process_tracks(vocal_path, accompaniment_path, sr=48000):
        y_vocal, _ = librosa.load(vocal_path, sr=sr, dtype='float64')
        y_accompaniment, _ = librosa.load(accompaniment_path, sr=sr, dtype='float64')
        chroma = librosa.feature.chroma_stft(y=y_accompaniment, sr=sr, hop_length=int(sr*0.005))
        return y_vocal, chroma

    def extract_pitch(y, sr):
        f0, t = pw.dio(y, sr)
        f0 = pw.stonemask(y, f0, t, sr)
        return f0, t

    def smooth_chromagram(chromagram, window_size=200):
        return uniform_filter1d(chromagram, size=window_size, axis=1)

    def calculate_output_probabilities(chroma_frame):
        total = np.sum(chroma_frame)
        if total == 0:
            return chroma_frame
        return chroma_frame / total

    # ハモリの方向に応じた音の生成
    def choose_harmony_note_viterbi(original_pitch, harmony_candidates, chroma_frame, prev_state, transition_probabilities, harmony_direction):
        if original_pitch == 0:
            return 0, "none"

        chroma_prob = calculate_output_probabilities(chroma_frame)
        max_prob = 0
        selected_harmony_pitch = None
        selected_state = None

        for state in harmony_candidates:
            if state == "none":
                candidate_pitch = 0
            elif state == "major_third":
                if harmony_direction == "up":
                    candidate_pitch = original_pitch * (2 ** (4/12))  # 上方向のハモリ
                else:
                    candidate_pitch = original_pitch * (2 ** (-4/12))  # 下方向のハモリ
            elif state == "minor_third":
                if harmony_direction == "up":
                    candidate_pitch = original_pitch * (2 ** (3/12))  # 上方向のハモリ
                else:
                    candidate_pitch = original_pitch * (2 ** (-3/12))  # 下方向のハモリ
            elif state == "perfect_fourth":
                if harmony_direction == "up":
                    candidate_pitch = original_pitch * (2 ** (5/12))  # 上方向のハモリ
                else:
                    candidate_pitch = original_pitch * (2 ** (-5/12))  # 下方向のハモリ

            if candidate_pitch <= 0:
                continue

            if prev_state in transition_probabilities:
                prob = transition_probabilities[prev_state][state] * chroma_prob[int(np.round(np.log2(candidate_pitch / 440) * 12) + 9) % 12]
                if prob > max_prob:
                    max_prob = prob
                    selected_harmony_pitch = candidate_pitch
                    selected_state = state
            else:
                selected_harmony_pitch = candidate_pitch
                selected_state = state

        if selected_harmony_pitch is None:
            return 0, "none"

        return selected_harmony_pitch, selected_state

    def generate_harmony(y_vocal, sr, chromagram, output_file, harmony_direction='down'):
        f0, t = extract_pitch(y_vocal, sr)
        sp = pw.cheaptrick(y_vocal, f0, t, sr)
        ap = pw.d4c(y_vocal, f0, t, sr)
        smoothed_chromagram = smooth_chromagram(chromagram)

        harmony_pitch = np.copy(f0)
        harmony_choices = []

        prev_state = "none"

        for i in range(len(f0)):
            if f0[i] == 0:
                harmony_pitch[i] = 0
                harmony_choices.append("none")
                continue

            harmony_candidates = ["major_third", "minor_third", "perfect_fourth", "none"]

            if np.sum(smoothed_chromagram[:, i]) == 0:
                harmony_candidates = [note for note in harmony_candidates]

            if i > 0 and f0[i-1] != 0 and (f0[i] / f0[i-1] > 0.99 and f0[i] / f0[i-1] < 1.01):
                selected_note, prev_state = choose_harmony_note_viterbi(f0[i], harmony_candidates, smoothed_chromagram[:, i], prev_state, transition_probabilities_same_pitch, harmony_direction)
            else:
                selected_note, prev_state = choose_harmony_note_viterbi(f0[i], harmony_candidates, smoothed_chromagram[:, i], prev_state, transition_probabilities_different_pitch, harmony_direction)

            harmony_choices.append(prev_state)
            harmony_pitch[i] = selected_note

        harmonic = pw.synthesize(harmony_pitch, sp, ap, sr)
        sf.write(output_file, harmonic, sr)

        return harmony_choices

    # ハモリを生成
    output_file = "generated_harmony.wav"
    y_vocal, chromagram_accompaniment = load_and_process_tracks(vocal_path, accompaniment_path)
    harmony_choices = generate_harmony(y_vocal, 48000, chromagram_accompaniment, output_file, harmony_direction)

    # ハモリ音声を表示
    st.audio(output_file, format='audio/wav')
    st.success("ハモリが生成されました！")


# ファイルアップロードウィジェット
st.title("ハモリ生成と音源の合成")

uploaded_harmony_file = st.file_uploader("生成したハモリをアップロードしてください (WAVファイル)", type="wav", key="harmony_uploader")
uploaded_accompaniment_file = st.file_uploader("伴奏トラックをアップロードしてください (WAVファイル)", type="wav", key="accompaniment_uploader")
uploaded_main_melody_file = st.file_uploader("主旋律をアップロードしてください (WAVファイル)", type="wav", key="main_melody_uploader")

# 音量調整用のスライダー
harmony_volume = st.slider("ハモリの音量を調整", min_value=-30, max_value=30, value=0, key="harmony_volume_slider")
accompaniment_volume = st.slider("伴奏の音量を調整", min_value=-30, max_value=30, value=0, key="accompaniment_volume_slider")
main_melody_volume = st.slider("主旋律の音量を調整", min_value=-30, max_value=30, value=0, key="main_melody_volume_slider")



# 主旋律、伴奏、ハモリを合成する部分
if uploaded_harmony_file and uploaded_accompaniment_file and uploaded_main_melody_file and st.button("主旋律、伴奏、ハモリを合成", key="combine_harmony_melody_accompaniment"):
    # アップロードされたファイルを保存
    harmony_path = "uploaded_harmony.wav"
    accompaniment_path = "uploaded_accompaniment.wav"
    main_melody_path = "uploaded_main_melody.wav"
    
    with open(harmony_path, "wb") as f:
        f.write(uploaded_harmony_file.getbuffer())
    with open(accompaniment_path, "wb") as f:
        f.write(uploaded_accompaniment_file.getbuffer())
    with open(main_melody_path, "wb") as f:
        f.write(uploaded_main_melody_file.getbuffer())
    
    harmony_audio = AudioSegment.from_file(harmony_path)
    accompaniment_audio = AudioSegment.from_file(accompaniment_path)
    main_melody_audio = AudioSegment.from_file(main_melody_path)
    
    # 音量調整
    harmony_audio = harmony_audio + harmony_volume
    accompaniment_audio = accompaniment_audio + accompaniment_volume
    main_melody_audio = main_melody_audio + main_melody_volume
    
    # 合成（主旋律、伴奏、ハモリ）
    combined_audio = main_melody_audio.overlay(accompaniment_audio).overlay(harmony_audio)
    combined_audio.export("combined_song_harmony_accompaniment.wav", format="wav")
    
    # 合成音声の再生とダウンロード
    st.audio("combined_song_harmony_accompaniment.wav", format="audio/wav")
    st.success("主旋律、伴奏、ハモリが合成されました！")
    st.download_button(
        label="合成された主旋律、伴奏、ハモリをダウンロード",
        data=open("combined_song_harmony_accompaniment.wav", "rb").read(),
        file_name="combined_song_harmony_accompaniment.wav",
        mime="audio/wav"
    )
