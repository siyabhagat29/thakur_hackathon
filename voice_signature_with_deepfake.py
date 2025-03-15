import random
import speech_recognition as sr
import librosa
import numpy as np
import assemblyai as aai
import time
import soundfile as sf
import noisereduce as nr
import torchaudio
import torch
from fuzzywuzzy import fuzz
from speechbrain.inference import SpeakerRecognition
from deepfake_proper import DeepfakeDetector



deepfake_detector = DeepfakeDetector("/Users/dhavalbhagat/Desktop/Thakur_hackathon/dataset/shuffled_file.csv")

def check_deepfake(audio_file):
    result = deepfake_detector.predict_audio_deepfake(audio_file)
    if result == "FAKE":
        print(f"🚨 Deepfake detected in {audio_file}. Please use a real voice recording.")
        return False
    print(f"✅ {audio_file} passed deepfake verification.")
    return True





# Load SpeechBrain speaker verification model
spkrec_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_spkrec"
)

# AssemblyAI API Key (Replace with a valid API key)
API_KEY = "d5a05d4271894a61ace9741605c8a7e8"
aai.settings.api_key = API_KEY

# Predefined sentences for training
SENTENCES = [
    "Technology is evolving every single day.",
    "The weather today is quite unpredictable.",
    "Artificial intelligence is shaping the future.",
    "Communication is key to building relationships.",
    "A healthy lifestyle requires balance and discipline.",
    "Reading books can expand your knowledge and creativity.",
    "Practice makes progress, not necessarily perfection.",
    "Traveling to new places broadens your perspective.",
    "Learning a new language takes time and dedication.",
    "Every challenge is an opportunity to grow stronger."
]


def get_random_sentences(n=3):
    return random.sample(SENTENCES, n)


def record_audio(filename, text=None, duration=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        if text:
            print(f"\n🎤 Say: \"{text}\"")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())
    return filename


def noise_reduction(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    sf.write(audio_file, reduced_noise, sr)
    return audio_file


def transcribe_audio(audio_file, max_retries=3):
    transcriber = aai.Transcriber()
    for attempt in range(max_retries):
        try:
            print("\U0001F504 Uploading audio for verification...")
            transcript = transcriber.transcribe(audio_file)
            if transcript.status == "completed":
                return transcript.text.strip().lower()
            print(f"⚠ Transcription failed. Attempt {attempt + 1}/{max_retries}")
        except Exception as e:
            print(f"⚠ Error: {e}. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(2)
    return None


def is_exact_match(transcribed_text, expected_text):
    return transcribed_text == expected_text.lower()


def extract_voice_features(audio_file):
    signal, _ = torchaudio.load(audio_file)
    return spkrec_model.encode_batch(signal).squeeze(0).detach().numpy()


def compare_voice_signatures(sig1, sig2, threshold=0.85):
    sig1 = sig1.flatten()
    sig2 = sig2.flatten()
    similarity_score = np.dot(sig1, sig2) / (np.linalg.norm(sig1) * np.linalg.norm(sig2))
    print(f"🔍 Voice Similarity: {similarity_score:.2f}")
    return similarity_score > threshold


def verify_speech(expected_text):
    filename = f"{expected_text.replace(' ', '_')}.wav"
    record_audio(filename, text=expected_text)
    noise_reduction(filename)

    # Deepfake check
    if not check_deepfake(filename):
        return False

    transcribed_text = transcribe_audio(filename)
    if not transcribed_text:
        return False
    print(f"🔍 Recognized: '{transcribed_text}'")
    return is_exact_match(transcribed_text, expected_text)


def main():
    print("\n🔹 **Voice Training Phase** 🔹\n")
    selected_sentences = get_random_sentences()
    voice_signatures = []

    for sentence in selected_sentences:
        if not verify_speech(sentence):
            print("❌ Verification failed. Please try again.")
            return

        # Deepfake check
        audio_file = f"{sentence.replace(' ', '_')}.wav"
        if not check_deepfake(audio_file):
            return  # Stop if a deepfake is detected

        voice_signatures.append(extract_voice_features(audio_file))

    avg_voice_signature = np.mean(voice_signatures, axis=0)

    print("\n🔹 **Voice Signature Generation Phase** 🔹\n")
    print("📢 Say anything you want to create your voice signature.")
    record_audio("user_voice_signature.wav")
    noise_reduction("user_voice_signature.wav")

    # Deepfake check
    if not check_deepfake("user_voice_signature.wav"):
        return

    transcribed_signature = transcribe_audio("user_voice_signature.wav")
    if not transcribed_signature:
        print("❌ Could not recognize the voice signature. Try again.")
        return
    print(f"🔍 Detected Signature: '{transcribed_signature}'")
    user_signature = extract_voice_features("user_voice_signature.wav")
    print("\n✅ Voice signature created successfully.")

    print("\n🔹 **Voice Verification Phase** 🔹\n")
    print(f"📢 Say your voice signature again: '{transcribed_signature}'")
    record_audio("verification_signature.wav")
    noise_reduction("verification_signature.wav")

    # Deepfake check
    if not check_deepfake("verification_signature.wav"):
        return

    verification_text = transcribe_audio("verification_signature.wav")
    if not verification_text:
        print("❌ Could not recognize verification attempt. Try again.")
        return
    print(f"🔍 Recognized: '{verification_text}'")

    if is_exact_match(verification_text, transcribed_signature):
        test_signature = extract_voice_features("verification_signature.wav")
        if compare_voice_signatures(user_signature, test_signature):
            print("✅ Voice signature matched. Access granted.")
        else:
            print("❌ Voice signature mismatch. Access denied.")
    else:
        print("❌ Voice signature text mismatch. Access denied.")


if __name__ == "__main__":
    main()
