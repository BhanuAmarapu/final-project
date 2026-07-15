import sys
import os
import wave
import numpy as np

# Ensure root directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentencebert_service import sentencebert_service
from whisper_service import whisper_service
from similarity_service import cosine_similarity, similarity_service

def test_normalization():
    print("\n--- Test 1: Text Normalization ---")
    original = "Hello World! This, is a test.   With multiple spaces."
    expected = "hello world this is a test with multiple spaces"
    result = sentencebert_service.normalize_text(original)
    print(f"Original: '{original}'")
    print(f"Result  : '{result}'")
    assert result == expected, f"Expected '{expected}', but got '{result}'"
    print("[OK] Normalization test passed!")

def test_embeddings_and_similarity():
    print("\n--- Test 2: Embeddings & Similarity ---")
    text1 = "Cloud computing provides scalable computing resources over the internet."
    text2 = "Using the internet, cloud systems offer expandable computation power."
    text3 = "The weather is very nice and sunny today in Washington."

    print("Generating embeddings...")
    emb1 = sentencebert_service.generate_embedding(text1)
    emb2 = sentencebert_service.generate_embedding(text2)
    emb3 = sentencebert_service.generate_embedding(text3)

    print(f"Embedding dimensions: {len(emb1)}")
    assert len(emb1) > 0, "Embedding should not be empty"

    sim_similar = cosine_similarity(emb1, emb2)
    sim_different = cosine_similarity(emb1, emb3)

    print(f"Similarity between similar texts: {sim_similar * 100:.2f}%")
    print(f"Similarity between different texts: {sim_different * 100:.2f}%")

    assert sim_similar > 0.60, f"Expected similarity to be >= 60%, got {sim_similar}"
    assert sim_different < 0.40, f"Expected similarity to be < 40%, got {sim_different}"
    print("[OK] Embeddings & Similarity test passed!")

def test_whisper_transcription():
    print("\n--- Test 3: Whisper Audio Processing ---")
    # Programmatically create a 2-second 440Hz sine wave WAV file
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Sine wave
    audio_data = np.sin(2 * np.pi * 440 * t)
    audio_data = (audio_data * 32767).astype(np.int16)

    wav_path = "test_sine.wav"
    with wave.open(wav_path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio_data.tobytes())

    print(f"Generated test audio: {wav_path}")

    try:
        print("Running transcription...")
        result = whisper_service.transcribe(wav_path)
        print(f"Transcript: '{result['transcript']}'")
        print(f"Language  : '{result['language']}'")
        print(f"Duration  : {result['duration']:.2f} seconds")

        assert result['duration'] == 2.0, f"Expected duration of 2.0s, got {result['duration']}"
        print("[OK] Whisper transcription test passed!")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
            print("Cleaned up test audio.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING AUDIO PIPELINE UNIT TESTS")
    print("=" * 60)
    try:
        test_normalization()
        test_embeddings_and_similarity()
        test_whisper_transcription()
        print("\n" + "=" * 60)
        print("[SUCCESS] All audio pipeline unit tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAILURE] Test failed: {e}")
        import traceback
        traceback.print_exc()
