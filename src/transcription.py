# IMPORTS
from pathlib import Path
import json
import mlx_whisper
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier
import soundfile as sf


# CONSTANTS
TEXT_DIR = Path(__file__).parent / "sample" / "text"
AUDIO_DIR = Path(__file__).parent / "sample" / "audio"
NAMES = ["", "Staiy", "Meme", "Friedrich Merz", "", "", "", "", "", "", ""]
SPEAKER_MAPPING = {
    f"SPEAKER_{i}": f"{NAMES[i]}" for i in range(1, 11)
}

# load speechbrain encoder on first boot
print("--> Lade SpeechBrain Speaker Model...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"}
)

# FUNCTIONS
# assign speakers to spoken text, LLM generated
def assign_speakers(speech_file: Path, segments: list, num_speakers: int = 2) -> list:
    """Extrahiert Stimm-Vektoren für jedes Whisper-Segment und ordnet sie Sprechern zu."""
    # Audio via soundfile statt torchaudio laden (vermeidet torchcodec-Fehler)
    data, sample_rate = sf.read(speech_file)

    # In PyTorch Tensor umwandeln (Shape: [channels, samples])
    if data.ndim == 1:
        signal = torch.from_numpy(data).unsqueeze(0).float()
    else:
        signal = torch.from_numpy(data.T).float()

    # Mono sicherstellen
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    embeddings = []
    valid_segments = []

    for seg in segments:
        start_sample = int(seg["start"] * sample_rate)
        end_sample = int(seg["end"] * sample_rate)

        # Ignoriere Schnipsel kürzer als 0.4 Sekunden
        if end_sample - start_sample < int(0.4 * sample_rate):
            seg["speaker"] = "SPEAKER_UNKNOWN"
            continue

        segment_audio = signal[:, start_sample:end_sample]

        with torch.no_grad():
            emb = classifier.encode_batch(segment_audio)
            embeddings.append(emb.squeeze().cpu().numpy())
            valid_segments.append(seg)

    # Vektoren in Sprecher-Gruppen clustern
    if embeddings:
        X = np.array(embeddings)
        actual_clusters = min(num_speakers, len(embeddings))
        clustering = AgglomerativeClustering(n_clusters=actual_clusters, metric="cosine", linkage="average")
        labels = clustering.fit_predict(X)

        for seg, label in zip(valid_segments, labels):
            seg["speaker"] = f"SPEAKER_{label + 1}"

    return segments

# transcribe audio and write to file with timestamps, return a dict that contains all spoken information
def transcribe_audio(name : str, num_speakers : int = 2,  to_text: bool = False):
    speech_file = AUDIO_DIR / (name + ".wav")
    name_json = name + "_json.json"
    speech_file_json = TEXT_DIR / name_json

    # if file does not exist, create new transcription from audio stream
    if not speech_file_json.is_file():
        # read mp4 file and save to dict and json
        audio_text = mlx_whisper.transcribe(str(speech_file),
                path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold= 2.4
        )

        # Sprecherzuordnung auf Segmenten ausführen
        audio_text["segments"] = assign_speakers(speech_file, audio_text["segments"], num_speakers=num_speakers)

        transcribe_audio_save_json(name_json, audio_text)
    # if file exists recreate from json
    else:
        audio_text = get_transcription_from_json(speech_file_json)

    # from dict save to txt
    if to_text:
        transcribe_audio_save_txt(name, audio_text)

    return audio_text

# TRANSCRIPTION
# save the dict of the audio stream to json, added timestamps
def transcribe_audio_save_json(filename : str, audio_text : dict):
    output_file_json = TEXT_DIR / filename
    with open(output_file_json, "w", encoding="utf-8") as f:
        json.dump(audio_text, f, ensure_ascii=False, indent=4)

# save the dict of the audio stream to txt, added timestamps
def transcribe_audio_save_txt(name: str, audio_text : dict):
    output_file_txt = TEXT_DIR / (name + "_text.txt")
    with open(output_file_txt, "w", encoding="utf-8") as f:
        for segment in audio_text["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
            display_name = SPEAKER_MAPPING.get(speaker, speaker)
            f.write(f"[{start:.2f}s - {end:.2f}s] [{display_name}] {text}\n")

# retrieve the audio stream information from an existing json file
def get_transcription_from_json(json_file : Path):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    result = transcribe_audio("test_1", num_speakers=3, to_text=True)
    print("Fertig! Transkription und Sprecherzuordnung gespeichert.")

