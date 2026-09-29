# Fact Check Deutschland
## macOS
### Voraussetzungen
#### FFmpeg
- **FFmpeg** (wird von `mlx-whisper` für das Audio-Decoding benötigt)
#### Installation
```bash
brew install ffmpeg
```
### Funktion
#### Transcription
Das Programm analysiert zunächst die Sprache anhand des Moduls 
**transcription.py**. Das Programm analysiert zwei Aspekte, das Gesprochene und
wer gesprochen hat, also wandelt Audio bzw. Videodateien (MP3/MP4)
in eine Textdatei umwandelt und deutet dann die Anzahl der Sprecher, um diese anhand einer
Nutzereingabe konkret zu belegen.
#### Extrahierung
Extrahiere den Inhalt des Transkriptes, und reduziere ihn auf wichtige Informationen, 
Zitate und zusammengefasste Nebenaussagen. Der Text wird von einem String in eine
json von einer LLM verarbeitet und zusammengetragen. Die LLM die hier genutzt wird ist 
Groq, dem System-Prompt findet man in **extraction.py** und der User-Prompt ist das
Transkript, welches aus **transcirption.py** generiert wird. Die Json wird dann in eine
Map konvertiert, um diese für die Analyse aufzuarbeiten.
#### Informationsanalyse

### Quellenanalyse




