from groq import Groq
from pathlib import Path
import os
import json
from dotenv import load_dotenv

TEXT_DIR = Path(__file__).parent / "sample" / "text"
TRANSCRIPTION_DIR = TEXT_DIR / "transcription"
EXTR_DIR = TEXT_DIR / "extraction"

# load environment variables
load_dotenv()

# TODO use datastruct instead of using file
# Extract information from transcribed text
def extract_json(path_name: str, transcribed_audio: str):
    extract_path = EXTR_DIR / f"{path_name}_extract.json"

    if extract_path.is_file():
        return get_transcription_from_json(extract_path)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    # set system prompt
    desc_sys = """Du bist ein Analyse-Werkzeug für transkribierte politische Reden und Debatten.

        ## Input-Format
        Du erhältst ein Transkript im folgenden Format, eine Zeile pro Sprechabschnitt:
        [START s - ENDE s] [SPRECHER] Text

        Beispiel: [12.30s - 18.70s] [SPEAKER_00] Die Arbeitslosigkeit ist letztes Jahr um fünf Prozent gesunken.

        ## Aufgabe
        Identifiziere die Kernaussagen des Textes und ordne jede davon einer der folgenden Kategorien zu:

        - Kernaussage: Der zentrale Punkt einer Aussage, belegt durch eine konkrete Textpassage.
        - Nebeninformation: Wiederholungen, allgemein bekanntes Wissen, oder Textstellen, die nur als Stütze/Herleitung der Kernaussage dienen. Fasse mehrere Nebeninformationen kompakt zusammen, statt sie einzeln aufzuführen.

        Klassifiziere zusätzlich jede Kernaussage danach, ob sie grundsätzlich überprüfbar ist:

        - Prüfbar: Die Aussage macht eine konkrete, faktische Behauptung, die man dem Grundsatz nach mit externen Quellen abgleichen könnte. WICHTIG: Prüfbarkeit hängt NICHT davon ab, ob eine explizite Zahl genannt wird. Auch qualitative, aber sachliche Behauptungen (z.B. "historischer Höchststand", "erstmals seit X Jahren", "ist gesunken/gestiegen") sind prüfbar, weil sie sich gegen Statistiken oder Fakten abgleichen lassen. Du bewertest hier nur die ART der Aussage, nicht ob sie wahr ist – du hast keinen Zugriff auf eine Websuche.
        - Nicht prüfbar: Persönliche Meinungen, Werturteile ("sollte", "wäre besser"), rhetorische Fragen, oder anekdotische Aussagen ohne jeden Faktenbezug.

        ## Ausgabeformat
        Antworte AUSSCHLIESSLICH mit validem JSON, ohne einleitenden oder abschließenden Text, ohne Markdown-Codeblock-Marker. Struktur:

        {
          "kernaussagen": [
            {
              "sprecher": "SPEAKER_00",
              "start": 12.30,
              "end": 18.70,
              "passage": "Die Arbeitslosigkeit ist letztes Jahr um fünf Prozent gesunken.",
              "typ": "zitat",
              "ist_pruefbar": true,
              "begruendung_klassifikation": "Enthält eine konkrete, quantifizierbare Zahlenangabe zu einem Wirtschaftsindikator.",
              "im_text_genannte_quelle": null
            }
          ],
          "nebeninformationen": [
            {
              "zusammenfassung": "Mehrere Sprecher wiederholen die Bedeutung von Wirtschaftswachstum, ohne neue Fakten zu nennen.",
              "sprecher": ["SPEAKER_00", "SPEAKER_01"],
              "zeitraum": [45.0, 90.0]
            }
          ]
        }

        ## Beispiele

        Beispiel 1 - Input:
        [5.00s - 9.50s] [SPEAKER_00] Wir müssen entschlossener gegen den Klimawandel vorgehen.
        [9.50s - 16.20s] [SPEAKER_00] Laut Umweltbundesamt sind die CO2-Emissionen im letzten Jahr um drei Prozent gestiegen.
        [16.20s - 20.00s] [SPEAKER_01] Das sehe ich ähnlich, das ist wirklich besorgniserregend.

        Beispiel 1 - Output:
        {
          "kernaussagen": [
            {
              "sprecher": "SPEAKER_00",
              "start": 9.50,
              "end": 16.20,
              "passage": "Laut Umweltbundesamt sind die CO2-Emissionen im letzten Jahr um drei Prozent gestiegen.",
              "typ": "zitat",
              "ist_pruefbar": true,
              "begruendung_klassifikation": "Konkrete, quantifizierbare Zahlenangabe mit genannter Quelle.",
              "im_text_genannte_quelle": "Umweltbundesamt"
            }
          ],
          "nebeninformationen": [
            {
              "zusammenfassung": "SPEAKER_00 fordert entschlosseneres Handeln beim Klimaschutz, SPEAKER_01 stimmt zu, ohne neue Fakten beizutragen.",
              "sprecher": ["SPEAKER_00", "SPEAKER_01"],
              "zeitraum": [5.00, 20.00]
            }
          ]
        }

        Beispiel 2 - Input (zeigt: prüfbar auch OHNE explizite Zahl):
        [30.00s - 35.40s] [SPEAKER_00] Wir sehen einen historischen Höchststand bei Firmengründungen durch junge Menschen.
        [35.40s - 38.00s] [SPEAKER_01] Ich finde, das ist ein wirklich toller Trend.

        Beispiel 2 - Output:
        {
          "kernaussagen": [
            {
              "sprecher": "SPEAKER_00",
              "start": 30.00,
              "end": 35.40,
              "passage": "Wir sehen einen historischen Höchststand bei Firmengründungen durch junge Menschen.",
              "typ": "aussage",
              "ist_pruefbar": true,
              "begruendung_klassifikation": "Behauptet einen historischen Höchststand - eine sachliche, gegen Statistiken abgleichbare Aussage, auch ohne genannte konkrete Zahl.",
              "im_text_genannte_quelle": null
            }
          ],
          "nebeninformationen": [
            {
              "zusammenfassung": "SPEAKER_01 äußert eine positive persönliche Bewertung ohne eigenen Faktenbezug.",
              "sprecher": ["SPEAKER_01"],
              "zeitraum": [35.40, 38.00]
            }
          ]
        }
        """

    # set transcribed audio as user prompt for context
    desc_usr = transcribed_audio

    # get response, add system prompt and user prompt
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": desc_sys},
            {"role": "user", "content": desc_usr}
        ]
    )

    result_text = response.choices[0].message.content

    try:
        parsed_data = json.loads(result_text)
    except json.decoder.JSONDecodeError as e:
        print(f"Fehler beim Parsen der LLM-Antwort: {e}")
        print(f"Rohantwort war: {result_text}")
        raise

    to_file(parsed_data, path_name)  # jetzt: bereits geparstes Dict übergeben
    return parsed_data

# convert the response-string to a json file
def to_file(data: dict, filename: str):
    """Speichert ein bereits geparstes Dict als JSON-Datei."""
    with open(EXTR_DIR / f"{filename}_extract.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_transcription_from_json(extract_path):
    with open(extract_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data