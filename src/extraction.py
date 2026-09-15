from groq import Groq
import os
import json
from dotenv import load_dotenv
from transcription import TEXT_DIR

EXTR_DIR = TEXT_DIR / "extraction"

load_dotenv()

# Extract information from transcribed text
def extract_json(path_file: str):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
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

    desc_usr = get_string_from_txt(path_file)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": desc_sys},
            {"role": "user", "content": desc_usr}
        ]
    )

    result_text = response.choices[0].message.content
    to_file(result_text)
    print(result_text)

def get_string_from_txt(file_path : str):
    txt_path = TEXT_DIR / file_path
    with open(txt_path, "r") as file:
        txt = file.read()
    return txt

def to_file(conv_text):
    output = json.loads(conv_text)
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    extract_json("test_1_text.txt")
