# Regenvorhersage mit DWD-Wetterdaten

## Forschungsfrage

Ab welchem Vorhersagehorizont ist ein einfaches Machine-Learning-Modell bei der Vorhersage von Regen besser als die simple Regel "das Wetter bleibt, wie es gerade ist" (Persistenz)?

Das Wetter ist ein chaotisches System und die Frage in wie weit ein so simples modell, wie logistische Regression, Regen vorhersagen kann hat mich zu diesem Projekt gebracht. Außerdem bietet es eine gute Möglichkeit meine ersten Erkenntnisse in Machine Learning anwenden zu können und ersten umgang mit GitHub zu lernen. 

## Daten

- Quelle: Deutscher Wetterdienst (DWD), Open Data, stündliche Stationsdaten
- Station: Angermünde (Stations-ID 164)
- Zeitraum: 2000 bis 2025
- Messgrößen: Temperatur, Luftfeuchte, Taupunkt, Luftdruck, Wind, Bewölkung, Niederschlag

## Vorgehen

1. **Datenaufbereitung:** DWD-Dateien laden, zusammenführen, fehlende Werte (-999) entfernen
2. **Features:** z. B. Taupunktdifferenz, Druckänderung über 3 Stunden, Bewölkung, Jahreszeit
3. **Modell:** Logistische Regression, die vorhersagt, ob es in 1 bis 24 Stunden regnet
4. **Zeitliche Aufteilung:** Training bis 2020, Test ab 2021 
5. **Vergleich:** Persistenz-Baseline auf exakt denselben Teststunden

## Ergebnis

| Horizont | F1 Modell | F1 Persistenz | Brier Modell | Brier Persistence | Onset Recall  | Climatology Brier |
|---|---|---|---|---|---|
| 1 h      |   0.74    |     0.74      |    0.077     |      0.097        |    0.01       |       0.149       |
| 3 h      |   0.57    |     0.56      |    0.111     |      0.165        |    0.17       |       0.149       |
| 6 h      |   0.49    |     0.44      |    0.125     |      0.210        |    0.29       |       0.149       |
| 12 h     |   0.42    |     0.34      |    0.134     |      0.248        |    0.31       |       0.149       |
| 24 h     |   0.32    |     0.27      |    0.143     |      0.273        |    0.20       |       0.149       |

**Antwort auf die Forschungsfrage:**
Die Ergebnisse zeigen, dass unser Modell unter betrachtung des F1-Scores, bei einer und drei Stunden noch keinen großen unterschied zu dem baseline Modell vorbringt. Erst ab ca. sechs Stunden hat das Modell, in bezug auf Prezision und Recall, einen leichten Vorteil. Der Brier-Score hingegen zeigt, dass das Modell schon ab Stunde 1 öfter richtig liegt wenn es sicher war und gegenüber dem Durchschnitt von ca. 
0.149 ebenfalls bessere Vorhersagen liefert. Dieser Vorteil baut sich bei größeren Horizonten noch weiter aus. Interessant ist auch der Onset Recall, welcher zeigt wie viele Regenbeginne, das Modell erkennt. Bei der Baseline sind das nach Definition 0. Das Modell hingegen zeigt einen Peak bei dem 8 Stunden Horizont mit 31%. Der Grund dafür könnte sein, dass in kurzen Horizonten die Regenwahrscheinlichkeit unter dem Threshold bleibt. Bei sehr langen Horizonten haben Faktoren wie Luftdruckänderung und andere Features keine Vorhersagekraft mehr. Um 8 Stunden könnte der Kompromiss liegen zwischen zu langem und zu kurzem Horizont.

## Grenzen

- Nur eine einzelne Station, keine Informationen über Wetter in der Umgebung
- Nur ein einfaches lineares Modell
- Die Schwellenwerte (ab welcher Wahrscheinlichkeit "Regen" gesagt wird) wurden nicht auf einem separaten Validierungszeitraum bestimmt

## Ausblick
- Vergleich mit einem komplexeren nicht linearen Modell wie Gradient Boosting 
- Messwerte von Nachbarstationen als zusätzliche Features, kombinierbar mit Windrichtung um Regenwahrscheinlichkeiten der Stationen zu gewichten
- Projekt mit eigenen Sensoren, die Features wie Luftdruck und Temperatur bereitstellen sollen, um die Frage zu beantworten ob und wie solche Messungen, zusammen mit den Vorhersagen echter Wetteranbieter in der Umgebung, die Vorhersagen verbessern 
## Projekt starten

```bash
pip install -r requirements.txt
python -m src.main
```

Die DWD-Rohdaten gehören in den Ordner `data/raw/`.

## Projektstruktur

```
src/
├── config.py          Einstellungen (Zeiträume, Vorhersage-Stunden)
├── data_loader.py     DWD-Dateien laden
├── preprocessing.py   Daten bereinigen und zusammenführen
├── features.py        Features berechnen
├── logistic_model.py  Logistische Regression
├── baseline.py       Persistenz-Baseline
├── evaluation.py      Bewertung (F1, Accuracy, ROC-AUC)
└── main.py            startet alles
```

## Hinweis zur Entwicklung

Bei der Strukturierung des Projekts und beim Verständnis einzelner Konzepte habe ich KI-Assistenz (Claude) genutzt. Dabei habe ich den Code selbst geschrieben und mir lediglich Tipps und Verbesserungsvorschläge geholt. Die Notebooks in denen Experimentiert wurde, wurden für die Übersichtlichkeit gelöscht.