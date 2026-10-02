# Regenvorhersage mit DWD-Wetterdaten

## Forschungsfrage

Ab welchem Vorhersagehorizont ist ein einfaches Machine-Learning-Modell bei der Vorhersage von Regen besser als die simple Regel "das Wetter bleibt, wie es gerade ist" (Persistenz)?

Die Frage öffnet möglichkeiten zu weiteren Forschungsfragen, wie "Verbessert eine eigene kleine Wetterstation aus wenigen Sensoren die lokale Wettervorhersage?". Mit diesem Projekt will ich meine ersten Erkenntnisse in Machine Learning anwenden.

## Daten

- Quelle: Deutscher Wetterdienst (DWD), Open Data, stündliche Stationsdaten
- Station: Angermünde (Stations-ID 164)
- Zeitraum: 2000 bis 2025
- Messgrößen: Temperatur, Luftfeuchte, Taupunkt, Luftdruck, Wind, Bewölkung, Niederschlag

## Vorgehen

1. **Datenaufbereitung:** DWD-Dateien laden, zusammenführen, fehlende Werte (-999) entfernen
2. **Features:** z. B. Taupunktdifferenz, Druckänderung über 3 Stunden, Bewölkung, Jahreszeit
3. **Modell:** Logistische Regression, die vorhersagt, ob es in 1 bis 24 Stunden regnet
4. **Zeitliche Aufteilung:** Training bis 2020, Test ab 2021 (kein zufälliges Mischen, damit das Modell nicht "in die Zukunft schaut")
5. **Vergleich:** Persistenz-Baseline auf exakt denselben Teststunden

<!-- Ergänze oder streiche Punkte, je nachdem, was du wirklich gemacht hast. -->

## Ergebnis

<!-- Hier die Zahlen aus deiner Tabelle eintragen (Spalten "F1 Score" und "Persistence F1").
     Nur ein paar Horizonte reichen, z. B. 1, 3, 6, 12, 24 Stunden. -->

| Horizont | F1 Modell | F1 Persistenz |
|---|---|---|
| 1 h | [ ] | [ ] |
| 3 h | [ ] | [ ] |
| 6 h | [ ] | [ ] |
| 12 h | [ ] | [ ] |
| 24 h | [ ] | [ ] |

**Antwort auf die Forschungsfrage:**
<!-- 2-3 Sätze in deinen Worten. Zum Beispiel: "Für die nächsten 1 bis 3 Stunden ist das Modell kaum besser als die Persistenz, weil Regen meist einfach weitergeht. Ab etwa 6 Stunden ist das Modell deutlich besser, weil es Veränderungen wie fallenden Luftdruck erkennt." -->
[Deine Antwort]

## Grenzen

- Nur eine einzelne Station, keine Informationen über Wetter in der Umgebung
- Nur ein einfaches lineares Modell
- Die Schwellenwerte (ab welcher Wahrscheinlichkeit "Regen" gesagt wird) wurden nicht auf einem separaten Validierungszeitraum bestimmt

## Ausblick

- Klimatologie als zweite Baseline (durchschnittliche Regenwahrscheinlichkeit pro Monat und Uhrzeit)
- Nichtlineares Modell (z. B. Gradient Boosting) im Vergleich
- Messwerte von Nachbarstationen als zusätzliche Features

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
├── baselines.py       Persistenz-Baseline
├── evaluation.py      Bewertung (F1, Accuracy, ROC-AUC)
└── main.py            startet alles
```

## Hinweis zur Entwicklung

Bei der Strukturierung des Projekts und beim Verständnis einzelner Konzepte habe ich KI-Assistenz (Claude) genutzt. Dabei habe ich den Code selbst geschrieben und mir lediglich Tipps und Verbesserungsvorschläge geholt. Die Notebooks in denen Experimentiert wurde, wurden für die Übersichtlichkeit gelöscht.