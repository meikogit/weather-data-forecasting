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

| Horizont | F1 Modell | F1 Persistenz | Brier Modell | Brier Persistence |
| 1 h      |   [0.74]  |     [0.74]    |    [0.078]   |      [0.097]      |
| 3 h      |   [0.57]  |     [0.56]    |    [0.111]   |      [0.165]      |
| 6 h      |   [0.54]  |     [0.44]    |    [0.125]   |      [0.210]      |
| 12 h     |   [0.42]  |     [0.34]    |    [0.134]   |      [0.248]      |
| 24 h     |   [0.32]  |     [0.27]    |    [0.143]   |      [0.273]      |

**Antwort auf die Forschungsfrage:**
Die Ergebnisse zeigen, dass unser Modell bei einer und drei Stunden noch keinen Signifikanten unterschied zu dem baseline Modell vorbringt. Erst ab ca. sechs Stunden hat das Modell einen leichten Vorteil erkennbar durch den höheren F1-Score. Der Brier-Score hingegen zeigt, dass in hinsicht auf Fehler des Modells schon ab Stunde 1 ein Vorteil gegenüber der Baseline besteht, welcher sich bei größeren Horizonten noch weiter Ausbaut.

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