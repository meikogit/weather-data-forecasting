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

## Ergebnis

| Horizont | F1 Modell | F1 Persistenz | Brier Modell | Brier Persistence | Onset Recall
|---|---|---|---|---|---|
| 1 h      |   [0.74]  |     [0.74]    |    [0.077]   |      [0.097]      |    [0.01]
| 3 h      |   [0.57]  |     [0.56]    |    [0.111]   |      [0.165]      |    [0.17]
| 6 h      |   [0.49]  |     [0.44]    |    [0.125]   |      [0.210]      |    [0.29]
| 12 h     |   [0.42]  |     [0.34]    |    [0.134]   |      [0.248]      |    [0.31]
| 24 h     |   [0.32]  |     [0.27]    |    [0.143]   |      [0.273]      |    [0.20]

**Antwort auf die Forschungsfrage:**
Die Ergebnisse zeigen, dass unser Modell bei einer und drei Stunden noch keinen Signifikanten unterschied zu dem baseline Modell vorbringt. Erst ab ca. sechs Stunden hat das Modell einen leichten Vorteil erkennbar durch den höheren F1-Score. Der Brier-Score hingegen zeigt, dass in hinsicht auf Fehler des Modells schon ab Stunde 1 ein Vorteil gegenüber der Baseline besteht, welcher sich bei größeren Horizonten noch weiter Ausbaut. Interessant ist der Onset Recall, welcher zeigt wie viele Regenstunden, in welchen es die jeweilige Stundenanzahl vorher nicht geregnet hat, das Modell erkennt. Bei der Baseline sind das nach Definition 0. Das Modell hingegen zeigt einen Peak bei dem 8 Stunden Horizont mit 31%. 

## Grenzen

- Nur eine einzelne Station, keine Informationen über Wetter in der Umgebung
- Nur ein einfaches lineares Modell
- Die Schwellenwerte (ab welcher Wahrscheinlichkeit "Regen" gesagt wird) wurden nicht auf einem separaten Validierungszeitraum bestimmt

## Ausblick

- Klimatologie als zweite Baseline (durchschnittliche Regenwahrscheinlichkeit pro Monat und Uhrzeit)
- Nichtlineares Modell (z. B. Gradient Boosting) im Vergleich
- Messwerte von Nachbarstationen als zusätzliche Features, kombinierbar mit Windrichtung um Regenwahrscheinlichkeiten der Stationen zu gewichten

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
