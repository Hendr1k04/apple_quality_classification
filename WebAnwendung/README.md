# Apple Quality Classifier Web-App mit Grad-CAM

Diese Streamlit-App lädt ein trainiertes Keras-Modell und klassifiziert ein hochgeladenes Apfelbild als `Apple__Healthy` oder `Apple__Rotten`.

Zusätzlich wird eine Grad-CAM-Heatmap erzeugt, die zeigt, welche Bildbereiche für die Modellentscheidung besonders wichtig waren.

## Dateien

Lege folgende Dateien in einen gemeinsamen Ordner:

```text
apple_classifier_app_gradcam/
├── app.py
├── requirements.txt
└── bestModel_mobilenet.keras
```

Die Datei `bestModel_mobilenet.keras` ist dein trainiertes Modell aus dem Notebook.

## Installation

```bash
pip install -r requirements.txt
```

## Starten

```bash
streamlit run app.py
```

## Wichtig

Die Klassenreihenfolge in `app.py` muss zu deinem Training passen:

```python
CLASS_NAMES = ["Apple__Healthy", "Apple__Rotten"]
```

Falls dein Notebook bei `train_ds.class_names` eine andere Reihenfolge ausgibt, musst du die Reihenfolge in `app.py` anpassen.

Für MobileNetV2 wird als letzter Convolutional Layer standardmäßig verwendet:

```python
LAST_CONV_LAYER_NAME = "Conv_1"
```
