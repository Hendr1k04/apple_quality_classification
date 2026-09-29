import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#Die APp unterstützt lediglich die vollständigen CNN Modelle
# Wichtig:
# Diese Reihenfolge muss exakt zu deinem Training passen.
# Falls train_ds.class_names bei dir anders ausgegeben wurde, hier entsprechend ändern.
CLASS_NAMES = ["Apple__Healthy", "Apple__Rotten"]
#Ordner Weban
# in neuem Terminal "pip install -r requirements.txt"
#dann "streamlit run app.py"
IMG_SIZE = (224, 224)
MODEL_PATH = "MobileNetV2FT.keras"

# Für MobileNetV2 ist der letzte Conv-Layer normalerweise "Conv_1".
LAST_CONV_LAYER_NAME = "Conv_1"


@st.cache_resource
def load_trained_model():
    """
    Lädt das trainierte Keras-Modell nur einmal beim Start der App.
    """
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def call_layer(layer, x):
    """
    Ruft eine Keras-Schicht robust im Inferenzmodus auf.
    Manche Layer akzeptieren training=False, andere nicht.
    """
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def prepare_image(image: Image.Image):
    """
    Bereitet das hochgeladene Bild für das Modell vor.
    Da die MobileNetV2-Vorverarbeitung in deinem Modell enthalten ist
    (Rescaling von 0..255 auf -1..1), wird hier NICHT zusätzlich normalisiert.
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_image(image: Image.Image, model):
    """
    Gibt die vorhergesagte Klasse, die Konfidenz und alle Wahrscheinlichkeiten zurück.
    """
    img_array = prepare_image(image)
    predictions = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_class, confidence, predictions, predicted_index


def find_nested_base_model(model):
    """
    Sucht in deinem Sequential-Modell das eingebettete MobileNetV2-Basismodell.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer

    raise ValueError(
        "Kein eingebettetes Basismodell gefunden. "
        "Prüfe, ob dein Modell wirklich MobileNetV2 als Layer enthält."
    )


def make_gradcam_heatmap(image_array, model, pred_index=None):
    """
    Erstellt eine Grad-CAM-Heatmap für das hochgeladene Bild.

    Diese Variante ist speziell für dein Modell geeignet:
    Input -> Rescaling -> MobileNetV2 -> GlobalAveragePooling -> Dense -> Output
    """

    base_model = find_nested_base_model(model)
    base_model_index = model.layers.index(base_model)

    pre_base_layers = model.layers[:base_model_index]
    post_base_layers = model.layers[base_model_index + 1:]

    last_conv_layer = base_model.get_layer(LAST_CONV_LAYER_NAME)

    # Modell vom MobileNetV2-Input bis zum letzten Conv-Layer und bis zum Base-Output
    conv_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output]
    )

    # Erst alle Layer vor MobileNetV2 anwenden, z. B. Rescaling
    x = tf.convert_to_tensor(image_array)

    for layer in pre_base_layers:
        x = call_layer(layer, x)

    with tf.GradientTape() as tape:
        conv_outputs, base_output = conv_model(x, training=False)

        # Danach alle Layer nach MobileNetV2 anwenden
        y = base_output
        for layer in post_base_layers:
            y = call_layer(layer, y)

        if pred_index is None:
            pred_index = tf.argmax(y[0])

        class_channel = y[:, pred_index]

    # Gradienten der Zielklasse bezogen auf die Feature Maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Durchschnittlicher Einfluss jedes Feature-Map-Kanals
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # Gewichtete Summe der Feature Maps
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Negative Werte entfernen und normalisieren
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    if max_value == 0:
        return np.zeros(heatmap.shape)

    heatmap = heatmap / max_value

    return heatmap.numpy()


def create_gradcam_overlay(original_image: Image.Image, heatmap, alpha=0.4):
    """
    Legt die Grad-CAM-Heatmap über das Originalbild.
    """
    original_image = original_image.convert("RGB").resize(IMG_SIZE)

    heatmap_uint8 = np.uint8(255 * heatmap)

    # Farbige Heatmap ohne matplotlib.cm.get_cmap()
    colormap = plt.get_cmap("jet")
    jet_colors = colormap(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap = jet_heatmap.resize(IMG_SIZE)

    jet_heatmap_array = np.array(jet_heatmap)
    original_array = np.array(original_image)

    overlay = jet_heatmap_array * alpha + original_array * (1 - alpha)
    overlay = np.uint8(overlay)

    return Image.fromarray(overlay), jet_heatmap

st.set_page_config(
    page_title="Apfel Klassifizierung",
    page_icon="🍎",
    layout="centered"
)

st.title("Apfel Klassifizierung")
st.write(
    "Lade ein Bild eines Apfels hoch. "
    "Das Modell klassifiziert den Apfel als **Healthy** oder **Rotten** "
    "und zeigt zusätzlich eine **Grad-CAM-Heatmap**."
)

try:
    model = load_trained_model()
except Exception as e:
    st.error("Das Modell konnte nicht geladen werden.")
    st.write("Prüfe, ob `MobileNetV2FT.keras` im gleichen Ordner wie `app.py` liegt.")
    st.exception(e)
    st.stop()


uploaded_file = st.file_uploader(
    "Apfelbild hochladen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    predicted_class, confidence, predictions, predicted_index = predict_image(image, model)

    st.subheader("Ergebnis")

    if predicted_class == "Apple__Healthy":
        st.success("Vorhersage: Healthy")
    else:
        st.error("Vorhersage: Rotten")

    st.write(f"**Modellklasse:** `{predicted_class}`")
    st.write(f"**Konfidenz:** {confidence:.2%}")

    st.subheader("Wahrscheinlichkeiten")

    for class_name, score in zip(CLASS_NAMES, predictions):
        st.write(f"**{class_name}:** {float(score):.2%}")

    st.subheader("Grad-CAM Heatmap")

    try:
        image_array = prepare_image(image)
        heatmap = make_gradcam_heatmap(image_array, model, predicted_index)
        overlay_image, heatmap_image = create_gradcam_overlay(image, heatmap, alpha=0.4)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image.resize(IMG_SIZE), caption="Original", use_container_width=True)

        with col2:
            st.image(heatmap_image, caption="Heatmap", use_container_width=True)

        with col3:
            st.image(overlay_image, caption="Overlay", use_container_width=True)

        st.caption(
            "Rote/gelbe Bereiche zeigen Bildregionen, die für die Entscheidung des Modells besonders relevant waren."
        )

    except Exception as e:
        st.error("Die Grad-CAM-Heatmap konnte nicht erstellt werden.")
        st.write(
            "Möglicherweise heißt der letzte Conv-Layer anders. "
            "Für MobileNetV2 ist es normalerweise `Conv_1`."
        )
        st.exception(e)

else:
    st.write("Noch kein Bild hochgeladen.")
