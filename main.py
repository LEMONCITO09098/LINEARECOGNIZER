"""
App Streamlit: reconocimiento de partes del cuerpo (pose, manos, rostro)
usando la cámara del NAVEGADOR vía WebRTC, para poder abrirla desde
cualquier dispositivo con una URL, no solo desde tu computadora.

Correr localmente:
    streamlit run app.py

Desplegar gratis en la nube (Streamlit Community Cloud):
    1. Sube esta carpeta a un repo de GitHub (incluye la carpeta modelos/,
       app.py y requirements.txt).
    2. Entra a https://share.streamlit.io, conecta el repo y despliega.
    3. Te da una URL pública tipo https://tuapp.streamlit.app
"""

import os
import time
import urllib.request

import av
import cv2
import mediapipe as mp
import streamlit as st
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- Descarga automática de modelos (una sola vez) ---
CARPETA_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos")
MODELOS = {
    "pose": {
        "archivo": os.path.join(CARPETA_MODELOS, "pose_landmarker.task"),
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    },
    "manos": {
        "archivo": os.path.join(CARPETA_MODELOS, "hand_landmarker.task"),
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    },
    "rostro": {
        "archivo": os.path.join(CARPETA_MODELOS, "face_landmarker.task"),
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task",
    },
}


def asegurar_modelos_descargados():
    os.makedirs(CARPETA_MODELOS, exist_ok=True)
    for nombre, info in MODELOS.items():
        if not os.path.exists(info["archivo"]) or os.path.getsize(info["archivo"]) < 1024:
            urllib.request.urlretrieve(info["url"], info["archivo"])


NOMBRES_PUNTOS_POSE = {
    0: "Nariz",
    11: "Hombro izquierdo", 12: "Hombro derecho",
    13: "Codo izquierdo", 14: "Codo derecho",
    15: "Muñeca izquierda", 16: "Muñeca derecha",
    23: "Cadera izquierda", 24: "Cadera derecha",
    25: "Rodilla izquierda", 26: "Rodilla derecha",
    27: "Tobillo izquierdo", 28: "Tobillo derecho",
}

CONEXIONES_POSE = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@st.cache_resource
def cargar_detectores():
    """Carga los 3 modelos una sola vez y los reutiliza entre frames."""
    asegurar_modelos_descargados()

    pose = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELOS["pose"]["archivo"]),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )
    manos = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELOS["manos"]["archivo"]),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )
    rostro = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELOS["rostro"]["archivo"]),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )
    return pose, manos, rostro


def dibujar_pose(frame, resultado):
    alto, ancho, _ = frame.shape
    for pose in resultado.pose_landmarks:
        puntos_px = [(int(p.x * ancho), int(p.y * alto)) for p in pose]
        for a, b in CONEXIONES_POSE:
            cv2.line(frame, puntos_px[a], puntos_px[b], (0, 255, 0), 2)
        for idx, nombre in NOMBRES_PUNTOS_POSE.items():
            x, y = puntos_px[idx]
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(frame, nombre, (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


def dibujar_manos(frame, resultado):
    alto, ancho, _ = frame.shape
    for mano, categoria in zip(resultado.hand_landmarks, resultado.handedness):
        puntos_px = [(int(p.x * ancho), int(p.y * alto)) for p in mano]
        for a, b in CONEXIONES_MANO:
            cv2.line(frame, puntos_px[a], puntos_px[b], (255, 0, 0), 2)
        for x, y in puntos_px:
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        etiqueta = categoria[0].category_name
        etiqueta_es = "Mano izquierda" if etiqueta == "Left" else "Mano derecha"
        x0, y0 = puntos_px[0]
        cv2.putText(frame, etiqueta_es, (x0, y0 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    return frame


def dibujar_rostro(frame, resultado):
    alto, ancho, _ = frame.shape
    for cara in resultado.face_landmarks:
        for p in cara[::4]:
            x, y = int(p.x * ancho), int(p.y * alto)
            cv2.circle(frame, (x, y), 1, (0, 200, 255), -1)
    return frame


# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Reconocimiento corporal", layout="wide")
st.title("🕺 Reconocimiento de partes del cuerpo")
st.caption("Pose, manos y rostro en tiempo real, procesado con MediaPipe.")

col1, col2, col3 = st.columns(3)
detectar_pose = col1.checkbox("Pose", value=True)
detectar_manos = col2.checkbox("Manos", value=True)
detectar_rostro = col3.checkbox("Rostro", value=True)

pose_landmarker, hand_landmarker, face_landmarker = cargar_detectores()

# Configuración RTC con servidor STUN público (necesario fuera de localhost,
# y no estorba corriendo local).
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def procesar_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Usamos el reloj real en vez de st.session_state: procesar_frame corre
    # en un hilo aparte del hilo principal de Streamlit, y tocar
    # session_state ahí puede causar comportamiento errático o bloqueos.
    marca = int(time.time() * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    if detectar_pose:
        resultado = pose_landmarker.detect_for_video(mp_image, marca)
        img = dibujar_pose(img, resultado)
    if detectar_manos:
        resultado = hand_landmarker.detect_for_video(mp_image, marca)
        img = dibujar_manos(img, resultado)
    if detectar_rostro:
        resultado = face_landmarker.detect_for_video(mp_image, marca)
        img = dibujar_rostro(img, resultado)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="reconocimiento-corporal",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=procesar_frame,
    media_stream_constraints={
        "video": {"width": 480, "height": 360},
        "audio": False,
    },
    async_processing=True,
)

st.info(
    "La primera vez que abras la app, tu navegador pedirá permiso para "
    "usar la cámara. Acéptalo para que funcione."
)