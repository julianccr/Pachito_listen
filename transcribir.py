import sounddevice as sd
from scipy.io.wavfile import write
import whisper

# ====== CONFIGURACIÓN ======
duracion = 5  # duración en segundos de la grabación
archivo_salida = "grabacion.wav"

'''

Modelo	Precisión	Velocidad
tiny	Baja	    Muy rápida
base	Media	    Rápida
small	Buena	    Media
medium	Muy buena	Lenta
large	Excelente	Más lenta

'''


# ====== GRABAR AUDIO ======
print("🎤 Comienza a hablar... (grabando por", duracion, "segundos)")
fs = 16000  # frecuencia de muestreo (16kHz recomendado para Whisper)
audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
sd.wait()  # espera a que termine la grabación
write(archivo_salida, fs, audio)
print("✅ Grabación finalizada y guardada como", archivo_salida)

# ====== TRANSCRIPCION 1 ======
print("--------------\n")
print("🧠 Transcribiendo... TINY")
model = whisper.load_model("tiny", device="cuda")  
result = model.transcribe(archivo_salida, language="es")
print("📄 Transcripción: TINY")
print(result["text"])

# ====== TRANSCRIPCION 2 ======
print("--------------\n")
print("🧠 Transcribiendo... BASE")
model = whisper.load_model("base", device="cuda")  
result = model.transcribe(archivo_salida, language="es")
print("📄 Transcripción: BASE")
print(result["text"])


# ====== TRANSCRIPCION 3 ======
print("--------------\n")
print("🧠 Transcribiendo... SMALL") 
model = whisper.load_model("small", device="cuda")  
result = model.transcribe(archivo_salida, language="es")
print("📄 Transcripción: SMALL")
print(result["text"])
print("--------------\n")

