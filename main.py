from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import io

model = tf.keras.models.load_model('perros_gatos.h5')
app = FastAPI()
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def create_upload_file(file: UploadFile):
    contenido_imagen = await file.read()
    imgBin = Image.open(io.BytesIO(contenido_imagen))
    img = np.array(imgBin)
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gris = cv2.resize(img_gris, (100,100))
    
    imarr = np.array(img_gris)
    imarr = imarr / 255
    imarr = np.expand_dims(imarr, axis=-1)
    imarr = np.expand_dims(imarr, axis=0)

    prediccion = model.predict(imarr, verbose=False)
    imgBin.close()
    if prediccion > 0.5:
        return {'mensaje': 'perro'}
    else:
        return {'mensaje': 'gato'}