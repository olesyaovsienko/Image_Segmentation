from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI()

# Предполагается, что глобально определён словарь:
CLASSES = {1: 'белое яйцо', 2: 'коричневое яйцо'}

def encode_image(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def clean_mask(mask, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return closed

def process_image(image):
    # Разделение на каналы
    b, g, r, *_ = cv2.split(image)

    # Бинаризация каждого канала
    _, white_red_mask = cv2.threshold(r, 200, 255, cv2.THRESH_BINARY)
    _, white_green_mask = cv2.threshold(g, 185, 255, cv2.THRESH_BINARY)
    _, white_blue_mask = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY)

    _, brown_down_red_mask = cv2.threshold(r, 60, 255, cv2.THRESH_BINARY)
    _, brown_up_red_mask = cv2.threshold(r, 210, 255, cv2.THRESH_BINARY_INV)
    _, brown_down_green_mask = cv2.threshold(g, 60, 255, cv2.THRESH_BINARY)
    _, brown_up_green_mask = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY_INV)
    _, brown_down_blue_mask = cv2.threshold(b, 110, 255, cv2.THRESH_BINARY)
    _, brown_up_blue_mask = cv2.threshold(b, 250, 255, cv2.THRESH_BINARY_INV)

    brown_red_mask = cv2.bitwise_and(brown_down_red_mask, brown_up_red_mask)
    brown_green_mask = cv2.bitwise_and(brown_down_green_mask, brown_up_green_mask)
    brown_blue_mask = cv2.bitwise_and(brown_down_blue_mask, brown_up_blue_mask)

    # Формирование объединённых масок
    white_combined_mask = cv2.bitwise_and(white_red_mask, cv2.bitwise_and(white_green_mask, white_blue_mask))
    brown_combined_mask = cv2.bitwise_and(brown_red_mask, cv2.bitwise_and(brown_green_mask, brown_blue_mask))

    white_closed = clean_mask(white_combined_mask, size=5)
    brown_closed = clean_mask(brown_combined_mask, size=10)

    white_contours, _ = cv2.findContours(white_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brown_contours, _ = cv2.findContours(brown_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров по площади
    min_area = 20000
    max_area = 150000
    filtered_white = [cnt for cnt in white_contours if min_area < cv2.contourArea(cnt) < max_area]
    filtered_brown = [cnt for cnt in brown_contours if min_area < cv2.contourArea(cnt) < max_area]

    # Рисование результатов
    result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res_white = result.copy()
    res_brown = result.copy()
    cv2.drawContours(res_white, filtered_white, -1, (255, 0, 0), 25)
    cv2.drawContours(res_brown, filtered_brown, -1, (50, 150, 50), 25)

    # Кодирование изображений
    return {
        "r": encode_image(r),
        "g": encode_image(g),
        "b": encode_image(b),
        "original": encode_image(result),
        "white_red_mask": encode_image(white_red_mask),
        "white_green_mask": encode_image(white_green_mask),
        "white_blue_mask": encode_image(white_blue_mask),
        "brown_down_red_mask": encode_image(brown_down_red_mask),
        "brown_up_red_mask": encode_image(brown_up_red_mask),
        "brown_down_blue_mask": encode_image(brown_down_blue_mask),
        "brown_up_blue_mask": encode_image(brown_up_blue_mask),
        "brown_down_green_mask": encode_image(brown_down_green_mask),
        "brown_up_green_mask": encode_image(brown_up_green_mask),
        "brown_red_mask": encode_image(brown_red_mask),
        "brown_green_mask": encode_image(brown_green_mask),
        "brown_blue_mask": encode_image(brown_blue_mask),
        "combined_white": encode_image(white_closed),
        "combined_brown": encode_image(brown_closed),
        "result_white": encode_image(res_white),
        "result_brown": encode_image(res_brown),
        "count_white": len(filtered_white),
        "count_brown": len(filtered_brown)
    }

@app.post("/process/")
async def process_image_route(file: UploadFile = File(...)):
    image = np.array(Image.open(io.BytesIO(await file.read())))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return JSONResponse(content=process_image(image))
