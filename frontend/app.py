import os
import cv2
import torch
import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from PIL import Image
import streamlit as st
import requests
from io import BytesIO
import base64

#########################################
# Нейросетевой метод (локально)
#########################################

CLASSES = ["фон", "белое яйцо", "коричневое яйцо"]
INFER_WIDTH = 256
INFER_HEIGHT = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Загрузка модели (предполагается, что файл модели находится в корне)
best_model = torch.jit.load('best_model_new.pt', map_location=DEVICE)


def get_validation_augmentation():
    test_transform = [
        albu.LongestMaxSize(max_size=INFER_HEIGHT, always_apply=True),
        albu.PadIfNeeded(min_height=INFER_HEIGHT, min_width=INFER_WIDTH,
                         always_apply=True),
        albu.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return albu.Compose(test_transform)


def infer_image(image):
    original_height, original_width, _ = image.shape
    augmentation = get_validation_augmentation()
    augmented = augmentation(image=image)
    image_transformed = augmented['image']
    x_tensor = torch.from_numpy(image_transformed).to(DEVICE).unsqueeze(
        0).permute(0, 3, 1, 2).float()

    best_model.eval()
    with torch.no_grad():
        pr_mask = best_model(x_tensor)
    pr_mask = pr_mask.squeeze().cpu().detach().numpy()
    label_mask = np.argmax(pr_mask, axis=0)

    if original_height > original_width:
        delta_pixels = int(((
                                        original_height - original_width) / 2 / original_height) * INFER_HEIGHT)
        mask_cropped = label_mask[:,
                       delta_pixels + 1: INFER_WIDTH - delta_pixels - 1]
    elif original_height < original_width:
        delta_pixels = int(((
                                        original_width - original_height) / 2 / original_width) * INFER_WIDTH)
        mask_cropped = label_mask[
                       delta_pixels + 1: INFER_HEIGHT - delta_pixels - 1, :]
    else:
        mask_cropped = label_mask

    label_mask_real_size = cv2.resize(mask_cropped,
                                      (original_width, original_height),
                                      interpolation=cv2.INTER_NEAREST)
    return label_mask_real_size


def process_mask(mask, sobel_threshold=30, erosion_iter=1, morph_kernel_size=3,
                 min_object_area=50, color_threshold=0.37):
    # Функция process_mask (из второго приложения) остается без изменений
    original_counts = {'белое яйцо': 0, 'коричневое яйцо': 0}
    processed_counts = {'белое яйцо': 0, 'коричневое яйцо': 0}

    for class_id in [1, 2]:
        class_mask = (mask == class_id).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(class_mask)
        original_counts[CLASSES[class_id]] = num_labels - 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (morph_kernel_size, morph_kernel_size))

    # refined_mask для белого яйца (класс 1)
    class_mask_white = (mask == 1).astype(np.uint8)
    sobelx = cv2.Sobel(class_mask_white, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(class_mask_white, cv2.CV_64F, 0, 1, ksize=3)
    grad_white = cv2.magnitude(sobelx, sobely)
    grad_white = cv2.normalize(grad_white, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
    _, border_mask_white = cv2.threshold(grad_white, sobel_threshold, 255,
                                         cv2.THRESH_BINARY)
    eroded_white = cv2.erode(class_mask_white, kernel, iterations=erosion_iter)
    refined_mask_white = cv2.bitwise_and(eroded_white,
                                         cv2.bitwise_not(border_mask_white))
    refined_mask_white = cv2.dilate(refined_mask_white, kernel,
                                    iterations=erosion_iter)

    # refined_mask для коричневого яйца (класс 2)
    class_mask_brown = (mask == 2).astype(np.uint8)
    sobelx = cv2.Sobel(class_mask_brown, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(class_mask_brown, cv2.CV_64F, 0, 1, ksize=3)
    grad_brown = cv2.magnitude(sobelx, sobely)
    grad_brown = cv2.normalize(grad_brown, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
    _, border_mask_brown = cv2.threshold(grad_brown, sobel_threshold, 255,
                                         cv2.THRESH_BINARY)
    eroded_brown = cv2.erode(class_mask_brown, kernel, iterations=erosion_iter)
    refined_mask_brown = cv2.bitwise_and(eroded_brown,
                                         cv2.bitwise_not(border_mask_brown))
    refined_mask_brown = cv2.dilate(refined_mask_brown, kernel,
                                    iterations=erosion_iter)

    combined_mask = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(combined_mask)
    new_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        comp_mask = (labels == label).astype(np.uint8)
        total_area = np.sum(comp_mask)
        if total_area < min_object_area:
            continue

        white_area = np.sum((mask == 1) & (comp_mask > 0))
        brown_area = np.sum((mask == 2) & (comp_mask > 0))
        fraction_white = white_area / total_area
        fraction_brown = brown_area / total_area

        if fraction_white >= color_threshold and fraction_brown >= color_threshold:
            white_submask = cv2.bitwise_and(refined_mask_white, comp_mask)
            num_white, white_labels = cv2.connectedComponents(white_submask)
            for w_label in range(1, num_white):
                sub_white = (white_labels == w_label).astype(np.uint8)
                if np.sum(sub_white) >= min_object_area:
                    new_mask[sub_white > 0] = 1
            brown_submask = cv2.bitwise_and(refined_mask_brown, comp_mask)
            num_brown, brown_labels = cv2.connectedComponents(brown_submask)
            for b_label in range(1, num_brown):
                sub_brown = (brown_labels == b_label).astype(np.uint8)
                if np.sum(sub_brown) >= min_object_area:
                    new_mask[sub_brown > 0] = 2
        else:
            dominant_class = 1 if white_area >= brown_area else 2
            new_mask[comp_mask > 0] = dominant_class

    for cls in [1, 2]:
        cls_mask = (new_mask == cls).astype(np.uint8)
        num_labels_cls, _ = cv2.connectedComponents(cls_mask)
        processed_counts[CLASSES[cls]] = num_labels_cls - 1

    new_mask = cv2.medianBlur(new_mask.astype(np.uint8), 3)
    return new_mask, original_counts, processed_counts


def mask_to_color(mask):
    color_map = {
        0: [0, 0, 0],
        1: [255, 255, 255],
        2: [139, 69, 19]
    }
    height, width = mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        colored_mask[mask == class_id] = color
    return colored_mask


#########################################
# Функции для работы с изображениями и API вызовами
#########################################

DATA_DIR = "data"  # Папка с изображениями для галереи
os.makedirs(DATA_DIR, exist_ok=True)


def load_image(uploaded_file):
    return np.array(Image.open(uploaded_file))


# Вызов API морфологического метода (первое приложение)
def call_api_morph(file_obj):
    try:
        response = requests.post("http://localhost:8000/process/",
                                 files={"file": file_obj})
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Ошибка обработки на сервере морфологического метода")
    except Exception as e:
        st.error(f"Ошибка соединения с API: {e}")
    return None


#########################################
# Основной блок приложения
#########################################

st.set_page_config(page_title="Анализатор яиц", page_icon="🥚",
                   layout="wide", initial_sidebar_state="expanded")
st.title("🥚 Анализатор яиц для идеальной яичницы 🍳")
st.subheader("by Овсиенко Олеся, 317 группа")

# Выбор метода обработки
method = st.sidebar.radio("Выберите метод обработки",
                          ("Модель UNet", "Морфологические преобразования"))

# Выбор источника изображения
source_option = st.sidebar.radio("Источник изображения",
                                 ("Галерея", "Загрузка нового"))
selected_image = None

if source_option == "Галерея":
    # st.sidebar.header("Галерея изображений")
    image_files = [f for f in sorted(os.listdir(DATA_DIR)) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if image_files:
        st.markdown("### Выберите изображение из галереи:")
        # Определяем количество колонок для сетки (например, 4)
        num_cols = 4
        cols = st.columns(num_cols)
        selected_file = None
        for idx, filename in enumerate(image_files):
            col = cols[idx % num_cols]
            img_path = os.path.join(DATA_DIR, filename)
            try:
                thumbnail = Image.open(img_path).resize((200, 200))
                col.image(thumbnail, use_container_width=True, caption=filename)
                if col.button(f"Выбрать {filename}", key=filename):
                    selected_file = filename
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")
        if selected_file:
            selected_image = np.array(
                Image.open(os.path.join(DATA_DIR, selected_file)))
    else:
        st.sidebar.warning("Папка с изображениями пуста!")
else:
    uploaded_file = st.sidebar.file_uploader("Загрузите изображение",
                                             type=["jpg", "jpeg", "png"])
    if uploaded_file:
        selected_image = load_image(uploaded_file)

if selected_image is not None:
    st.subheader("Исходное изображение")
    st.image(selected_image, use_container_width=True)

    if method == "Морфологические преобразования":
        st.header(
            "Обработка методом морфологических преобразований")
        # Преобразуем изображение в поток байт
        image_pil = Image.fromarray(selected_image)
        buf = BytesIO()
        image_pil.save(buf, format="PNG")
        buf.seek(0)
        result = call_api_morph(buf)
        if result:
            # Отображение каналов
            cols = st.columns(3)
            for col, key, title in zip(cols, ["r", "g", "b"],
                                       ["Красный канал", "Зелёный канал",
                                        "Синий канал"]):
                with col:
                    st.image(
                        Image.open(BytesIO(base64.b64decode(result[key]))),
                        caption=title, use_container_width=True)

            # Отображение масок для белых яиц
            st.markdown("#### Маски для белых яиц")
            cols = st.columns(3)
            for col, key, title in zip(cols,
                                       ["white_red_mask", "white_green_mask",
                                        "white_blue_mask"],
                                       ["Красная маска", "Зеленая маска",
                                        "Синяя маска"]):
                with col:
                    st.image(
                        Image.open(BytesIO(base64.b64decode(result[key]))),
                        caption=title, use_container_width=True)

            cols = st.columns(2)
            with cols[0]:
                st.image(Image.open(
                    BytesIO(base64.b64decode(result["combined_white"]))),
                    caption="Объединенная маска (белые)",
                    use_container_width=True)
            with cols[1]:
                st.image(Image.open(
                    BytesIO(base64.b64decode(result["result_white"]))),
                    caption="Результат (белые)", use_container_width=True)
            cols = st.columns(2)

            # Отображение масок для коричневых яиц (нижние и верхние)
            st.markdown("#### Маски для коричневых яиц")
            for section in [[("brown_red_mask", "Красная маска"), ("brown_green_mask", "Зелёная маска"),
                             ("brown_blue_mask", "Синяя маска")]]:
                cols = st.columns(3)
                for col, key in zip(cols, section):
                    with col:
                        st.image(
                            Image.open(BytesIO(base64.b64decode(result[key[0]]))),
                            caption=key[1], use_container_width=True)
            cols = st.columns(2)
            # Отображение объединенных масок и результатов
            with cols[0]:
                st.image(Image.open(
                    BytesIO(base64.b64decode(result["combined_brown"]))),
                         caption="Объединенная маска (коричневые)",
                         use_container_width=True)
            with cols[1]:
                st.image(Image.open(
                    BytesIO(base64.b64decode(result["result_brown"]))),
                         caption="Результат (коричневые)",
                         use_container_width=True)

            st.info(f"Количество белых яиц: {result['count_white']}")
            st.info(f"Количество коричневых яиц: {result['count_brown']}")
    else:
        st.header("Обработка UNet")
        mask = infer_image(selected_image)
        processed_mask, original_counts, processed_counts = process_mask(mask)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Исходное изображение")
            st.image(selected_image, use_container_width=True)
        with col2:
            st.subheader("Маска от UNet")
            st.image(mask_to_color(mask), use_container_width=True)
            # Если оригинальные и обработанные счетчики совпадают, фон зелёный, иначе красный
            bg_color = "#d4edda" if original_counts == processed_counts else "#f8d7da"
            st.markdown(
                f"<div style='background-color:{bg_color}; padding:10px; border-radius:5px;'>"
                f"<strong>Белых яиц:</strong> {original_counts['белое яйцо']}<br>"
                f"<strong>Коричневых яиц:</strong> {original_counts['коричневое яйцо']}"
                f"</div>",
                unsafe_allow_html=True
            )
        with col3:
            st.subheader("Обработанная маска")
            st.image(mask_to_color(processed_mask), use_container_width=True)
            st.markdown(
                f"<div style='background-color:#d4edda; padding:10px; border-radius:5px;'>"
                f"<strong>Белых яиц:</strong> {processed_counts['белое яйцо']}<br>"
                f"<strong>Коричневых яиц:</strong> {processed_counts['коричневое яйцо']}"
                f"</div>",
                unsafe_allow_html=True
            )



else:
    st.info("Пожалуйста, выберите или загрузите изображение для анализа.")
