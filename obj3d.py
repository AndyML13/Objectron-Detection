import cv2
import mediapipe as mp
import time
import os
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron


IMAGE_FILES = [
      "img/1.jpeg",
      "img/2.jpeg",
      "img/3.jpeg",
]

# Инициализация метрик качества
precision = 0
recall = 0
F1 = 0

# Параметры для расчета метрик
total_original_objects = 0
total_detected_objects = 0
correctly_detected_objects = 0


with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Shoe') as objectron:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.detected_objects:
      print(f'No box landmarks detected on {file}')
      continue
    print(f'Box landmarks of {file}:')
    # Получение количества детектированных объектов на текущем изображении
    num_detected_objects = len(results.detected_objects)
    total_detected_objects += num_detected_objects
    # Подсчет метрик
    original_objects = 1  # Ваш код для определения числа объектов на оригинальном изображении
    total_original_objects += original_objects
    # Проверяем, есть ли детектированные объекты на текущем изображении
    if num_detected_objects > 0:
       correctly_detected_objects += 1 

    annotated_image = image.copy()
    for detected_object in results.detected_objects:
      mp_drawing.draw_landmarks(
          annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
      mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                           detected_object.translation)
      cv2.imwrite('result/annotated_image' + str(idx) + '.png', annotated_image)

# Рассчитываем метрики точности, полноты и F1-меры
if total_detected_objects > 0:
    precision = correctly_detected_objects / total_detected_objects
if total_original_objects > 0:
    recall = correctly_detected_objects / total_original_objects
if precision + recall > 0:
    F1 = 2 * precision * recall / (precision + recall)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", F1)


# Путь к папке с оригинальными изображениями
image_folder = "img"
# Путь к папке с изображениями с обнаруженными объектами
result_folder = "result"


# Загрузка изображений с обнаруженными объектами из папки 'result'
result_files = [os.path.join(result_folder, f"annotated_image{idx}.png") for idx in range(len(IMAGE_FILES))]

# Объединение изображений
for idx, (image_file, result_file) in enumerate(zip(IMAGE_FILES, result_files)):
    # Загрузка оригинального изображения
    original_image = cv2.imread(image_file)
    # Загрузка изображения с обнаруженными объектами
    detected_image = cv2.imread(result_file)
    
    # Проверка, что изображения загружены успешно
    if original_image is None or detected_image is None:
        print(f"Ошибка при загрузке изображений: {image_file}, {result_file}")
        continue
    
    # Объединение изображений
    combined_image = cv2.hconcat([original_image, detected_image])
    
    window_size = (1280, 720) # Ширина и высота окна в пикселях
    cv2.namedWindow("Combined Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Combined Image", window_size[0], window_size[1])
    # Сохранение или отображение объединенного изображения
    cv2.imwrite(f"combined_images/combined_image{idx}.png", combined_image)
    cv2.imshow("Combined Image", combined_image)
    cv2.waitKey(0)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Shoe') as objectron:
  while cap.isOpened():
    success, image = cap.read()
    start = time.time()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = objectron.process(image)

    # Draw the box landmarks on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(
              image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation,
                                 detected_object.translation)
    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(image, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (1, 50, 32), 2)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Objectron', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()