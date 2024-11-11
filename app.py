
# Author: vlarobbyk
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.


from flask import Flask, render_template, Response, request, send_from_directory
from io import BytesIO
import cv2
import numpy as np
import requests
import threading
import time
import os

app = Flask(__name__)
# IP Address
_URL = 'http://192.168.18.46'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])

start_time = time.time()
count_frame = 0

# Objeto de bloqueo para el hilo de la cámara
lock = threading.Lock()
current_frame = None  # Variable para almacenar el frame actual
back_sub = cv2.createBackgroundSubtractorMOG2()

# Sal y pimienta
sal_level = 0
pepper_level = 0

#FILTROS:
median_ksize = 5
blur_ksize = 5
gaussian_ksize = 5

IMAGE_FOLDER = os.path.join(os.getcwd(), 'static', 'images')
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER


mask_sizes = [37, 55, 75]

def video_capture():
    global start_time, count_frame, sal_level, pepper_level

    res = requests.get(stream_url, stream=True)
    for chunk in res.iter_content(chunk_size=100000):

        if len(chunk) > 100:
            try:
                # Decodificación de la imagen
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)

                # Detección de movimiento
                fg_mask = detection_motion(cv_img, back_sub)
                cv2.putText(fg_mask, 'Detector de Movimiento', (10, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 155), 1, cv2.LINE_AA)
                
                # Ecualización de histograma
                img_yuv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                ecualizada = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                cv2.putText(ecualizada, 'Ecualización histograma', (10, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Método CLAHE
                lab_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab_image)
                clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
                clahe_l = clahe.apply(l_channel)
                updated_lab_image = cv2.merge((clahe_l, a_channel, b_channel))
                clahe_image = cv2.cvtColor(updated_lab_image, cv2.COLOR_LAB2BGR)
                cv2.putText(clahe_image, 'CLAHE', (10, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Desenfoque Gaussiano
                gaussian_blurred_image = cv2.GaussianBlur(cv_img, (5, 5), 0)
                cv2.putText(gaussian_blurred_image, 'Desenfoque GaussianoBlur', (10, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                #Imagen en gris
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape

                # Aplicar ruido de sal (píxeles blancos)
                noise = np.copy(cv_img)
                if sal_level > 0:
                    random_positions_salt = (np.random.randint(0, height, sal_level), np.random.randint(0, width, sal_level))
                    noise[random_positions_salt] = 255

                # Aplicar ruido de pimienta (píxeles negros)
                if pepper_level > 0:
                    random_positions_pepper = (np.random.randint(0, height, pepper_level), np.random.randint(0, width, pepper_level))
                    noise[random_positions_pepper] = 0

                noise_image = noise
                cv2.putText(noise_image, 'Sal y pimienta', (10, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                 # Filtros
                median_filtered = cv2.medianBlur(cv_img, median_ksize)  # Filtro de mediana
                blur_filtered = cv2.blur(cv_img, (blur_ksize, blur_ksize))    # Filtro de desenfoque
                gaussian_blurred_image = cv2.GaussianBlur(cv_img, (gaussian_ksize, gaussian_ksize), 0)  # Filtro Gaussiano

                cv2.putText(median_filtered, 'Mediana', (10, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(blur_filtered, 'Desenfoque', (10, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(gaussian_blurred_image, 'Gaussiano', (10, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                #Canny
                edges = cv2.Canny(cv_img, 100, 150)
                canny = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                cv2.putText(canny, 'Canny', (10, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                

                # Imagen total
                total_image = np.zeros((height*2, width * 5, 3), dtype=np.uint8)

                total_image[:height, :width] = cv_img
                total_image[:height, width:width*2] = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                total_image[:height, width*2:width*3] = ecualizada
                total_image[:height, width*3:width*4] = clahe_image
                total_image[:height, width*4:width*5] = gaussian_blurred_image
                total_image[height:height*2, :width] = noise_image
                total_image[height:height*2, width:width*2] = median_filtered
                total_image[height:height*2, width*2:width*3] = blur_filtered
                total_image[height:height*2, width*3:width*4] = gaussian_blurred_image
                total_image[height:height*2, width*4:width*5] = canny

                # Actualizar FPS
                total_image, start_time, count_frame = show_fps(total_image, start_time, count_frame)

                flag, encodedImage = cv2.imencode(".jpg", total_image)

                if not flag:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            except Exception as e:
                print(e)
                continue

def show_fps(cv_img, start_time, count_frame):
    elapsed_time = time.time() - start_time
    count_frame += 1
    fps = count_frame / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(cv_img, f'FPS: {fps:.2f}', (10, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return cv_img, start_time, count_frame

def detection_motion(cv_img, back_sub):
    fg_mask = back_sub.apply(cv_img)
    return fg_mask

def apply_morphological_operations(image_path, mask_size):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        kernel = np.ones((mask_size, mask_size), np.uint8)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((mask_size, mask_size), np.uint8)

        erosion = cv2.erode(img, kernel, iterations=1)
        dilation = cv2.dilate(img, kernel, iterations=1)
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        
        # Operación imagen original + (Top Hat - Black Hat)
        combined = cv2.add(img, cv2.subtract(tophat, blackhat))

        result_files = {}
        result_files['erosion'] = 'erosion_{}.jpg'.format(mask_size)
        result_files['dilation'] = 'dilation_{}.jpg'.format(mask_size)
        result_files['tophat'] = 'tophat_{}.jpg'.format(mask_size)
        result_files['blackhat'] = 'blackhat_{}.jpg'.format(mask_size)
        result_files['combined'] = 'combined_{}.jpg'.format(mask_size)

        cv2.imwrite(os.path.join(IMAGE_FOLDER, result_files['erosion']), erosion)
        cv2.imwrite(os.path.join(IMAGE_FOLDER, result_files['dilation']), dilation)
        cv2.imwrite(os.path.join(IMAGE_FOLDER, result_files['tophat']), tophat)
        cv2.imwrite(os.path.join(IMAGE_FOLDER, result_files['blackhat']), blackhat)
        cv2.imwrite(os.path.join(IMAGE_FOLDER, result_files['combined']), combined)

        return result_files
    except Exception as e:
        print(f"Error al aplicar operaciones morfológicas: {e}")
        raise

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_stream")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/update_noise", methods=["POST"])
def update_noise():
    global sal_level, pepper_level
    sal_level = int(request.form.get("sal", 0))
    pepper_level = int(request.form.get("pimienta", 0))
    return "OK", 200

@app.route("/update_filters", methods=["POST"])
def update_filters():
    global median_ksize, blur_ksize, gaussian_ksize
    median_ksize = [3, 5, 7][int(request.form.get("median", 0))]
    blur_ksize = [3, 5, 7][int(request.form.get("blur", 0))]
    gaussian_ksize = [3, 5, 7][int(request.form.get("gaussian", 0))]
    return "OK", 200

@app.route("/apply_operations", methods=['POST'])
def apply_operations():
    image_name = request.form.get('image_name')
    mask_size = int(request.form.get('mask_size'))
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    result_files = apply_morphological_operations(image_path, mask_size)
    return render_template('index.html', result_files=result_files, image_name=image_name)

@app.route("/images/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=False)
