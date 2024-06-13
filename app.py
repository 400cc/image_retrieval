from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import torch
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# 모델 및 함수 불러오기
from image_search import process_image_and_feature

# 이미지 업로드를 위한 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        # 이미지 저장
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 입력 데이터 가져오기
        input_data = request.form.get('input_data')
        
        image, image_feature = process_image_and_feature(filepath, input_data)
        
        # 이미지를 base64로 인코딩하여 전송
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()
        img_base64 = base64.b64encode(img_byte_array).decode('utf-8')
        
        # 이미지 특징을 JSON 형태로 반환
        image_features_list = [feat.tolist() for feat in image_feature]
        return jsonify({'image': img_base64, 'image_features': image_features_list})
    else:
        return jsonify({'error': 'File not allowed'})
    
if __name__ == '__main__':
    app.run(debug=True)
