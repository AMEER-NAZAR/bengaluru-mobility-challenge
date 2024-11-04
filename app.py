from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from supervision.geometry.dataclasses import Point
from countVehicle import count_vehicles

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'

# Load YOLOv8 model
model = YOLO('best.pt')  # Replace with your model path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('process_video', filename=file.filename))

@app.route('/process/<filename>')
def process_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(video_path)
    LINE_START = Point(300, 300)
    LINE_END = Point(1600, 700)
    in_count,out_count = count_vehicles(model=model,LINE_START=LINE_START,LINE_END=LINE_END,SOURCE_VIDEO_PATH=video_path)

    counts = {
        'in_count': in_count,
        'out_count': out_count
    }
    
    counts_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'counts.txt')

    with open(counts_file_path, 'w') as f:
        json.dump(counts, f)

    # Redirect to the /result route with the video filename
    return redirect(url_for('result_counter', video_file=filename))

@app.route('/result/<video_file>')
def result_counter(video_file):
    counts_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'counts.txt')
    with open(counts_file_path, 'r') as f:
        counts = json.load(f)
    
    in_count = counts['in_count']
    out_count = counts['out_count']

    return render_template('result.html', in_count=in_count, out_count=out_count, video_file=video_file)

@app.route('/static/uploads/<filename>')
def send_processed_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
