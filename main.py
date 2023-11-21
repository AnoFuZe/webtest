# app.py

from flask import Flask, render_template, request, send_file
import os
from PIL import Image, ImageDraw, ImageFont  # Pillow library for image manipulation
import textwrap
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    generated_image_path = generate_image(prompt)
    return render_template('index.html', generated_image_path=generated_image_path)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join('static', 'generated_images', filename), as_attachment=True)

def generate_image(prompt):
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    adapter_id = "latent-consistency/lcm-lora-sdxl"

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()
    image = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Wrap text to fit in the image
    wrapped_text = textwrap.fill(prompt, width=20)
    
    draw.text((10, 10), wrapped_text, font=font, fill='black')

    # Save the generated image
    generated_image_path = os.path.join('static', 'generated_images', 'generated_image.png')
    image.save(generated_image_path)
    return generated_image_path

if __name__ == '__main__':
    os.makedirs(os.path.join('static', 'generated_images'), exist_ok=True)
    app.run(debug=True)
