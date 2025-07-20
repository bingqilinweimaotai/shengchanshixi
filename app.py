from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import numpy as np
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import logging
from rembg import remove
from typing import Optional, Union

# 修复PIL.Image.LANCZOS兼容性问题
try:
    LANCZOS = Image.LANCZOS
except AttributeError:
    try:
        LANCZOS = Image.ANTIALIAS
    except AttributeError:
        LANCZOS = Image.BICUBIC

# MODNet相关导入
try:
    from src.models.modnet import MODNet
    MODNET_AVAILABLE = True
except ImportError:
    print("警告: MODNet模型未找到，将使用rembg作为备选方案")
    from rembg import remove
    MODNET_AVAILABLE = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_FOLDER'] = 'pretrained/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 全局变量存储模型
modnet_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_modnet_model():
    """加载MODNet模型"""
    global modnet_model
    
    if not MODNET_AVAILABLE:
        return None
        
    if modnet_model is not None:
        return modnet_model
    
    try:
        ckpt_path = os.path.join(app.config['MODEL_FOLDER'], 'modnet_photographic_portrait_matting.ckpt')
        
        if not os.path.exists(ckpt_path):
            print(f"MODNet模型文件不存在: {ckpt_path}")
            return None
        
        # 初始化模型
        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet)
        
        # 加载预训练权重
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        
        # 处理不同的checkpoint格式
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        
        # 处理DataParallel的键名前缀问题
        if any(k.startswith('module.') for k in state_dict.keys()):
            modnet.load_state_dict(state_dict)
        else:
            # 如果没有module前缀，需要添加
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[f'module.{k}'] = v
            modnet.load_state_dict(new_state_dict)
        
        modnet.eval()
        modnet_model = modnet
        print("MODNet模型加载成功")
        return modnet_model
        
    except Exception as e:
        print(f"MODNet模型加载失败: {e}")
        return None

def preprocess_image_for_modnet(image_path, ref_size=512):
    """为MODNet预处理图像，自动调整为32的倍数"""
    try:
        im = Image.open(image_path)

        if im.mode != 'RGB':
            im = im.convert('RGB')

        # resize 保持比例 + 裁剪成32的倍数
        im_size = im.size
        ratio = min(ref_size / max(im_size), 1.0)
        new_size = tuple([int(x * ratio) for x in im_size])
        new_size = (new_size[0] // 32 * 32, new_size[1] // 32 * 32)  # 裁剪到32的倍数

        im = im.resize(new_size, LANCZOS)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return transform(im).unsqueeze(0), new_size
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None, None


def modnet_remove_background(image_path):
    """使用MODNet移除背景"""
    model = load_modnet_model()
    if model is None:
        return None
    
    try:
        # 预处理图像
        im_tensor, im_size = preprocess_image_for_modnet(image_path)
        if im_tensor is None:
            return None
        
        # 推理
        with torch.no_grad():
            _, _, matte = model(im_tensor, True)
        
        # 后处理遮罩
        matte = matte[0][0].data.cpu().numpy()
        matte = np.clip(matte, 0, 1)
        
        # 调整回原始尺寸
        matte_resized = Image.fromarray((matte * 255).astype(np.uint8)).resize(im_size, LANCZOS)
        matte_np = np.array(matte_resized) / 255.0
        
        # 加载原始图像
        original_img = Image.open(image_path).convert('RGB')
        original_np = np.array(original_img)
        
        # 创建RGBA图像
        rgba_img = np.zeros((original_np.shape[0], original_np.shape[1], 4), dtype=np.uint8)
        rgba_img[:, :, :3] = original_np  # RGB通道
        rgba_img[:, :, 3] = (matte_np * 255).astype(np.uint8)  # Alpha通道
        
        # 转换为PIL图像
        result_img = Image.fromarray(rgba_img, 'RGBA')
        return result_img
        
    except Exception as e:
        print(f"MODNet背景移除失败: {e}")
        return None

def remove_background(image_path):
    """移除背景的统一接口"""
    # 优先使用MODNet
    if MODNET_AVAILABLE:
        result = modnet_remove_background(image_path)
        if result is not None:
            return result
        print("MODNet处理失败，回退到rembg")
    
    # 备选方案：使用rembg
    try:
        with open(image_path, 'rb') as f:
            input_data = f.read()
        output_data = remove(input_data)
        # 修复类型兼容性问题
        if isinstance(output_data, bytes):
            return Image.open(io.BytesIO(output_data)).convert('RGBA')
        else:
            return output_data
    except Exception as e:
        print(f"rembg处理也失败: {e}")
        return None

def save_image_high_quality(img, output_path, format='JPEG'):
    """高质量保存图片，保持原图品质"""
    try:
        if format.upper() == 'JPEG':
            # JPEG格式：最高质量，无优化压缩
            img.save(output_path, 'JPEG', quality=100, optimize=False, progressive=False)
        elif format.upper() == 'PNG':
            # PNG格式：无损压缩，最高质量
            img.save(output_path, 'PNG', optimize=False, compress_level=0)
        else:
            # 其他格式：默认高质量
            img.save(output_path, format, quality=100)
        return True
    except Exception as e:
        print(f"保存图片失败: {e}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/background', methods=['GET', 'POST'])
def background():
    original_img = None
    removed_bg_img = None
    final_img = None
    processing_method = "rembg"

    if request.method == 'POST':
        file = request.files.get('file')
        bg_color = request.form.get('bg_color', '#ffffff')

        if file and allowed_file(file.filename):
            try:
                # 1. 保存原图 - 保持原格式和质量
                filename = secure_filename(file.filename) if file.filename else "uploaded_image"
                original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'original_{filename}')
                file.save(original_path)

                # 2. 使用 rembg 移除背景
                with open(original_path, 'rb') as f:
                    input_bytes = f.read()

                output_bytes = remove(input_bytes)  # 调用 rembg
                # 修复类型兼容性问题
                if isinstance(output_bytes, bytes):
                    removed_img = Image.open(io.BytesIO(output_bytes)).convert('RGBA')
                else:
                    removed_img = output_bytes

                # 保存去背景图片（用于显示）- 高质量保存
                removed_path = os.path.join(app.config['UPLOAD_FOLDER'], f'removed_{filename}')
                preview_bg = Image.new("RGB", removed_img.size, (255, 255, 255))
                # 确保removed_img是PIL.Image对象
                if hasattr(removed_img, 'split'):
                    preview_bg.paste(removed_img, mask=removed_img.split()[3])
                save_image_high_quality(preview_bg, removed_path, 'JPEG')

                # 3. 应用新背景色
                if bg_color.startswith('#'):
                    bg_color = bg_color[1:]
                bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))

                final_bg = Image.new('RGBA', removed_img.size, bg_rgb + (255,))
                # 确保removed_img是PIL.Image对象
                if hasattr(removed_img, 'split'):
                    final_bg.paste(removed_img, mask=removed_img.split()[3])

                # 保存最终图 - 高质量保存
                final_path = os.path.join(app.config['UPLOAD_FOLDER'], f'final_{filename}')
                save_image_high_quality(final_bg.convert('RGB'), final_path, 'JPEG')

                # 设置图片路径供前端展示
                original_img = '/' + original_path
                removed_bg_img = '/' + removed_path
                final_img = '/' + final_path

            except Exception as e:
                print(f"处理过程中出错: {e}")
                return render_template('background.html', error=f"处理失败: {str(e)}")

    return render_template(
        'background.html',
        original_img=original_img,
        removed_bg_img=removed_bg_img,
        final_img=final_img,
        processing_method=processing_method
    )

@app.route('/crop', methods=['GET', 'POST'])
def crop():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = "crop_" + (file.filename or "image")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            output = crop_image(filepath)
            return send_file(output, mimetype='image/png', as_attachment=True, download_name='cropped.png')
    return render_template('crop.html')

@app.route('/watermark', methods=['GET', 'POST'])
def watermark():
    if request.method == 'POST':
        file = request.files['file']
        watermark_text = request.form.get('watermark', '')
        if file and allowed_file(file.filename):
            filename = "watermark_" + (file.filename or "image")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            output = add_watermark(filepath, watermark_text)
            return send_file(output, mimetype='image/png', as_attachment=True, download_name='watermarked.png')
    return render_template('watermark.html')

@app.route('/remove_watermark_advanced', methods=['GET', 'POST'])
def remove_watermark_advanced():
    if request.method == 'POST':
        file = request.files.get('file')
        mask_file = request.files.get('mask')
        
        if file and allowed_file(file.filename):
            filename = 'advanced_watermark_' + (file.filename or "image")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 高级去水印功能需要用户提供mask
            if mask_file and allowed_file(mask_file.filename):
                mask_filename = 'mask_' + (mask_file.filename or "mask")
                mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
                mask_file.save(mask_path)
                output = remove_watermark_with_mask(filepath, mask_path)
            else:
                # 如果没有提供mask，返回错误
                return render_template('remove_watermark_advanced.html', error='请使用画笔工具标注要去除的水印区域')
            
            if output:
                # 使用最高质量参数返回文件
                return send_file(output, mimetype='image/png', as_attachment=True, download_name='no_watermark.png')
            else:
                return render_template('remove_watermark_advanced.html', error='处理失败')
    
    return render_template('remove_watermark_advanced.html')


def remove_watermark_with_mask(image_path, mask_path):
    """使用用户提供的mask去除水印"""
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None:
        return None
    
    # 确保mask和图片尺寸一致
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    
    # 使用inpaint算法
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    # 边缘保持滤波优化
    inpainted = cv2.bilateralFilter(inpainted, 9, 75, 75)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'no_watermark_advanced.png')
    # 使用最高质量保存
    cv2.imwrite(output_path, inpainted, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return output_path

@app.route('/api/model_status')
def model_status():
    """API接口：返回模型状态"""
    status = {
        'modnet_available': MODNET_AVAILABLE,
        'modnet_loaded': modnet_model is not None,
        'fallback': 'rembg'
    }
    return jsonify(status)

@app.route('/portrait_cutout', methods=['GET', 'POST'])
def portrait_cutout():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) if file.filename else "uploaded_image"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 调用背景去除或抠图函数
            result_img = remove_background(filepath)
            if result_img is None:
                return render_template('portrait_cutout.html', error='背景去除失败')

            # 保存结果图，方便前端展示
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cutout_' + filename)
            
            # 如果图片带透明通道，保存PNG；否则保存JPEG
            if hasattr(result_img, 'mode') and result_img.mode == 'RGBA':
                # 保存为PNG格式（保留透明）- 高质量保存
                base, ext = os.path.splitext(result_path)
                result_path = base + '.png'
                save_image_high_quality(result_img, result_path, 'PNG')
            else:
                # 非RGBA直接保存JPEG - 高质量保存
                save_image_high_quality(result_img, result_path, 'JPEG')

            return render_template('portrait_cutout.html',
                                   original_img='/' + filepath.replace('\\', '/'),
                                   cutout_img='/' + result_path.replace('\\', '/'))

    return render_template('portrait_cutout.html')

def change_background(filepath, bg_color):
    """更改背景颜色（保持兼容性）"""
    removed_img = remove_background(filepath)
    if removed_img is None:
        return None
    
    bg = Image.new('RGBA', removed_img.size, bg_color)
    # 确保removed_img是PIL.Image对象
    if hasattr(removed_img, 'split'):
        bg.paste(removed_img, mask=removed_img.split()[3])
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'background_changed.png')
    save_image_high_quality(bg.convert('RGB'), output_path, 'JPEG')
    return output_path

def crop_image(filepath):
    """裁剪图像"""
    img = Image.open(filepath)
    width, height = img.size
    left = width * 0.1
    top = height * 0.1
    right = width * 0.9
    bottom = height * 0.9
    cropped_img = img.crop((left, top, right, bottom))
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped.png')
    # 高质量保存PNG
    save_image_high_quality(cropped_img, output_path, 'PNG')
    return output_path

def add_watermark(filepath, watermark):
    """添加水印"""
    img = Image.open(filepath).convert("RGBA")
    txt = Image.new('RGBA', img.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt)
    try:
        font = ImageFont.truetype("static/fonts/Tuffy.ttf", 40)
    except:
        font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), watermark, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = img.width - text_width - 10
    y = img.height - text_height - 10
    draw.text((x, y), watermark, fill=(255, 255, 255, 128), font=font)
    combined = Image.alpha_composite(img, txt)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'watermarked.png')
    # 高质量保存
    save_image_high_quality(combined.convert('RGB'), output_path, 'JPEG')
    return output_path

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    
    # 预加载MODNet模型
    if MODNET_AVAILABLE:
        print("正在预加载MODNet模型...")
        load_modnet_model()
    
    app.run(debug=True, port=5001)
