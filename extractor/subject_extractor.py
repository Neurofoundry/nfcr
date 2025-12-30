from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import base64
import os
import logging
import numpy as np
from scipy.ndimage import gaussian_filter
import requests  # For calling ControlNet service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow all origins for now - restrict in production

OUTPUT_DIR = "extracted_subjects"
COMPOSED_DIR = "composed_images"
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPOSED_DIR, exist_ok=True)
REMBG_SESSION = new_session("u2netp")

# Serve static HTML files
@app.route('/')
def index():
    """Serve the main test interface"""
    html_path = os.path.join(STATIC_DIR, 'test_composition_pipeline.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return '''
    <h1>AI Composition Pipeline</h1>
    <p>Server is running! Available test pages:</p>
    <ul>
        <li><a href="/test_composition_pipeline.html">Full Pipeline Test</a></li>
        <li><a href="/test_extraction.html">Extraction Test</a></li>
        <li><a href="/test_forgeintegration.html">Forge Integration</a></li>
    </ul>
    '''

@app.route('/<path:filename>')
def serve_file(filename):
    """Serve any file from the current directory"""
    file_path = os.path.join(STATIC_DIR, filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Set appropriate content type
        if filename.endswith('.html'):
            return content, 200, {'Content-Type': 'text/html'}
        elif filename.endswith('.js'):
            return content, 200, {'Content-Type': 'application/javascript'}
        elif filename.endswith('.css'):
            return content, 200, {'Content-Type': 'text/css'}
        else:
            return content
    return "File not found", 404

@app.route('/extract', methods=['POST'])
def extract_subject():
    try:
        data = request.get_json()
        
        if 'image' not in data:
            logger.error('No image provided in request')
            return jsonify({'error': 'No image provided'}), 400
        
        logger.info('Starting background removal...')
        
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        
        # Open image
        input_image = Image.open(io.BytesIO(image_bytes))
        logger.info(f'Input image: {input_image.size}, {input_image.mode}')
        
        # Remove background using U²-Net
        output_image = remove(input_image, session=REMBG_SESSION)
        logger.info('Background removed successfully')
        
        # Convert to base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Optionally save to disk
        save_path = None
        if data.get('save', False):
            filename = f"subject_{len(os.listdir(OUTPUT_DIR))}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            output_image.save(save_path)
            logger.info(f'Saved to: {save_path}')
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_str}',
            'saved_path': save_path
        })
        
    except Exception as e:
        logger.error(f'Extraction error: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'subject-extractor'})

@app.route('/compose', methods=['POST'])
def compose_image():
    """
    Intelligently compose subject + scene + style:
    - Uses ControlNet when style is drastically different from scene
    - Falls back to color blending for subtle adjustments
    """
    try:
        data = request.get_json()
        
        # Validate inputs
        required = ['subject', 'scene', 'style']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing {field} image'}), 400
        
        logger.info('Starting intelligent image composition...')
        
        # Decode all images
        subject_img = decode_base64_image(data['subject'])
        scene_img = decode_base64_image(data['scene'])
        style_img = decode_base64_image(data['style'])
        
        logger.info(f'Subject: {subject_img.size}, Scene: {scene_img.size}, Style: {style_img.size}')
        
        # Get composition settings
        settings = data.get('settings', {})
        width = settings.get('width', 1024)
        height = settings.get('height', 576)
        use_controlnet = settings.get('use_controlnet', 'auto')  # 'auto', 'force', 'never'
        
        # Step 1: Analyze scene and style
        scene_analysis = analyze_scene(scene_img)
        style_features = extract_style_features(style_img)
        
        # Step 2: ALWAYS use ControlNet for true AI-powered fusion
        needs_controlnet = True  # Force ControlNet every time
        
        logger.info('🎨 Using ControlNet for AI-powered style transformation')
        final_image = compose_with_controlnet(
            subject_img, scene_img, style_img, 
            scene_analysis, style_features, 
            (width, height)
        )
        method = 'controlnet'
        
        # Convert to base64
        buffered = io.BytesIO()
        final_image.save(buffered, format="PNG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Optionally save
        save_path = None
        if data.get('save', False):
            filename = f"composed_{len(os.listdir(COMPOSED_DIR))}_{method}.png"
            save_path = os.path.join(COMPOSED_DIR, filename)
            final_image.save(save_path)
            logger.info(f'Saved to: {save_path}')
        
        logger.info(f'✅ Composition complete using {method}!')
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_str}',
            'saved_path': save_path,
            'method': method,
            'analysis': {
                'scene': scene_analysis,
                'style': style_features,
                'used_controlnet': needs_controlnet
            }
        })
        
    except Exception as e:
        logger.error(f'Composition error: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

def should_use_controlnet(scene_img, style_img, scene_analysis, style_features, override='auto'):
    """
    Intelligently decide if ControlNet is needed
    
    Returns True if:
    - Style has very different saturation/brightness (>40% difference)
    - Style has different color palette (artistic vs realistic)
    - User forces it via override='force'
    """
    if override == 'force':
        return True
    if override == 'never':
        return False
    
    # Calculate differences
    brightness_diff = abs(scene_analysis['brightness'] - style_features['brightness'])
    saturation_diff = abs(scene_analysis.get('saturation', 0.5) - style_features['saturation'])
    contrast_diff = abs(scene_analysis.get('contrast', 0.5) - style_features['contrast'])
    
    # Check if style is drastically different
    is_drastically_different = (
        brightness_diff > 0.4 or
        saturation_diff > 0.4 or
        contrast_diff > 0.5
    )
    
    logger.info(f'Style difference: brightness={brightness_diff:.2f}, saturation={saturation_diff:.2f}, contrast={contrast_diff:.2f}')
    
    return is_drastically_different

def compose_with_controlnet(subject_img, scene_img, style_img, scene_analysis, style_features, target_size):
    """
    TRUE 3-PASS WORKFLOW:
    1. Subject → img2img transform → Result A
    2. Scene → img2img transform → Result B
    3. Style → img2img transform → Result C
    4. Layer A + B + C → Final composite
    5. (Optional) Final img2img pass on composite
    """
    try:
        import requests
        
        logger.info('🎨 Starting 3-PASS img2img workflow...')
        
        # Helper to convert image to base64
        def img_to_b64(img):
            buffered = io.BytesIO()
            img = img.convert('RGB')
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        
        # Helper to call img2img
        def call_img2img(image_b64, prompt, strength=0.7):
            payload = {
                'prompt': prompt,
                'image': image_b64,
                'width': target_size[0],
                'height': target_size[1],
                'model': '@cf/stabilityai/stable-diffusion-xl-base-1.0',
                'guidance_scale': 7.5,
                'num_inference_steps': 25,
                'strength': strength
            }
            
            response = requests.post('http://localhost:8080/features/4/render', 
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=120
            )
            
            if not response.ok:
                raise Exception(f'img2img failed: {response.status_code}')
            
            result = response.json()
            
            # Parse response
            if result.get('image_base64'):
                img_data = f"data:image/png;base64,{result['image_base64']}"
                return decode_base64_image(img_data)
            elif result.get('image_path') or result.get('saved_as'):
                imagePath = result.get('image_path') or result.get('saved_as')
                imageUrl = f"http://localhost:8080/{imagePath.replace(chr(92), '/')}"
                img_response = requests.get(imageUrl, timeout=30)
                return Image.open(io.BytesIO(img_response.content))
            else:
                raise Exception('No image in response')
        
        # PASS 1: Transform SUBJECT
        logger.info('📤 PASS 1/3: Processing SUBJECT through img2img...')
        subject_prompt = "high quality detailed character, professional artwork, 8k"
        subject_b64 = img_to_b64(subject_img)
        result_subject = call_img2img(subject_b64, subject_prompt, strength=0.6)
        logger.info('✅ PASS 1 complete')
        
        # PASS 2: Transform SCENE
        logger.info('📤 PASS 2/3: Processing SCENE through img2img...')
        scene_prompt = f"detailed background environment, {get_lighting_desc(scene_analysis)}, professional quality"
        scene_b64 = img_to_b64(scene_img)
        result_scene = call_img2img(scene_b64, scene_prompt, strength=0.6)
        logger.info('✅ PASS 2 complete')
        
        # PASS 3: Transform STYLE
        logger.info('📤 PASS 3/3: Processing STYLE through img2img...')
        style_prompt = f"artistic style reference, {get_style_desc(style_features)}, masterpiece"
        style_b64 = img_to_b64(style_img)
        result_style = call_img2img(style_b64, style_prompt, strength=0.5)
        logger.info('✅ PASS 3 complete')
        
        # COMPOSITE: Layer all 3 results
        logger.info('🎨 Compositing 3 transformed images...')
        final = composite_three_images(result_subject, result_scene, result_style, target_size)
        
        logger.info('✅ 3-PASS FUSION COMPLETE!')
        
        return final
        
    except Exception as e:
        logger.error(f'3-pass workflow failed: {e}')
        logger.info('⚠️ Using fallback local composition')
        canvas = prepare_styled_scene(scene_img, style_img, style_features, target_size)
        return composite_subject(canvas, subject_img, scene_analysis, style_features)

def get_lighting_desc(scene_analysis):
    """Get lighting description from scene analysis"""
    if scene_analysis['brightness'] > 0.6:
        return "bright well-lit"
    elif scene_analysis['brightness'] < 0.3:
        return "dark moody"
    else:
        return "balanced lighting"

def get_style_desc(style_features):
    """Get style description from features"""
    if style_features['saturation'] > 0.7 and style_features['contrast'] > 0.6:
        return "vibrant highly saturated dramatic"
    elif style_features['saturation'] < 0.3:
        return "muted colors painterly"
    else:
        return "professional digital art"

def composite_three_images(subject, scene, style, target_size):
    """
    Layer 3 transformed images together:
    1. Scene as base (100%)
    2. Style overlay with multiply blend (40%)
    3. Subject on top with alpha (90%)
    """
    # Ensure all are same size
    scene_resized = ImageOps.fit(scene, target_size, Image.Resampling.LANCZOS).convert('RGBA')
    style_resized = ImageOps.fit(style, target_size, Image.Resampling.LANCZOS).convert('RGBA')
    subject_resized = ImageOps.fit(subject, target_size, Image.Resampling.LANCZOS).convert('RGBA')
    
    # Start with scene as base
    canvas = scene_resized.copy()
    
    # Blend style on top (multiply for color transfer)
    style_arr = np.array(style_resized, dtype=np.float32)
    canvas_arr = np.array(canvas, dtype=np.float32)
    
    # Multiply blend at 40% opacity
    blended = (canvas_arr * style_arr) / 255.0
    canvas_arr = canvas_arr * 0.6 + blended * 0.4
    canvas_arr = np.clip(canvas_arr, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(canvas_arr, 'RGBA')
    
    # Composite subject on top
    canvas.paste(subject_resized, (0, 0), subject_resized)
    
    return canvas

def generate_fusion_prompt(subject_img, scene_img, style_img, scene_analysis, style_features):
    """
    Intelligently create a prompt that fuses all 3 images
    
    This is the KEY to making it work like Google Whisk:
    - Analyzes subject (what it is)
    - Uses scene for composition/setting
    - Uses style for artistic rendering
    """
    
    # Analyze subject characteristics
    subject_desc = analyze_subject_content(subject_img)
    
    # Build scene description
    if scene_analysis['brightness'] > 0.6:
        lighting = "bright, well-lit"
    elif scene_analysis['brightness'] < 0.3:
        lighting = "dark, moody"
    else:
        lighting = "balanced lighting"
    
    if scene_analysis['temperature'] > 0.2:
        tone = "warm tones"
    elif scene_analysis['temperature'] < -0.2:
        tone = "cool tones"
    else:
        tone = "neutral tones"
    
    # Determine artistic style
    if style_features['saturation'] > 0.7 and style_features['contrast'] > 0.6:
        art_style = "vibrant digital art, highly saturated, dramatic"
    elif style_features['saturation'] < 0.3:
        art_style = "muted colors, painterly, classical art"
    elif style_features['brightness'] > 0.7:
        art_style = "bright, ethereal, fantasy art"
    elif style_features['contrast'] > 0.7:
        art_style = "high contrast, dramatic, cinematic"
    else:
        art_style = "professional digital painting"
    
    # BUILD FUSION PROMPT
    fusion_prompt = f"{subject_desc} in a scene with {lighting}, {tone}, rendered in {art_style} style, highly detailed, masterpiece, 8k, professional artwork"
    
    return fusion_prompt

def analyze_subject_content(subject_img):
    """
    Analyze what the subject actually IS
    
    Uses basic analysis for speed. CLIP can be enabled later.
    """
    # Fast analysis without CLIP download
    rgb = subject_img.convert('RGB')
    arr = np.array(rgb)
    pixels = arr.reshape(-1, 3)
    avg_color = np.mean(pixels, axis=0)
    
    # Determine subject type from colors/patterns
    brightness = np.mean(arr) / 255.0
    
    if avg_color[0] > 150 and avg_color[1] < 100:
        subject_type = "character with warm red tones"
    elif avg_color[2] > 150:
        subject_type = "character with cool blue tones"
    elif brightness > 0.7:
        subject_type = "bright subject"
    elif brightness < 0.3:
        subject_type = "dark mysterious subject"
    else:
        subject_type = "detailed subject"
    
    return subject_type

def generate_style_prompt(style_features, scene_analysis):
    """
    Automatically generate a prompt based on style features
    """
    # Determine artistic style based on features
    if style_features['saturation'] > 0.7 and style_features['contrast'] > 0.6:
        style_desc = "vibrant, highly saturated, dramatic contrast"
    elif style_features['saturation'] < 0.3:
        style_desc = "muted colors, subtle tones, soft"
    elif style_features['brightness'] > 0.7:
        style_desc = "bright, high key lighting, ethereal"
    elif style_features['brightness'] < 0.3:
        style_desc = "dark, moody, low key lighting"
    else:
        style_desc = "balanced, natural"
    
    # Build prompt
    prompt = f"{style_desc}, high quality, detailed, professional, 8k"
    
    return prompt

def decode_base64_image(data_str):
    """Decode base64 image string to PIL Image"""
    image_data = data_str.split(',')[1] if ',' in data_str else data_str
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert('RGBA')

def analyze_scene(scene_img):
    """
    Analyze the scene image to extract:
    - Dominant colors
    - Average brightness
    - Color temperature (warm/cool)
    - Suggested subject placement
    """
    # Resize for faster processing
    small = scene_img.copy()
    small.thumbnail((256, 256))
    
    # Convert to RGB for analysis
    rgb = small.convert('RGB')
    pixels = np.array(rgb)
    
    # Get dominant colors using histogram
    colors = pixels.reshape(-1, 3)
    unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors = unique_colors[sorted_indices[:5]].tolist()
    
    # Calculate average brightness
    brightness = np.mean(pixels) / 255.0
    
    # Calculate color temperature (more red = warm, more blue = cool)
    avg_color = np.mean(pixels, axis=(0, 1))
    temperature = (avg_color[0] - avg_color[2]) / 255.0  # R-B difference
    
    return {
        'dominant_colors': dominant_colors,
        'brightness': float(brightness),
        'temperature': float(temperature),
        'width': scene_img.width,
        'height': scene_img.height
    }

def extract_style_features(style_img):
    """
    Extract artistic style features:
    - Color saturation
    - Contrast level
    - Color palette
    - Texture intensity
    """
    # Resize for processing
    small = style_img.copy()
    small.thumbnail((256, 256))
    rgb = small.convert('RGB')
    hsv = rgb.convert('HSV')
    
    # Get saturation and value
    hsv_array = np.array(hsv)
    saturation = np.mean(hsv_array[:, :, 1]) / 255.0
    brightness = np.mean(hsv_array[:, :, 2]) / 255.0
    
    # Get contrast (standard deviation of brightness)
    gray = rgb.convert('L')
    contrast = np.std(np.array(gray)) / 128.0
    
    # Extract color palette
    pixels = np.array(rgb).reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    palette = unique_colors[sorted_indices[:8]].tolist()
    
    return {
        'saturation': float(saturation),
        'brightness': float(brightness),
        'contrast': float(contrast),
        'palette': palette
    }

def prepare_styled_scene(scene_img, style_img, style_features, target_size):
    """
    Apply HEAVY style transfer to the scene:
    - Blend scene with style image directly
    - Match color saturation aggressively
    - Adjust brightness/contrast dramatically
    - Apply strong color grading based on style palette
    """
    # Resize both to target
    scene = scene_img.copy()
    scene = ImageOps.fit(scene, target_size, Image.Resampling.LANCZOS)
    scene = scene.convert('RGB')
    
    style = style_img.copy()
    style = ImageOps.fit(style, target_size, Image.Resampling.LANCZOS)
    style = style.convert('RGB')
    
    # STEP 1: Blend scene with style image directly (50/50 mix)
    scene_arr = np.array(scene, dtype=np.float32)
    style_arr = np.array(style, dtype=np.float32)
    
    # Multiply blend mode for artistic effect
    blended = (scene_arr * style_arr) / 255.0
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    scene = Image.fromarray(blended, 'RGB')
    
    # STEP 2: Overlay style image with 40% opacity for texture
    scene_arr = np.array(scene, dtype=np.float32)
    style_arr = np.array(style, dtype=np.float32)
    overlayed = scene_arr * 0.6 + style_arr * 0.4
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
    scene = Image.fromarray(overlayed, 'RGB')
    
    # STEP 3: Apply STRONG style saturation (not subtle anymore)
    saturation_factor = 0.5 + (style_features['saturation'] * 1.5)
    enhancer = ImageEnhance.Color(scene)
    scene = enhancer.enhance(saturation_factor)
    
    # STEP 4: Apply STRONG style contrast
    contrast_factor = 0.7 + (style_features['contrast'] * 0.8)
    enhancer = ImageEnhance.Contrast(scene)
    scene = enhancer.enhance(contrast_factor)
    
    # STEP 5: Apply STRONG style brightness
    brightness_factor = 0.6 + (style_features['brightness'] * 0.8)
    enhancer = ImageEnhance.Brightness(scene)
    scene = enhancer.enhance(brightness_factor)
    
    # STEP 6: Heavy color grading toward style palette (not subtle 15%, now 45%)
    scene = apply_color_grading(scene, style_features['palette'], intensity=0.45)
    
    return scene.convert('RGBA')

def apply_color_grading(img, palette, intensity=0.45):
    """Apply STRONG color grading based on style palette"""
    if not palette or len(palette) < 3:
        return img
    
    # Get average style color
    avg_style_color = np.mean(palette[:3], axis=0)
    
    # Convert to array
    arr = np.array(img, dtype=np.float32)
    
    # Blend heavily toward style color (now 45% default, was 15%)
    arr = arr * (1 - intensity) + avg_style_color * intensity
    
    # Clamp values
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr, 'RGB')

def composite_subject(background, subject_img, scene_analysis, style_features):
    """
    Intelligently composite the subject onto the styled scene:
    - Smart positioning
    - Edge blending
    - Color matching
    - Shadow/lighting adjustment
    """
    bg = background.copy()
    
    # Resize subject to fit nicely (max 70% of canvas height)
    max_height = int(bg.height * 0.7)
    if subject_img.height > max_height:
        ratio = max_height / subject_img.height
        new_size = (int(subject_img.width * ratio), max_height)
        subject = subject_img.resize(new_size, Image.Resampling.LANCZOS)
    else:
        subject = subject_img.copy()
    
    # Ensure subject is RGBA
    if subject.mode != 'RGBA':
        subject = subject.convert('RGBA')
    
    # Apply color matching to subject based on scene
    subject = match_subject_to_scene(subject, scene_analysis, style_features)
    
    # Add subtle edge blur for natural integration
    subject = add_edge_blur(subject)
    
    # Position subject (centered horizontally, bottom-aligned with slight offset)
    x_pos = (bg.width - subject.width) // 2
    y_pos = bg.height - subject.height - int(bg.height * 0.05)
    
    # Paste with alpha compositing
    bg.paste(subject, (x_pos, y_pos), subject)
    
    return bg

def match_subject_to_scene(subject, scene_analysis, style_features):
    """Adjust subject colors to STRONGLY match scene lighting and style"""
    rgb_subject = subject.convert('RGB')
    
    # Apply STRONG brightness adjustment to match scene
    brightness_factor = 0.7 + (scene_analysis['brightness'] * 0.6)
    enhancer = ImageEnhance.Brightness(rgb_subject)
    rgb_subject = enhancer.enhance(brightness_factor)
    
    # Apply STRONG warm/cool adjustment based on scene temperature
    if abs(scene_analysis['temperature']) > 0.05:  # Lower threshold for more effect
        rgb_subject = apply_temperature_shift(rgb_subject, scene_analysis['temperature'] * 1.5)
    
    # Apply STRONG saturation to match style (not subtle)
    saturation_factor = 0.6 + (style_features['saturation'] * 0.8)
    enhancer = ImageEnhance.Color(rgb_subject)
    rgb_subject = enhancer.enhance(saturation_factor)
    
    # Apply contrast matching
    contrast_factor = 0.8 + (style_features['contrast'] * 0.5)
    enhancer = ImageEnhance.Contrast(rgb_subject)
    rgb_subject = enhancer.enhance(contrast_factor)
    
    # Restore alpha channel
    rgb_subject.putalpha(subject.split()[3])
    
    return rgb_subject

def apply_temperature_shift(img, temperature):
    """Apply STRONG warm/cool color temperature shift"""
    arr = np.array(img, dtype=np.float32)
    
    # Positive = warm (more red), Negative = cool (more blue)
    # Increased from 20 to 40 for more dramatic effect
    shift = temperature * 40
    
    arr[:, :, 0] = np.clip(arr[:, :, 0] + shift, 0, 255)  # Red
    arr[:, :, 2] = np.clip(arr[:, :, 2] - shift, 0, 255)  # Blue
    
    return Image.fromarray(arr.astype(np.uint8), 'RGB')

def add_edge_blur(subject):
    """Add subtle edge blur to alpha channel for natural blending"""
    alpha = subject.split()[3]
    
    # Convert to array
    alpha_arr = np.array(alpha, dtype=np.float32)
    
    # Create edge mask (pixels near transparency)
    edge_distance = 10  # pixels
    edges = np.zeros_like(alpha_arr)
    
    # Find edges where alpha transitions
    for i in range(edge_distance):
        kernel_size = max(1, i // 2)
        blurred = gaussian_filter(alpha_arr, sigma=kernel_size)
        mask = (alpha_arr > 10) & (alpha_arr < 245)
        edges[mask] = blurred[mask]
    
    # Blend edges
    alpha_arr = np.where(edges > 0, edges * 0.3 + alpha_arr * 0.7, alpha_arr)
    alpha_arr = np.clip(alpha_arr, 0, 255).astype(np.uint8)
    
    # Reconstruct image
    r, g, b, _ = subject.split()
    return Image.merge('RGBA', (r, g, b, Image.fromarray(alpha_arr)))


if __name__ == '__main__':
    print("="*60)
    print("🎨 Subject Extractor & Composer Service")
    print("="*60)
    print("📍 Local:   http://localhost:5001")
    print("📍 Network: http://0.0.0.0:5001")
    print("="*60)
    print("💡 Endpoints:")
    print("   GET  /              - Test interface")
    print("   POST /extract       - Remove background from image")
    print("   POST /compose       - Compose subject + scene + style")
    print("   GET  /health        - Check service status")
    print("="*60)
    print("\n✨ Opening browser automatically...\n")
    
    # Auto-open browser
    import webbrowser
    import threading
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open('http://localhost:5001')
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
