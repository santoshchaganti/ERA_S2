from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from google import genai
import os
from dotenv import load_dotenv
from PIL import Image
import io
import base64
import re

# Load environment variables
load_dotenv(".env")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this to a random secret key

def format_analysis_text(text):
    """Convert markdown-style formatting to HTML"""
    # Convert **bold** to <strong>bold</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Convert *italic* to <em>italic</em> (but not the bullet points)
    text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<em>\1</em>', text)
    
    # Convert bullet points * to proper HTML list items
    text = re.sub(r'^\*\s+', '• ', text, flags=re.MULTILINE)
    
    # Convert line breaks to HTML breaks for better formatting
    text = text.replace('\n', '<br>')
    
    return text

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.route('/')
def index():
    """Main page with file upload form"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_food():
    """Analyze uploaded food image for psoriasis dietary recommendations"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Read and process the image
            image_data = file.read()
            
            # Convert image to PIL Image for Gemini
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Create a detailed prompt for psoriasis dietary analysis
            prompt = """
            Analyze this food image for someone with psoriasis. Please provide:
            
            1. **Food Identification**: What food(s) do you see in the image?
            
            2. **Psoriasis Impact**: Rate this food as:
               - GOOD ✅: Anti-inflammatory, beneficial for psoriasis
               - MODERATE ⚠️: Neutral, consume in moderation  
               - BAD ❌: Pro-inflammatory, may trigger psoriasis flares
            
            3. **Explanation**: Why is this food good/moderate/bad for psoriasis? Consider:
               - Inflammatory properties
               - Common psoriasis triggers
               - Nutritional benefits/concerns
            
            4. **Recommendations**: 
               - If GOOD: How to incorporate it into diet
               - If MODERATE: Portion control or preparation tips
               - If BAD: Healthier alternatives to suggest
            
            Please be specific and helpful for someone managing psoriasis through diet.
            """
            
            # Get response from Gemini
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[prompt, pil_image]
            )
            
            # # Save analysis to text file
            # import datetime
            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename_clean = file.filename.replace(" ", "_").replace(".", "_")
            # txt_filename = f"analysis_{filename_clean}_{timestamp}.txt"
            
            # with open(txt_filename, 'w', encoding='utf-8') as txt_file:
            #     txt_file.write(f"Food Image Analysis - {file.filename}\n")
            #     txt_file.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            #     txt_file.write("=" * 50 + "\n\n")
            #     txt_file.write(response.text)
            
            # print(f"Analysis saved to: {txt_filename}")
            
            # Convert image to base64 for display
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            return render_template('result.html', 
                                 analysis=format_analysis_text(response.text),
                                 image_data=image_base64,
                                 filename=file.filename)
            
        except Exception as e:
            print(f"Error details: {str(e)}")  # This will show in your terminal
            flash(f'Error analyzing image: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Please upload a valid image file (JPG, JPEG, PNG, GIF)')
        return redirect(url_for('index'))

def allowed_file(filename):
    """Check if file has allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)