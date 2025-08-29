# 🔍 Psoriasis Food Scout

## Project Motivation

The idea is to develop a AI based mobile application on psoriasis food analysis. For psoriasis patients, it is important to know which foods are good for them and which are not. As a proof of concept, an AI-powered web application that helps individuals with psoriasis make informed dietary choices by analyzing food images and providing personalized recommendations.

## Features

- **Image Analysis**: Upload food images for instant AI-powered analysis
- **AI-Driven Recommendations**: Uses Google's Gemini 2.0 Flash model for intelligent food assessment
- **Psoriasis-Specific Guidance**: Categorizes foods as GOOD ✅, MODERATE ⚠️, or BAD ❌ for psoriasis management
- **Detailed Explanations**: Provides scientific reasoning behind each recommendation
- **Alternative Suggestions**: Offers healthier alternatives for problematic foods
- **Responsive Design**: Modern, user-friendly interface that works on all devices
- **Cloud Deployment**: Ready-to-deploy on AWS with Terraform infrastructure

## How It Works

1. **Upload**: Take a photo or upload an image of food
2. **Analyze**: AI examines the food for inflammatory properties and psoriasis triggers
3. **Receive**: Get categorized recommendations with detailed explanations
4. **Act**: Make informed dietary choices based on evidence-based advice

### Rating System

- **✅ GOOD**: Anti-inflammatory foods that may help reduce psoriasis symptoms
- **⚠️ MODERATE**: Neutral foods that should be consumed mindfully
- **❌ BAD**: Pro-inflammatory foods that may trigger psoriasis flares

## Quick Start

### Prerequisites

- Python 3.12+
- Google Gemini API key
- uv (Python package manager)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd psoriasis_food_scout
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   uv run python main.py
   ```

5. **Access the app**
   Open your browser and navigate to `http://localhost:5000`

## Cloud Deployment

### AWS Deployment with Terraform

The project includes complete Terraform configuration for AWS deployment.

#### Prerequisites
- AWS account with appropriate permissions
- Terraform installed
- AWS CLI configured

#### Deploy to AWS

1. **Set up Terraform variables**
   ```bash
   # Create terraform.tfvars file
   cat << EOF > terraform.tfvars
   gemini_api_key = "your_gemini_api_key"
   aws_access_key = "your_aws_access_key"
   aws_secret_key = "your_aws_secret_key"
   aws_region = "us-east-1"
   key_pair_name = "psoriasis-app-key"
   vpc_id = "your_vpc_id"
   EOF
   ```

2. **Initialize and deploy**
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

3. **Access deployed application**
   The application will be available on the created EC2 instance at port 5000.

## Technology Stack

- **Backend**: Flask (Python web framework)
- **AI/ML**: Google Gemini 2.0 Flash model
- **Frontend**: HTML5, CSS3, Jinja2 templates
- **Image Processing**: Pillow (PIL)
- **Infrastructure**: Terraform + AWS
- **Package Management**: uv

## Project Structure

```
psoriasis_food_scout/
├── main.py                 # Flask application
├── main.tf                 # Terraform infrastructure
├── pyproject.toml          # Python dependencies
├── templates/              # HTML templates
│   ├── base.html          # Base template with styling
│   ├── index.html         # Upload page
│   └── result.html        # Results page
├── terraform.tfvars       # Terraform variables (create this)
├── .env                   # Environment variables (create this)
└── README.md              # This file
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Supported Image Formats

- PNG
- JPG/JPEG
- GIF



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This application provides educational information only and should not replace professional medical advice. Always consult with healthcare providers before making significant dietary changes for psoriasis management.

## Resources

- [National Psoriasis Foundation](https://www.psoriasis.org/)
- [Google Gemini API Documentation](https://ai.google.dev/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest)