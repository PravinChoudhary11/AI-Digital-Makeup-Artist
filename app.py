from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import base64
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# API configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the .env file")

# Initialize Gemini AI client
genai.configure(api_key=GOOGLE_API_KEY)

# Set safety settings for Gemini
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_and_query")
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
    try:
        # Read and validate image
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()  # Verify image integrity
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        # Convert image to format compatible with Gemini
        img = Image.open(io.BytesIO(image_content))
        
        # Create system prompts for more specific responses
        skin_analysis_prompt = """
        You are GlowU, an expert beauty advisor specializing in skincare analysis and product recommendations.
        
        Analyze the uploaded facial selfie and provide:
        1. A detailed skin type assessment (dry, oily, combination, normal, sensitive)
        2. Identification of visible skin concerns (acne, hyperpigmentation, wrinkles, redness, etc.)
        3. Current skin condition evaluation
        
        Format your response in markdown with appropriate sections and be specific with your observations.
        """
        
        product_recommendation_prompt = """
        You are GlowU, an expert beauty advisor specializing in skincare analysis and product recommendations.
        
        Based on the uploaded facial selfie and the user's query, provide:
        1. A brief analysis of their skin type and concerns
        2. Personalized product recommendations with specific product names in these categories:
           - Cleanser
           - Treatment (serums, spot treatments)
           - Moisturizer
           - Sunscreen
           - Any specialty products addressing their specific concerns
        3. Suggest a simple morning and evening skincare routine
        
        For each product recommendation, include:
        - Product name and brand
        - Key ingredients that address their concerns
        - Why it's suitable for their skin type
        - Price range (budget, mid-range, or luxury)
        
        Format your response in markdown with clear sections:
        ## Skin Analysis
        ## Recommended Products
        ## Skincare Routine
        """

        def make_gemini_request(prompt, user_query, image):
            try:
                # Create Gemini model instance - using Gemini Pro Vision for image processing capabilities
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-pro",
                    generation_config={
                        "max_output_tokens": 2048,
                        "temperature": 0.4,
                    },
                    safety_settings=safety_settings
                )
                
                # Create a combined prompt with system instructions and user query
                combined_prompt = f"{prompt}\n\nUser query: {user_query}"
                
                # Generate content with Gemini using both text and image
                response = model.generate_content([combined_prompt, image])
                
                return response.text
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                return f"We apologize, but we encountered an issue with our AI service: {str(e)}. Please try again later."

        # Make requests with different prompts
        logger.info("Making request for skin analysis")
        analysis_content = make_gemini_request(skin_analysis_prompt, query, img)
        
        logger.info("Making request for product recommendations")
        recommendations_content = make_gemini_request(product_recommendation_prompt, query, img)

        responses = {
            "analysis": analysis_content,
            "recommendations": recommendations_content
        }

        return JSONResponse(status_code=200, content=responses)

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)