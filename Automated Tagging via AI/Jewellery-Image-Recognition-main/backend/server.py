from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from groq import Groq
import os
import base64
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any
import asyncio
from dotenv import load_dotenv
from PIL import Image
import io
import logging
import httpx
import requests # Added requests for Imagga API

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
mongo_url = os.getenv("MONGO_URL")
db_name = os.getenv("DB_NAME")
try:
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
    logger.info("MongoDB connection established")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise Exception(f"Failed to connect to MongoDB: {str(e)}")

# Groq client
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("Groq client initialized successfully")
    test_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=10,
    )
    logger.info("Groq API key validated successfully")
except Exception as e:
    logger.error(f"Error initializing or validating Groq client: {str(e)}")
    groq_client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com",
        http_client=httpx.Client(timeout=60.0),
    )

# Collections
images_collection = db.images
taxonomies_collection = db.taxonomies

# Lifespan events
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await images_collection.create_index([("session_id", 1)])
        await taxonomies_collection.create_index([("session_id", 1)])
        logger.info("Database indexes created")
    except Exception as e:
        logger.error(f"Error creating database indexes: {str(e)}")
        raise Exception(f"Error creating database indexes: {str(e)}")
    yield
    client.close()
    logger.info("Database connection closed")


app.lifespan = lifespan


def optimize_image(image_data: bytes, max_size: tuple = (800, 600)) -> str:
    try:
        if not image_data:
            raise ValueError("Empty image data")
        image = Image.open(io.BytesIO(image_data))
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        logger.error(f"Error optimizing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


async def analyze_image_with_groq(image_base64: str, filename: str) -> Dict[str, Any]:
    logger.info(f"Analyzing image: {filename}")
    try:
        # Step 1: Use Imagga API to analyze the image
        imagga_api_key = os.getenv("IMAGGA_API_KEY")
        imagga_api_secret = os.getenv("IMAGGA_API_SECRET")

        if not imagga_api_key or not imagga_api_secret:
            logger.error("Imagga API key or secret not found in environment variables.")
            # Fallback or raise error - for now, proceed with empty tags
            imagga_tags = []
            imagga_error = "Imagga API credentials not configured."
        else:
            imagga_tags = []
            imagga_error = None
            https_api_url = "https://api.imagga.com/v2/tags"
            try:
                # image_base64 is already a string, no need to decode image_content
                response = requests.post(
                    https_api_url,
                    auth=(imagga_api_key, imagga_api_secret),
                    data={'image_base64': image_base64} # Send as form data
                )
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                imagga_response = response.json()

                if imagga_response.get('status', {}).get('type') == 'success':
                    imagga_tags = [tag['tag']['en'] for tag in imagga_response.get('result', {}).get('tags', []) if tag.get('confidence', 0) > 20]
                else:
                    imagga_error = f"Imagga API error: {imagga_response.get('status', {}).get('text', 'Unknown error')}"
                    logger.error(imagga_error)
            except requests.exceptions.RequestException as req_e:
                imagga_error = f"Error calling Imagga API (requests): {str(req_e)}"
                logger.error(imagga_error)
            except Exception as img_e: # Catch other potential errors like JSON parsing
                imagga_error = f"Error processing Imagga response: {str(img_e)}"
                logger.error(imagga_error)

        logger.info(f"Imagga API tags: {imagga_tags}")
        if imagga_error:
            logger.error(f"Imagga processing failed: {imagga_error}")

        # Step 2: Create a description for Groq
        if imagga_tags:
            description = f"""
            Jewelry image analysis:
            - Detected tags from Imagga: {', '.join(imagga_tags)}
            """
        elif imagga_error:
            description = f"""
            Jewelry image analysis:
            - Tag extraction from Imagga failed: {imagga_error}. Please analyze based on visual features.
            """
        else:
            description = f"""
            Jewelry image analysis:
            - No tags detected by Imagga. Please analyze based on visual features.
            """
        logger.info(f"Description for Groq (using Imagga): {description}")

        # Step 3: Use Groq to generate structured tags
        analysis_prompt = f"""
        Based on the following jewelry image analysis, extract comprehensive visual features for hashtag generation:

        {description}

        Please identify and categorize:
        1. MATERIALS & TEXTURES:
        - Metals (gold, silver, platinum, copper, brass, etc.)
        - Gemstones (diamond, ruby, emerald, etc.)
        - Other materials (pearl, bead, etc.)
        - Surfaces (matte, glossy, textured, smooth, etc.)
        2. COLORS:
        - Dominant colors in the jewelry (e.g., gold, silver, red, blue)
        3. PATTERNS:
        - Any visible patterns (e.g., filigree, engraved, plain)
        4. MOTIFS:
        - Design elements (e.g., floral, geometric, abstract)
        5. STYLES:
        - Style categories (e.g., vintage, modern, minimalist, ornate)
        6. FUNCTIONAL:
        - Jewelry type (e.g., ring, necklace, bracelet, earrings)

        Provide the response in JSON format with the following structure. Ensure the JSON is valid and complete:
        {{
            "materials": [],
            "colors": [],
            "patterns": [],
            "motifs": [],
            "styles": [],
            "functional": []
        }}
        """

        logger.info("Making Groq API call")
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=500,
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content
        logger.info(f"Raw Groq response content: {raw_content}")

        analysis_data = json.loads(raw_content)
        logger.info(f"Parsed analysis data: {analysis_data}")

        return {
            "filename": filename,
            "analysis": analysis_data,
            "raw_response": raw_content,
        }

    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        fallback_analysis = {
            "materials": ["unknown"],
            "colors": ["unknown"],
            "patterns": ["unknown"],
            "motifs": ["unknown"],
            "styles": ["unknown"],
            "functional": ["unknown"],
            "overall_description": f"Analysis failed: {str(e)}",
        }
        return {
            "filename": filename,
            "analysis": fallback_analysis,
            "raw_response": f"Error: {str(e)}",
        }


async def generate_smart_hashtags(
    analyses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    try:

        def generate_hashtags_for_analysis(
            data: Dict[str, Any],
        ) -> Dict[str, List[str]]:
            all_features = {
                "materials": [],
                "colors": [],
                "patterns": [],
                "motifs": [],
                "styles": [],
                "functional": [],
            }
            for category in all_features.keys():
                if category in data:
                    all_features[category] = list(set(data[category]))
            return all_features

        individual_hashtags = []
        for analysis in analyses:
            data = analysis["analysis"]
            hashtags = generate_hashtags_for_analysis(data)
            individual_hashtags.append(
                {
                    "filename": analysis["filename"],
                    "hashtags": hashtags,
                    "image_base64": analysis.get("image_base64", ""),
                }
            )

        return individual_hashtags

    except Exception as e:
        logger.error(f"Error generating hashtags: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating hashtags: {str(e)}"
        )


@app.post("/api/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    logger.info("Received upload request")
    try:
        if not files:
            logger.warning("No files provided in the request")
            raise HTTPException(status_code=400, detail="No files provided")

        session_id = str(uuid.uuid4())
        analyses = []
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            if not file.content_type.startswith("image/"):
                logger.warning(f"Skipping non-image file: {file.filename}")
                continue

            try:
                image_data = await file.read()
                if not image_data:
                    logger.error(f"Empty file received: {file.filename}")
                    raise HTTPException(
                        status_code=400, detail=f"Empty file: {file.filename}"
                    )

                image_base64 = optimize_image(image_data)
                analysis = await analyze_image_with_groq(image_base64, file.filename)
                analysis["image_base64"] = image_base64
                analyses.append(analysis)

                image_doc = {
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "filename": file.filename,
                    "analysis": analysis,
                    "uploaded_at": datetime.utcnow(),
                    "image_base64": image_base64,
                }
                await images_collection.insert_one(image_doc)
                logger.info(f"Stored analysis for {file.filename} in database")

            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                analyses.append(
                    {
                        "filename": file.filename,
                        "image_base64": "",
                        "analysis": {
                            "materials": ["error"],
                            "colors": ["error"],
                            "patterns": [],
                            "motifs": [],
                            "styles": [],
                            "functional": ["error"],
                            "overall_description": f"Failed to process: {str(e)}",
                        },
                        "raw_response": f"Error: {str(e)}",
                    }
                )
                continue

        if not analyses:
            logger.error("No images were successfully processed")
            raise HTTPException(
                status_code=400, detail="No images were successfully processed"
            )

        individual_hashtags = await generate_smart_hashtags(analyses)

        taxonomy_doc = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "taxonomy": {},
            "image_count": len(analyses),
            "created_at": datetime.utcnow(),
        }
        await taxonomies_collection.insert_one(taxonomy_doc)
        logger.info(f"Stored taxonomy for session {session_id}")

        return {
            "session_id": session_id,
            "images_processed": len(analyses),
            "individual_hashtags": individual_hashtags,
            "individual_analyses": [
                {
                    **analysis,
                    "image_base64": analysis.get("image_base64", ""),
                }
                for analysis in analyses
            ],
        }

    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing images: {str(e)}"
        )


@app.get("/api/taxonomy/{session_id}")
async def get_taxonomy(session_id: str):
    try:
        taxonomy = await taxonomies_collection.find_one({"session_id": session_id})
        if not taxonomy:
            raise HTTPException(status_code=404, detail="Taxonomy not found")

        images = await images_collection.find({"session_id": session_id}).to_list(None)
        individual_hashtags = [
            {
                "filename": img["filename"],
                "hashtags": img["analysis"]["analysis"],
                "image_base64": img.get("image_base64", ""),
            }
            for img in images
        ]
        taxonomy["individual_hashtags"] = individual_hashtags
        taxonomy.pop("_id", None)
        return taxonomy

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving taxonomy: {str(e)}")
        raise HTTPException(
            status_code=404, detail=f"Error retrieving taxonomy: {str(e)}"
        )


@app.get("/api/sessions")
async def get_sessions():
    try:
        sessions = []
        async for taxonomy in taxonomies_collection.find().sort("created_at", -1):
            taxonomy.pop("_id", None)
            sessions.append(
                {
                    "session_id": taxonomy["session_id"],
                    "image_count": taxonomy["image_count"],
                    "created_at": taxonomy["created_at"],
                    "hashtag_count": taxonomy.get("taxonomy", {}).get(
                        "hashtag_count", {}
                    ),
                }
            )

        return {"sessions": sessions}

    except Exception as e:
        logger.error(f"Error retrieving sessions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving sessions: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "groq_api": "connected" if groq_client else "disconnected",
        "database": "connected",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
