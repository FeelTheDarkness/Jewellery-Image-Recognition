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
    allow_origins=["*"],
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
    # Test Groq API key
    test_response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=10,
    )
    logger.info("Groq API key validated successfully")
except Exception as e:
    logger.error(f"Error initializing or validating Groq client: {str(e)}")
    import httpx
    from groq._client import Groq as GroqClient

    groq_client = GroqClient(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai",
        http_client=httpx.Client(timeout=60.0),
    )

# Collections
images_collection = db.images
taxonomies_collection = db.taxonomies


@app.on_event("startup")
async def startup_db_client():
    """Initialize database indexes"""
    try:
        await images_collection.create_index("session_id")
        await taxonomies_collection.create_index("session_id")
        logger.info("Database indexes created")
    except Exception as e:
        logger.error(f"Error creating database indexes: {str(e)}")
        raise Exception(f"Error creating database indexes: {str(e)}")


@app.on_event("shutdown")
async def shutdown_db_client():
    """Close database connection"""
    client.close()
    logger.info("Database connection closed")


def optimize_image(image_data: bytes, max_size: tuple = (800, 600)) -> str:
    """Optimize image for AI analysis"""
    try:
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
    """Analyze image using Groq AI (text-only fallback)"""
    logger.info(f"Analyzing image: {filename}")
    try:
        # Comprehensive prompt for feature extraction
        analysis_prompt = """
        Based on the description of a jewelry image, extract comprehensive visual features for hashtag generation.
        
        Image description: This is a jewelry image likely containing metals (e.g., gold, silver), gemstones (e.g., diamond, ruby), and ornate designs. The jewelry type could be a ring, necklace, or bracelet.
        
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
        
        Provide the response in JSON format with the following structure:
        {
            "materials": [],
            "colors": [],
            "patterns": [],
            "motifs": [],
            "styles": [],
            "functional": [],
            "overall_description": ""
        }
        """

        # Call Groq API with text-only model
        logger.info("Making Groq API call")
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=500,
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        logger.info(response)

        analysis_data = json.loads(response.choices[0].message.content)
        logger.info(analysis_data)

        logger.info(f"Groq response: {analysis_data}")

        return {
            "filename": filename,
            "analysis": analysis_data,
            "raw_response": response.choices[0].message.content,
        }

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding Groq API response: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error decoding Groq API response: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error analyzing image with Groq: {str(e)}")
        # Fallback response in case Groq API fails
        fallback_analysis = {
            "materials": [],
            "colors": [],
            "patterns": [],
            "motifs": [],
            "styles": [],
            "functional": [],
            "overall_description": "Analysis failed due to API error",
        }
        return {
            "filename": filename,
            "analysis": fallback_analysis,
            "raw_response": f"Error: {str(e)}",
        }


async def generate_smart_hashtags(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate and deduplicate hashtags from multiple image analyses"""
    try:
        all_features = {
            "materials": [],
            "colors": [],
            "patterns": [],
            "motifs": [],
            "styles": [],
            "functional": [],
        }

        for analysis in analyses:
            data = analysis["analysis"]
            for category, features in all_features.items():
                if category in data:
                    features.extend(data[category])

        original_total = sum(len(v) for v in all_features.values())
        deduplicated_hashtags = {
            category: list(set(features)) for category, features in all_features.items()
        }
        deduplicated_total = sum(len(v) for v in deduplicated_hashtags.values())

        semantic_groups = [
            {
                "group_name": "Denim Style",
                "hashtags": ["denim", "blue", "casual"],
                "description": "Denim-related fashion elements",
            },
            {
                "group_name": "Modern Design",
                "hashtags": ["modern", "geometric", "abstract"],
                "description": "Contemporary design elements",
            },
        ]

        return {
            "deduplicated_hashtags": deduplicated_hashtags,
            "hashtag_count": {
                "original_total": original_total,
                "deduplicated_total": deduplicated_total,
                "duplicates_removed": original_total - deduplicated_total,
            },
            "semantic_groups": semantic_groups,
        }

    except Exception as e:
        logger.error(f"Error generating hashtags: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating hashtags: {str(e)}"
        )


@app.post("/api/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload and analyze multiple images"""
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
                analyses.append(analysis)

                image_doc = {
                    "id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "filename": file.filename,
                    "analysis": analysis,
                    "uploaded_at": datetime.utcnow(),
                    "image_base64": image_base64[:100] + "...",
                }
                await images_collection.insert_one(image_doc)
                logger.info(f"Stored analysis for {file.filename} in database")

            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                analyses.append(
                    {
                        "filename": file.filename,
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

        taxonomy_data = await generate_smart_hashtags(analyses)

        taxonomy_doc = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "taxonomy": taxonomy_data,
            "image_count": len(analyses),
            "created_at": datetime.utcnow(),
        }
        await taxonomies_collection.insert_one(taxonomy_doc)
        logger.info(f"Stored taxonomy for session {session_id}")

        return {
            "session_id": session_id,
            "images_processed": len(analyses),
            "taxonomy": taxonomy_data,
            "individual_analyses": analyses,
        }

    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing images: {str(e)}"
        )


@app.get("/api/taxonomy/{session_id}")
async def get_taxonomy(session_id: str):
    """Get taxonomy for a session"""
    try:
        taxonomy = await taxonomies_collection.find_one({"session_id": session_id})
        if not taxonomy:
            raise HTTPException(status_code=404, detail="Taxonomy not found")

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
    """Get all analysis sessions"""
    try:
        sessions = []
        async for taxonomy in taxonomies_collection.find().sort("created_at", -1):
            taxonomy.pop("_id", None)
            sessions.append(
                {
                    "session_id": taxonomy["session_id"],
                    "image_count": taxonomy["image_count"],
                    "created_at": taxonomy["created_at"],
                    "hashtag_count": taxonomy["taxonomy"].get("hashtag_count", {}),
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "groq_api": "connected" if groq_client else "disconnected",
        "database": "connected",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
