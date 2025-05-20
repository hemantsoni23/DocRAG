from fastapi import APIRouter
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from google.genai import types
import json
import os
from google import genai
import logging
import re

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv("GEMINI_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

client = genai.Client(api_key=GOOGLE_API_KEY)

# Define models for request and response
class PrivacyClassificationRequest(BaseModel):
    user_data: Dict[str, Any]
    context: Optional[str] = None  

class PrivacyClassificationResponse(BaseModel):
    safe_fields: Dict[str, Any]  
    excluded_fields: List[str]  
    reasoning: Dict[str, str]  

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to get more detailed logs

# Add a stream handler if not already added
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

router = APIRouter(prefix="/api", tags=["classifier"])

def classify_privacy(user_data: Dict[str, Any], context: Optional[str] = None) -> PrivacyClassificationResponse:
    """
    Classify user data fields as either safe or private using Google Gemini LLM.
    
    Args:
        user_data: Dictionary containing user data
        context: Optional context about the application/domain
    
    Returns:
        PrivacyClassificationResponse with safe fields, excluded fields, and reasoning
    """
    # Check if input data is empty
    if not user_data:
        return PrivacyClassificationResponse(
            safe_fields={},
            excluded_fields=[],
            reasoning={}
        )
    
    # Prepare prompt for Gemini with a more structured output format
    prompt = """
    You are a privacy classifier for a RAG-based chatbot system. Your task is to analyze user data fields and determine 
    which fields can safely be included in the RAG system without compromising user privacy.

    Rules for classification:
    1. EXCLUDE any personally identifiable information (PII) such as:
       - Full names
       - Email addresses
       - Phone numbers
       - Addresses (home, work, etc.)
       - Social security numbers, tax IDs, or other government identifiers
       - Credit card numbers, bank account details
       - Passwords or security credentials
       - Medical information and health records
       - Biometric data
       - Precise geolocation data
       - IP addresses

    2. INCLUDE non-private information such as:
       - User preferences (e.g., theme settings, language preferences)
       - General demographics (age range, not exact birthdate)
       - Product usage statistics (features used, time spent)
       - Broad location data (country, city, but not specific addresses)
       - Professional roles or job titles (without company names for small companies)
       - General interests and topic preferences
       - Shopping preferences and categories (but not specific purchase history)

    3. Consider CONTEXT:
       - A field might be safe in one context but private in another
       - Some fields may need generalization rather than exclusion
       - Consider the potential for identification when fields are combined

    The data will be used for personalizing a RAG-based chatbot to provide user-specific assistance.

    IMPORTANT: Your response MUST be in the following JSON format:
    {
      "safe_fields": {
        "field_name": field_value,
        ...
      },
      "excluded_fields": [
        "field_name1",
        "field_name2",
        ...
      ],
      "reasoning": {
        "field_name1": "reason for classification",
        "field_name2": "reason for classification",
        ...
      }
    }

    You must return the actual safe_fields with their values, not just the field names.
    For nested objects, you can include the entire object if it's safe, or exclude the entire object if it contains private information.
    """
    
    if context:
        prompt += f"\n\nAdditional context about the application: {context}"
    
    prompt += f"\n\nUser data to classify: {json.dumps(user_data, indent=2)}"

    logger.debug(f"Sending prompt to Gemini: {prompt}")
    
    # Get response from Gemini
    try:
        response = client.models.generate_content(model='gemini-2.0-flash-lite',contents=prompt)
        response_text = response.text
        logger.debug(f"Received response from Gemini: {response_text}")
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        # Return conservative approach if API fails
        return PrivacyClassificationResponse(
            safe_fields={},
            excluded_fields=list(user_data.keys()),
            reasoning={key: "API error - taking conservative approach" for key in user_data.keys()}
        )
    
    try:
        # First attempt: Check for JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            logger.debug(f"Found JSON in code block: {json_str}")
        else:
            # Second attempt: Try to find JSON-like structures
            json_match = re.search(r'(\{[\s\S]*\})', response_text)
            if json_match:
                json_str = json_match.group(1).strip()
                logger.debug(f"Found JSON-like structure: {json_str}")
            else:
                # If still no JSON found, fallback to manual parsing
                logger.warning("No JSON structure found in response, using manual parsing")
                return manual_parse_response(response_text, user_data)
        
        # Try to parse JSON and handle different formats
        classification_data = json.loads(json_str)
        
        # Handle different response structures
        safe_fields = {}
        excluded_fields = []
        reasoning = {}
        
        # Process safe fields
        if "safe_fields" in classification_data:
            # If safe_fields is a dictionary with values
            if isinstance(classification_data["safe_fields"], dict):
                safe_fields = classification_data["safe_fields"]
            # If safe_fields is a list of field names
            elif isinstance(classification_data["safe_fields"], list):
                for key in classification_data["safe_fields"]:
                    if key in user_data:
                        safe_fields[key] = user_data[key]
        
        # Process excluded fields
        if "excluded_fields" in classification_data:
            excluded_fields = classification_data["excluded_fields"]
        else:
            # If excluded_fields is not provided, infer from keys not in safe_fields
            excluded_fields = [key for key in user_data.keys() if key not in safe_fields]
        
        # Process reasoning
        if "reasoning" in classification_data:
            reasoning = classification_data["reasoning"]
        else:
            # If reasoning is not provided, generate generic explanations
            for key in safe_fields:
                reasoning[key] = "Considered safe for RAG"
            for key in excluded_fields:
                reasoning[key] = "Excluded due to privacy concerns"
        
        # Make sure all keys have a reasoning
        for key in user_data.keys():
            if key not in reasoning:
                if key in safe_fields:
                    reasoning[key] = "Considered safe for RAG"
                elif key in excluded_fields:
                    reasoning[key] = "Excluded due to privacy concerns"
                else:
                    excluded_fields.append(key)
                    reasoning[key] = "Excluded by default due to uncertainty"
        
        # Ensure all fields are either in safe_fields or excluded_fields
        all_fields = set(user_data.keys())
        classified_fields = set(safe_fields.keys()).union(set(excluded_fields))
        
        unclassified_fields = all_fields - classified_fields
        for field in unclassified_fields:
            # By default, exclude unclassified fields
            excluded_fields.append(field)
            reasoning[field] = "Excluded by default - field was not classified"
        
        return PrivacyClassificationResponse(
            safe_fields=safe_fields,
            excluded_fields=excluded_fields,
            reasoning=reasoning
        )
    
    except Exception as e:
        logger.error(f"Error parsing Gemini response: {str(e)}")
        logger.error(f"Problematic response text: {response_text}")
        
        # Fallback to manual parsing if JSON parsing fails
        return manual_parse_response(response_text, user_data)

def manual_parse_response(response_text: str, user_data: Dict[str, Any]) -> PrivacyClassificationResponse:
    """
    Manual parsing of response text when JSON parsing fails.
    """
    logger.info("Using manual parsing for response")
    
    safe_fields = {}
    excluded_fields = []
    reasoning = {}
    
    # Define patterns to look for
    safe_patterns = [
        r"safe fields[:\s]+(?:include|contains)?[:\s]*([\w\s,\"']+)",
        r"([\w_]+)[:\s]+(?:is\s+)?safe",
        r"can (?:safely )?include[:\s]+([\w\s,\"']+)"
    ]
    
    exclude_patterns = [
        r"exclude[d]?[:\s]+(?:fields|include)?[:\s]*([\w\s,\"']+)",
        r"([\w_]+)[:\s]+(?:is\s+)?(?:private|excluded)",
        r"should (?:be )?exclude[d]?[:\s]+([\w\s,\"']+)"
    ]
    
    # Look for safe fields patterns
    for pattern in safe_patterns:
        matches = re.finditer(pattern, response_text, re.IGNORECASE)
        for match in matches:
            field_text = match.group(1).strip()
            # Split by commas, quotes, spaces
            fields = re.findall(r'[\w_]+', field_text)
            for field in fields:
                if field in user_data and field not in safe_fields:
                    safe_fields[field] = user_data[field]
    
    # Look for excluded fields patterns
    for pattern in exclude_patterns:
        matches = re.finditer(pattern, response_text, re.IGNORECASE)
        for match in matches:
            field_text = match.group(1).strip()
            # Split by commas, quotes, spaces
            fields = re.findall(r'[\w_]+', field_text)
            for field in fields:
                if field in user_data and field not in excluded_fields:
                    excluded_fields.append(field)
    
    # For fields we've found, try to extract reasoning
    all_found_fields = list(safe_fields.keys()) + excluded_fields
    for field in all_found_fields:
        reason_pattern = rf"{field}.*?(?:reason|because)[:\s]+(.*?)(?:\.\s|\n|$)"
        reason_matches = re.search(reason_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if reason_matches:
            reasoning[field] = reason_matches.group(1).strip()
        else:
            reasoning[field] = "Safe for RAG" if field in safe_fields else "Privacy concern"
    
    # Check for any fields that weren't explicitly classified
    for field in user_data.keys():
        if field not in safe_fields and field not in excluded_fields:
            # Default to excluding unclassified fields to be conservative
            excluded_fields.append(field)
            reasoning[field] = "Excluded by default - field was not classified"
    
    # Special case for common PII fields - if we don't explicitly identify them via parsing
    # but they are common PII fields, explicitly exclude them
    common_pii = ["email", "name", "phone", "address", "ip_address", "ssn", "password", "credit_card"]
    for field in user_data.keys():
        for pii in common_pii:
            if pii in field.lower() and field not in safe_fields:
                if field not in excluded_fields:
                    excluded_fields.append(field)
                reasoning[field] = f"Excluded as potential PII (contains '{pii}')"
    
    # Handle nested objects specially
    for field, value in user_data.items():
        if isinstance(value, dict):
            # Check if any nested field looks like PII
            for nested_field in value.keys():
                for pii in common_pii:
                    if pii in nested_field.lower():
                        # If we find potential PII in nested object, exclude the whole object
                        if field in safe_fields:
                            del safe_fields[field]
                        if field not in excluded_fields:
                            excluded_fields.append(field)
                        reasoning[field] = f"Excluded because nested field '{nested_field}' may contain PII"
    
    # If we didn't find any safe fields, add some common ones that are usually safe
    if not safe_fields:
        safe_candidates = ["preferences", "theme", "language", "interests", "age_range"]
        for field in safe_candidates:
            if field in user_data and field not in excluded_fields:
                safe_fields[field] = user_data[field]
                reasoning[field] = "Common safe field for personalization"
                if field in excluded_fields:
                    excluded_fields.remove(field)
    
    logger.info(f"Manual parsing results - Safe fields: {list(safe_fields.keys())}, Excluded: {excluded_fields}")
    
    return PrivacyClassificationResponse(
        safe_fields=safe_fields,
        excluded_fields=excluded_fields,
        reasoning=reasoning
    )

@router.post("/privacy-classifier", response_model=PrivacyClassificationResponse)
async def privacy_classifier_endpoint(request: PrivacyClassificationRequest = Body(...)):
    try:
        logger.info(f"Received privacy classification request with {len(request.user_data)} fields")
        result = classify_privacy(request.user_data, request.context)
        logger.info(f"Classification complete: {len(result.safe_fields)} safe fields, {len(result.excluded_fields)} excluded fields")
        return result
    except Exception as e:
        logger.exception("Error processing privacy classification request")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")