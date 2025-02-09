from ctypes import Array
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from googletrans import Translator, LANGUAGES
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId  # Add this import
import asyncio  # Add this import
from checker import (
    encrypt_priority_score , check_patients
)
from contextlib import asynccontextmanager


# Initialize FastAPI
# app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(check_patients())  # Start the background task
    try:
        yield  # Yield control back to FastAPI
    finally:
        task.cancel()  # Cancel the task when the app shuts down

app = FastAPI(lifespan=lifespan)

# MongoDB URI and database connection
MONGO_URI = "mongodb+srv://rushikesh22320064:clntvsuLSF67UFTz@cluster0.hjilt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Initialize MongoDB Client
client = AsyncIOMotorClient(MONGO_URI)
db = client["test"]  # Replace with your database name

doctors_collection = db["doctors"]  # Add a new collection for doctors
patients_collection = db["patients"]  # Add a new collection for patients

# Initialize Google Translator
translator = Translator()

# Step 1: Preprocess the Dataset
def preprocess_symptoms_dataset(filepath):
    df = pd.read_csv(filepath)
    # Combine all symptoms into a single string for each disease
    df["Symptoms"] = df["Symptoms"].apply(lambda x: ' '.join(x.split(', ')))
    # Drop duplicate rows based on Disease and Symptoms
    df = df[["Diseases", "Symptoms", "Specialist"]].drop_duplicates()
    
    # Ensure the Weight column exists and has default values if missing
    if "Weight" not in df.columns:
        df["Weight"] = 1  # Assign a default weight of 1 to each symptom
    
    return df

# Step 2: Load and Preprocess Data
symptoms_df = preprocess_symptoms_dataset("Dataset/updated_specialists.csv")

# Prepare Disease-Speciality Mapping
speciality_mapping = {row["Diseases"]: row["Specialist"] for _, row in symptoms_df.iterrows()}

# Step 3: Initialize Sentence Transformer and Load FAISS Index
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Load FAISS index from file
faiss_index = faiss.read_index("Dataset/faiss_index.bin")

# Input Models
class SymptomsInput(BaseModel):
    symptoms: str  # Accept a list of symptoms
    language: str = "en"  # Language of input symptoms, default is English
    timeSlot :int

class DoctorInput(BaseModel):
    name: str
    specialist: str
    language: list[str]

class PatientQuery(BaseModel):
    history: int
    age: int
    gender: str
    symptoms: str

# Step 4: API Endpoints
@app.post("/predict")
async def predict_disease(input: SymptomsInput):
    # Combine all symptoms into a single string
    original_query = input.symptoms
    translated_query = original_query  # Default to the original if no translation is needed

    # Translate symptoms to English if needed
    if input.language != "en":
        try:
            # Translate the query to English
            translated = translator.translate(original_query, src=input.language, dest="en").text
            translated_query = ' '.join(translated.split())  # Clean up the translated text
            print(f"Original Query: {original_query}")
            print(f"Translated Query: {translated_query}")
        except Exception as e:
            print(f"Translation Error: {e}")
            return {"error": "Translation failed", "details": str(e)}

    # Log the translated query for verification
    # print(f"Final Query in English: {translated_query}")

    # Encode the translated query
    query_vector = encoder.encode([translated_query])

    # Search the FAISS index for the closest matches
    _, indices = faiss_index.search(query_vector, k=10)

    doctors = []
    doctor_set = set()
    priority_score = 0

    predictions = []
    seen_diseases = set()  # Set to keep track of seen diseases
    for idx in indices[0]:
        if idx < len(symptoms_df):
            disease = symptoms_df.iloc[idx]["Diseases"]
            weight = symptoms_df.iloc[idx]["Weight"]

            print(disease)
            if disease in seen_diseases:
                continue  # Skip if the disease has already been added
            seen_diseases.add(disease)
            priority_score += weight
            specialist = speciality_mapping.get(disease, "Not Found")
            
            # Fetch doctor details based on the speciality and input language
            print(specialist)
            
            if specialist != "Not Found":
                doctor_docs = await doctors_collection.find({
                    "specialist": specialist,
                    "language": {"$elemMatch": {"$in": [input.language, LANGUAGES.get(input.language, input.language)]}}
                }).to_list(length=10)

                # print(f"Fetched doctor documents for specialist '{specialist}' and language '{input.language}': {doctor_docs}")  # Print fetched doctor documents
                doctor_info = []
                
                for doctor_doc in doctor_docs:
                    if doctor_doc["language"] != "Not specified":
                        # print(doctor_doc["name"]," ",doctor_doc["time_slot"][input.timeSlot*2-1]," ",doctor_doc["time_slot"][input.timeSlot*2-2]," ",input.timeSlot*2-1," ",input.timeSlot*2-2)
                        
                        if doctor_doc["time_slot"][input.timeSlot*2-1] > 0 or doctor_doc["time_slot"][input.timeSlot*2-2] > 0:
                            
                            doctor_doc["_id"] = str(doctor_doc["_id"])  # Convert ObjectId to string
                            
                            if doctor_doc["_id"] not in doctor_set :
                                doctor_set.add(doctor_doc["_id"])
                                doctors.append({
                                    "doctorId" : doctor_doc["_id"],
                                    "name" : doctor_doc["name"],
                                    "specialist": doctor_doc["specialist"],
                                    "profile_image" : doctor_doc["profile_image"],
                                    "recommendationScore" : doctor_doc["time_slot"][input.timeSlot]
                                })

                            doctor_info.append({
                                "doctor_id": doctor_doc["_id"],
                                "doctor_name": doctor_doc["name"],
                                "specialist": doctor_doc["specialist"],
                                "language": doctor_doc["language"]  # Include language if available
                            })
                
                predictions.append({"disease": disease, "speciality": specialist, "doctors": doctor_info})
            else:
                predictions.append({"disease": disease, "speciality": specialist, "doctors": []})

    # Optionally, save predictions to MongoDB
    prediction_data = {
        "original_query": original_query,
        "translated_query": translated_query,
        "predictions": predictions,
    }

    priority_score = await encrypt_priority_score(priority_score)

    recommended_doctors = sorted(doctors, key=lambda x: x["recommendationScore"], reverse=True)

    # print(recommended_doctors)

    final_result = {
        "doctors" : recommended_doctors,
        "priority score" : priority_score
    }

    # print(final_result)

    # Insert prediction into MongoDB collection (this is an async operation)
    # result = await symptoms_collection.insert_one(prediction_data)
    # prediction_data["_id"] = str(result.inserted_id)  # Convert ObjectId to string

    return final_result

async def calculate_priority_score(query):
    # Fetch the patient history
   
    history_value = query['history']

    # Extract symptom weights
    symptoms_list = [symptom.strip() for symptom in query["symptoms"].split(",")]
    symptoms_weight = sum(
        symptoms_df[symptoms_df["Symptoms"].str.contains(symptom, case=False, na=False)]["Weight"].sum()
        for symptom in symptoms_list
    )

    # Calculate Priority Score
    f1 = query["Age"]
    f2 = 1 if query["Gender"].lower() == "f" else 0
    f3 = history_value
    f4 = symptoms_weight
    print(f1," ",f2," ",f3," ",f4)
    return f1 * 0.4 + f2 * 0.2 + f3 * 0.2 + f4 * 0.2

@app.get("/")
def read_root():
    return {"message": "Disease Prediction API is running!"}



# Run the FastAPI server on a different port
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  



