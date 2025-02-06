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
    update_doctor_availability, get_available_doctors, book_appointment,
    check_and_assign_appointments, cancel_appointment, recommend_doctor,
    schedule_appointment, check_availability, process_queue, notify_user, check_patients,encrypt_priority_score 
)


# Initialize FastAPI
app = FastAPI()

# MongoDB URI and database connection
MONGO_URI = "mongodb+srv://kunal:kunal@cluster0.azapi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Initialize MongoDB Client
client = AsyncIOMotorClient(MONGO_URI)
db = client["temp"]  # Replace with your database name
symptoms_collection = db["temp"]  # Replace with your collection name
doctors_collection = db["doctor"]  # Add a new collection for doctors
patients_collection = db["patients"]  # Add a new collection for patients
queries_collection = db["queries"]  # Add a new collection for queries

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

            
            if disease in seen_diseases:
                continue  # Skip if the disease has already been added
            seen_diseases.add(disease)
            priority_score += weight
            specialist = speciality_mapping.get(disease, "Not Found")
            
            # Fetch doctor details based on the speciality and input language
            
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
                                    "recommendationScore" : doctor_doc["time_slot"][input.timeSlot*2-1] + doctor_doc["time_slot"][input.timeSlot*2-2]
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

@app.post("/add_doctor")
async def add_doctor(doctor: DoctorInput):
    # Check if the doctor already exists
    existing_doctor = await doctors_collection.find_one({"name": doctor.name, "specialist": doctor.specialist})
    if existing_doctor:
        raise HTTPException(status_code=400, detail="Doctor already exists")

    # Insert the new doctor into the collection
    doctor_data = doctor.dict()
    result = await doctors_collection.insert_one(doctor_data)
    doctor_data["_id"] = str(result.inserted_id)  # Convert ObjectId to string

    return doctor_data

@app.post("/assign_appointments")
async def assign_appointments():
    # Fetch all patient queries from the queries collection
    queries = await queries_collection.find().to_list(length=1000)

    # Calculate priority score for each query
    for query in queries:
        print(query)
        if all(key in query for key in ["Age", "Gender", "History", "symptoms"]):
            print("kunal")
            query["priority_score"] = await calculate_priority_score(query)
        else:
            query["priority_score"] = 0  # Assign a default priority score if any key is missing

    # Sort queries by priority score in descending order
    sorted_queries = sorted(queries, key=lambda x: x["priority_score"], reverse=True)

    # Convert ObjectId to string before returning the response
    for query in sorted_queries:
        query["_id"] = str(query["_id"])

    return sorted_queries

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


@app.post("/update_availability")
async def update_availability(doctor_id: str, availability: list[dict]):
    await update_doctor_availability(doctor_id, availability)
    return {"status": "success"}

@app.post("/book_appointment")
async def book_appointment_endpoint(patient_id: str, doctor_id: str, time_slot: str):
    appointment_id = await book_appointment(patient_id, doctor_id, time_slot)
    return {"status": "success", "appointment_id": appointment_id}

@app.post("/check_appointments")
async def check_appointments():
    await check_and_assign_appointments()
    return {"status": "success"}

@app.post("/cancel_appointment")
async def cancel_appointment_endpoint(appointment_id: str):
    await cancel_appointment(appointment_id)
    return {"status": "success"}

@app.post("/recommend_doctor")
async def recommend_doctor_endpoint(symptoms: str, language: str = "en"):
    recommendations = await recommend_doctor(symptoms, language)
    return {"recommendations": recommendations}

@app.post("/schedule_appointment")
async def schedule_appointment_endpoint(patient_id: str, doctor_id: str, time_slot: str):
    appointment_id = await schedule_appointment(patient_id, doctor_id, time_slot)
    return {"status": "success", "appointment_id": appointment_id}

@app.get("/check_availability")
async def check_availability_endpoint(doctor_id: str, time_slot: str):
    available = await check_availability(doctor_id, time_slot)
    return {"available": available}

@app.post("/process_queue")
async def process_queue_endpoint():
    await process_queue()
    return {"status": "success"}

@app.post("/notify_user")
async def notify_user_endpoint(appointment_id: str):
    await notify_user(appointment_id)
    return {"status": "success"}

# Step 5: Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Disease Prediction API is running!"}

# Step 6: Supported Languages Endpoint
@app.get("/languages")
def supported_languages():
    return {"supported_languages": LANGUAGES}


@app.get("/check_db")
async def check_db():
    # Try to fetch one document from the collection
    test_doc = await symptoms_collection.find_one({"test": "connection"})
    if test_doc:
        test_doc["_id"] = str(test_doc["_id"])  # Convert ObjectId to string
        return {"status": "success", "data": test_doc}
    
    # Insert a test document if not found
    test_doc = {"test": "connection", "status": "working"}
    result = await symptoms_collection.insert_one(test_doc)
    test_doc["_id"] = str(result.inserted_id)  # Convert ObjectId to string
    return {"status": "success", "message": "Test document inserted", "data": test_doc}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(check_patients())

# Run the FastAPI server on a different port
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)  # Change the port number to 8001
