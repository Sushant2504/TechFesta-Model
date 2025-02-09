from datetime import datetime, timedelta
import pytz
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from datetime import time
import asyncio

from cryptography.fernet import Fernet

# MongoDB URI and database connection
MONGO_URI = "mongodb+srv://rushikesh22320064:clntvsuLSF67UFTz@cluster0.hjilt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = AsyncIOMotorClient(MONGO_URI)
db = client["test"]
doctors_collection = db["doctors"]
patients_collection = db["patients"]
appointments_collection = db["appointments"]


queries_collection = db["queries"]
appointments_query = db["appointment_query"]
appointment_booking = db["appointment_booking"]

# Initialize Sentence Transformer and Load FAISS Index
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
faiss_index = faiss.read_index("Dataset/faiss_index.bin")

# Load and preprocess data
symptoms_df = pd.read_csv("Dataset/updated_specialists.csv")
speciality_mapping = {row["Diseases"]: row["Specialist"] for _, row in symptoms_df.iterrows()}


key = b'PLBP6UzOYhzs7PiTBOrXsdKYM5d14IPx0TPWfawB81k='
# print(key)
cipher_suite = Fernet(key)


def get_time_range(value: int, subvalue: int):
    
    time_slots = {
        0: 6,  1: 9,  2: 12, 3: 15, 4: 18, 
        5: 21, 6: 0,  7: 3
    }

    if value not in time_slots or subvalue < 0 or subvalue > 5:
        raise ValueError("Invalid value or subvalue. Value must be 0-7 and subvalue must be 0-5.")

    # Get the base start hour
    start_hour = time_slots[value]
    
    # Calculate minutes
    extra_hours = (subvalue * 30) // 60  # Convert extra minutes to hours
    minutes = (subvalue * 30) % 60       # Get remaining minutes
    
    # Compute start and end times
    start_time = time((start_hour + extra_hours) % 24, minutes)
    end_minutes = (minutes + 30) % 60
    end_hour = (start_hour + extra_hours + ((minutes + 30) // 60)) % 24
    end_time = time(end_hour, end_minutes)

    return start_time.strftime("%I:%M %p"), end_time.strftime("%I:%M %p")

async def encrypt_priority_score(score: int) -> str:
    """Encrypts an integer priority score."""
    score_bytes = str(score).encode()
    encrypted_score = cipher_suite.encrypt(score_bytes)
    return encrypted_score.decode()

# Function to decrypt priority score safely
async def decrypt_priority_score(encrypted_score: str) -> int:
    """Decrypts the priority score and returns 0 if tampering is detected."""
    try:
        decrypted_bytes = cipher_suite.decrypt(encrypted_score.encode())
        return int(decrypted_bytes.decode())  # Convert back to integer
    except Exception:
        # print("Warning: Tampered data detected. Assigning default priority score 0.")
        return 0  # Default priority score in case of tampering



async def calculate_priority_score(query):
    # Fetch the patient history
   
    # history_value = query['History']

    # Calculate Priority Score
    f1 = query["Age"]
    f2 = 1 if query["Gender"].lower() == "f" else 0
    # f3 = history_value
    f4 = query["timeIn"]
    
    # print(f1," ",f2," ",f3," ")
    return f1 * 0.1 + f2 * 0.1 + f4 * 0.15

async def check_patients():
    while True:
        
        appointment_count = await appointments_collection.count_documents({ "isScheduled": False })

        print(f"Patients count: {appointment_count}")
        
        if appointment_count > 0:
            
            appointments = await appointments_collection.find({ "isScheduled" : False }).sort("createdAt", -1).to_list(None)

            count = 1
            appointment_list = []

            # Set the priority
            for appointent in appointments:

                patientId = appointent["patientId"]
                doctorId = appointent["doctorId"]

                patientData = await patients_collection.find_one({"_id": ObjectId(patientId)})

                if patientData :

                    query = {
                        "Age" : patientData["age"],
                        "Gender" : patientData["gender"],
                        "timeIn" : count
                    }

                    prio_score = appointent["priority_score"]

                    priority_score = await decrypt_priority_score(prio_score)

                    priority_score *= 0.4

                    priority_score += await calculate_priority_score(query)

                    priority_score += patientData["Ageing"] * 0.1

                    # print(priority_score)

                    appointment_list.append({
                        "appointmentId" : appointent["_id"],
                        "patientId" : patientId,
                        "Ageing" : patientData["Ageing"],
                        "doctorId" : doctorId,
                        "timeSlot" : appointent["timeSlot"],
                        "priority_score" : priority_score ,
                    })


                count += 0.1

            
            #sort the appoint ment list in descending order according to priority
            appointment_list = sorted(appointment_list, key=lambda x: x["priority_score"], reverse=True)

            # print(appointment_list)


            #Assign the doctors
            for appointment in appointment_list:
                doctorId = appointment["doctorId"]
                patientId = appointment["patientId"]
                timeSlot = appointment["timeSlot"]
                appointmentId = appointment["appointmentId"]
                

                finalBooking = {
                    "doctorId" : doctorId,
                    "patientId" : patientId,
                    "finalize_booking" : False,
                }

                doctorData = await doctors_collection.find_one({"_id": ObjectId(doctorId)})

                if(doctorData) :
                    available = doctorData['available']
                    booked_slot = doctorData['booked_slot']

                    if(available[timeSlot] > 0):
                        if(booked_slot[timeSlot] < 6):
                            value = timeSlot
                            subvalue = booked_slot[timeSlot]

                            alloted_slot_start,alloted_slot_end = get_time_range(value, subvalue)

                            booked_slot[timeSlot]+=1

                            finalBooking["Start"] = alloted_slot_start
                            finalBooking["End"] = alloted_slot_end
                            finalBooking["finalize_booking"] = True
                    
                    
                if finalBooking["finalize_booking"]: 

                    booked_appointment = await appointments_collection.update_one(
                        {"_id": ObjectId(appointmentId)},
                        {"$set": {
                            "start" : finalBooking["Start"],
                            "end" : finalBooking["End"],
                            "finalize_booking" : True,
                            "isScheduled" : True,
                            "isNotified" : False,
                        }}
                    )

                    booking_id = appointmentId

                    result = await doctors_collection.update_one(
                        {"_id": ObjectId(doctorId)}, 
                        {"$set": {
                            "booked_slot" : booked_slot,
                        },"$push" : {
                            "booking_ref" : ObjectId(booking_id),
                        }}
                    )

                    patient = await patients_collection.update_one(
                        {"_id": ObjectId(patientId)}, 
                        {"$set": {
                            "booking_ref" : ObjectId(booking_id),
                            "Ageing" : 0,
                        },
                        }
                    )
                else :

                    booked_appointment = await appointments_collection.update_one(
                        {"_id": ObjectId(appointmentId)},
                        {"$set": { 
                            "isScheduled" : True,
                            "finalize_booking" : False,
                            "isNotified" : False,
                        }}
                    )


                    patient = await patients_collection.update_one(
                        {"_id": ObjectId(patientId)}, 
                        {"$set": {
                            "Ageing" : appointment["Ageing"]+1,
                        },
                        }
                    )


                # print(finalBooking)
               
        await asyncio.sleep(900)  # Sleep for 10 seconds