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
MONGO_URI = "mongodb+srv://kunal:kunal@cluster0.azapi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = AsyncIOMotorClient(MONGO_URI)
db = client["temp"]
doctors_collection = db["doctor"]
patients_collection = db["patient"]
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

async def update_doctor_availability(doctor_id, availability):
    await doctors_collection.update_one(
        {"_id": ObjectId(doctor_id)},
        {"$set": {"availability": availability}}
    )

async def get_available_doctors(time_range, patient_timezone):
    current_time_utc = datetime.now(pytz.utc)
    patient_time = current_time_utc.astimezone(pytz.timezone(patient_timezone))
    
    available_doctors = await doctors_collection.find({
        "availability": {"$elemMatch": {"time_range": time_range, "available": True}}
    }).to_list(length=100)
    
    return available_doctors

async def book_appointment(patient_id, doctor_id, time_slot):
    appointment_data = {
        "patient_id": ObjectId(patient_id),
        "doctor_id": ObjectId(doctor_id),
        "time_slot": time_slot,
        "status": "pending"
    }
    result = await appointments_collection.insert_one(appointment_data)
    return str(result.inserted_id)

async def check_and_assign_appointments():
    pending_appointments = await appointments_collection.find({"status": "pending"}).to_list(length=1000)
    
    for appointment in pending_appointments:
        doctor = await doctors_collection.find_one({"_id": appointment["doctor_id"]})
        if doctor and "availability" in doctor and any(slot["time_slot"] == appointment["time_slot"] and slot["available"] for slot in doctor["availability"]):
            await appointments_collection.update_one(
                {"_id": appointment["_id"]},
                {"$set": {"status": "confirmed"}}
            )
            await doctors_collection.update_one(
                {"_id": doctor["_id"], "availability.time_slot": appointment["time_slot"]},
                {"$set": {"availability.$.available": False}}
            )
        else:
            await appointments_collection.update_one(
                {"_id": appointment["_id"]},
                {"$set": {"status": "failed"}}
            )

async def cancel_appointment(appointment_id):
    appointment = await appointments_collection.find_one({"_id": ObjectId(appointment_id)})
    if appointment:
        await appointments_collection.update_one(
            {"_id": appointment["_id"]},
            {"$set": {"status": "cancelled"}}
        )
        await doctors_collection.update_one(
            {"_id": appointment["doctor_id"], "availability.time_slot": appointment["time_slot"]},
            {"$set": {"availability.$.available": True}}
        )

async def recommend_doctor(symptoms, language="en"):
    query_vector = encoder.encode([symptoms])
    _, indices = faiss_index.search(query_vector, k=10)
    
    recommendations = []
    seen_diseases = set()
    for idx in indices[0]:
        if idx < len(symptoms_df):
            disease = symptoms_df.iloc[idx]["Diseases"]
            if disease in seen_diseases:
                continue
            seen_diseases.add(disease)
            specialist = speciality_mapping.get(disease, "Not Found")
            if specialist != "Not Found":
                doctor_docs = await doctors_collection.find({
                    "specialist": specialist,
                    "language": {"$in": [language]}
                }).to_list(length=10)
                doctor_info = []
                for doc in doctor_docs:
                    if await check_availability(doc["_id"], "time_slot"):  # Check availability
                        doctor_info.append({
                            "doctor_id": str(doc["_id"]),
                            "doctor_name": doc["name"],
                            "specialist": doc["specialist"],
                            "language": doc["language"]
                        })
                recommendations.append({"disease": disease, "speciality": specialist, "doctors": doctor_info})
            else:
                recommendations.append({"disease": disease, "speciality": specialist, "doctors": []})
    return recommendations

async def schedule_appointment(patient_id, doctor_id, time_slot):
    appointment_data = {
        "patient_id": ObjectId(patient_id),
        "doctor_id": ObjectId(doctor_id),
        "time_slot": time_slot,
        "status": "pending"
    }
    result = await appointments_collection.insert_one(appointment_data)
    return str(result.inserted_id)

async def check_availability(doctor_id, time_slot):
    doctor = await doctors_collection.find_one({"_id": ObjectId(doctor_id)})
    if doctor:
        print(f"Checking availability for doctor: {doctor['name']}")
        if "availability" in doctor:
            for slot in doctor["availability"]:
                print(f"Checking slot: {slot}")
                if slot["time_slot"] == time_slot and slot["available"]:
                    return True
    return False

async def process_queue():
    pending_appointments = await appointments_collection.find({"status": "pending"}).to_list(length=1000)
    for appointment in pending_appointments:
        if await check_availability(appointment["doctor_id"], appointment["time_slot"]):
            await appointments_collection.update_one(
                {"_id": appointment["_id"]},
                {"$set": {"status": "confirmed"}}
            )
            await doctors_collection.update_one(
                {"_id": appointment["doctor_id"], "availability.time_slot": appointment["time_slot"]},
                {"$set": {"availability.$.available": False}}
            )
        else:
            await appointments_collection.update_one(
                {"_id": appointment["_id"]},
                {"$set": {"status": "failed"}}
            )

async def notify_user(appointment_id):
    appointment = await appointments_collection.find_one({"_id": ObjectId(appointment_id)})
    if appointment:
        patient = await patients_collection.find_one({"_id": appointment["patient_id"]})
        if patient:
            if appointment["status"] == "confirmed":
                message = f"Your appointment with doctor {appointment['doctor_id']} at {appointment['time_slot']} is confirmed."
            else:
                message = f"Your appointment with doctor {appointment['doctor_id']} at {appointment['time_slot']} could not be confirmed. Please choose another slot or doctor."
            print(f"Notification sent to {patient['email']}: {message}")

async def check_queue():
    while True:
        queue_length = await appointments_collection.count_documents({"status": "pending"})
        print(f"Queue length: {queue_length}")
        if queue_length > 0:
            pending_appointments = await appointments_collection.find({"status": "pending"}).to_list(length=1000)
            for appointment in pending_appointments:
                print(f"Appointment ID: {appointment['_id']}, Patient ID: {appointment['patient_id']}, Doctor ID: {appointment['doctor_id']}, Time Slot: {appointment['time_slot']}")
            await process_queue()
        await asyncio.sleep(10)  # Sleep for 10 seconds


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
   
    history_value = query['History']

    # Calculate Priority Score
    f1 = query["Age"]
    f2 = 1 if query["Gender"].lower() == "f" else 0
    f3 = history_value
    f4 = query["timeIn"]
    
    # print(f1," ",f2," ",f3," ")
    return f1 * 0.1 + f2 * 0.1 + f3 * 0.25 + f4 * 0.15

async def check_patients():
    while True:
        appointment_count = await appointments_query.count_documents({})
        # print(f"Patients count: {appointment_count}")
        
        if appointment_count > 0:
            appointments = await appointments_query.find().sort("createdAt", -1).to_list(length=6)
            count = 1
            appointment_list = []

            # Set the priority
            for appointent in appointments:

                #delete the all queries from the collection
                # await appointments_query.delete_one({"_id": appointent["_id"]})
                
                patientId = appointent["patientId"]
                doctorId = appointent["doctorId"]

                # print(patientId)
                # print(doctorId)

                # doctorData = await doctors_collection.find_one({"_id": ObjectId(doctorId)}).to_list()
                patientData = await patients_collection.find_one({"_id": ObjectId(patientId)})

                if patientData :

                    query = {
                        "Age" : patientData["age"],
                        "Gender" : patientData["gender"],
                        "History" : patientData["history"],
                        "timeIn" : count
                    }

                    prio_score = appointent["priority_score"]

                    priority_score = await decrypt_priority_score(prio_score)

                    priority_score *= 0.4

                    priority_score += await calculate_priority_score(query)

                    priority_score += patientData["Ageing"] * 0.1

                    print(priority_score)

                    appointment_list.append({
                        "patientId" : patientId,
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
                

                finalBooking = {
                    "doctorId" : doctorId,
                    "patientId" : patientId,
                    "finalize_booking" : False,
                }

                doctorData = await doctors_collection.find_one({"_id": ObjectId(doctorId)})

                if(doctorData) :
                    available = doctorData['available']
                    booked_slot = doctorData['booked_slot']

                    if(available[timeSlot*2-1] > 0):
                        if(booked_slot[timeSlot*2-1] < 6):
                            value = timeSlot*2-1
                            subvalue = booked_slot[timeSlot*2-1]

                            alloted_slot_start,alloted_slot_end = get_time_range(value, subvalue)

                            booked_slot[timeSlot*2-1]+=1

                            finalBooking["Start"] = alloted_slot_start
                            finalBooking["End"] = alloted_slot_end
                            finalBooking["finalize_booking"] = True
                    elif (available[timeSlot*2-2] > 0):
                        if(booked_slot[timeSlot*2-2] < 6):
                            value = timeSlot*2-2
                            subvalue = booked_slot[timeSlot*2-2]

                            alloted_slot_start,alloted_slot_end = get_time_range(value, subvalue)

                            booked_slot[timeSlot*2-2]+=1

                            finalBooking["Start"] = alloted_slot_start
                            finalBooking["End"] = alloted_slot_end
                            finalBooking["finalize_booking"] = True
                            

                        
                    
                
                booked_appointment = await appointment_booking.insert_one(finalBooking)

                booking_id = booked_appointment.inserted_id

                if finalBooking["finalize_booking"]: 

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
                        },
                        }
                    )

                print(finalBooking)
               
        await asyncio.sleep(30)  # Sleep for 10 seconds