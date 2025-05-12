from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
import pandas as pd
import random
import json
import os
import matplotlib.pyplot as plt
from datetime import timedelta

from config import (
    CGM_READING_INTERVAL,
    CGM_TOTAL_READINGS,
    PATIENT_DATA_DIR,
    HYPOGLYCEMIA_THRESHOLD,
    HYPERGLYCEMIA_THRESHOLD,
    PATIENT_PROFILE_FROM_CONFIG
)

from logger import setup_logger
logger = setup_logger()

class PatientIdenitfier(BaseModel):
    id: str
    name: Optional[str] = None
    age: Optional[int] = None

    def __init__(self, **data):
        data['id'] = self._get_next_patient_id(PATIENT_DATA_DIR) if 'id' not in data else data['id']
        data['name'] = data.get('name', "John Doe")
        data['age'] = data.get('age', random.randint(20, 80))
        super().__init__(**data)
    
    @staticmethod
    def _get_next_patient_id(data_dir: str) -> str:
        patient_files = [f for f in os.listdir(data_dir) if f.startswith("patient_") and f.endswith(".json")]
        patient_ids = [int(f.split("_")[1].split(".")[0]) for f in patient_files]
        next_id = max(patient_ids, default=0) + 1
        return f"{next_id:03d}"
        
class PatientClinicalData(BaseModel):
    diabetes_type: Optional[str] = None
    years_with_diabetes: Optional[int] = None
    medications: Optional[List[str]] = None
    medical_history: Optional[List[str]] = None

class PatientCGMData(BaseModel):
    timestamps: List[datetime] = Field(..., description="Timestamps of CGM readings")
    glucose_levels: List[float] = Field(..., description="Glucose levels in mg/dL")

    def __init__(self, **data):
        try:
            baseline_glucose = data.get('baseline_glucose', PATIENT_PROFILE_FROM_CONFIG['baseline_glucose'])
            variability = data.get('variability', PATIENT_PROFILE_FROM_CONFIG['variability'])
            meal_impact = data.get('meal_impact', PATIENT_PROFILE_FROM_CONFIG['meal_impact'])
            activity_impact = data.get('activity_impact', PATIENT_PROFILE_FROM_CONFIG['activity_impact'])
            meal_times = data.get('meal_times', PATIENT_PROFILE_FROM_CONFIG['meal_times'])
            activity_times = data.get('activity_times', PATIENT_PROFILE_FROM_CONFIG['activity_times'])
            has_dawn_effect = data.get('has_dawn_effect', PATIENT_PROFILE_FROM_CONFIG['has_dawn_effect'])
        except Exception as e:
            logger.warning(f"Missing key in patient profile: {e}")
            logger.info("Using default values for CGM data generation.")
            baseline_glucose = 140
            variability = 30
            meal_impact = 60
            activity_impact = -35
            meal_times = [8, 13, 19]
            activity_times = [10, 16]
            has_dawn_effect = True

        # Generate timestamps (5-minute intervals)
        start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        timestamps = [start_time + timedelta(minutes=CGM_READING_INTERVAL*i) for i in range(CGM_TOTAL_READINGS)]

        # Generate glucose levels based on the profile
        glucose_levels = []
         # Initialize glucose values with baseline and random noise
        glucose_levels = [baseline_glucose + np.random.normal(0, variability) for _ in range(CGM_TOTAL_READINGS)]
        
        # Apply meal and activity impacts
        for meal_hour in meal_times:
            for i in range(CGM_TOTAL_READINGS):
                if timestamps[i].hour == meal_hour:
                    glucose_levels[i] += meal_impact

        for activity_hour in activity_times:
            for i in range(CGM_TOTAL_READINGS):
                if timestamps[i].hour == activity_hour:
                    glucose_levels[i] += activity_impact

        # Apply dawn phenomenon
        if has_dawn_effect:
            for i in range(CGM_TOTAL_READINGS):
                if 4 <= timestamps[i].hour < 8:
                    glucose_levels[i] += 20  # Dawn phenomenon

        # Add random noise to glucose levels
        glucose_levels = [glucose + np.random.normal(0, 5) for glucose in glucose_levels]

        # Add low glucose levels for hypoglycemia simulation
        min_index = random.randint(0,  5)
        max_index = random.randint(min_index+1, 10)
        hypoglycemia_indices = random.sample(range(CGM_TOTAL_READINGS), k=random.randint(min_index, max_index))
        for i in hypoglycemia_indices:
            glucose_levels[i] = random.uniform(50, 70)  # Simulate hypoglycemia

        # Ensure glucose levels are within physiological limits
        glucose_levels = [max(40, min(400, glucose)) for glucose in glucose_levels]

        # Convert timestamps to datetime objects
        # Store the generated data
        data['timestamps'] = timestamps
        data['glucose_levels'] = glucose_levels
        super().__init__(**data)

    def get_stats(self) -> Dict[str, float]:
        mean_glucose = np.mean(self.glucose_levels)
        median_glucose = np.median(self.glucose_levels)
        std_dev_glucose = np.std(self.glucose_levels)
        min_glucose = np.min(self.glucose_levels)
        max_glucose = np.max(self.glucose_levels)
        stats = {
            'mean': mean_glucose,
            'median': median_glucose,
            'std_dev': std_dev_glucose,
            'min': min_glucose,
            'max': max_glucose,
            'hypoglycemia_events': len([g for g in self.glucose_levels if g < HYPOGLYCEMIA_THRESHOLD]),
            'hyperglycemia_events': len([g for g in self.glucose_levels if g > HYPERGLYCEMIA_THRESHOLD])
        }
        return stats

    def to_dataframe(self) -> pd.DataFrame:
        data = {
            'timestamp': self.timestamps,
            'glucose_mg_dl': self.glucose_levels
        }
        df = pd.DataFrame(data)
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['glucose_mg_dl'] = df['glucose_mg_dl'].round(1)
        return df

    def plot_glucose_data(self):
        df = self.to_dataframe()
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['glucose_mg_dl'], label='Glucose Levels', color='blue')
        plt.axhline(y=HYPOGLYCEMIA_THRESHOLD, color='red', linestyle='--', label='Hypoglycemia Threshold')
        plt.axhline(y=HYPERGLYCEMIA_THRESHOLD, color='orange', linestyle='--', label='Hyperglycemia Threshold')
        plt.title('Continuous Glucose Monitoring Data')
        plt.xlabel('Timestamp')
        plt.ylabel('Glucose Levels (mg/dL)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_to_json(self, filename: str):
        data = {
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'glucose_levels': self.glucose_levels
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

class Patient:
    def __init__(self):
        data = {
            'name': "John Doe",
            'age': random.randint(20, 80)
            }
        self.patient_identifier = PatientIdenitfier(**data)
        self.patient_clinical_data = PatientClinicalData()
        self.patient_cgm_data = PatientCGMData()

    def save(self, output_dir=PATIENT_DATA_DIR):
        os.makedirs(output_dir, exist_ok=True)
        patient_id = self.patient_identifier.id
        patient_profile = {
            'id': patient_id,
            'idenitifier': {
                'name': self.patient_identifier.name,
                'age': self.patient_identifier.age
            },
            'clinical_data': {
                'diabetes_type': self.patient_clinical_data.diabetes_type,
                'years_with_diabetes': self.patient_clinical_data.years_with_diabetes,
                'medications': self.patient_clinical_data.medications,
                'medical_history': self.patient_clinical_data.medical_history
            },
            'cgm_data': {
                'timestamps': [ts.isoformat() for ts in self.patient_cgm_data.timestamps],
                'glucose_levels': self.patient_cgm_data.glucose_levels
            }
        }
        json_path = os.path.join(output_dir, f"patient_{patient_id}.json")
        with open(json_path, 'w') as f:
            json.dump(patient_profile, f, indent=2)
        logger.info(f"Saved patient data to {json_path}")

    def get_formatted_data(self):
        """Return a formatted dictionary of patient data for easy use in prompts"""
        stats = self.patient_cgm_data.get_stats()
        
        return {
            "id": self.patient_identifier.id,
            "name": self.patient_identifier.name,
            "age": self.patient_identifier.age,
            "diabetes_type": self.patient_clinical_data.diabetes_type,
            "years_with_diabetes": self.patient_clinical_data.years_with_diabetes,
            "medications": self.patient_clinical_data.medications,
            "medical_history": self.patient_clinical_data.medical_history,
            "cgm_stats": stats,
            "glucose_readings": self.patient_cgm_data.glucose_levels,
            "monitoring_period": f"{self.patient_cgm_data.timestamps[0]} to {self.patient_cgm_data.timestamps[-1]}"
        }


def load_patient_data(patient_id:str=None, output_dir=PATIENT_DATA_DIR):
    if not patient_id:
        # choose the latest patient id
        patient_files = [f for f in os.listdir(output_dir) if f.startswith("patient_") and f.endswith(".json")]
        patient_ids = [int(f.split("_")[1].split(".")[0]) for f in patient_files]
        if not patient_ids:
            logger.warning("No patient data files found.")
            return None
        patient_id = f"{max(patient_ids):03d}"
    logger.info(f"Loading patient data for ID: {patient_id}")
    json_path = os.path.join(output_dir, f"patient_{patient_id}.json")
    if not os.path.exists(json_path):
        logger.warning(f"Patient data file {json_path} does not exist.")
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
    patient = Patient()
    patient.patient_identifier.id = patient_id
    patient.patient_identifier.name = data['idenitifier']['name']
    patient.patient_identifier.age = data['idenitifier']['age']
    patient.patient_clinical_data.diabetes_type = data['clinical_data']['diabetes_type']
    patient.patient_clinical_data.years_with_diabetes = data['clinical_data']['years_with_diabetes']
    patient.patient_clinical_data.medications = data['clinical_data']['medications']
    patient.patient_clinical_data.medical_history = data['clinical_data']['medical_history']
    patient.patient_cgm_data.timestamps = [datetime.fromisoformat(ts) for ts in data['cgm_data']['timestamps']]
    patient.patient_cgm_data.glucose_levels = data['cgm_data']['glucose_levels']
    return patient

if __name__ == "__main__":
   loaded_patient = load_patient_data()
   if loaded_patient:
       print(f"Loaded patient ID: {loaded_patient.patient_identifier.id}")
       #print some of cgm data
       print(f"CGM Data: {loaded_patient.patient_cgm_data.glucose_levels[:5]}")
       #loaded_patient.patient_cgm_data.plot_glucose_data()

