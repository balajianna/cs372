from patient import Patient, load_patient_data
from config import PATIENT_PROFILE_ONE
from llm_agents import LLMAgents
import os

from logger import setup_logger
import json
logger = setup_logger()
sample_every_n = 4  # Sample every 20 minutes
readings_limit = 30  # Limit to 30 readings to save tokens

class LLMAdvisor:
    def __init__(self):
        self.logger = logger
        self.patient = None
        self.prompt = None
        self.responses = {
            "openai": None,
            "anthropic": None,
            "gemini": None
        }

    def set_patient_data(self, patient:Patient):
        if patient is None or not isinstance(patient, Patient):
            raise ValueError("Invalid patient data provided. Please provide a valid Patient object.")
        self.logger.info(f"setting patient {patient.get_formatted_data()} for LLMAdvisor")
        self.patient = patient
        self.logger.info(f"Patient data set successfully: {self.patient.get_formatted_data()}")
        
    def set_prompt(self):
        patient_name = self.patient.patient_identifier.name
        patient_age = self.patient.patient_identifier.age
        patient_diabetes_type = self.patient.patient_clinical_data.diabetes_type
        patient_years_with_diabetes = self.patient.patient_clinical_data.years_with_diabetes
        patient_medications = self.patient.patient_clinical_data.medications
        patient_medical_history = self.patient.patient_clinical_data.medical_history

        # Get the data in a DataFrame format first
        if hasattr(self.patient.patient_cgm_data, 'get_dataframe'):
            # If there's a method to get a DataFrame
            cgm_data = self.patient.patient_cgm_data.get_dataframe()[::sample_every_n]
        else:
            # Create a DataFrame from the glucose readings
            import pandas as pd
            timestamps = self.patient.patient_cgm_data.timestamps  # Assuming this exists
            readings = self.patient.patient_cgm_data.glucose_levels[::sample_every_n]
            timestamps = timestamps[::sample_every_n]  # Sample timestamps at the same rate
            cgm_data = pd.DataFrame({
                'timestamp': timestamps,
                'glucose_mg_dl': readings
            })

        stats = self.patient.patient_cgm_data.get_stats()
        time_series = cgm_data.dropna().apply(lambda row: f"{row['timestamp'].strftime('%H:%M')}: {row['glucose_mg_dl']:.1f} mg/dL", axis=1).tolist()
        formatted_data = {"statistics": stats, "time_series_sample": time_series[:readings_limit]}  # Limit to x readings to save tokens

        prompt = f"""
        You are a diabetes care specialist analyzing Continuous Glucose Monitoring (CGM) data for a patient. 
        
        Patient Information:
        - Name: {patient_name}
        - Age: {patient_age}
        - Diabetes Type: {patient_diabetes_type}
        - Years with Diabetes: {patient_years_with_diabetes}
        - Current Medications: {patient_medications}
        - Medical History: {patient_medical_history}
        
        CGM Data:
        
        Below is 10 hours of CGM data (sampled every 20 minutes) along with summary statistics:
        
        {formatted_data}
        
        Based on this CGM data, please conduct a thorough analysis on the following dimensions:
        1. A brief assessment of the patient's glucose control
        2. Identification of any concerning patterns
        3. Specific recommendations for medication, nutrition, activity, and sleep
        4. Suggestions for improving time in range (70-180 mg/dL)
        
        You should create a JSON output that includes the following fields:
        - assessment: A brief assessment of the patient's glucose control
        - patterns: A list of concerning patterns identified in the CGM data
        - recommendations: A list of specific recommendations for medication, nutrition, activity, and sleep
        - time_in_range: A percentage of time spent in the target range (70-180 mg/dL)
        
        
        Please note that the analysis should be thorough and consider the patient's entire medical history, current medications, and lifestyle factors. 
        The recommendations should be evidence-based and tailored to the patient's specific needs.
        The analysis should be clear and concise, avoiding medical jargon where possible. 
        The recommendations should be actionable and easy for the patient to understand.
        The JSON output should be well-structured and easy to parse. 
        Please ensure that the output is in JSON format and does not include any other text.
        The analysis should be based on the latest clinical guidelines and best practices in diabetes care.
        The recommendations should be practical and feasible for the patient to implement in their daily life.
        
        """
        self.prompt = prompt
        self.logger.info(f"Prompt set successfully")
        
    def get_openai_response(self):
        llm_agents = LLMAgents()
        agent_id = llm_agents.create_agent(provider="openai", system_prompt="You are a diabetes care specialist analyzing Continuous Glucose Monitoring (CGM) data for a patient.")
        response = llm_agents.call_agent(agent_id, self.prompt)
        self.responses["openai"] = response
        self.logger.info(f"openAI response set: {response}")
        return response

    def get_anthropic_response(self):
        llm_agents = LLMAgents()
        agent_id = llm_agents.create_agent(provider="anthropic", system_prompt="You are a diabetes care specialist analyzing Continuous Glucose Monitoring (CGM) data for a patient.")
        response = llm_agents.call_agent(agent_id, self.prompt)
        self.responses["anthropic"] = response
        self.logger.info(f"Anthropic response set: {response}")
        return response

    def get_gemini_response(self):
        llm_agents = LLMAgents()
        agent_id = llm_agents.create_agent(provider="gemini", system_prompt="You are a diabetes care specialist analyzing Continuous Glucose Monitoring (CGM) data for a patient.")
        response = llm_agents.call_agent(agent_id, self.prompt)
        self.responses["gemini"] = response
        self.logger.info(f"Gemini response set: {response}")
        return response

    def save_prompt_and_responses(self):
        try:        
            openai_response = self.responses["openai"]
            anthropic_response = self.responses["anthropic"]
            gemini_response = self.responses["gemini"]
            output_dir = "output/results"
            os.makedirs(output_dir, exist_ok=True)
            # Create a dictionary with all the data
            output_data = {
                "patient_id": self.patient.patient_identifier.id,
                "prompt": self.prompt,
                "responses": {}
            }
            
            # Add responses if available
            if openai_response:
                output_data["responses"]["openai"] = openai_response
            if anthropic_response:
                output_data["responses"]["anthropic"] = anthropic_response
            if gemini_response:
                output_data["responses"]["gemini"] = gemini_response
                
            # Write to JSON file
            file_path = os.path.join(output_dir, "all_responses.json")
            with open(file_path, "w") as f:
                json.dump(output_data, f, indent=4, default=str)
            # Log the file path
            self.logger.info(f"Prompt and responses saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error retrieving responses: {e}")
            return "Error in response retrieval. Please check the input data."

if __name__ == "__main__":
    patient = load_patient_data()
    if not patient:
        logger.error(f"Failed to load patient data")
        exit(1)

    # Initialize LLMAdvisor
    llm_advisor = LLMAdvisor()
    llm_advisor.set_patient_data(patient)
    llm_advisor.set_prompt()
    llm_advisor.get_openai_response()
    llm_advisor.get_anthropic_response()
    llm_advisor.save_prompt_and_responses()
    logger.info("LLMAdvisor process completed successfully.")
