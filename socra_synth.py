from patient import Patient, load_patient_data
from llm_agents import LLMAgents
import json
import os
import pandas as pd
from pathlib import Path
from logger import setup_logger

logger = setup_logger()

class SocraSynth:
    def __init__(self):
        self.logger = logger
        self.llm_agents = LLMAgents()
        self.get_patient_data()
        self.output_dir = "output/results"
        os.makedirs(self.output_dir, exist_ok=True)

    def get_patient_data(self): 
        patient_data_dir = "output/patient_data"
        patient_files = [f for f in os.listdir(patient_data_dir) if f.startswith("patient_") and f.endswith(".json")]
        patient_ids = [int(f.split("_")[1].split(".")[0]) for f in patient_files]       
        if not patient_ids:
            logger.warning("No patient data files found.")
            return None
        patient_id = f"{max(patient_ids):03d}"

        self.patient_id = patient_id
        self.logger.info(f"Patient ID selected: {self.patient_id}")

        # Load patient data
        self.patient = load_patient_data(self.patient_id)
        self.patient_profile = self.patient.get_formatted_data()
        self.cgm_data = self.patient_profile.get('glucose_readings', None)
        self.statistics = self.patient_profile.get('cgm_stats', None)

    def create_agent_prompt(self, role, contentiousness_level):
        """
        Create a prompt for a specific agent role with the specified contentiousness level.
        
        Args:
            role (str): The role of the agent (e.g., "endocrinologist", "diabetes_educator")
            contentiousness_level (float): Level of contentiousness (0.0 to 1.0)
        """
        # Base prompt with patient data
        base_prompt = f"""
        You are an AI assistant playing the role of a {role} in a structured dialogue about diabetes management.
        
        Patient Information:
        {json.dumps(self.patient_profile, indent=2)}
        
        CGM Data Summary:
        {self.statistics}
        
        Your contentiousness level is set to {contentiousness_level} (on a scale of 0-1).
        - At level 0.0: Be completely agreeable and supportive of others' views
        - At level 0.5: Be moderately critical, questioning assumptions but remaining constructive
        - At level 1.0: Be highly critical, challenge every claim, and propose alternative viewpoints
        
        Analyze the patient's CGM data and provide your professional assessment.
        """
        
        # Role-specific additions
        if role == "endocrinologist":
            base_prompt += """
            As an endocrinologist, focus on:
            - Medical interpretation of the glucose patterns
            - Medication adjustments that might be needed
            - Physiological explanations for the observed patterns
            - Potential complications if patterns continue
            
            Be particularly attentive to the hyperglycemic episodes exceeding 350 mg/dL and the concerning hypoglycemic event.
            """
        elif role == "diabetes_educator":
            base_prompt += """
            As a diabetes educator, focus on:
            - Behavioral factors that might explain the observed patterns
            - Educational opportunities for the patient
            - Practical lifestyle modifications
            - Self-management techniques to improve glucose control
            
            Pay special attention to meal timing, physical activity patterns, and how they might relate to the glucose fluctuations.
            """
        elif role == "nutritionist":
            base_prompt += """
            As a nutritionist, focus on:
            - Meal composition and timing
            - Carbohydrate intake patterns
            - Nutritional strategies to reduce glucose variability
            - Dietary recommendations to prevent hypoglycemic events
            
            Consider how the meal times at 8, 13, and 19 hours might be affecting the glucose patterns.
            """
            
        return base_prompt
    
    def create_moderator_prompt(self, previous_discussion):
        """Create a prompt for the moderator agent."""
        return f"""
        You are a moderator in a multi-agent dialogue about diabetes management.
        
        Patient Information:
        {json.dumps(self.patient_profile, indent=2)}
        
        CGM Data Summary:
        {self.statistics}
        
        Previous discussion:
        {previous_discussion}
        
        Your task is to:
        1. Identify the key points of agreement and disagreement among the specialists
        2. Highlight any overlooked aspects of the patient's condition
        3. Guide the discussion toward consensus on the most important issues
        4. Suggest specific questions for the next round of dialogue
        
        Provide a structured summary and 2-3 specific questions to advance the discussion.
        """
    
    def create_synthesis_prompt(self, full_discussion):
        """Create a prompt for the final synthesis of the discussion."""
        return f"""
        You are tasked with synthesizing a multi-specialist discussion about a diabetes patient.
        
        Patient Information:
        {json.dumps(self.patient_profile, indent=2)}
        
        CGM Data Summary:
        {self.statistics}
        
        Full discussion transcript:
        {full_discussion}
        
        Your task is to:
        1. Synthesize the key insights from all specialists
        2. Identify the most probable explanations for the patient's glucose patterns
        3. Provide a comprehensive set of recommendations addressing:
           - Medication adjustments
           - Meal timing and composition
           - Physical activity
           - Self-monitoring practices
           - Education needs
        4. Highlight areas where further data or assessment might be needed
        
        Structure your response as a clinical assessment and management plan that could be presented to the patient's healthcare team.
        """
    
    def run_dialogue(self):
        """Run the multi-agent SocraSynth dialogue."""
        self.logger.info("Starting SocraSynth dialogue")
        
        # Initialize agents with different roles and contentiousness levels
        endocrinologist_id = self.llm_agents.create_agent(
            provider="openai", 
            system_prompt="You are an endocrinologist specializing in diabetes management."
        )
        
        diabetes_educator_id = self.llm_agents.create_agent(
            provider="openai", 
            system_prompt="You are a diabetes educator with expertise in self-management education."
        )
        
        nutritionist_id = self.llm_agents.create_agent(
            provider="openai", 
            system_prompt="You are a nutritionist specializing in diabetes meal planning."
        )
        
        moderator_id = self.llm_agents.create_agent(
            provider="openai", 
            system_prompt="You are a moderator facilitating a clinical discussion."
        )
        
        synthesizer_id = self.llm_agents.create_agent(
            provider="openai", 
            system_prompt="You are synthesizing insights from multiple specialists."
        )
        
        # Initial prompts for each specialist
        endocrinologist_prompt = self.create_agent_prompt("endocrinologist", 0.7)
        diabetes_educator_prompt = self.create_agent_prompt("diabetes_educator", 0.3)
        nutritionist_prompt = self.create_agent_prompt("nutritionist", 0.5)
        
        # Get initial responses
        self.logger.info("Getting initial specialist responses")
        endocrinologist_response = self.llm_agents.call_agent(endocrinologist_id, endocrinologist_prompt)
        diabetes_educator_response = self.llm_agents.call_agent(diabetes_educator_id, diabetes_educator_prompt)
        nutritionist_response = self.llm_agents.call_agent(nutritionist_id, nutritionist_prompt)
        
        # Compile first round discussion
        round1_discussion = f"""
        ROUND 1 DISCUSSION:
        
        Endocrinologist:
        {endocrinologist_response}
        
        Diabetes Educator:
        {diabetes_educator_response}
        
        Nutritionist:
        {nutritionist_response}
        """
        
        # Get moderator's summary and questions
        self.logger.info("Getting moderator's input after round 1")
        moderator_prompt = self.create_moderator_prompt(round1_discussion)
        moderator_response = self.llm_agents.call_agent(moderator_id, moderator_prompt)
        
        # Second round with adjusted contentiousness (moving toward consensus)
        self.logger.info("Starting round 2 with adjusted contentiousness levels")
        endocrinologist_prompt2 = f"""
        {round1_discussion}
        
        Moderator:
        {moderator_response}
        
        You are an endocrinologist. Your contentiousness level is now 0.5 (more moderate).
        Respond to the points raised by other specialists and the moderator's questions.
        Focus on finding common ground while maintaining your medical perspective.
        """
        
        diabetes_educator_prompt2 = f"""
        {round1_discussion}
        
        Moderator:
        {moderator_response}
        
        You are a diabetes educator. Your contentiousness level is now 0.5 (more balanced).
        Respond to the points raised by other specialists and the moderator's questions.
        Focus on finding common ground while maintaining your educational perspective.
        """
        
        nutritionist_prompt2 = f"""
        {round1_discussion}
        
        Moderator:
        {moderator_response}
        
        You are a nutritionist. Your contentiousness level remains at 0.5.
        Respond to the points raised by other specialists and the moderator's questions.
        Focus on finding common ground while maintaining your nutritional perspective.
        """
        
        # Get second round responses
        endocrinologist_response2 = self.llm_agents.call_agent(endocrinologist_id, endocrinologist_prompt2)
        diabetes_educator_response2 = self.llm_agents.call_agent(diabetes_educator_id, diabetes_educator_prompt2)
        nutritionist_response2 = self.llm_agents.call_agent(nutritionist_id, nutritionist_prompt2)
        
        # Compile full discussion
        full_discussion = f"""
        {round1_discussion}
        
        Moderator:
        {moderator_response}
        
        ROUND 2 DISCUSSION:
        
        Endocrinologist:
        {endocrinologist_response2}
        
        Diabetes Educator:
        {diabetes_educator_response2}
        
        Nutritionist:
        {nutritionist_response2}
        """
        
        # Final synthesis
        self.logger.info("Creating final synthesis")
        synthesis_prompt = self.create_synthesis_prompt(full_discussion)
        final_synthesis = self.llm_agents.call_agent(synthesizer_id, synthesis_prompt)
        
        # Save the complete dialogue and synthesis
        self.save_results(full_discussion, final_synthesis)
        
        return final_synthesis
    
    def save_results(self, full_discussion, final_synthesis):
        """Save the dialogue and synthesis to the output directory."""
        self.logger.info(f"Saving SocraSynth results to {self.output_dir}")
        
        # Save the full discussion
        discussion_path = Path(self.output_dir).joinpath('socrasynth_discussion.txt')
        with open(discussion_path, 'w') as f:
            f.write(full_discussion)
        
        # Save the final synthesis
        synthesis_path = Path(self.output_dir).joinpath('socrasynth_synthesis.txt')
        with open(synthesis_path, 'w') as f:
            f.write(final_synthesis)
        
        # Save a combined JSON with all data
        results = {
            "patient_profile": self.patient_profile,
            "statistics": self.statistics,
            "full_discussion": full_discussion,
            "final_synthesis": final_synthesis
        }
        
        json_path = Path(self.output_dir).joinpath('socrasynth_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"SocraSynth results saved to {self.output_dir}")

if __name__ == "__main__":
    # Initialize SocraSynth
    socrasynth = SocraSynth()
    
    # Run the dialogue
    final_synthesis = socrasynth.run_dialogue()
    
    # Print the final synthesis
    print("\nFINAL SYNTHESIS:")
    print(final_synthesis)
