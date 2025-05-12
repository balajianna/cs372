from patient import Patient
from llm_agents import LLMAgents
from patient import load_patient_data
import json
import os
import pandas as pd

from logger import setup_logger
logger = setup_logger()

class Evaluator:
    def __init__(self):
        self.logger = logger        
        self.valid_responses = {}
        self.load_response_data()

    def load_response_data(self):
        """Load LLM responses from JSON file."""
        data_file = "output/results/all_responses.json"
        if not os.path.exists(data_file):
            self.logger.error(f"File {data_file} does not exist.")
            return
        with open(data_file, 'r') as file:
            data = json.load(file)
        self.logger.info(f"Loading LLM responses from {data_file}")
        
        if not data:
            self.logger.error("Failed to load data.")
            return
        
        self.patient_id = data.get('patient_id', None)
        self.patient = load_patient_data(self.patient_id)
        self.patient_profile = self.patient.get_formatted_data()
        self.cgm_data = self.patient_profile.get('glucose_readings', None)
        self.statistics = self.patient_profile.get('cgm_stats', None)
        self.original_prompt = data.get('prompt', None)
        responses = data.get('responses', {})
        self.valid_responses = {f"response_{i+1}": value for i, (key, value) in enumerate(responses.items())}

        self.logger.info(f"responses loaded: {len(self.valid_responses)}")

    def create_socratic_prompt(self):
        eval_prompt = f"""
        You are an expert diabetes care specialist tasked with evaluating AI-generated clinical guidance notes.
        
        The following were the original inputs provided to the AI systems:
        Patient Profile:
        {self.patient_profile}
        
        CGM Data:
        {self.cgm_data}
        
        Patient Statistics:
        {self.statistics}
        
        The AI systems were asked to provide clinical guidance based on the following prompt:
        {self.original_prompt}

        The AI systems generated the following responses:
        """

        for i, (model, response) in enumerate(self.valid_responses.items(), 1):
            eval_prompt += f"""
            Response {i} ({model}):
            {response}
            
            """

        eval_prompt += f"""

        Your task is to evaluate the responses based on the following criteria:
        
        Evaluation Criteria:
        
        1. Definition (1-5): 
        - How clear and understandable is the response for a patient?
        - Is medical jargon appropriately explained?
        - Is the tone supportive and encouraging?
        
        2. Generalization (1-5): 
        - Is the guidance making appropriate generalizations? 
        - Does it avoid overgeneralizing where personalization is needed?
        - How specific and implementable are the recommendations?
        - Are the suggestions tailored to this specific patient?

        3. Induction (1-5): 
        - Would these recommendations be practical in real life?
        - Is there sufficient evidence to support the recommendations? 
        - Are the conclusions drawn from the CGM data reasonable?
        
        4. Elenchus (1-5): 
        - Are there contradictions in the reasoning? 
        - Are any recommendations inconsistent with others?

        5. Hypothesis Elimination (1-5): 
        - Have alternative explanations or approaches been considered? 
        - Are there other potential causes for glucose patterns?

        6. Maieutic (1-5): 
        - How useful would this guidance be for improving the patient's diabetes management?
        - Does it address the most critical issues shown in the data?
        - Would it likely lead to improved outcomes?

        7. Dialectic (1-5): 
        - How would this guidance stand up to counterarguments? 
        - What are potential objections to these recommendations?

        For each response, provide:
        1. Numerical scores for each criterion
        2. Brief justification for each score
        3. Specific strengths and weaknesses
        4. A final recommendation on which response would be more helpful and why
        
        Use the following scoring system:
        1 = Poor
        2 = Fair
        3 = Good
        4 = Very Good
        5 = Excellent
        
        You cannoat assign the same score to both responses, for each criterion. You must differentiate between the two responses.

        IMPORTANT: You must format your response as a valid JSON object with the following structure:
        
        {{
        "evaluations": {{
        """

        # Dynamically generate the JSON structure for each provider
        providers = list(self.valid_responses.keys())
        for i, provider in enumerate(providers):
            eval_prompt += f"""
                "{provider}": {{
                    "definition": {{
                        "score": 0,
                        "rationale": "Your assessment here..."
                    }},
                    "generalization": {{
                        "score": 0,
                        "rationale": "Your assessment here..."
                    }},
                    "induction": {{
                        "score": 0,
                        "rationale": "Your assessment here..."
                    }},
                    "elenchus": {{
                        "score": 0,
                        "rationale": "Your assessment here..."
                    }},
                    "hypothesis_elimination": {{
                        "score": 0,
                        "rationale": "Your assessment here..."
                    }},
                    "maieutic": {{
                        "score": 0,
                        "rationale": "Your assessment here..."
                    }},
                    "dialectic": {{
                        "score": 0,
                        "rationale": "Your assessment here..."
                    }},
                }},
            """
        
        # Complete the JSON structure with comparison section
        eval_prompt += f"""
        
        All scores must be integers between 1 and 5.
        Do not include any text outside of this JSON structure.
        """
        return eval_prompt

    def _extract_scores(self, evaluation_json):
        try:
            results = {}            
            for provider, scores in evaluation_json["evaluations"].items():
                provider_total = 0
                provider_scores = {}
                number_of_criteria = len(scores)
                for criterion, details in scores.items():
                    score = details["score"]
                    rationale = details["rationale"]
                    provider_scores[criterion] = {
                        "score": score,
                        "rationale": rationale
                    }
                    provider_total += score
                
                results[provider] = {
                    "scores": provider_scores,
                    "total_score": provider_total,
                    "average_score": provider_total / number_of_criteria
                }
            return results
        except Exception as e:
            self.logger.error(f"Error extracting scores: {str(e)}")
            return None
    
    def evaluate(self, prompt, provider="openai"):
        llm_agents = LLMAgents()
        agent_id = llm_agents.create_agent(provider=provider, system_prompt="Evaluate the AI-generated clinical guidance notes.")
        response = llm_agents.call_agent(agent_id, prompt)
        self.logger.info(f"Evaluation response: {response}")
        # use the response to extract scores
        try:
            evaluation_json = json.loads(response)
            extracted_scores = self._extract_scores(evaluation_json)
            if extracted_scores:
                output_dir = "output/results"
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, "socratic_eval.json")
                with open(file_path, "w") as f:
                    json.dump(extracted_scores, f, indent=4)
                self.logger.info(f"Evaluation results saved to {file_path}")
            else:
                self.logger.error("Failed to extract scores from evaluation response.")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON response: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            self.logger.error("The response was not in the expected JSON format.")
            self.logger.error(f"Response content: {response}")

    def create_df(self):
        try:
            output_dir = "output/results"
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, "socratic_eval.json")
            
            with open(file_path, "r") as f:
                results = json.load(f)
                
            # Check if results are empty
            if not results:
                return "No evaluation results found."
                
            # Create DataFrame structure
            data = []
            
            for provider, provider_data in results.items():
                row = {'provider': provider}
                
                # Add individual criteria scores
                for criterion, criterion_data in provider_data["scores"].items():
                    # if score is not an integer or not in range or not a number, set to 0
                    if not isinstance(criterion_data["score"], int) or criterion_data["score"] < 1 or criterion_data["score"] > 5:
                        row[criterion] = 0
                        self.logger.warning(f"Invalid score for {provider} - {criterion}: {criterion_data['score']}")
                    else:
                        row[criterion] = criterion_data["score"]
                    
                # Add total and average
                # if total is not an integer or not in range or not a number, set to 0
                if not isinstance(provider_data["total_score"], (int, float)):
                    row['total'] = 0
                    self.logger.warning(f"Invalid total score for {provider}: {provider_data['total_score']}")
                else:
                    row['total'] = provider_data["total_score"]

                # if average is not an integer or not in range or not a number, set to 0
                if not isinstance(provider_data["average_score"], (int, float)) or provider_data["average_score"] < 0 or provider_data["average_score"] > 5:
                    row['average'] = 0
                    self.logger.warning(f"Invalid average score for {provider}: {provider_data['average_score']}")
                else:
                    row['average'] = provider_data["average_score"]
                
                data.append(row)
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Reorder columns to have provider first, then criteria, then total and average
            criteria_cols = [col for col in df.columns if col not in ['provider', 'total', 'average']]
            column_order = ['provider'] + criteria_cols + ['total', 'average']
            df = df[column_order]
            
            # Save to CSV
            df.to_csv("output/results/scores_table.csv", index=False)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating scores table: {str(e)}")
            return "Error creating table"


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.load_response_data()
    socratic_prompt = evaluator.create_socratic_prompt()
    evaluator.evaluate(socratic_prompt, provider="anthropic")
    socratic_scores = evaluator.create_df()
    print(socratic_scores)
   
    evaluator.evaluate(socratic_prompt, provider="openai")
    socratic_scores = evaluator.create_df()
    print(socratic_scores)
