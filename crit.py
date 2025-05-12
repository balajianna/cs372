from patient import Patient, load_patient_data
from llm_agents import LLMAgents
import json
import os
import pandas as pd
from pathlib import Path
from logger import setup_logger

logger = setup_logger()

class CritFramework:
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

    def create_definition_prompt(self):
        """Create the initial definition prompt for the CRIT framework with JSON output."""
        return f"""
        You are a diabetes care specialist using the CRIT (Critical Inquisitive Template) framework to analyze CGM data.

        Patient Information:
        {json.dumps(self.patient_profile, indent=2)}

        CGM Data Summary:
        {self.statistics}

        This is the Definition phase of the CRIT framework. Your task is to:
        1. Define what constitutes normal blood glucose range, hyperglycemia, and hypoglycemia in diabetes management
        2. Analyze the attached CGM data for {self.patient_profile['name']}, identifying the key patterns visible in the glucose readings
        3. Establish clear definitions for terms you will use in your analysis

        IMPORTANT: Format your response as a valid JSON object with the following structure:
        {{
            "normal_range": {{
                "lower_bound": 70,
                "upper_bound": 180
            }},
            "hyperglycemia": {{
                "definition": "Blood glucose above 180 mg/dL",
                "severity_levels": [
                    {{
                        "level": "mild",
                        "range": "180-250 mg/dL"
                    }},
                    {{
                        "level": "moderate",
                        "range": "250-350 mg/dL"
                    }},
                    {{
                        "level": "severe",
                        "range": ">350 mg/dL"
                    }}
                ]
            }},
            "hypoglycemia": {{
                "definition": "Blood glucose below 70 mg/dL",
                "severity_levels": [
                    {{
                        "level": "mild",
                        "range": "54-70 mg/dL"
                    }},
                    {{
                        "level": "severe",
                        "range": "<54 mg/dL"
                    }}
                ]
            }},
            "observed_patterns": [
                {{
                    "pattern_name": "Morning hyperglycemia",
                    "description": "Elevated glucose levels observed consistently in morning hours",
                    "frequency": "daily",
                    "average_magnitude": "210 mg/dL"
                }}
            ],
            "term_definitions": [
                {{
                    "term": "Dawn phenomenon",
                    "definition": "A surge in blood glucose levels in the early morning hours"
                }}
            ]
        }}

        Focus on being precise with medical terminology while ensuring clarity. This will form the foundation for subsequent analysis.
        Respond with valid JSON only. Do not include any explanation or extra text.
        """

    def create_inductive_prompt(self, previous_response):
        """Create the inductive reasoning prompt based on the previous response with JSON output."""
        # Create meal and activity correlation text based on available data
        meal_correlation_text = ""
        activity_correlation_text = ""
        
        if 'meal_times' in self.patient_profile and self.patient_profile['meal_times']:
            meal_times_str = ', '.join(map(str, self.patient_profile['meal_times']))
            meal_correlation_text = f"meal times ({meal_times_str})"
        else:
            meal_correlation_text = "meal times (not provided)"
        
        if 'activity_times' in self.patient_profile and self.patient_profile['activity_times']:
            activity_times_str = ', '.join(map(str, self.patient_profile['activity_times']))
            activity_correlation_text = f"activity times ({activity_times_str})"
        else:
            activity_correlation_text = "activity times (not provided)"

        return f"""
        You are a diabetes care specialist using the CRIT (Critical Inquisitive Template) framework to analyze CGM data.

        Patient Information:
        {json.dumps(self.patient_profile, indent=2)}

        CGM Data Summary:
        {self.statistics}

        Previous analysis (Definition phase):
        {previous_response}

        This is the Inductive Reasoning phase of the CRIT framework. Your task is to:
        1. Based on the patterns identified in the patient's CGM data, what general patterns can you identify?
        2. What might these patterns suggest about the patient's diabetes management?
        3. Look for correlations between glucose levels and {meal_correlation_text} or {activity_correlation_text}
        4. Identify any recurring patterns in the data that might indicate specific issues

        IMPORTANT: Format your response as a valid JSON object with the following structure:
        {{
            "general_patterns": [
                {{
                    "pattern": "Post-meal hyperglycemia",
                    "evidence": "Glucose rises 80-120 mg/dL after recorded meal times",
                    "frequency": "75% of meals",
                    "significance": "Indicates potential issues with meal insulin timing or dosing"
                }}
            ],
            "management_insights": [
                {{
                    "insight": "Inadequate mealtime insulin coverage",
                    "supporting_evidence": "Consistent post-prandial spikes",
                    "confidence_level": "high"
                }}
            ],
            "meal_correlations": [
                {{
                    "meal_time": "8:00",
                    "observed_pattern": "Glucose rise of approximately 100 mg/dL within 2 hours",
                    "potential_cause": "Insufficient insulin-to-carb ratio for breakfast",
                    "consistency": "Observed on 6 of 7 days"
                }}
            ],
            "activity_correlations": [
                {{
                    "activity_time": "17:00",
                    "observed_pattern": "Glucose drop of approximately 50 mg/dL during activity",
                    "recommendation": "Consider reducing pre-exercise insulin or consuming carbs before activity",
                    "consistency": "Observed on all days with recorded activity"
                }}
            ],
            "recurring_issues": [
                {{
                    "issue": "Nocturnal hypoglycemia",
                    "time_pattern": "Between 2:00-4:00 AM",
                    "frequency": "3 of 7 nights",
                    "potential_causes": [
                        "Excessive basal insulin overnight",
                        "Delayed effect of evening exercise"
                    ]
                }}
            ]
        }}

        Focus on drawing reasonable inferences from the observed data without making assumptions beyond what the data shows.
        Respond with valid JSON only. Do not include any explanation or extra text.
        """

    def create_elenchus_prompt(self, previous_responses):
        """Create the elenchus (critical challenge) prompt based on previous responses with JSON output."""
        return f"""
        You are a diabetes care specialist using the CRIT (Critical Inquisitive Template) framework to analyze CGM data.

        Patient Information:
        {json.dumps(self.patient_profile, indent=2)}

        CGM Data Summary:
        {self.statistics}

        Previous analysis:
        {previous_responses}

        This is the Elenchus (Critical Challenge) phase of the CRIT framework. Your task is to:
        1. Challenge the assumptions made in the previous analysis
        2. Consider alternative explanations for the glucose patterns observed
        3. Question whether the patterns identified might have causes other than those initially suggested
        4. Examine if the correlation between meal/activity times and glucose changes might be coincidental
        5. Consider if there are other factors not mentioned in the patient profile that could explain the data

        IMPORTANT: Format your response as a valid JSON object with the following structure:
        {{
            "challenged_assumptions": [
                {{
                    "original_assumption": "Morning hyperglycemia is due to dawn phenomenon",
                    "challenge": "Could be related to insufficient overnight basal insulin rather than dawn phenomenon",
                    "alternative_explanation": "Patient may be taking evening basal insulin too early",
                    "evidence_needed": "Basal insulin timing and overnight glucose trend details"
                }}
            ],
            "alternative_explanations": [
                {{
                    "pattern": "Post-meal hyperglycemia",
                    "original_explanation": "Insufficient insulin-to-carb ratio",
                    "alternative_explanations": [
                        {{
                            "explanation": "Delayed insulin administration relative to meal start",
                            "plausibility": "high",
                            "supporting_evidence": "Sharp initial rise in glucose before eventual decrease"
                        }},
                        {{
                            "explanation": "Underestimated carbohydrate content in meals",
                            "plausibility": "medium",
                            "supporting_evidence": "Inconsistent meal-to-meal glucose response"
                        }}
                    ]
                }}
            ],
            "correlation_challenges": [
                {{
                    "claimed_correlation": "Exercise at 17:00 causes glucose drop",
                    "challenge": "Glucose may be dropping due to delayed lunch insulin action rather than exercise",
                    "test_to_confirm": "Compare days with and without exercise at this time"
                }}
            ],
            "missing_factors": [
                {{
                    "factor": "Stress levels",
                    "potential_impact": "Increased cortisol from stress can raise blood glucose",
                    "relevance": "Could explain seemingly random glucose elevations"
                }},
                {{
                    "factor": "Menstrual cycle (if applicable)",
                    "potential_impact": "Hormonal fluctuations can affect insulin sensitivity",
                    "relevance": "Could explain cyclical patterns in glucose control"
                }}
            ]
        }}

        Be constructively critical of the previous analysis, looking for potential flaws in reasoning or alternative interpretations.
        Respond with valid JSON only. Do not include any explanation or extra text.
        """


    def create_maieutic_prompt(self, previous_responses):
        """Create the maieutic (synthesis) prompt based on all previous responses with JSON output."""
        return f"""
        You are a diabetes care specialist using the CRIT (Critical Inquisitive Template) framework to analyze CGM data.

        Patient Information:
        {json.dumps(self.patient_profile, indent=2)}

        CGM Data Summary:
        {self.statistics}

        Previous analysis:
        {previous_responses}

        This is the Maieutic (Synthesis) phase of the CRIT framework. Your task is to:
        1. Synthesize all the insights from the previous phases of analysis
        2. Develop a comprehensive assessment of the patient's glucose management challenges
        3. Identify the most likely explanations for the observed patterns, acknowledging any uncertainty
        4. Formulate specific, actionable recommendations for improving glucose control
        5. Prioritize interventions based on the severity and clarity of the identified issues

        IMPORTANT: Format your response as a valid JSON object with the following structure:
        {{
            "comprehensive_assessment": {{
                "overall_glucose_control": "suboptimal",
                "primary_challenges": [
                    "Post-meal hyperglycemia",
                    "Nocturnal hypoglycemia",
                    "High glucose variability"
                ],
                "risk_assessment": {{
                    "hypoglycemia_risk": "moderate",
                    "hyperglycemia_risk": "high",
                    "long_term_complications_risk": "elevated due to sustained periods above target range"
                }}
            }},
            "pattern_explanations": [
                {{
                    "pattern": "Morning hyperglycemia",
                    "most_likely_explanation": "Combination of dawn phenomenon and insufficient overnight basal insulin",
                    "confidence_level": "high",
                    "alternative_explanations": [
                        "Late evening snack without insulin coverage",
                        "Somogyi effect following undetected nocturnal hypoglycemia"
                    ],
                    "supporting_evidence": "Consistent rise between 4:00-7:00 AM even on days without nocturnal hypoglycemia"
                }}
            ],
            "recommendations": [
                {{
                    "area": "Insulin therapy",
                    "specific_actions": [
                        {{
                            "action": "Increase breakfast insulin-to-carb ratio from current ratio to 1:X",
                            "rationale": "Consistent post-breakfast hyperglycemia",
                            "priority": "high",
                            "expected_outcome": "Reduced post-breakfast glucose excursions"
                        }}
                    ]
                }},
                {{
                    "area": "Meal planning",
                    "specific_actions": [
                        {{
                            "action": "Redistribute carbohydrate intake with fewer carbs at breakfast",
                            "rationale": "Morning insulin resistance making breakfast spikes difficult to manage",
                            "priority": "medium",
                            "expected_outcome": "Smaller post-breakfast glucose excursions"
                        }}
                    ]
                }},
                {{
                    "area": "Physical activity",
                    "specific_actions": [
                        {{
                            "action": "Adjust pre-exercise insulin reduction from X% to Y%",
                            "rationale": "Current exercise adjustments resulting in hypoglycemia",
                            "priority": "medium",
                            "expected_outcome": "Reduced exercise-related hypoglycemia"
                        }}
                    ]
                }},
                {{
                    "area": "Monitoring",
                    "specific_actions": [
                        {{
                            "action": "Increase overnight CGM alerts for readings below 80 mg/dL",
                            "rationale": "Early detection of dropping glucose to prevent severe nocturnal hypoglycemia",
                            "priority": "high",
                            "expected_outcome": "Reduced frequency and severity of nocturnal hypoglycemia"
                        }}
                    ]
                }}
            ],
            "follow_up": {{
                "recommended_timeframe": "2 weeks",
                "key_metrics_to_evaluate": [
                    "Time in range (70-180 mg/dL)",
                    "Frequency of hypoglycemic events",
                    "Post-meal glucose excursions"
                ],
                "additional_data_needed": [
                    "Detailed food log with carbohydrate counts",
                    "Insulin dosing times relative to meals",
                    "Stress levels and sleep quality"
                ]
            }}
        }}

        Create a coherent synthesis that integrates the definitions, patterns, and critical challenges into a useful clinical assessment.
        Respond with valid JSON only. Do not include any explanation or extra text.
        """

    def run_crit_analysis(self):
        """Run the complete CRIT framework analysis with structured JSON output."""
        self.logger.info("Starting CRIT framework analysis")
        
        # Create an agent for the CRIT analysis
        agent_id = self.llm_agents.create_agent(
            provider="openai",
            system_prompt="You are a diabetes care specialist using the CRIT framework for systematic analysis. Always respond with valid JSON."
        )
        
        # Phase 1: Definition
        self.logger.info("Running Definition phase")
        definition_prompt = self.create_definition_prompt()
        definition_response = self.llm_agents.call_agent(agent_id, definition_prompt)
        
        # Parse JSON response
        try:
            definition_json = json.loads(definition_response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse Definition phase response as JSON")
            # Attempt to clean the response by removing code block markers
            definition_response = definition_response.strip()
            if "```" in definition_response:
                definition_response = definition_response.split("```json")[1].split("```")[0].strip()
            elif "```" in definition_response:
                definition_response = definition_response.split("``````")[0].strip()
            
            try:
                definition_json = json.loads(definition_response)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse cleaned Definition phase response as JSON")
                definition_json = {"error": "Invalid JSON response"}
        
        # Phase 2: Inductive Reasoning
        self.logger.info("Running Inductive Reasoning phase")
        inductive_prompt = self.create_inductive_prompt(json.dumps(definition_json, indent=2))
        inductive_response = self.llm_agents.call_agent(agent_id, inductive_prompt)
        
        # Parse JSON response
        try:
            inductive_json = json.loads(inductive_response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse Inductive phase response as JSON")
            # Attempt to clean the response
            inductive_response = inductive_response.strip()
            if "```" in inductive_response:
                inductive_response = inductive_response.split("```json")[1].split("```")[0].strip()
            elif "```" in inductive_response:
                inductive_response = inductive_response.split("``````")[0].strip()
            
            try:
                inductive_json = json.loads(inductive_response)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse cleaned Inductive phase response as JSON")
                inductive_json = {"error": "Invalid JSON response"}
        
        # Combine responses for context in next phase
        combined_json = {
            "definition_phase": definition_json,
            "inductive_phase": inductive_json
        }
        
        # Phase 3: Elenchus (Critical Challenge)
        self.logger.info("Running Elenchus phase")
        elenchus_prompt = self.create_elenchus_prompt(json.dumps(combined_json, indent=2))
        elenchus_response = self.llm_agents.call_agent(agent_id, elenchus_prompt)
        
        # Parse JSON response
        try:
            elenchus_json = json.loads(elenchus_response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse Elenchus phase response as JSON")
            # Attempt to clean the response
            elenchus_response = elenchus_response.strip()
            if "```" in elenchus_response:
                elenchus_response = elenchus_response.split("```json")[1].split("```")[0].strip()
            elif "```" in elenchus_response:
                elenchus_response = elenchus_response.split("``````")[0].strip()
            
            try:
                elenchus_json = json.loads(elenchus_response)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse cleaned Elenchus phase response as JSON")
                elenchus_json = {"error": "Invalid JSON response"}
        
        # Update combined responses
        combined_json["elenchus_phase"] = elenchus_json
        
        # Phase 4: Maieutic (Synthesis)
        self.logger.info("Running Maieutic phase")
        maieutic_prompt = self.create_maieutic_prompt(json.dumps(combined_json, indent=2))
        maieutic_response = self.llm_agents.call_agent(agent_id, maieutic_prompt)
        
        # Parse JSON response
        try:
            maieutic_json = json.loads(maieutic_response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse Maieutic phase response as JSON")
            # Attempt to clean the response
            maieutic_response = maieutic_response.strip()
            if "```" in maieutic_response:
                maieutic_response = maieutic_response.split("```json")[1].split("```")[0].strip()
            elif "```" in maieutic_response:
                maieutic_response = maieutic_response.split("``````")[0].strip()
            
            try:
                maieutic_json = json.loads(maieutic_response)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse cleaned Maieutic phase response as JSON")
                maieutic_json = {"error": "Invalid JSON response"}
        
        # Final combined analysis as JSON
        final_analysis_json = {
            "patient_name": self.patient_profile['name'],
            "analysis_date": str(pd.Timestamp.now()),
            "crit_framework": {
                "definition_phase": definition_json,
                "inductive_phase": inductive_json,
                "elenchus_phase": elenchus_json,
                "maieutic_phase": maieutic_json
            }
        }
        
        # Save the complete analysis
        self.save_results(final_analysis_json)
        
        return final_analysis_json

    def save_results(self, final_analysis_json):
        """Save the CRIT analysis to the output directory."""
        self.logger.info(f"Saving CRIT analysis results to {self.output_dir}")
        
        # Save the full analysis as JSON
        json_path = Path(self.output_dir).joinpath('crit_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(final_analysis_json, f, indent=2)
        
        # Also save a human-readable text version
        text_analysis = self._generate_text_from_json(final_analysis_json)
        text_path = Path(self.output_dir).joinpath('crit_analysis.txt')
        with open(text_path, 'w') as f:
            f.write(text_analysis)
        
        # Save a JSON with all data including patient profile
        results = {
            "patient_profile": self.patient_profile,
            "statistics": self.statistics,
            "crit_analysis": final_analysis_json,
            "analysis_date": str(pd.Timestamp.now())
        }
        
        full_json_path = Path(self.output_dir).joinpath('crit_results.json')
        with open(full_json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"CRIT analysis results saved to {self.output_dir}")

    def _generate_text_from_json(self, analysis_json):
        """Generate a human-readable text version of the JSON analysis."""
        text = f"# CRIT Framework Analysis for {analysis_json['patient_name']}\n\n"
        text += f"Analysis Date: {analysis_json['analysis_date']}\n\n"
        
        # Definition Phase
        text += "## Phase 1: Definition\n\n"
        def_phase = analysis_json['crit_framework']['definition_phase']
        text += f"Normal blood glucose range: {def_phase['normal_range']['lower_bound']}-{def_phase['normal_range']['upper_bound']} mg/dL\n\n"
        
        text += "### Hyperglycemia\n"
        text += f"{def_phase['hyperglycemia']['definition']}\n"
        text += "Severity levels:\n"
        for level in def_phase['hyperglycemia']['severity_levels']:
            text += f"- {level['level']}: {level['range']}\n"
        text += "\n"
        
        text += "### Hypoglycemia\n"
        text += f"{def_phase['hypoglycemia']['definition']}\n"
        text += "Severity levels:\n"
        for level in def_phase['hypoglycemia']['severity_levels']:
            text += f"- {level['level']}: {level['range']}\n"
        text += "\n"
        
        text += "### Observed Patterns\n"
        for pattern in def_phase['observed_patterns']:
            text += f"- {pattern['pattern_name']}: {pattern['description']} (Frequency: {pattern['frequency']})\n"
        text += "\n"
        
        text += "### Term Definitions\n"
        for term in def_phase['term_definitions']:
            text += f"- {term['term']}: {term['definition']}\n"
        text += "\n"
        
        # Inductive Reasoning Phase
        text += "## Phase 2: Inductive Reasoning\n\n"
        ind_phase = analysis_json['crit_framework']['inductive_phase']
        
        text += "### General Patterns\n"
        for pattern in ind_phase['general_patterns']:
            text += f"- **{pattern['pattern']}**: {pattern['evidence']}\n"
            text += f"  Significance: {pattern['significance']}\n"
        text += "\n"
        
        text += "### Management Insights\n"
        for insight in ind_phase['management_insights']:
            text += f"- **{insight['insight']}** (Confidence: {insight['confidence_level']})\n"
            text += f"  Supporting Evidence: {insight['supporting_evidence']}\n"
        text += "\n"
        
        text += "### Meal Correlations\n"
        for meal in ind_phase['meal_correlations']:
            text += f"- **Meal at {meal['meal_time']}**: {meal['observed_pattern']}\n"
            text += f"  Potential Cause: {meal['potential_cause']}\n"
            text += f"  Consistency: {meal['consistency']}\n"
        text += "\n"
        
        text += "### Activity Correlations\n"
        for activity in ind_phase['activity_correlations']:
            text += f"- **Activity at {activity['activity_time']}**: {activity['observed_pattern']}\n"
            text += f"  Recommendation: {activity['recommendation']}\n"
            text += f"  Consistency: {activity['consistency']}\n"
        text += "\n"
        
        # Elenchus Phase
        text += "## Phase 3: Elenchus (Critical Challenge)\n\n"
        ele_phase = analysis_json['crit_framework']['elenchus_phase']
        
        text += "### Challenged Assumptions\n"
        for challenge in ele_phase['challenged_assumptions']:
            text += f"- **Original Assumption**: {challenge['original_assumption']}\n"
            text += f"  Challenge: {challenge['challenge']}\n"
            text += f"  Alternative Explanation: {challenge['alternative_explanation']}\n"
            text += f"  Evidence Needed: {challenge['evidence_needed']}\n"
        text += "\n"
        
        text += "### Alternative Explanations\n"
        for alt in ele_phase['alternative_explanations']:
            text += f"- **Pattern**: {alt['pattern']}\n"
            text += f"  Original Explanation: {alt['original_explanation']}\n"
            text += "  Alternative Explanations:\n"
            for exp in alt['alternative_explanations']:
                text += f"  - {exp['explanation']} (Plausibility: {exp['plausibility']})\n"
                text += f"    Supporting Evidence: {exp['supporting_evidence']}\n"
        text += "\n"
        
        text += "### Correlation Challenges\n"
        for corr in ele_phase['correlation_challenges']:
            text += f"- **Claimed Correlation**: {corr['claimed_correlation']}\n"
            text += f"  Challenge: {corr['challenge']}\n"
            text += f"  Test to Confirm: {corr['test_to_confirm']}\n"
        text += "\n"
        
        text += "### Missing Factors\n"
        for factor in ele_phase['missing_factors']:
            text += f"- **{factor['factor']}**\n"
            text += f"  Potential Impact: {factor['potential_impact']}\n"
            text += f"  Relevance: {factor['relevance']}\n"
        text += "\n"
        
        # Maieutic Phase
        text += "## Phase 4: Maieutic (Synthesis)\n\n"
        mai_phase = analysis_json['crit_framework']['maieutic_phase']
        
        text += "### Comprehensive Assessment\n"
        assess = mai_phase['comprehensive_assessment']
        text += f"Overall Glucose Control: **{assess['overall_glucose_control']}**\n\n"
        
        text += "Primary Challenges:\n"
        for challenge in assess['primary_challenges']:
            text += f"- {challenge}\n"
        text += "\n"
        
        text += "Risk Assessment:\n"
        text += f"- Hypoglycemia Risk: {assess['risk_assessment']['hypoglycemia_risk']}\n"
        text += f"- Hyperglycemia Risk: {assess['risk_assessment']['hyperglycemia_risk']}\n"
        text += f"- Long-term Complications Risk: {assess['risk_assessment']['long_term_complications_risk']}\n"
        text += "\n"
        
        text += "### Pattern Explanations\n"
        for pattern in mai_phase['pattern_explanations']:
            text += f"- **{pattern['pattern']}** (Confidence: {pattern['confidence_level']})\n"
            text += f"  Most Likely Explanation: {pattern['most_likely_explanation']}\n"
            text += "  Alternative Explanations:\n"
            for alt in pattern['alternative_explanations']:
                text += f"  - {alt}\n"
            text += f"  Supporting Evidence: {pattern['supporting_evidence']}\n"
        text += "\n"
        
        text += "### Recommendations\n"
        for area in mai_phase['recommendations']:
            text += f"#### {area['area']}\n"
            for action in area['specific_actions']:
                text += f"- **{action['action']}** (Priority: {action['priority']})\n"
                text += f"  Rationale: {action['rationale']}\n"
                text += f"  Expected Outcome: {action['expected_outcome']}\n"
            text += "\n"
        
        text += "### Follow-up Plan\n"
        follow = mai_phase['follow_up']
        text += f"Recommended Timeframe: **{follow['recommended_timeframe']}**\n\n"
        
        text += "Key Metrics to Evaluate:\n"
        for metric in follow['key_metrics_to_evaluate']:
            text += f"- {metric}\n"
        text += "\n"
        
        text += "Additional Data Needed:\n"
        for data in follow['additional_data_needed']:
            text += f"- {data}\n"
        return text
        
if __name__ == "__main__":
    # Initialize CRIT Framework
    crit = CritFramework()
    final_analysis = crit.run_crit_analysis()
    print("\nFINAL CRIT ANALYSIS:")
    print(final_analysis)
