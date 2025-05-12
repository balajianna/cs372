from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

from config import (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_MODEL, ANTHROPIC_MODEL, GEMINI_MODEL)

from logger import setup_logger
logger = setup_logger()

class LLMAgents:
    def __init__(self):
        self.logger = logger

        self.logger.info("Initializing LLMAgents")
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
        self.gemini_model = genai.GenerativeModel(GEMINI_MODEL) if GEMINI_API_KEY else None
        self.logger.info("LLMAgents initialized successfully")
        self.agents = {}
        
    def create_agent(self, provider="openai", system_prompt=None):
        agent_id = f"{provider}_{len(self.agents)}"
        self.logger.info(f"Creating agent with ID: {agent_id}, provider: {provider}")
        self.agents[agent_id] = {"provider": provider, "system_prompt": system_prompt}
        return agent_id
    
    def call_agent(self, agent_id, prompt):
        agent_config = self.agents[agent_id]
        provider = agent_config["provider"]
        system_prompt = agent_config["system_prompt"]
        self.logger.info(f"Calling agent with ID: {agent_id}, provider: {provider}")

        return self._get_openai_response(system_prompt, prompt) if provider == "openai" else \
               self._get_anthropic_response(system_prompt, prompt) if provider == "anthropic" else \
               self._get_gemini_response(system_prompt, prompt) if provider == "gemini" else \
               None
    
    def _get_openai_response(self, system_prompt=None, prompt=None):
        messages = []
        messages.append({"role": "system", "content": system_prompt}) if system_prompt else None
        messages.append({"role": "user", "content": prompt}) if prompt else None
        response = self.openai_client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.3, max_tokens=1000)
        return response.choices[0].message.content
        
    def _get_anthropic_response(self, system_prompt=None, prompt=None):
        response = self.anthropic_client.messages.create(
                    model=ANTHROPIC_MODEL, system=system_prompt if system_prompt else "",
                    messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1000)
        return response.content[0].text
    
    def _get_gemini_response(self, system_prompt=None, prompt=None):
        content = [prompt]
        content = [system_prompt, prompt] if system_prompt else content            
        response = self.gemini_model.generate_content(content, generation_config={"temperature": 0.3})
        return response.text

if __name__ == "__main__":
    llm_agents = LLMAgents()
    agent_id = llm_agents.create_agent(provider="openai", system_prompt="You are a helpful assistant.")
    response = llm_agents.call_agent(agent_id, "What is the capital of France?")
    print(response)
    # Expected output: "The capital of France is Paris."

