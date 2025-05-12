from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Flask app
app = Flask(__name__)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.model = os.getenv("OPENAI_MODEL")

@app.route('/webhook', methods=['POST'])
def webhook():
    # Get the message from the user
    incoming_msg = request.values.get('Body', '').lower()
    
    # Create Twilio response object
    resp = MessagingResponse()
    
    try:
        # Call OpenAI API
        completion = openai.ChatCompletion.create(
            model=openai.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": incoming_msg}
            ]
        )
        
        # Get the response text
        ai_response = completion.choices[0].message.content
        
        # Send the response back to the user
        resp.message(ai_response)
    except Exception as e:
        # Handle errors
        resp.message(f"Sorry, I encountered an error: {str(e)}")
    
    return str(resp)

if __name__ == '__main__':
    app.run(debug=True)
