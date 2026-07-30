import os
import dotenv
from google import genai
from google.api_core import retry


class ChatBot(object):
    def __init__(self, initial_prompt, model="gemini-3.5-flash"):
        """
        Define model and reset chat history
        """
        self.create_genai_client()
        self.gen_config = genai.types.GenerateContentConfig(temperature=0.6,
                                                            top_p=0.95)
        self.conversation = self.client.chats.create(model=model,
                                                     config=self.gen_config,
                                                     history=[])
        self.send(initial_prompt)
        
    def create_genai_client(self):
        """
        Set up access to Google Gen AI models.
        """
        if not "GOOGLE_API_KEY" in os.environ:
            dotenv.load_dotenv()
        is_retriable = lambda e: (
            isinstance(e, genai.errors.APIError) and e.code in {429, 503}
        )
        genai.models.Models.generate_content = retry.Retry(
            predicate=is_retriable
        )(genai.models.Models.generate_content)
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
    def send(self, prompt):
        """
        Submit prompt, update chat history, return response
        """
        return self.conversation.send_message(prompt).text
