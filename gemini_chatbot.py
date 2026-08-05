import os
import dotenv
from google import genai
from google.api_core import retry


class ChatBot(object):
    def __init__(self,
                 model: str = "gemini-3.1-pro-preview") -> None:
        """
        Define model and reset chat history
        """
        self.create_genai_client()
        self.gen_config = genai.types.GenerateContentConfig(temperature=0.6,
                                                            top_p=0.95)
        self.model = model
        
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
    
    def send(self, 
             prompt: str) -> str:
        """
        Submit prompt, return response
        """
        response = self.client.models.generate_content(model=self.model, 
                                                       config=self.gen_config, 
                                                       contents=prompt)
        return response.text
