import os
from datetime import datetime, timedelta
from agents.weather_agent import WeatherAgent
from agents.rag_agent import RAGAgent
from agents.sql_agent import SQLAgent
from agents.recommender_agent import RecommenderAgent
from agents.image_agent import ImageAgent
from config import client


class ControllerAgent:
    def __init__(self):
        self.weather_agent = WeatherAgent()
        self.sql_agent = SQLAgent()
        self.image_agent = ImageAgent()
        self.recommender_agent = RecommenderAgent()
        self.doc_folder = "document-upload/"
        self.rag_agent = self._initialize_rag()

    def _initialize_rag(self):
        """Pre-loads the PDF so the AI is ready to answer immediately."""
        if not os.path.exists(self.doc_folder):
            os.makedirs(self.doc_folder)

        pdfs = [f for f in os.listdir(self.doc_folder) if f.endswith('.pdf')]

        if pdfs:
            pdf_path = os.path.join(self.doc_folder, pdfs[0])
            # No need to pass the token here anymore!
            return RAGAgent(pdf_path)

        return None

    def _extract_date(self, query):
        """Extracts a date from the query using LLM. Handles relative dates like 'tomorrow'."""
        try:
            response = client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=[
                    {"role": "system", "content": "Extract the date from the user's query. Handle relative dates: if 'tomorrow' return the date for tomorrow, if 'today' return today. Return in YYYY-MM-DD format. If no date is mentioned, return 'none'."},
                    {"role": "user", "content": query}
                ],
                max_completion_tokens=20
            )
            date_str = response.choices[0].message.content.strip()

            if date_str == 'none':
                return None

            # Try to parse as YYYY-MM-DD
            try:
                return date_str
            except:
                pass

            # Handle relative dates
            today = datetime.now().date()
            query_lower = query.lower()

            if 'tomorrow' in query_lower:
                return (today + timedelta(days=1)).strftime('%Y-%m-%d')
            elif 'today' in query_lower:
                return today.strftime('%Y-%m-%d')

            return date_str if date_str != 'none' else None
        except Exception as e:
            return None

    def route_request(self, user_query):
        """
        Uses an LLM to decide which tool to use.
        """

        intent_prompt = f"""
        Analyze the user's request and categorize it into ONE of these tools:

        - WEATHER: Current weather, forecasts, temperature, conditions. 
          Examples: "What's the weather in Singapore?", "Will it rain tomorrow?"

        - SQL: Queries about events, database records, listings, availability.
          Examples: "List all events on July 15th", "What indoor events are happening?", "Show me upcoming concerts"

        - IMAGE: Generate, create, visualize, draw, or design images.
          Examples: "Generate an image of...", "Create a picture of...", "Show me a cinematic view of..."

        - RESEARCH: Questions about documents, papers, research, analysis, learning.
          Examples: "What does the paper say about...", "Summarize the ImageNet paper", "Explain ReLU from the paper"

        - RECOMMENDER: Requests for recommendations combining multiple factors (weather + events + date).
          Examples: "Recommend activities for tomorrow", "What should I do given the weather?", "Suggest events considering the rain", "I want to go cycling but worried about rain"

        - GENERAL: Chitchat, greetings, general conversation not fitting above categories.

        Request: "{user_query}"
        Return ONLY the category name (one word).
        Category:"""

        try:
            # Classification
            response = client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=[{"role": "user", "content": intent_prompt}],
                max_completion_tokens=10
            )
            category = response.choices[0].message.content.strip().upper()

            # Routing
            if "WEATHER" in category:
                return self.weather_agent.get_weather(user_query)

            elif "SQL" in category:
                return self.sql_agent.run_query(user_query)

            elif "RESEARCH" in category:
                if self.rag_agent:
                    return self.rag_agent.ask(user_query)

            elif "RECOMMENDER" in category:
                weather_info = self.weather_agent.get_weather(user_query)
                event_info = self.sql_agent.run_query(user_query)
                date = self._extract_date(user_query) or "2025-07-15"
                return self.recommender_agent.get_recommendation(weather_info, event_info, date)

            elif "IMAGE" in category:
                return self.image_agent.generate_image(user_query)

            else:
                return "I'm not sure which tool to use"

        except Exception as e:
            return f"Controller Error: {str(e)}"


if __name__ == "__main__":
    controller = ControllerAgent()

    test_queries = [
        # 1. Test the Weather Agent (Live API)
        "What is the current weather in Singapore?",

        # 2. Test the SQL Agent (Structured Data)
        "List all indoor events for July 15th 2025.",

        # 3. Test the Image Agent (Generative / Prompt Engineering)
        "Generate a cinematic 8k image of a futuristic music festival in a park.",

        # 4. Test the RAG Agent (Unstructured PDF Research)
        "According to the ImageNet paper, what is ReLU?",
        "Then what are the benefits of ReLU",

        # 5. Test the Recommender Logic (Synthesized Intelligence)
        "What do you recommend for an activity on July 15 2025?",
    ]

    print(f"{'='*20} SYSTEM INTEGRATION TEST {'='*20}")
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}] User Query: {query}")
        response = controller.route_request(query)
        print(f"Controller Response:\n{response}")
        print("-" * 65)
