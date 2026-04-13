import openai
from config import OPENAI_API_KEY, client


class RecommenderAgent:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def get_recommendation(self, weather_data, events, target_date):
        """
        Synthesizes weather and event data into a professional recommendation.
        weather_data: dict from WeatherAgent
        events: list of tuples from SQL/EventAgent
        target_date: string
        """
        if not events or isinstance(events, str):
            return f"I couldn't find any specific events scheduled for {target_date}."

        if isinstance(weather_data, str) or "error" in str(weather_data):
            weather_context = "Weather data is currently unavailable."
            cond = "Unknown"
            temp = "Unknown"
        else:
            try:
                cond = weather_data['current']['condition']['text']
                temp = weather_data['current']['temp_c']
                weather_context = f"Weather: {cond}, Temperature: {temp}°C"
            except (KeyError, TypeError):
                weather_context = "Weather data is currently unavailable."
                cond = "Unknown"
                temp = "Unknown"

        event_list = ""
        for e in events:
            event_list += f"- {e[1]} [{e[2]}]: {e[3]} at {e[4]}\n"

        system_prompt = f"""
        You are a Singapore-based Event Concierge. 
        Current Status: {weather_context}.

        REASONING RULES:
        1. TRANSPORT: If raining or >31°C, suggest Grab/Taxi. If clear/cool, suggest walking or MRT.
        2. SAFETY: If an event is 'outdoor' and it is raining, advise caution or suggest indoor alternatives.
        3. LOCAL TONE: Use a professional yet helpful tone suitable for a Singapore context.
        4. SUMMARY: Provide a concise recommendation on which events are best suited for the current conditions.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Date: {target_date}\nAvailable Events:\n{event_list}"}
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"Recommender Error: {str(e)}"
