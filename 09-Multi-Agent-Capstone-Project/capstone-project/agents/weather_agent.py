import requests
from config import WEATHER_API_KEY, OPENAI_API_KEY, client
import openai
import json
import time

openai.api_key = OPENAI_API_KEY


class WeatherAgent:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

        self.last_call_time = 0
        self.min_interval = 1.0

        self.functions_metadata = [{
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }]

    def _fetch_from_api(self, location):
        """Internal method to handle the actual HTTP request to WeatherAPI."""
        # Simple Throttle
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        try:
            url = "http://api.weatherapi.com/v1/current.json"
            params = {
                "key": WEATHER_API_KEY,
                "q": location,
                "aqi": "no"
            }

            response = requests.get(url, params=params, timeout=10)
            self.last_call_time = time.time()

            if response.status_code == 400:
                error_data = response.json().get('error', {})
                if error_data.get('code') == 1006:
                    return {"error": f"City '{location}' not found."}
                return {"error": error_data.get('message')}

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def get_weather(self, user_query):
        """Main entry point: Decide if we need weather data and return the formatted answer."""
        try:
            response = client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=[{"role": "user", "content": user_query}],
                functions=self.functions_metadata,
                function_call="auto"
            )

            message = response.choices[0].message

            if message.function_call:
                args = json.loads(message.function_call.arguments)
                weather_data = self._fetch_from_api(args["location"])

                if "error" in weather_data:
                    return f"Weather Error: {weather_data['error']}"

                return (
                    f"Current weather in {weather_data['location']['name']}: "
                    f"{weather_data['current']['temp_c']}°C, "
                    f"{weather_data['current']['condition']['text']}"
                )

            # If no tool was needed, just return the AI's text
            return message.content

        except Exception as e:
            return f"Agent Error: {str(e)}"
