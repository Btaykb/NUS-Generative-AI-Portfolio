import sqlite3
import openai
from config import OPENAI_API_KEY, client


class SQLAgent:

    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        self.db_path = 'events.db'

    def get_schema(self):
        return """ 
        Table: events 
        Columns: 
        - id (INTEGER PRIMARY KEY), 
        - name (TEXT), 
        - type (TEXT),
        - description (TEXT), 
        - location (TEXT), 
        - date (TEXT)
        """

    def generate_sql(self, question):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"""You are a SQL expert. Use this schema:\n{self.get_schema()} 
                Return ONLY the SQL query without any explanation or markdown formatting."""},
                {"role": "user", "content": f"Generate SQL for: {question}"}
            ]
        )
        sql = response.choices[0].message.content.strip()

        # Remove any markdown code block syntax
        sql = sql.replace('```sql', '').replace(
            '```SQL', '').replace('```', '')

        # Remove any explanatory text before or after the SQL
        sql_lines = [line.strip() for line in sql.split('\n') if line.strip()]
        sql = ' '.join(sql_lines)

        return sql

    def validate_sql(self, sql):
        # Basic safety checks
        sql_lower = sql.lower()
        if any(word in sql_lower for word in ['drop', 'delete', 'update', 'insert']):
            raise ValueError("Only SELECT queries are allowed")
        return sql

    def execute_query(self, sql):
        sql = self.validate_sql(sql)
        conn = sqlite3.connect('company.db')
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            conn.close()

    def format_results(self, results):
        if not isinstance(results, list):
            return str(results)
        if not results:
            return "No results found"

        # If it's a single column result
        if len(results[0]) == 1:
            return "\n".join([str(row[0]) for row in results])

        # For multiple columns, try to format as a table
        # Get column names from the first result
        if isinstance(results[0], tuple):
            # Format each row with proper spacing
            formatted_rows = []
            for row in results:
                row_items = []
                for item in row:
                    if isinstance(item, float):
                        row_items.append(f"${item:,.2f}" if "salary" in str(
                            row) or "budget" in str(row) else f"{item:.2f}")
                    else:
                        row_items.append(str(item))
                formatted_rows.append("\t".join(row_items))
            return "\n".join(formatted_rows)

        return "\n".join([str(row) for row in results])

    def query_agent(self, question):
        try:
            # Generate SQL
            sql = self.generate_sql(question)
            print(f"Generated SQL: {sql}\n")

            # Execute and format results
            results = self.execute_query(sql)
            return self.format_results(results)
        except Exception as e:
            return f"Error: {str(e)}"

    def run_query(self, question):
        try:
            sql = self.generate_sql(question)
            sql = self.validate_sql(sql)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()

            return results if results else "No events found."
        except Exception as e:
            return f"SQL Error: {str(e)}"
