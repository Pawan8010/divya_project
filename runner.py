import subprocess
import sys

def run_model(user_input):
    try:
        result = subprocess.run(
            [sys.executable, "LLM_Project.py", user_input],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return result.stdout.strip()

    except Exception as e:
        return f"Exception: {str(e)}"