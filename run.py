import os
import sys
import subprocess
from dotenv import load_dotenv

def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "flan"
    print(f"Modo seleccionado: {mode}")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    
    if mode == "gpt4o-mini":
        os.environ["BOT_BACKEND"] = "gpt4o-mini"
        
        credentials_path = os.path.join(base_dir, "data", "credentials.env")
        if not os.path.exists(credentials_path):
            print("ERROR: No se encontró el archivo credentials.env")
            sys.exit(1)

        load_dotenv(credentials_path)

        print("Variables de entorno para GPT4o-mini cargadas desde credentials.env.")

    else:
        os.environ["BOT_BACKEND"] = "flan"
        os.environ["AZURE_OPENAI_ENDPOINT"] = ""
        os.environ["AZURE_OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = ""
        print("Startup en modo FLAN local: variables limpiadas.")

    # ejecuta main.py
    subprocess.run([sys.executable, "-m", "app.main"], cwd=base_dir)

if __name__ == "__main__":
    main()
