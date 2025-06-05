import os
import sys
import subprocess
from dotenv import load_dotenv

def main():
    # Obtener modo desde argumento
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "flan"
    print(f"Modo seleccionado: {mode}")

    if mode == "gpt4o":
        os.environ["BOT_BACKEND"] = "gpt4o"
        
        # Cargar credenciales desde archivo .env
        if not os.path.exists("credentials.env"):
            print("ERROR: No se encontró el archivo credentials.env")
            sys.exit(1)

        load_dotenv("credentials.env")

        print("Variables de entorno para GPT-4o cargadas desde credentials.env.")

    else:
        os.environ["BOT_BACKEND"] = "flan"
        os.environ["AZURE_OPENAI_ENDPOINT"] = ""
        os.environ["AZURE_OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = ""
        print("Startup en modo FLAN local: variables limpiadas.")

    # Ejecutar app.py con el mismo interprete que ejecuta este script
    subprocess.run([sys.executable, "app.py"])

if __name__ == "__main__":
    main()
