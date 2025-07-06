
## 🛠️ Usage

To run the application, if you want to use gpt4o-mini you must first create a file named `credentials.env` in the data directory. This file should contain your Azure OpenAI credentials in the following format:

```env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
```

> ⚠️ **Important:** Do not include quotes (`"`) around the values. Each line must follow the format `KEY=VALUE` with no spaces around the `=` sign.

---

## ▶️ Running the Application

Use the following command to start the app in **FLAN** mode (default):

```bash
python run.py flan
```

To use the **GPT4o-mini** backend (requires `credentials.env` to be present and correctly filled):

```bash
python run.py gpt4o-mini
```

If no argument is provided, the script defaults to `flan`.

---

## ⚠️ Important Notes

- **FLAN Mode**: The application does not work properly in the FLAN environment. It is recommended to run it locally.

- **Local visualization**: To correctly view the frontend (`index.html`), it is recommended to use the **"Live Server" extension in VS Code** and click on **"Go Live"** to launch the local server.

- **Spotify registration (within the app)**:  
  To test the Spotify sign-up functionality, your account must first be registered in the app’s development environment.  
  Please send an email with your **full name** and the **email associated with your Spotify account** to any of the following addresses:

  - lluquematerazzi@udesa.edu.ar  
  - cguerrero@udesa.edu.ar  
  - tkaucher@udesa.edu.ar  
  - bdrexler@udesa.edu.ar