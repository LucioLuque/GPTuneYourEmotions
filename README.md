
## 🛠️ Usage

To run the application, if you want to use gpt4o-mini you must first create a file named `credentials.env` in the root directory of the project. This file should contain your Azure OpenAI credentials in the following format:

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
python run_app.py flan
```

To use the **GPT-4o** backend (requires `credentials.env` to be present and correctly filled):

```bash
python run_app.py gpt4o-mini
```

If no argument is provided, the script defaults to `flan`.

---
