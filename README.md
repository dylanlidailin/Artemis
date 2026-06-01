# Artemis

Artemis is a chatbot developed using langchain and OpenAI API. It's designed to analyze pdf files and provide AI-generated answers.

## Workflow

1. Open the web app.
2. Upload a PDF.
3. Artemis builds an in-memory retrieval session for that PDF.
4. Ask questions or extract keywords from the uploaded PDF.

Sessions are stored in server memory, so they are cleared when the server restarts or redeploys.

## Setup

Create a virtual environment, install dependencies, and configure your OpenAI API key:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Then edit `.env` and set `OPENAI_API_KEY`.

## Run

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Verify

```powershell
pytest
```

## Notes

- The app answers from the uploaded PDF only.
- The web app test suite lives in `tests/`.