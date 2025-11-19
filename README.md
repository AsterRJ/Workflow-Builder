# Workflow-Builder



# Workflow Builder

**Convert informal text instructions into structured, operational workflows.**

This project combines:

* A **Python backend + HTML UI** (`server.py` + `static/index.html`)
* A local **Ollama** LLM server (tested with `llama3`)
* An **n8n** workflow (imported from JSON)
* A lightweight **ontology + skills layer** (`config/reference_ontology.json`, `config/skills.json`)

The system takes free-text (specs, WPS text, SOPs, task descriptions) and converts it into a structured JSON workflow containing tasks, sub-tasks, skills, and ontology labels.

---

## 1. Repository Structure

```text
.
├── config/
│   ├── reference_ontology.json
│   └── skills.json
├── knowledge/
│   ├── db_manager.py
│   ├── ont_manager.py
│   └── __pycache__/
├── n8n/
│   └── Llama Builder - Combined (Main + Sub-workflow).json
├── outputs/
│   ├── state_*.json
│   └── workflow_*.json
├── requirements.txt
├── server.py
├── static/
│   └── index.html
└── README.md
```

`outputs/` is git-ignored and contains runtime artefacts created when workflows are generated.

---

## 2. What the System Does

1. You paste free-text into the UI.
2. The backend sends this to an **n8n webhook**.
3. n8n orchestrates:

   * LLM calls via **Ollama**
   * Task + sub-task extraction
   * Skill assignment
   * Ontology mapping
4. The backend stores the resulting structured workflow in `outputs/`.

Example output:

```json
{
  "workflow": [
    {
      "task": "Check incoming files",
      "subtasks": [
        {"description": "Verify existence", "skill": "inspection"},
        {"description": "Flag invalid entries", "skill": "quality-control"}
      ]
    }
  ]
}
```

---

## 3. Prerequisites

* Python **3.11+**
* Docker
* Ollama
  Install from: [https://ollama.com/download](https://ollama.com/download)

Pull the model:

```bash
ollama pull llama3
```

---

## 4. Step-by-Step: Start Everything

### 4.1 Start Ollama (LLM Server)

1. Open a terminal.

2. Start the Ollama server:

   ```bash
   ollama serve
   ```

3. (Optional) In a second terminal, verify the model runs:

   ```bash
   ollama run llama3
   ```

4. Visit the local Ollama web UI (if enabled) at:

   * [http://localhost:11434](http://localhost:11434)

By default, Ollama exposes an HTTP API on `http://localhost:11434` which n8n uses to call the model.

---

### 4.2 Start n8n via Docker

1. Open another terminal.

2. Run n8n with Docker:

   ```bash
   docker run -it --rm \
     --name n8n \
     --network host \
     -e GENERIC_TIMEZONE="Europe/London" \
     -e TZ="Europe/London" \
     -e N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true \
     -e N8N_RUNNERS_ENABLED=true \
     -v n8n_data:/home/node/.n8n \
     docker.n8n.io/n8nio/n8n
   ```

3. Open the n8n UI in your browser:

   * [http://localhost:5678](http://localhost:5678)

4. Create an account if prompted.

#### 4.2.1 Import the Workflow into n8n

1. In the n8n UI, click **Import from File**.

2. Select:

   ```text
   n8n/Llama Builder - Combined (Main + Sub-workflow).json
   ```

3. Save and **activate** the workflow.

#### 4.2.2 Test the n8n Webhook

1. In a new terminal, send a test payload:

   ```bash
   curl -X POST "http://localhost:5678/webhook-test/wf_builder_agent" \
     -H "Content-Type: application/json" \
     -d '{"spec_text": "Operator checks files."}'
   ```

2. If configured correctly, you will receive structured JSON in response.

---

### 4.3 Install Python Dependencies

1. From the repository root, create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

### 4.4 Configure Backend Environment

If your `server.py` uses environment variables, set them before running:

```bash
export N8N_BASE_URL="http://localhost:5678"
export N8N_WEBHOOK_PATH="/webhook-test/wf_builder_agent"
export ONTOLOGY_FILE="config/reference_ontology.json"
export SKILLS_FILE="config/skills.json"
export OUTPUT_DIR="outputs"
```

Alternatively, hard-code these values inside `server.py` while developing.

---

### 4.5 Run the Backend + UI

1. With the virtualenv active, start the backend:

   ```bash
   python server.py
   ```

2. Check the terminal output; by default, it will usually listen on:

   * [http://localhost:5000](http://localhost:5000)

3. Open the UI in your browser:

   * [http://localhost:5000](http://localhost:5000)

4. Paste your free-text instructions into the form and submit.

5. The backend calls n8n, which calls Ollama, and returns a structured workflow.

6. Generated workflows and state files will be saved under:

   ```text
   outputs/workflow_<JOB_ID>.json
   outputs/state_<JOB_ID>.json
   ```

---

## 5. Ontology and Skills Configuration

The following files control how tasks and sub-tasks are annotated:

* `config/reference_ontology.json` – defines the ontology of processes, entities and labels.
* `config/skills.json` – maps sub-tasks to skills or capabilities.

These are loaded and used by:

* `knowledge/ont_manager.py`
* `knowledge/db_manager.py`

To adapt the system to a new domain:

1. Edit `reference_ontology.json` with your domain-specific concepts.
2. Edit `skills.json` with skills relevant to your workflows.
3. Restart `server.py` so changes are picked up.

---

## 6. Troubleshooting

### n8n returns no data

* Check that the workflow is **active** in n8n.
* Confirm the webhook path in n8n matches `N8N_WEBHOOK_PATH` in `server.py`.
* Use the **Executions** view in n8n ([http://localhost:5678](http://localhost:5678)) to inspect runs.

### Backend cannot reach n8n

* Confirm n8n is running at [http://localhost:5678](http://localhost:5678).
* Check for typos in `N8N_BASE_URL`.
* Ensure `--network host` was used in the Docker run command (Linux / WSL).

### Ollama errors

* Confirm `ollama serve` is running.
* Make sure `ollama pull llama3` has completed.
* Check that the n8n HTTP Request node points to `http://localhost:11434`.

---

## 7. Possible Extensions

* Include training the local LLM of choice on your specific datatsets to ensure responses are structured in ways you may expect.
* Add in a RAG setup to create an internal chat assistant based on previous knowledge.
* Add authentication and multi-user support.
* Add tests that call the n8n webhook and validate expected JSON structure.

Definitely will be adding secondary nodes to critique responses from all LLM calls.
---

## 8. Licence

Add your preferred licence information here (for example MIT, Apache 2.0, or proprietary).
