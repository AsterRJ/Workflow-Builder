from typing import Any, Dict, Optional
import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import time
import queue

import requests
from flask import Flask, request, jsonify, send_from_directory, Response
import uuid
import threading
from knowledge.db_manager import CONTEXT_DB
from knowledge.ont_manager import Ontology


# --- Global ontology -------------------------------------------------


REFERENCE_ONTOLOGY_PATH = os.getenv(
    "REFERENCE_ONTOLOGY_PATH", "config/reference_ontology.json")
GLOBAL_ONTOLOGY = Ontology.from_path(REFERENCE_ONTOLOGY_PATH)


# --------------------------------------------------------------------------------------
# FLASK SERVER FOR LLAMA BUILDER N8N WORKFLOW - ENHANCED VERSION
# --------------------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="")

# ------------------------- CONFIGURATION SYSTEM ----------------------
CONFIG = {
    "notification_defaults": {
        "default_recipient": "operator@company.com",
        "system_admin": "admin@company.com",
        "data_team": "data-team@company.com"
    },
    "ontology_mappings": {
        "batch_validation": {
            "success_recipient": "data-team@company.com",
            "failure_recipient": "data-team@company.com",
            "critical_failure_recipient": "admin@company.com"
        },
        "file_operations": {
            "success_recipient": "operations@company.com",
            "failure_recipient": "operations@company.com"
        }
    },
    "error_handling_defaults": {
        "continueOnFail": True,
        "retryOnFail": True,
        "maxTries": 3,
        "waitBetweenTries": 5000
    }
}

# ------------------------- JOB AND WORKFLOW STATE ----------------------
JOB_STORE: Dict[str, Dict[str, Any]] = {}
JOB_LOCK = threading.Lock()

N8N_BUILDER_URL = os.getenv(
    "N8N_BUILDER_URL", "http://localhost:5678/webhook/wf_builder_agent")

# ------------------------- WORKFLOW STATE WITH ENHANCED STORAGE ---------------------------
WORKFLOW_STATE = {
    "jobs": {},  # job_id -> job metadata
    "ontology": {},  # job_id -> ontology data
    "rules": {},  # job_id -> rules data
    "tasks": {},  # job_id -> {task_id -> task data}
    "transitions": {},  # job_id -> list of transitions
    "subtasks": {},  # job_id -> {task_id -> list of subtasks}
}
STATE_LOCK = threading.Lock()

# SSE (Server-Sent Events) for live updates
SSE_CLIENTS: Dict[str, queue.Queue] = {}
SSE_LOCK = threading.Lock()

# --------------------------------------------------------------------------------------
# SKILLS REGISTRY
# --------------------------------------------------------------------------------------


@dataclass
class SkillDefinition:
    id: str                     # "messaging.email_notification"
    node_type: str              # "n8n-nodes-base.emailSend"
    description: str            # human readable
    # "messaging" | "etl" | "remote" | "human" | ...
    category: str
    io: Dict[str, Any]          # semantic IO spec: inputs / outputs
    n8n_template: Dict[str, Any]  # base node JSON: parameters, settings, etc.
    error_handling: Dict[str, Any]  # optional: retry/continue defaults


def load_skill_registry() -> Dict[str, SkillDefinition]:
    """
    Load skill definitions from config/skills.json (or SKILLS_PATH env).

    Expected shape:
      {
        "<skill_id>": {
          "type": "n8n-nodes-base.xxx",
          "description": "...",
          "category": "...",
          "io": {...},
          "n8n_template": {...},
          "error_handling": {...}
        },
        ...
      }
    """
    path = os.getenv("SKILLS_PATH", "config/skills.json")

    if not os.path.exists(path):
        # Safe default: empty registry, you can also seed with hard-coded skills here
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        # If file is corrupt, fail safe and log / print
        print(f"[skills] Failed to load skills from {path}: {e}")
        return {}

    registry: Dict[str, SkillDefinition] = {}
    for sid, spec in raw.items():
        if not isinstance(spec, dict):
            continue

        node_type = spec.get("type")
        if not node_type:
            # Skip malformed entry
            print(f"[skills] Skill '{sid}' missing 'type'; skipping")
            continue

        description = spec.get("description", "")
        category = spec.get("category", "generic")
        io_block = spec.get("io", {}) or {}
        template = spec.get("n8n_template", {}) or {}
        error_handling = spec.get("error_handling", {}) or {}

        registry[sid] = SkillDefinition(
            id=sid,
            node_type=node_type,
            description=description,
            category=category,
            io=io_block,
            n8n_template=template,
            error_handling=error_handling,
        )

    return registry


SKILL_REGISTRY: Dict[str, SkillDefinition] = load_skill_registry()


def resolve_skill(skill_id: Optional[str]) -> Optional[SkillDefinition]:
    if not skill_id:
        return None
    return SKILL_REGISTRY.get(skill_id)


# --------------------------------------------------------------------------------------
# DATA MODELS FOR N8N WORKFLOW GENERATION
# --------------------------------------------------------------------------------------

def prune_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively drop keys with value None from a dict."""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            out[k] = prune_none(v)
        elif isinstance(v, list):
            # Recursively prune dicts inside lists
            out[k] = [
                prune_none(x) if isinstance(x, dict) else x
                for x in v
            ]
        else:
            out[k] = v
    return out


@dataclass
class N8NNode:
    """Represents an n8n workflow node"""
    id: str
    name: str
    type: str
    position: List[int]
    parameters: Dict[str, Any]
    credentials: Optional[Dict[str, Any]] = None
    onError: Optional[str] = None
    continueOnFail: Optional[bool] = None
    retryOnFail: Optional[bool] = None
    maxTries: Optional[int] = None
    waitBetweenTries: Optional[int] = None


@dataclass
class N8NConnection:
    """Represents a connection between nodes"""
    source_node: str
    source_output: int
    target_node: str
    target_input: int


@dataclass
class N8NWorkflow:
    """Complete n8n workflow structure"""
    name: str
    nodes: List[N8NNode]
    connections: Dict[str, Any]
    active: bool = False
    settings: Optional[Dict[str, Any]] = None

    def to_json(self):
        workflow = {
            "name": self.name,
            "nodes": [prune_none(asdict(node)) for node in self.nodes],
            "connections": self.connections,
            "active": self.active,
        }
        if self.settings:
            workflow["settings"] = self.settings
        return workflow


# --------------------------------------------------------------------------------------
# COMPREHENSIVE SKILL TO N8N NODE TYPE MAPPING WITH ERROR HANDLING
# --------------------------------------------------------------------------------------
SKILL_TO_N8N_TYPE = {
    "filesystem.check_expected_files": {
        "type": "n8n-nodes-base.executeCommand",
        "parameters": {
            "command": "python3",
            "arguments": [
                "/scripts/check_files.py",
                "{{$json.input_folder || '/data/input'}}",
                "{{$json.expected_patterns || '[]'}}"
            ]
        },
        "error_handling": {
            "continueOnFail": True,
            "retryOnFail": True,
            "maxTries": 2,
            "waitBetweenTries": 3000
        },
        "outputs": {
            "success": {
                "state": "inputs_OK",
                "next_node": "continue"
            },
            "failure": {
                "state": "inputs_FAILED",
                "next_node": "email_notification",
                "notification_config": {
                    "recipient": "{{CONFIG.ontology_mappings.file_operations.failure_recipient}}",
                    "subject": "File Check Failed - Missing Expected Files",
                    "template": "file_check_failure"
                }
            }
        }
    },

    "etl.flag_invalid_files": {
        "type": "n8n-nodes-base.code",
        "parameters": {
            "mode": "runOnceForAllItems",
            "jsCode": """
const items = $input.all();
const files = items[0].json.file_list || [];
const batch_id = items[0].json.batch_id;

const flagged = [];
const issues = [];

for (const file of files) {
    if (!file.exists) {
        flagged.push(file);
        issues.push({file: file.name, reason: 'File does not exist'});
    } else if (file.size === 0) {
        flagged.push(file);
        issues.push({file: file.name, reason: 'Zero-byte file'});
    } else if (file.name && !file.name.match(/^[a-zA-Z0-9_\\-\\.]+$/)) {
        flagged.push(file);
        issues.push({file: file.name, reason: 'Invalid filename'});
    }
}

return [{
    json: {
        batch_id: batch_id,
        files_flagged: flagged.length > 0,
        flagged_files: flagged,
        flagged_count: flagged.length,
        issues: issues,
        timestamp: new Date().toISOString()
    }
}];
"""
        },
        "error_handling": {
            "continueOnFail": True
        },
        "outputs": {
            "success": {
                "state": "files_flagged",
                "conditional": "{{$json.files_flagged}}",
                "routes": {
                    "true": "etl.basic_cleaning",
                    "false": "etl.structural_validation"
                }
            },
            "failure": {
                "state": "flagging_FAILED",
                "next_node": "email_notification"
            }
        }
    },

    "etl.basic_cleaning": {
        "type": "n8n-nodes-base.executeCommand",
        "parameters": {
            "command": "python3",
            "arguments": [
                "/scripts/basic_cleaning.py",
                "--files={{$json.flagged_files}}",
                "--batch-id={{$json.batch_id}}"
            ]
        },
        "error_handling": {
            "continueOnFail": True,
            "retryOnFail": True,
            "maxTries": 3,
            "waitBetweenTries": 5000
        },
        "outputs": {
            "success": {
                "state": "cleaned_files",
                "next_node": "etl.structural_validation"
            },
            "failure": {
                "state": "cleaning_FAILED",
                "next_node": "email_notification",
                "notification_config": {
                    "recipient": "{{CONFIG.ontology_mappings.batch_validation.failure_recipient}}",
                    "subject": "ETL Cleaning Failed - Batch {{$json.batch_id}}",
                    "template": "cleaning_failure"
                }
            }
        }
    },

    "etl.structural_validation": {
        "type": "n8n-nodes-base.httpRequest",
        "parameters": {
            "method": "POST",
            "url": "{{$env.ETL_SERVICE_URL}}/structural_validate",
            "authentication": "genericCredentialType",
            "genericAuthType": "httpBasicAuth",
            "sendBody": True,
            "bodyParameters": {
                "parameters": [
                    {
                        "name": "files",
                        "value": "={{$json.cleaned_file_list}}"
                    },
                    {
                        "name": "schema",
                        "value": "={{$json.schema_definition}}"
                    },
                    {
                        "name": "batch_id",
                        "value": "={{$json.batch_id}}"
                    }
                ]
            },
            "options": {
                "timeout": 300000,
                "response": {
                    "response": {
                        "responseFormat": "json"
                    }
                }
            }
        },
        "error_handling": {
            "continueOnFail": True,
            "retryOnFail": True,
            "maxTries": 3,
            "waitBetweenTries": 10000
        },
        "outputs": {
            "success": {
                "conditional": "{{$json.validation_passed}}",
                "routes": {
                    "true": "batch_state.mark_validated",
                    "false": "batch_state.mark_requires_investigation"
                }
            },
            "failure": {
                "state": "validation_FAILED",
                "next_node": "email_notification",
                "notification_config": {
                    "recipient": "{{CONFIG.ontology_mappings.batch_validation.critical_failure_recipient}}",
                    "subject": "CRITICAL: Validation Service Failed - Batch {{$json.batch_id}}",
                    "template": "validation_service_failure",
                    "priority": "high"
                }
            }
        }
    },

    "batch_state.mark_validated": {
        "type": "n8n-nodes-base.postgres",
        "parameters": {
            "operation": "executeQuery",
            "query": """
UPDATE batch_log 
SET 
    status = 'validated',
    validated_at = NOW(),
    validated_by = '{{$json.validated_by || 'system'}}',
    validation_notes = '{{$json.validation_notes || ''}}'
WHERE batch_id = '{{$json.batch_id}}';
""",
            "options": {}
        },
        "error_handling": {
            "continueOnFail": False,
            "retryOnFail": True,
            "maxTries": 5,
            "waitBetweenTries": 2000
        },
        "outputs": {
            "success": {
                "state": "validated",
                "next_node": "email_notification",
                "notification_config": {
                    "recipient": "{{CONFIG.ontology_mappings.batch_validation.success_recipient}}",
                    "subject": "Batch Validated Successfully - {{$json.batch_id}}",
                    "template": "batch_validated"
                }
            },
            "failure": {
                "state": "db_update_FAILED",
                "next_node": "email_notification",
                "notification_config": {
                    "recipient": "{{CONFIG.notification_defaults.system_admin}}",
                    "subject": "CRITICAL: Database Update Failed - {{$json.batch_id}}",
                    "template": "db_failure",
                    "priority": "critical"
                }
            }
        }
    },

    "batch_state.mark_requires_investigation": {
        "type": "n8n-nodes-base.postgres",
        "parameters": {
            "operation": "executeQuery",
            "query": """
UPDATE batch_log 
SET 
    status = 'requires_investigation',
    flagged_at = NOW(),
    anomaly_count = {{$json.anomaly_count || 0}},
    anomaly_details = '{{$json.quarantine_files || '[]'}}'
WHERE batch_id = '{{$json.batch_id}}';
""",
            "options": {}
        },
        "error_handling": {
            "continueOnFail": False,
            "retryOnFail": True,
            "maxTries": 5,
            "waitBetweenTries": 2000
        },
        "outputs": {
            "success": {
                "state": "requires_investigation",
                "next_node": "human.manual_confirmation"
            },
            "failure": {
                "state": "db_update_FAILED",
                "next_node": "email_notification",
                "notification_config": {
                    "recipient": "{{CONFIG.notification_defaults.system_admin}}",
                    "subject": "CRITICAL: Database Update Failed - {{$json.batch_id}}",
                    "template": "db_failure",
                    "priority": "critical"
                }
            }
        }
    },

    "messaging.email_notification": {
        "type": "n8n-nodes-base.emailSend",
        "parameters": {
            "fromEmail": "{{$env.SMTP_FROM_EMAIL || 'workflow@company.com'}}",
            "toEmail": "={{$json.recipient_email || CONFIG.notification_defaults.default_recipient}}",
            "subject": "={{$json.subject || 'Workflow Notification'}}",
            "emailType": "html",
            "message": "={{$json.email_body || $json.body}}",
            "options": {
                "attachments": "={{$json.attachments || ''}}",
                "ccEmail": "={{$json.cc_email || ''}}",
                "bccEmail": "={{$json.bcc_email || ''}}"
            }
        },
        "error_handling": {
            "continueOnFail": True,
            "retryOnFail": True,
            "maxTries": 3,
            "waitBetweenTries": 5000
        },
        "outputs": {
            "success": {
                "state": "email_sent",
                "next_node": "continue"
            },
            "failure": {
                "state": "email_FAILED",
                "next_node": "log_to_file"
            }
        }
    },

    "human.manual_confirmation": {
        "type": "n8n-nodes-base.wait",
        "parameters": {
            "resume": "webhook",
            "options": {
                "webhook": {
                    "httpMethod": "POST",
                    "path": "confirm-{{$json.batch_id}}",
                    "responseCode": 200,
                    "responseData": "={{$json}}"
                }
            },
            "limit": {
                "unit": "hours",
                "value": 48
            }
        },
        "error_handling": {
            "continueOnFail": True
        },
        "outputs": {
            "success": {
                "conditional": "={{$json.decision}}",
                "routes": {
                    "validated": "batch_state.mark_validated",
                    "rejected": "batch_state.mark_rejected",
                    "escalate": "email_notification"
                }
            },
            "timeout": {
                "next_node": "email_notification",
                "notification_config": {
                    "recipient": "{{CONFIG.notification_defaults.system_admin}}",
                    "subject": "Manual Confirmation Timeout - {{$json.batch_id}}",
                    "template": "confirmation_timeout"
                }
            }
        }
    },

    "kb.resolve_ontology_object": {
        "type": "n8n-nodes-base.httpRequest",
        "parameters": {
            "method": "POST",
            "url": "{{$env.KB_SERVICE_URL}}/kb/object/get",
            "sendBody": True,
            "bodyParameters": {
                "parameters": [
                    {
                        "name": "id_or_title",
                        "value": "={{$json.object_ref}}"
                    }
                ]
            },
            "options": {
                "timeout": 30000
            }
        },
        "error_handling": {
            "continueOnFail": True,
            "retryOnFail": True,
            "maxTries": 3
        },
        "outputs": {
            "success": {
                "state": "object_resolved",
                "next_node": "continue"
            },
            "failure": {
                "state": "object_not_found",
                "next_node": "log_error"
            }
        }
    },

    "kb.check_permission": {
        "type": "n8n-nodes-base.httpRequest",
        "parameters": {
            "method": "POST",
            "url": "{{$env.KB_SERVICE_URL}}/kb/check_permission",
            "sendBody": True,
            "bodyParameters": {
                "parameters": [
                    {
                        "name": "actor_id",
                        "value": "={{$json.actor_id}}"
                    },
                    {
                        "name": "permission",
                        "value": "={{$json.permission}}"
                    }
                ]
            }
        },
        "error_handling": {
            "continueOnFail": False
        },
        "outputs": {
            "success": {
                "conditional": "={{$json.allowed}}",
                "routes": {
                    "true": "continue",
                    "false": "email_notification"
                },
                "notification_config_on_false": {
                    "recipient": "{{$json.actor_id}}",
                    "subject": "Permission Denied - {{$json.permission}}",
                    "template": "permission_denied"
                }
            },
            "failure": {
                "state": "permission_check_FAILED",
                "next_node": "email_notification"
            }
        }
    }
}

# EMAIL TEMPLATES
EMAIL_TEMPLATES = {
    "file_check_failure": {
        "subject": "File Check Failed - Missing Expected Files",
        "body": """
<html>
<body>
<h2>File Validation Failed</h2>
<p><strong>Batch ID:</strong> {{$json.batch_id}}</p>
<p><strong>Expected Files:</strong> {{$json.expected_patterns}}</p>
<p><strong>Missing Files:</strong></p>
<ul>
{{#each $json.missing_files}}
    <li>{{this}}</li>
{{/each}}
</ul>
<p><strong>Action Required:</strong> Please verify the input folder and ensure all expected files are present.</p>
<p><strong>Timestamp:</strong> {{$json.timestamp}}</p>
</body>
</html>
"""
    },
    "cleaning_failure": {
        "subject": "ETL Cleaning Failed - Batch {{$json.batch_id}}",
        "body": """
<html>
<body>
<h2>Data Cleaning Process Failed</h2>
<p><strong>Batch ID:</strong> {{$json.batch_id}}</p>
<p><strong>Error:</strong> {{$json.error}}</p>
<p><strong>Files Affected:</strong> {{$json.flagged_files}}</p>
<p><strong>Action Required:</strong> Manual intervention required. Check logs for details.</p>
</body>
</html>
"""
    },
    "validation_service_failure": {
        "subject": "CRITICAL: Validation Service Failed - Batch {{$json.batch_id}}",
        "body": """
<html>
<body>
<h2 style="color: red;">CRITICAL: Validation Service Failure</h2>
<p><strong>Batch ID:</strong> {{$json.batch_id}}</p>
<p><strong>Error:</strong> {{$json.error}}</p>
<p><strong>Service Status:</strong> {{$json.service_status}}</p>
<p><strong>Action Required:</strong> Immediate investigation required. Service may be down.</p>
<p><strong>Escalation:</strong> System administrator has been notified.</p>
</body>
</html>
"""
    },
    "batch_validated": {
        "subject": "Batch Validated Successfully - {{$json.batch_id}}",
        "body": """
<html>
<body>
<h2>Batch Validation Successful</h2>
<p><strong>Batch ID:</strong> {{$json.batch_id}}</p>
<p><strong>Validation Date:</strong> {{$json.validated_at}}</p>
<p><strong>Records Processed:</strong> {{$json.record_count}}</p>
<p><strong>Status:</strong> Ready for ingestion</p>
</body>
</html>
"""
    },
    "db_failure": {
        "subject": "CRITICAL: Database Update Failed - {{$json.batch_id}}",
        "body": """
<html>
<body>
<h2 style="color: red;">CRITICAL: Database Operation Failed</h2>
<p><strong>Operation:</strong> {{$json.operation}}</p>
<p><strong>Batch ID:</strong> {{$json.batch_id}}</p>
<p><strong>Error:</strong> {{$json.error}}</p>
<p><strong>Action Required:</strong> Immediate database administrator intervention required.</p>
</body>
</html>
"""
    },
    "confirmation_timeout": {
        "subject": "Manual Confirmation Timeout - {{$json.batch_id}}",
        "body": """
<html>
<body>
<h2>Manual Confirmation Timeout</h2>
<p><strong>Batch ID:</strong> {{$json.batch_id}}</p>
<p><strong>Waiting Since:</strong> {{$json.waiting_since}}</p>
<p><strong>Action Required:</strong> This batch has been waiting for manual confirmation for 48 hours. Please review and take action.</p>
</body>
</html>
"""
    },
    "permission_denied": {
        "subject": "Permission Denied - {{$json.permission}}",
        "body": """
<html>
<body>
<h2>Access Denied</h2>
<p><strong>User:</strong> {{$json.actor_id}}</p>
<p><strong>Requested Permission:</strong> {{$json.permission}}</p>
<p><strong>Action:</strong> {{$json.attempted_action}}</p>
<p>If you believe you should have access, please contact your system administrator.</p>
</body>
</html>
"""
    }
}

# --------------------------------------------------------------------------------------
#  UI ROUTES
# --------------------------------------------------------------------------------------


@app.route("/")
def root():
    return send_from_directory("static", "index.html")


@app.route("/status/<job_id>", methods=["GET"])
def job_status(job_id: str):
    with STATE_LOCK:
        job = WORKFLOW_STATE["jobs"].get(job_id)

    if not job:
        return jsonify({"error": "unknown job_id"}), 404

    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error"),
        "created_at": job.get("created_at"),
    })


@app.route("/workflow/<job_id>", methods=["GET"])
def get_workflow(job_id: str):
    """Get complete workflow state for a job"""
    with STATE_LOCK:
        if job_id not in WORKFLOW_STATE["jobs"]:
            return jsonify({"error": "unknown job_id"}), 404

        workflow_data = {
            "job": WORKFLOW_STATE["jobs"].get(job_id, {}),
            "ontology": WORKFLOW_STATE["ontology"].get(job_id, {}),
            "rules": WORKFLOW_STATE["rules"].get(job_id, {}),
            "tasks": WORKFLOW_STATE["tasks"].get(job_id, {}),
            "transitions": WORKFLOW_STATE["transitions"].get(job_id, []),
            "subtasks": WORKFLOW_STATE["subtasks"].get(job_id, {}),
        }

    return jsonify(workflow_data)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accept workflow files and trigger n8n builder.
    """
    if "workflow_file" not in request.files:
        return jsonify({"error": "Missing workflow_file upload"}), 400

    workflow_file = request.files["workflow_file"]
    spec_text = workflow_file.read().decode("utf-8")

    wps_text = None
    datasheet_text = None

    if "wps_file" in request.files:
        wps_file = request.files["wps_file"]
        if wps_file.filename:
            wps_text = wps_file.read().decode("utf-8")

    if "datasheet_file" in request.files:
        datasheet_file = request.files["datasheet_file"]
        if datasheet_file.filename:
            datasheet_text = datasheet_file.read().decode("utf-8")

    job_id = str(uuid.uuid4())

    with STATE_LOCK:
        WORKFLOW_STATE["jobs"][job_id] = {
            "job_id": job_id,
            "status": "processing",
            "spec_text": spec_text,
            "wps_text": wps_text,
            "datasheet_text": datasheet_text,
            "created_at": time.time(),
            "error": None
        }
        WORKFLOW_STATE["tasks"][job_id] = {}
        WORKFLOW_STATE["subtasks"][job_id] = {}
        WORKFLOW_STATE["transitions"][job_id] = []

    def fire_n8n():
        try:
            payload = {
                "job_id": job_id,
                "spec_text": spec_text,
                "wps_text": wps_text,
                "datasheet_text": datasheet_text
            }

            response = requests.post(
                N8N_BUILDER_URL, json=payload, timeout=300)

            if not response.ok:
                raise Exception(f"n8n returned status {response.status_code}")

            with STATE_LOCK:
                WORKFLOW_STATE["jobs"][job_id]["status"] = "completed"

            broadcast_sse_event("job_complete", {"job_id": job_id})

        except Exception as e:
            with STATE_LOCK:
                WORKFLOW_STATE["jobs"][job_id]["status"] = "error"
                WORKFLOW_STATE["jobs"][job_id]["error"] = str(e)
            broadcast_sse_event("error", {"job_id": job_id, "error": str(e)})

    threading.Thread(target=fire_n8n, daemon=True).start()

    return jsonify({"job_id": job_id, "status": "processing"})

# --------------------------------------------------------------------------------------
#  N8N CALLBACK ENDPOINTS
# --------------------------------------------------------------------------------------

# --------------- GET --------------------------------------


@app.route("/skills", methods=["GET"])
def list_skills():
    """
    Return a lightweight view of the skill registry for n8n:
    id + description + category.
    """
    skills_payload = [
        {
            "id": skill.id,
            "description": skill.description,
            "category": skill.category,
        }
        for skill in SKILL_REGISTRY.values()
    ]
    return jsonify({"skills": skills_payload})


@app.route("/reference-ontology", methods=["GET"])
def reference_ontology_summary():
    """
    Compact ontology summary for n8n / LLM prompts.
    """
    payload = GLOBAL_ONTOLOGY.to_llm_payload()
    return jsonify(payload)

# --------------- SET --------------------------------------


@app.route("/task-update", methods=["POST"])
def task_update():
    """Called by n8n Task Feedback node."""
    data = request.get_json(silent=True) or {}

    task_id = data.get("task_id")
    description = data.get("description")
    label = data.get("label")

    if not task_id:
        return jsonify({"error": "task_id required"}), 400

    with STATE_LOCK:
        job_ids = list(WORKFLOW_STATE["jobs"].keys())
        if not job_ids:
            return jsonify({"error": "No active jobs"}), 400

        job_id = None
        for jid in reversed(job_ids):
            if WORKFLOW_STATE["jobs"][jid]["status"] == "processing":
                job_id = jid
                break

        if not job_id and job_ids:
            job_id = job_ids[-1]

        if not job_id:
            return jsonify({"error": "No active job found"}), 400

        if job_id not in WORKFLOW_STATE["tasks"]:
            WORKFLOW_STATE["tasks"][job_id] = {}

        WORKFLOW_STATE["tasks"][job_id][task_id] = {
            "task_id": task_id,
            "label": label,
            "description": description,
            "created_at": time.time(),
            "subtasks": []
        }

    broadcast_sse_event("task_update", {
        "job_id": job_id,
        "task_id": task_id,
        "label": label,
        "description": description
    })

    return jsonify({"ok": True})


@app.route("/subtask-update", methods=["POST"])
def subtask_update():
    """Called by n8n Subtask Feedback node."""
    data = request.get_json(silent=True) or {}

    substep_index = data.get("substep_index")
    description = data.get("description")
    # NOTE: this was `bladerunner_index` before – leaving as-is,
    # but you probably want data.get("kind", "system") here.
    kind = data.get("bladerunner_index", "system")
    raw_skill = (data.get("skill") or "").strip()
    skill = normalise_skill_id(raw_skill)

    with STATE_LOCK:
        job_ids = list(WORKFLOW_STATE["jobs"].keys())
        job_id = None

        # Prefer most recent "processing" job
        for jid in reversed(job_ids):
            if WORKFLOW_STATE["jobs"][jid]["status"] == "processing":
                job_id = jid
                break

        # Fallback: last job if none marked processing
        if not job_id and job_ids:
            job_id = job_ids[-1]

        if not job_id:
            return jsonify({"error": "No active job"}), 400

        tasks_for_job = WORKFLOW_STATE["tasks"].get(job_id, {})
        if not tasks_for_job:
            return jsonify({"error": "No tasks found for job"}), 400

        # For now, attach subtask to the most recent task
        task_id = list(tasks_for_job.keys())[-1]

        # Ensure nested structure exists: subtasks[job_id][task_id] = []
        if job_id not in WORKFLOW_STATE["subtasks"]:
            WORKFLOW_STATE["subtasks"][job_id] = {}
        if task_id not in WORKFLOW_STATE["subtasks"][job_id]:
            WORKFLOW_STATE["subtasks"][job_id][task_id] = []

        # Build a single subtask object including diagnostics
        subtask = {
            "id": data.get("substep_id"),
            "substep_index": substep_index,
            "description": description,
            "kind": kind,
            "skill": skill,
            "created_at": time.time(),
        }

        diagnostic = validate_skill_for_subtask(skill, subtask)
        subtask["diagnostic"] = diagnostic

        # Store in global subtasks structure
        WORKFLOW_STATE["subtasks"][job_id][task_id].append(subtask)

        # Also mirror onto the task itself for convenience
        if "subtasks" not in WORKFLOW_STATE["tasks"][job_id][task_id]:
            WORKFLOW_STATE["tasks"][job_id][task_id]["subtasks"] = []
        WORKFLOW_STATE["tasks"][job_id][task_id]["subtasks"].append(subtask)

    broadcast_sse_event(
        "subtask_update",
        {
            "job_id": job_id,
            "task_id": task_id,
            "subtask": subtask,
        },
    )

    return jsonify({"ok": True})


@app.route("/relations", methods=["POST"])
def relations():
    """Called by n8n Return Relations node."""
    data = request.get_json(silent=True) or {}
    links = data.get("links", [])

    with STATE_LOCK:
        job_ids = list(WORKFLOW_STATE["jobs"].keys())
        job_id = None
        for jid in reversed(job_ids):
            if WORKFLOW_STATE["jobs"][jid]["status"] in ["processing", "completed"]:
                job_id = jid
                break

        if not job_id and job_ids:
            job_id = job_ids[-1]

        if not job_id:
            return jsonify({"error": "No active job"}), 400

        tasks = WORKFLOW_STATE["tasks"].get(job_id, {})
        task_list = sorted(
            tasks.items(), key=lambda x: x[1].get("created_at", 0))
        task_ids = [task_id for task_id, _ in task_list]

        existing_connections = set()
        for link in links:
            src = link.get("source_step_id")
            tgt = link.get("target_step_id")
            if src and tgt:
                existing_connections.add((src, tgt))

        inferred_links = []
        for i in range(len(task_ids) - 1):
            current_task = task_ids[i]
            next_task = task_ids[i + 1]

            has_outgoing = any(link.get("source_step_id") ==
                               current_task for link in links)

            if not has_outgoing and (current_task, next_task) not in existing_connections:
                inferred_link = {
                    "source_step_id": current_task,
                    "target_step_id": next_task,
                    "condition_text": None,
                    "outcome_name": None
                }
                inferred_links.append(inferred_link)
                existing_connections.add((current_task, next_task))

        all_links = links + inferred_links

        WORKFLOW_STATE["transitions"][job_id] = all_links
        WORKFLOW_STATE["jobs"][job_id]["status"] = "completed"

        relations_map = {}
        for link in all_links:
            src = link.get("source_step_id")
            if src not in relations_map:
                relations_map[src] = []
            relations_map[src].append({
                "target_step_id": link.get("target_step_id"),
                "condition_text": link.get("condition_text"),
                "outcome_name": link.get("outcome_name"),
                "inferred": link in inferred_links
            })

        tasks_copy = dict(WORKFLOW_STATE["tasks"].get(job_id, {}))

    broadcast_sse_event("relations_update", {
        "job_id": job_id,
        "relations": relations_map,
        "links": all_links,
        "inferred_count": len(inferred_links)
    })

    broadcast_sse_event("job_complete", {"job_id": job_id})

    return jsonify({
        "ok": True,
        "tasks": tasks_copy,
        "relations": relations_map,
        "inferred_links": inferred_links,
        "total_links": len(all_links)
    })


@app.route("/ontology-update", methods=["POST"])
def ontology_update():
    data = request.get_json(silent=True) or {}

    # Accept both wrapped and unwrapped payloads
    ontology = data.get("ontology", data)

    with STATE_LOCK:
        job_ids = list(WORKFLOW_STATE["jobs"].keys())
        job_id = None
        for jid in reversed(job_ids):
            if WORKFLOW_STATE["jobs"][jid]["status"] == "processing":
                job_id = jid
                break
        if not job_id and job_ids:
            job_id = job_ids[-1]
        if not job_id:
            return jsonify({"error": "No active job"}), 400

        WORKFLOW_STATE["ontology"][job_id] = ontology

    broadcast_sse_event("ontology_update", {
        "job_id": job_id,
        "ontology": ontology,
    })

    return jsonify({"ok": True})


@app.route("/rules-update", methods=["POST"])
def rules_update():
    """
    Called by n8n Extract Rules node.
    Expects: {
        "rules": [...]
    }
    """
    data = request.get_json(silent=True) or {}
    rules = data.get("rules", [])

    with STATE_LOCK:
        # Find most recent processing job
        job_ids = list(WORKFLOW_STATE["jobs"].keys())
        job_id = None
        for jid in reversed(job_ids):
            if WORKFLOW_STATE["jobs"][jid]["status"] == "processing":
                job_id = jid
                break

        if not job_id and job_ids:
            job_id = job_ids[-1]

        if not job_id:
            return jsonify({"error": "No active job"}), 400

        # Store rules
        WORKFLOW_STATE["rules"][job_id] = {"rules": rules}

    # Broadcast to SSE clients
    broadcast_sse_event("rules_update", {
        "job_id": job_id,
        "rules": rules
    })

    return jsonify({"ok": True})


# --------------------------------------------------------------------------------------
# SSE ENDPOINT FOR LIVE UPDATES
# --------------------------------------------------------------------------------------


@app.route("/events/<job_id>")
def sse_stream(job_id: str):
    """Server-Sent Events stream for live updates"""

    def event_stream():
        client_queue = queue.Queue()
        client_id = str(uuid.uuid4())

        with SSE_LOCK:
            SSE_CLIENTS[client_id] = client_queue

        try:
            yield f"data: {json.dumps({'type': 'connected', 'job_id': job_id})}\n\n"

            with STATE_LOCK:
                current_state = {
                    "type": "initial_state",
                    "job": WORKFLOW_STATE["jobs"].get(job_id, {}),
                    "tasks": WORKFLOW_STATE["tasks"].get(job_id, {}),
                    "subtasks": WORKFLOW_STATE["subtasks"].get(job_id, {}),
                    "transitions": WORKFLOW_STATE["transitions"].get(job_id, [])
                }
            yield f"data: {json.dumps(current_state)}\n\n"

            while True:
                try:
                    event = client_queue.get(timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    yield f": keepalive\n\n"
        finally:
            with SSE_LOCK:
                SSE_CLIENTS.pop(client_id, None)

    return Response(event_stream(), mimetype="text/event-stream")


def broadcast_sse_event(event_type: str, data: dict):
    """Broadcast an event to all SSE clients"""
    event = {"type": event_type, **data}

    with SSE_LOCK:
        dead_clients = []
        for client_id, client_queue in SSE_CLIENTS.items():
            try:
                client_queue.put_nowait(event)
            except queue.Full:
                dead_clients.append(client_id)

        for client_id in dead_clients:
            SSE_CLIENTS.pop(client_id, None)

# --------------------------------------------------------------------------------------
# ENHANCED N8N WORKFLOW BUILDER
# --------------------------------------------------------------------------------------


@app.route("/build-n8n-workflow/<job_id>", methods=["POST"])
def build_n8n_workflow(job_id: str):
    """Generate an n8n workflow JSON from collected workflow state"""

    with STATE_LOCK:
        if job_id not in WORKFLOW_STATE["jobs"]:
            return jsonify({"error": "Unknown job_id"}), 404

        tasks = WORKFLOW_STATE["tasks"].get(job_id, {})
        subtasks = WORKFLOW_STATE["subtasks"].get(job_id, {})
        transitions = WORKFLOW_STATE["transitions"].get(job_id, [])

    if not tasks:
        return jsonify({"error": "No tasks found for job_id"}), 404

    try:
        workflow = build_workflow_from_state(
            job_id, tasks, subtasks, transitions)

        output_path = f"outputs/workflow_{job_id}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(workflow.to_json(), f, indent=2)

        return jsonify({
            "ok": True,
            "workflow": workflow.to_json(),
            "file": output_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def build_workflow_from_state(
    job_id: str,
    tasks: Dict[str, Any],
    subtasks: Dict[str, List[Any]],
    transitions: List[Dict[str, Any]]
) -> N8NWorkflow:
    """
    Build an n8n workflow from the collected server state.

    Features:
      • Node-per-subtask if available, otherwise node-per-task
      • Skill-aware IO binding using ontology + CONTEXT_DB
      • Handles databases, emails, webhooks, topics, PLCs, state vars
      • Inserts IF nodes for conditional transitions
      • Human subtasks become WAIT nodes
    """

    nodes: List[N8NNode] = []
    connections: Dict[str, Any] = {}

    node_counter = 0
    task_entry_node: Dict[str, str] = {}

    # Job ontology (may be empty)
    ontology = WORKFLOW_STATE["ontology"].get(job_id, {})

    # -------------------------------------------------------------------------
    # 1. Manual Trigger
    # -------------------------------------------------------------------------
    trigger = N8NNode(
        id="trigger",
        name="Manual Trigger",
        type="n8n-nodes-base.manualTrigger",
        position=[0, 0],
        parameters={}
    )
    nodes.append(trigger)

    # -------------------------------------------------------------------------
    # 2. Create nodes for tasks & subtasks
    # -------------------------------------------------------------------------
    x_spacing = 380
    y_spacing = 170
    base_x = 300

    for task_idx, (task_id, task_data) in enumerate(tasks.items()):
        x = base_x + task_idx * x_spacing
        y = 0

        st_list = subtasks.get(task_id) or []
        st_list = [s for s in st_list if s]  # remove nulls

        # ---------------------------------------------------------------------
        # No subtasks → single node for whole task
        # ---------------------------------------------------------------------
        if not st_list:
            node_id = f"node_{node_counter}"
            node_counter += 1

            node = create_node_from_subtask(
                node_id=node_id,
                task_data=task_data,
                subtask_data=task_data,
                position=[x, y],
                skill=task_data.get("skill")
            )

            # Bind IO via ontology & context DB
            skill_id = task_data.get("skill")
            skill_def = resolve_skill(skill_id)
            if skill_def:
                bound_params = bind_skill_io_to_node(
                    skill=skill_def,
                    subtask=task_data,
                    ontology=ontology,
                    context_db=CONTEXT_DB
                )
                node.parameters.update(bound_params)

            nodes.append(node)
            task_entry_node[task_id] = node.id
            continue

        # ---------------------------------------------------------------------
        # Subtasks exist → chain them vertically
        # ---------------------------------------------------------------------
        previous_node_id = None
        first_node_id = None

        for st_idx, subtask in enumerate(st_list):
            node_id = f"node_{node_counter}"
            node_counter += 1

            node = create_node_from_subtask(
                node_id=node_id,
                task_data=task_data,
                subtask_data=subtask,
                position=[x, y + st_idx * y_spacing],
                skill=subtask.get("skill")
            )

            # Skill-aware IO binding
            skill_id = subtask.get("skill")
            # Normalise format “a.b.c [category] → a.b.c”
            if skill_id and "[" in skill_id:
                skill_id = skill_id.split("[")[0].strip()

            skill_def = resolve_skill(skill_id)
            if skill_def:
                bound_params = bind_skill_io_to_node(
                    skill=skill_def,
                    subtask=subtask,
                    ontology=ontology,
                    context_db=CONTEXT_DB
                )
                node.parameters.update(bound_params)

            nodes.append(node)

            # Chain subtasks sequentially
            if previous_node_id:
                add_connection(connections, previous_node_id, node_id, 0, 0)

            previous_node_id = node_id
            if first_node_id is None:
                first_node_id = node_id

        if first_node_id:
            task_entry_node[task_id] = first_node_id

    # -------------------------------------------------------------------------
    # 3. Create transitions between tasks
    # -------------------------------------------------------------------------
    for tr in transitions:
        src_id = tr.get("source_step_id")
        tgt_id = tr.get("target_step_id")
        if not src_id or not tgt_id:
            continue
        src = task_entry_node.get(src_id)
        tgt = task_entry_node.get(tgt_id)
        if not src or not tgt:
            continue

        condition = tr.get("condition_text") or tr.get("outcome_name")

        if condition:
            # Insert IF node
            if_node_id = f"node_{node_counter}"
            node_counter += 1

            if_node = N8NNode(
                id=if_node_id,
                name=f"If {condition}",
                type="n8n-nodes-base.if",
                position=[150, 150],  # later: layout engine
                parameters={
                    "conditions": {
                        "string": [
                            {
                                "value1": "={{$json.state}}",
                                "operation": "equal",
                                "value2": condition
                            }
                        ]
                    }
                }
            )
            nodes.append(if_node)

            add_connection(connections, src, if_node_id, 0, 0)
            add_connection(connections, if_node_id, tgt, 0, 0)

        else:
            add_connection(connections, src, tgt, 0, 0)

    # -------------------------------------------------------------------------
    # 4. Trigger → first node
    # -------------------------------------------------------------------------
    if len(nodes) > 1:
        add_connection(connections, "trigger", nodes[1].id, 0, 0)

    # -------------------------------------------------------------------------
    # 5. Build workflow object
    # -------------------------------------------------------------------------
    return N8NWorkflow(
        name=f"Generated Workflow {job_id}",
        nodes=nodes,
        connections=connections,
        active=False,
        settings={"executionOrder": "v1"}
    )


def create_node_from_task(
    node_id: str,
    task_data: Dict[str, Any],
    position: List[int]
) -> N8NNode:
    """Create n8n node from task"""
    return N8NNode(
        id=node_id,
        name=task_data.get("label", "Task"),
        type="n8n-nodes-base.function",
        position=position,
        parameters={
            "functionCode": f"// {task_data.get('description', '')}",
            "description": task_data.get("description", "")
        },
        continueOnFail=True
    )


def create_node_from_subtask(
    node_id: str,
    task_data: Dict[str, Any],
    subtask_data: Dict[str, Any],
    position: List[int],
    skill: Optional[str] = None
) -> N8NNode:
    """
    Create an n8n node from a subtask.
    Handles:
      - Skill registry lookup
      - Human approval nodes
      - Generic fallback nodes
      - Standardised error-handling injection
    """

    # Normalise skill format “skill_id [category]”
    if skill and "[" in skill:
        skill = skill.split("[")[0].strip()

    skill_def = resolve_skill(skill) if skill else None

    # -------------------------------------------------------------------------
    # Human operator → WAIT node
    # -------------------------------------------------------------------------
    if subtask_data.get("kind") == "human":
        return N8NNode(
            id=node_id,
            name=subtask_data.get("description", "Human Step")[:50],
            type="n8n-nodes-base.wait",
            position=position,
            parameters={
                "resume": "webhook",
                "options": {
                    "webhook": {
                        "responseCode": "200",
                        "responseData": "firstEntryJson"
                    }
                },
                "description": subtask_data.get("description", "")
            },
            continueOnFail=False,
            retryOnFail=False
        )

    # -------------------------------------------------------------------------
    # Skill-based node
    # -------------------------------------------------------------------------
    if skill_def:
        node_type = skill_def.n8n_template.get(
            "type", "n8n-nodes-base.function")
        init_params = dict(skill_def.n8n_template.get("parameters", {}))
        eh = skill_def.n8n_template.get(
            "error_handling", CONFIG["error_handling_defaults"])

    else:
        # ---------------------------------------------------------------------
        # Generic fallback node
        # ---------------------------------------------------------------------
        node_type = "n8n-nodes-base.function"
        init_params = {
            "functionCode": f"// {subtask_data.get('description','')}"
        }
        eh = CONFIG["error_handling_defaults"]

    return N8NNode(
        id=node_id,
        name=subtask_data.get("description", "Subtask")[:50],
        type=node_type,
        position=position,
        parameters={
            **init_params,
            "description": subtask_data.get("description", "")
        },
        continueOnFail=eh.get("continueOnFail"),
        retryOnFail=eh.get("retryOnFail"),
        maxTries=eh.get("maxTries"),
        waitBetweenTries=eh.get("waitBetweenTries")
    )


def create_error_handling_node(
    task_id: str,
    task_data: Dict[str, Any],
    source_node_id: str,
    position: List[int],
    node_counter: int
) -> Optional[N8NNode]:
    """Create an error handling node for a task if needed"""
    # This is a placeholder - implement based on task requirements
    return None


def create_notification_node(
    node_counter: int,
    notification_config: Dict[str, Any],
    position: List[int]
) -> N8NNode:
    """Create an email notification node"""

    recipient = notification_config.get(
        "recipient", CONFIG["notification_defaults"]["default_recipient"])
    subject = notification_config.get("subject", "Workflow Notification")
    template_name = notification_config.get("template", "default")

    template = EMAIL_TEMPLATES.get(template_name, {})
    email_body = template.get("body", "{{$json.message}}")

    return N8NNode(
        id=f"email_notif_{node_counter}",
        name=f"Email: {subject[:30]}",
        type="n8n-nodes-base.emailSend",
        position=position,
        parameters={
            "fromEmail": "workflow@company.com",
            "toEmail": recipient,
            "subject": subject,
            "emailType": "html",
            "message": email_body
        },
        continueOnFail=True,
        retryOnFail=True,
        maxTries=3
    )


def add_connection(
    connections: Dict[str, Any],
    source_node: str,
    target_node: str,
    source_output: int = 0,
    target_input: int = 0
):
    """Add connection between nodes"""
    if source_node not in connections:
        connections[source_node] = {}

    if "main" not in connections[source_node]:
        connections[source_node]["main"] = []

    while len(connections[source_node]["main"]) <= source_output:
        connections[source_node]["main"].append([])

    connections[source_node]["main"][source_output].append({
        "node": target_node,
        "type": "main",
        "index": target_input
    })


def validate_skill_for_subtask(skill_id: str, subtask: Dict[str, Any]) -> Dict[str, Any]:
    """Return a small diagnostics dict to attach to state."""
    skill_def = resolve_skill(skill_id)
    if not skill_def:
        return {
            "status": "unknown",
            "message": f"Skill '{skill_id}' not found in registry"
        }
    # Optionally: check required IO keys vs subtask/ontology
    return {
        "status": "ok",
        "message": ""
    }


def normalise_skill_id(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    # Strip trailing category like " [messaging]"
    if "[" in s and s.endswith("]"):
        s = s[:s.rfind("[")].strip()
    return s


# ONTOLOGY REFS for N8N

def resolve_database(
    name: str,
    context_db: Dict[str, Any],
    ontology: Optional[Ontology] = None,
) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    ontology = ontology or GLOBAL_ONTOLOGY
    dbs = context_db.get("databases", {})
    ctx_key = ontology.find_context_key(
        name, kind="database", context_section=dbs)
    if not ctx_key:
        return None
    return dbs.get(ctx_key)


def resolve_webhook(
    name: str,
    context_db: Dict[str, Any],
    ontology: Optional[Ontology] = None,
) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    ontology = ontology or GLOBAL_ONTOLOGY
    whs = context_db.get("webhooks", {})
    ctx_key = ontology.find_context_key(
        name, kind="webhook", context_section=whs)
    if not ctx_key:
        return None
    return whs.get(ctx_key)


def resolve_topic(
    name: str,
    context_db: Dict[str, Any],
    ontology: Optional[Ontology] = None,
) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    ontology = ontology or GLOBAL_ONTOLOGY
    topics = context_db.get("topics", {})
    ctx_key = ontology.find_context_key(
        name, kind="topic", context_section=topics)
    if not ctx_key:
        return None
    return topics.get(ctx_key)


def resolve_machine(
    name: str,
    context_db: Dict[str, Any],
    ontology: Optional[Ontology] = None,
) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    ontology = ontology or GLOBAL_ONTOLOGY
    machines = context_db.get("machines", {})
    ctx_key = ontology.find_context_key(
        name, kind="machine", context_section=machines)
    if not ctx_key:
        return None
    return machines.get(ctx_key)


def resolve_contact_email(
    role_or_entity: Optional[str],
    ontology: Ontology,
    contacts_section: Dict[str, Any],
) -> Optional[str]:
    if not role_or_entity:
        return None
    return ontology.resolve_contact_email(role_or_entity, contacts_section)


def bind_skill_io_to_node(
    skill: SkillDefinition,
    subtask: Dict[str, Any],
    ontology: Ontology,
    context_db: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Map semantic IO (email, db, webhooks, topics, state vars, etc.)
    to actual n8n node parameters using ontology + context DB.
    """

    base_template = getattr(skill, "n8n_template", {}) or {}
    params = dict(base_template.get("parameters", {}) or {})

    inputs_spec = {}
    io_block = getattr(skill, "io", None)
    if isinstance(io_block, dict):
        inputs_spec = io_block.get("inputs", {}) or {}

    for logical_name, spec in inputs_spec.items():
        if not isinstance(spec, dict):
            continue

        io_type = (spec.get("type") or "").strip()
        if not io_type:
            continue

        param_name = spec.get("param") or logical_name

        if io_type == "contact.email":
            role_field = spec.get("from_field", "target_role")
            role = subtask.get(role_field) or subtask.get("owner_role")
            email = resolve_contact_email(
                role_or_entity=role,
                ontology=ontology,
                contacts_section=context_db.get("contacts", {}),
            )
            if email:
                params[param_name] = email

        elif io_type == "database.by_name":
            name_field = spec.get("from_field", "target_system")
            db_name = subtask.get(name_field) or spec.get("default")
            db_cfg = resolve_database(
                db_name, context_db, ontology=ontology) if db_name else None
            if db_cfg:
                if db_cfg.get("database"):
                    params[param_name] = db_cfg["database"]
                if "schema" in db_cfg:
                    params.setdefault("schema", db_cfg["schema"])

        elif io_type == "webhook.by_name":
            action_field = spec.get("from_field", "target_action")
            wh_name = subtask.get(action_field) or spec.get("default")
            wh_cfg = resolve_webhook(
                wh_name, context_db, ontology=ontology) if wh_name else None
            if wh_cfg and "url" in wh_cfg:
                params[param_name] = wh_cfg["url"]

        elif io_type == "topic.by_name":
            topic_field = spec.get("from_field", "topic_name")
            t_name = subtask.get(topic_field) or spec.get("default")
            t_cfg = resolve_topic(t_name, context_db,
                                  ontology=ontology) if t_name else None
            if t_cfg:
                if "mqtt_topic" in t_cfg:
                    params[param_name] = t_cfg["mqtt_topic"]
                elif "kafka_topic" in t_cfg:
                    params[param_name] = t_cfg["kafka_topic"]

        elif io_type == "machine.by_name":
            machine_field = spec.get("from_field", "machine_name")
            m_name = subtask.get(machine_field) or spec.get("default")
            m_cfg = resolve_machine(
                m_name, context_db, ontology=ontology) if m_name else None
            if m_cfg and "endpoint" in m_cfg:
                params[param_name] = m_cfg["endpoint"]

        elif io_type == "state_var":
            state_key = spec.get("state_key")
            if state_key:
                json_path = spec.get("json_path") or f"$json['{state_key}']"
                params[param_name] = f"={{{{ {json_path} }}}}"

        elif io_type == "literal":
            if "value" in spec:
                params[param_name] = spec["value"]

        elif io_type == "subtask.description":
            params[param_name] = subtask.get("description", "") or ""

        elif io_type == "subtask.id":
            params[param_name] = subtask.get("id")

    if "description" not in params:
        params["description"] = subtask.get("description", "")

    return params


# --------------------------------------------------------------------------------------
# EXPORT STATE
# --------------------------------------------------------------------------------------


@app.route("/export-state/<job_id>", methods=["GET"])
def export_state(job_id: str):
    """Export complete workflow state"""

    with STATE_LOCK:
        if job_id not in WORKFLOW_STATE["jobs"]:
            return jsonify({"error": "Unknown job_id"}), 404

        state = {
            "job_id": job_id,
            "job": WORKFLOW_STATE["jobs"].get(job_id, {}),
            "ontology": WORKFLOW_STATE["ontology"].get(job_id, {}),
            "rules": WORKFLOW_STATE["rules"].get(job_id, {}),
            "tasks": WORKFLOW_STATE["tasks"].get(job_id, {}),
            "subtasks": WORKFLOW_STATE["subtasks"].get(job_id, {}),
            "transitions": WORKFLOW_STATE["transitions"].get(job_id, [])
        }

    output_path = f"outputs/state_{job_id}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(state, f, indent=2)

    return jsonify({
        "ok": True,
        "state": state,
        "file": output_path
    })

# --------------------------------------------------------------------------------------
# CONFIGURATION ENDPOINTS
# --------------------------------------------------------------------------------------


@app.route("/config", methods=["GET"])
def get_config():
    """Get current configuration"""
    return jsonify(CONFIG)


@app.route("/config", methods=["POST"])
def update_config():
    """Update configuration"""
    data = request.get_json(silent=True) or {}

    if "notification_defaults" in data:
        CONFIG["notification_defaults"].update(data["notification_defaults"])

    if "ontology_mappings" in data:
        CONFIG["ontology_mappings"].update(data["ontology_mappings"])

    if "error_handling_defaults" in data:
        CONFIG["error_handling_defaults"].update(
            data["error_handling_defaults"])

    return jsonify({"ok": True, "config": CONFIG})

# --------------------------------------------------------------------------------------
# DEBUG ENDPOINTS
# --------------------------------------------------------------------------------------


@app.route("/debug/state", methods=["GET"])
def debug_state():
    """Get complete server state for debugging"""
    with STATE_LOCK:
        return jsonify(WORKFLOW_STATE)


@app.route("/debug/clients", methods=["GET"])
def debug_clients():
    """Get connected SSE clients"""
    with SSE_LOCK:
        return jsonify({
            "client_count": len(SSE_CLIENTS),
            "client_ids": list(SSE_CLIENTS.keys())
        })


@app.route("/debug/skills", methods=["GET"])
def debug_skills():
    """Get available skills and their configurations"""
    return jsonify({
        "skills": list(SKILL_TO_N8N_TYPE.keys()),
        "skill_details": SKILL_TO_N8N_TYPE
    })

# --------------------------------------------------------------------------------------
# START SERVER
# --------------------------------------------------------------------------------------


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "8011"))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
