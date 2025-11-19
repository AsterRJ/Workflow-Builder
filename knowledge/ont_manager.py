from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional
import os


@dataclass
class Ontology:
    """
    Handles reference + job-specific ontology information.

    Expected base structure for each ontology dict:

      {
        "objects": [
          {
            "id": "db.etl_batch_log",
            "kind": "database",
            "label": "ETL Batch Log Database",
            "context_key": "etl_batch_log",   # key into CONTEXT_DB["databases"]
            "aliases": ["ETL batch DB", "batch log DB"],
            "tags": ["etl", "logging"]
          },
          {
            "id": "person.batch_manager",
            "kind": "person",
            "label": "Batch Manager",
            "context_key": "batch_manager",   # key into CONTEXT_DB["contacts"]
            "aliases": ["batch supervisor"],
            "tags": ["role", "ops"]
          },
          ...
        ],
        "processes": [...],
        "properties": [...],
        "events": [...]
      }

    - `reference` is loaded from a static JSON file (shared across jobs).
    - `overlay` is a job-specific ontology (from /ontology-update), merged on top.
    """
    reference: Dict[str, Any] = field(default_factory=dict)
    overlay: Dict[str, Any] = field(default_factory=dict)

    # --- constructors -------------------------------------------------

    @classmethod
    def from_path(cls, path: str) -> "Ontology":
        if not os.path.exists(path):
            print(f"[ontology] No reference ontology found at {path}")
            return cls(reference=cls._empty_ontology())

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ontology] Failed to load {path}: {e}")
            data = {}

        if not isinstance(data, dict):
            print(
                f"[ontology] Reference ontology at {path} must be a JSON object")
            data = {}

        return cls(reference=cls._normalise(data), overlay=cls._empty_ontology())

    @classmethod
    def empty(cls) -> "Ontology":
        return cls(reference=cls._empty_ontology(), overlay=cls._empty_ontology())

    @staticmethod
    def _empty_ontology() -> Dict[str, Any]:
        return {"objects": [], "processes": [], "properties": [], "events": []}

    @staticmethod
    def _normalise(data: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(data)
        for key in ("objects", "processes", "properties", "events"):
            if key not in out or not isinstance(out[key], list):
                out[key] = []
        return out

    # --- overlay handling ---------------------------------------------

    def with_overlay(self, overlay: Optional[Dict[str, Any]]) -> "Ontology":
        """
        Return a NEW Ontology instance with the given overlay merged on top.
        Safe to call with None.
        """
        if not overlay:
            return Ontology(reference=self.reference, overlay=self.overlay)

        norm_overlay = self._normalise(overlay)
        return Ontology(reference=self.reference, overlay=norm_overlay)

    # --- accessors ----------------------------------------------------

    def _objects_source_order(self) -> List[Dict[str, Any]]:
        """
        Overlay objects override reference objects with the same id.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for obj in self.reference.get("objects", []):
            oid = obj.get("id") or obj.get("label") or obj.get("name")
            if oid:
                result[oid] = obj
        for obj in self.overlay.get("objects", []):
            oid = obj.get("id") or obj.get("label") or obj.get("name")
            if oid:
                result[oid] = obj
        return list(result.values())

    def iter_objects(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        objs = self._objects_source_order()
        if kind:
            return [o for o in objs if o.get("kind") == kind]
        return objs

    def get_object(self, ontology_id: str) -> Optional[Dict[str, Any]]:
        if not ontology_id:
            return None
        # overlay wins
        for obj in self.overlay.get("objects", []):
            if obj.get("id") == ontology_id:
                return obj
        for obj in self.reference.get("objects", []):
            if obj.get("id") == ontology_id:
                return obj
        return None

    def find_object(
        self,
        id_or_label: str,
        kind: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Try to find an object by id, label or alias.
        Overlay takes precedence over reference.
        """
        if not id_or_label:
            return None

        candidates = self._objects_source_order()
        # 1) exact id match
        for obj in candidates:
            if kind and obj.get("kind") != kind:
                continue
            if obj.get("id") == id_or_label:
                return obj

        # 2) label / name match
        for obj in candidates:
            if kind and obj.get("kind") != kind:
                continue
            label = obj.get("label") or obj.get("name")
            if label and label == id_or_label:
                return obj

        # 3) alias match
        for obj in candidates:
            if kind and obj.get("kind") != kind:
                continue
            aliases = obj.get("aliases") or []
            if id_or_label in aliases:
                return obj

        return None

    # --- context binding helpers -------------------------------------

    def find_context_key(
        self,
        id_or_label: str,
        kind: str,
        context_section: Dict[str, Any],
    ) -> Optional[str]:
        """
        Given an ontology id/label and a kind ('database', 'webhook', 'topic',
        'machine', 'person', 'role'), return the key into the appropriate
        context_db section.

        Typical pattern:
          ctx_key = ontology.find_context_key(name, "database", context_db["databases"])
          if ctx_key and ctx_key in context_db["databases"]:
              db_cfg = context_db["databases"][ctx_key]
        """
        if not id_or_label:
            return None

        obj = self.find_object(id_or_label, kind=kind)
        if obj:
            ctx_key = obj.get("context_key") or obj.get("id")
            if ctx_key in context_section:
                return ctx_key

        # As a fallback, if id_or_label IS directly a context key, allow that.
        if id_or_label in context_section:
            return id_or_label

        return None

    def resolve_contact_email(
        self,
        role_or_entity: Optional[str],
        contacts_section: Dict[str, Any],
    ) -> Optional[str]:
        """
        Resolve a role/person identifier to an email using contacts_section and
        ontology objects (kind 'person' or 'role').
        """
        if not role_or_entity:
            return None

        # 1) direct lookup
        entry = contacts_section.get(role_or_entity)
        if isinstance(entry, dict):
            email = entry.get("email")
            if isinstance(email, str) and email:
                return email

        # 2) ontology-backed lookup
        obj = self.find_object(role_or_entity, kind=None)
        if obj and obj.get("kind") in ("person", "role"):
            ctx_key = obj.get("context_key") or obj.get("id")
            if ctx_key:
                entry2 = contacts_section.get(ctx_key)
                if isinstance(entry2, dict):
                    email2 = entry2.get("email")
                    if isinstance(email2, str) and email2:
                        return email2

        return None

    # --- LLM payload helper -------------------------------------------

    def to_llm_payload(self, max_objects: int = 200) -> Dict[str, Any]:
        """
        Return a compact view of the ontology suitable for including in prompts.

        We do not dump everything: just enough to allow reuse of existing IDs.
        """
        objs = self._objects_source_order()[:max_objects]
        slim_objs = []
        for obj in objs:
            slim_objs.append({
                "id": obj.get("id"),
                "label": obj.get("label") or obj.get("name"),
                "kind": obj.get("kind"),
                "context_key": obj.get("context_key"),
                "aliases": obj.get("aliases", []),
                "tags": obj.get("tags", []),
            })
        return {
            "objects": slim_objs,
            # you can extend this later with processes/properties/events if needed
        }
