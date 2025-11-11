from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional

from google import genai
from google.genai import types as gtypes
from google.cloud import bigquery

from context import sales, stocr, traffic

from decimal import Decimal
from datetime import date, datetime

# --- BigQuery setup (ADC: env, WI, or SA on the host) ---
bq = bigquery.Client()

# --- Gemini (AI Studio) ---
# Uses GOOGLE_API_KEY if set, otherwise falls back to application-default creds when supported.
gclient = genai.Client()

# ---------- Tool implementations ----------

def json_safe(obj):
	"""Recursively convert BigQuery / Python types to JSON-serializable types."""
	if isinstance(obj, Decimal):
		# choose float or str depending on how precise you need
		return float(obj)
	if isinstance(obj, (datetime, date)):
		return obj.isoformat()
	if isinstance(obj, dict):
		return {k: json_safe(v) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [json_safe(v) for v in obj]
	# BigQuery Row and similar should already be converted before calling this
	return obj

def list_datasets(project_id: Optional[str] = None) -> List[str]:
	"""List accessible BigQuery datasets in a project (defaults to client's project)."""
	proj = project_id or bq.project
	return [d.dataset_id for d in bq.list_datasets(project=proj)]

def list_tables(dataset: str, project_id: Optional[str] = None) -> List[str]:
	"""List tables in a dataset."""
	proj = project_id or bq.project
	return [t.table_id for t in bq.list_tables(f"{proj}.{dataset}")]

def get_table_schema(table: str, dataset: str, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
	"""Return table schema (name, type, mode) as JSON."""
	proj = project_id or bq.project
	t = bq.get_table(f"{proj}.{dataset}.{table}")
	return [{"name": f.name, "type": f.field_type, "mode": f.mode} for f in t.schema]

def run_query(sql: str, params: Optional[Dict[str, Any]] = None, dry_run: bool = False) -> Dict[str, Any]:
	lowered = sql.strip().lower()
	if lowered.startswith(("insert", "update", "delete", "merge", "create", "drop", "alter")):
		return {
			"error": "Refusing to run DDL/DML in this agent. SELECT-only allowed.",
			"sql": sql,
		}

	qparams = []
	if params:
		for k, v in params.items():
			if isinstance(v, bool):
				typ = "BOOL"
			elif isinstance(v, int):
				typ = "INT64"
			elif isinstance(v, float):
				typ = "FLOAT64"
			else:
				typ = "STRING"
			qparams.append(bigquery.ScalarQueryParameter(k, typ, v))

	job_config = bigquery.QueryJobConfig(
		dry_run=dry_run,
		use_query_cache=True,
	)
	if qparams:
		job_config.query_parameters = qparams

	try:
		job = bq.query(sql, job_config=job_config)

		if dry_run:
			return json_safe({
				"rows": [],
				"total_rows": 0,
				"slot_ms": job.slot_millis,
				"job_id": job.job_id,
				"cache_hit": getattr(job, "cache_hit", None),
				"sql": sql,
			})

		raw_rows = [dict(r) for r in job.result()]
		return json_safe({
			"rows": raw_rows,
			"total_rows": len(raw_rows),
			"slot_ms": job.slot_millis,
			"job_id": job.job_id,
			"cache_hit": getattr(job, "cache_hit", None),
			"sql": sql,
		})

	except Exception as e:
		return json_safe({
			"error": str(e),
			"sql": sql,
		})

# ---------- Tool declarations for Gemini ----------

bq_tools = gtypes.Tool(function_declarations=[
	{
		"name": "list_datasets",
		"description": "List accessible BigQuery datasets for a project.",
		"parameters": {
			"type": "object",
			"properties": {
	"project_id": {"type": "string", "description": "GCP project ID"},
			}
		},
	},
	{
		"name": "list_tables",
		"description": "List tables in a dataset.",
		"parameters": {
			"type": "object",
			"properties": {
	"dataset": {"type": "string"},
	"project_id": {"type": "string"},
			},
			"required": ["dataset"]
		},
	},
	{
		"name": "get_table_schema",
		"description": "Get schema for a table as JSON.",
		"parameters": {
			"type": "object",
			"properties": {
	"table": {"type": "string"},
	"dataset": {"type": "string"},
	"project_id": {"type": "string"},
			},
			"required": ["table", "dataset"]
		},
	},
	{
		"name": "run_query",
		"description": "Run a SELECT-only BigQuery query with optional named parameters.",
		"parameters": {
			"type": "object",
			"properties": {
	"sql": {"type": "string"},
	"params": {"type": "object"},
	"dry_run": {"type": "boolean"}
			},
			"required": ["sql"]
		},
	},
])



SYSTEM_PROMPT = """You are a data analyst assistant for BigQuery.
- Prefer SELECT-only SQL.
- When missing a column/table name, use list_datasets/list_tables/get_table_schema.
- For final answers, provide a brief natural-language summary, include the SQL you ran in a fenced code block, and include the output from the SQL, styled as an HTML <table>, in a fenced code block.
- The rest of what follows in this prompt are the data schema and hints on how to build the SQL queries for your data analysis.
"""
def dispatch_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
	try:
		if name == "list_datasets":
			return json_safe({"datasets": list_datasets(**args)})
		if name == "list_tables":
			return json_safe({"tables": list_tables(**args)})
		if name == "get_table_schema":
			return json_safe({"schema": get_table_schema(**args)})
		if name == "run_query":
			return json_safe(run_query(**args))
		return {"error": f"Unknown tool: {name}"}
	except Exception as e:
		return {"error": str(e)}

def chat(user_data: list[str],
			history: list[gtypes.Content] | None = None,
			modelTemperature = 0.4,
			modelMode = "AUTO",
			model: str = "gemini-2.5-pro") -> gtypes.GenerateContentResponse:

	# Optional: force the model to use tools when needed (or try AUTO first).
	tool_config_any = gtypes.ToolConfig(
		function_calling_config=gtypes.FunctionCallingConfig(mode="AUTO")
	)

	gen_config = gtypes.GenerateContentConfig(
		tools=[bq_tools],
		tool_config=tool_config_any,
		temperature=modelTemperature,
	)

	user_prompt = user_data[0]
	additional_instructions = user_data[1]
	if history is None:
		history = []

	if not history:
		history.append(
			gtypes.Content(
				role="user",
				parts=[gtypes.Part(text=f"{SYSTEM_PROMPT}\n{additional_instructions}")]
			)
		)

	# Add the new user message
	history.append(
		gtypes.Content(
			role="user",
			parts=[gtypes.Part(text=user_prompt)]
		)
	)

	# First call: model may respond with text + function_call parts
	resp = gclient.models.generate_content(
		model=model,
		contents=history,
		config=gen_config,
	)

	# Collect any tool calls from parts
	tool_calls = []
	for part in resp.candidates[0].content.parts:
		if part.function_call:
			tool_calls.append(part.function_call)

	if not tool_calls:
		# No tools requested → just return; caller can use resp.text safely
		return resp

	# Execute each requested tool
	tool_response_contents: list[gtypes.Content] = []
	for fc in tool_calls:
		result = dispatch_tool(fc.name, dict(fc.args))
		tool_response_contents.append(
			gtypes.Content(
				role="tool",
				parts=[
					gtypes.Part(
						function_response=gtypes.FunctionResponse(
							name=fc.name,
							response=result,
						)
					)
				],
			)
		)

	# Extend history with the model's function_call turn + tool responses
	history.append(resp.candidates[0].content)
	history.extend(tool_response_contents)

	# Second call: model now sees tool outputs and should produce final answer text
	final_resp = gclient.models.generate_content(
		model=model,
		contents=history,
		config=gen_config,
	)

	return final_resp
