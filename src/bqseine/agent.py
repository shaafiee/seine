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

code_exec_tool = gtypes.Tool(
    code_execution=gtypes.ToolCodeExecution()
)

# 2. Define the BigQuery Function Tool using proper types
bq_functions_tool = gtypes.Tool(
    function_declarations=[
        gtypes.FunctionDeclaration(
            name="list_datasets",
            description="List accessible BigQuery datasets for a project.",
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "project_id": gtypes.Schema(
                        type=gtypes.Type.STRING, 
                        description="GCP project ID"
                    ),
                },
            ),
        ),
        gtypes.FunctionDeclaration(
            name="list_tables",
            description="List tables in a dataset.",
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "dataset": gtypes.Schema(type=gtypes.Type.STRING),
                    "project_id": gtypes.Schema(type=gtypes.Type.STRING),
                },
                required=["dataset"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="get_table_schema",
            description="Get schema for a table as JSON.",
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "table": gtypes.Schema(type=gtypes.Type.STRING),
                    "dataset": gtypes.Schema(type=gtypes.Type.STRING),
                    "project_id": gtypes.Schema(type=gtypes.Type.STRING),
                },
                required=["table", "dataset"],
            ),
        ),
        gtypes.FunctionDeclaration(
            name="run_query",
            description="Run a SELECT-only BigQuery query with optional named parameters.",
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "sql": gtypes.Schema(type=gtypes.Type.STRING),
                    "params": gtypes.Schema(type=gtypes.Type.OBJECT),
                    "dry_run": gtypes.Schema(type=gtypes.Type.BOOLEAN),
                },
                required=["sql"],
            ),
        )
    ]
)

chart_tool = gtypes.Tool(
    function_declarations=[
        gtypes.FunctionDeclaration(
            name="render_complex_chart",
            description="Generates advanced visualizations including maps, heatmaps, and comparative graphs.",
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    # 1. The Expanded Enum
                    "chart_type": gtypes.Schema(
                        type=gtypes.Type.STRING,
                        enum=[
                            "bar", "line", "pie", "scatter", # Basic
                            "heatmap",                       # Matrix
                            "pictograph",                    # Icon-based
                            "spatial_map",                   # Geo
                            "grouped_bar", "stacked_area"    # Comparative
                        ],
                        description="The specific type of visualization to render."
                    ),
                    "title": gtypes.Schema(type=gtypes.Type.STRING),
                    
                    # 2. Standard Data (Simple X/Y)
                    "x_labels": gtypes.Schema(
                        type=gtypes.Type.ARRAY,
                        items=gtypes.Schema(type=gtypes.Type.STRING),
                        description="Labels for the X axis or Categories."
                    ),
                    
                    # 3. Comparative / Multi-Series Data
                    # We use a list of objects for multiple datasets (e.g., '2023 Sales', '2024 Sales')
                    "datasets": gtypes.Schema(
                        type=gtypes.Type.ARRAY,
                        description="List of data series for comparative graphs.",
                        items=gtypes.Schema(
                            type=gtypes.Type.OBJECT,
                            properties={
                                "label": gtypes.Schema(type=gtypes.Type.STRING, description="Name of the series (e.g., 'Revenue')"),
                                "data": gtypes.Schema(type=gtypes.Type.ARRAY, items=gtypes.Schema(type=gtypes.Type.NUMBER)),
                                "color": gtypes.Schema(type=gtypes.Type.STRING, description="Optional hex color.")
                            }
                        )
                    ),

                    # 4. Heatmap Specifics (The Z-Matrix)
                    "z_matrix": gtypes.Schema(
                        type=gtypes.Type.ARRAY,
                        description="2D array of numbers for heatmaps.",
                        items=gtypes.Schema(
                            type=gtypes.Type.ARRAY,
                            items=gtypes.Schema(type=gtypes.Type.NUMBER)
                        )
                    ),

                    # 5. Spatial/Map Specifics
                    "geo_data": gtypes.Schema(
                        type=gtypes.Type.OBJECT,
                        properties={
                            "latitudes": gtypes.Schema(type=gtypes.Type.ARRAY, items=gtypes.Schema(type=gtypes.Type.NUMBER)),
                            "longitudes": gtypes.Schema(type=gtypes.Type.ARRAY, items=gtypes.Schema(type=gtypes.Type.NUMBER)),
                            "location_names": gtypes.Schema(type=gtypes.Type.ARRAY, items=gtypes.Schema(type=gtypes.Type.STRING), description="Country or City names for Choropleths"),
                        }
                    ),

                    # 6. Pictograph Specifics
                    "icon_config": gtypes.Schema(
                        type=gtypes.Type.OBJECT,
                        properties={
                            "icon_name": gtypes.Schema(type=gtypes.Type.STRING, description="e.g., 'person', 'car', 'house'"),
                            "items_per_icon": gtypes.Schema(type=gtypes.Type.NUMBER, description="Scale, e.g., 1 icon = 100 units")
                        }
                    )
                },
                required=["chart_type", "title"]
            )
        )
    ]
)

SYSTEM_PROMPT = """You are a data analyst assistant for BigQuery.
- Prefer SELECT-only SQL.
- When missing a column/table name, use list_datasets/list_tables/get_table_schema.
- For final answers, provide a brief natural-language summary, include the SQL you ran in a fenced code block, and include the output from the SQL, styled as an HTML <table>, in a fenced code block.
- If the user asks for a chart illustrating the data, then generate it in JPEG format and add it to the end of the answer in BASE64 format as a fenced JPEG code block. Spend no more than 45 seconds to do this and if it takes longer then skip this request.
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
			model: str = "gemini-2.5-pro",
			thinking = 1024) -> gtypes.GenerateContentResponse:

	# Optional: force the model to use tools when needed (or try AUTO first).
	tool_config_any = gtypes.ToolConfig(
		function_calling_config=gtypes.FunctionCallingConfig(mode="AUTO")
	)

	gen_config = gtypes.GenerateContentConfig(
		tools=[bq_functions_tool, chart_tool],
		tool_config=tool_config_any,
		temperature=modelTemperature
		thinking_config=gtypes.ThinkingConfig(thinking_budget=thinking) 
	)

	user_prompt = user_data[0]
	additional_instructions = user_data[1]
	if history is None:
		history = []

	"""
	if not history:
		history.append(
			gtypes.Content(
				role="user",
				parts=[gtypes.Part(text=f"{SYSTEM_PROMPT}\n{additional_instructions}")]
			)
		)
	"""

	# Add the new user message
	history.append(
		gtypes.Content(
			role="user",
			parts=[gtypes.Part(text=user_prompt)]
		)
	)

	history.insert(
		0,
		gtypes.Content(
			role="user",
			parts=[gtypes.Part(text=f"{SYSTEM_PROMPT}\n{additional_instructions}")]
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
