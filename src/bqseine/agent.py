from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional

from google import genai
from google.genai import types as gtypes
from google.cloud import bigquery

#from context import sales, stocr, traffic

from decimal import Decimal
from datetime import date, datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
import numpy as np

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
			return {
				"rows": [],
				"total_rows": 0,
				"slot_ms": job.slot_millis,
				"job_id": job.job_id,
				"cache_hit": getattr(job, "cache_hit", None),
				"sql": sql,
			}

		raw_rows = [dict(r) for r in job.result()]
		return {
			"rows": raw_rows,
			"total_rows": len(raw_rows),
			"slot_ms": job.slot_millis,
			"job_id": job.job_id,
			"cache_hit": getattr(job, "cache_hit", None),
			"sql": sql,
		}

	except Exception as e:
		return {
			"error": str(e),
			"sql": sql,
		}

def render_complex_chart(chart_type, title, x_labels=None, datasets=None, z_matrix=None, geo_data=None, icon_config=None):
	"""
	Generates a chart based on the 'Super-Schema' and returns a Base64 image string.
	"""
	plt.figure(figsize=(7, 5))
	plt.title(title, fontsize=12, pad=20)
	
	# --- 1. Basic Charts (Bar, Line, Scatter) ---
	if chart_type in ["bar", "line", "scatter"]:
		# Check if we have comparative datasets or simple x/y
		if datasets:
			for series in datasets:
				label = series.get("label", "Data")
				color = series.get("color", 'black')
				data = series.get("data", [])
				
				if chart_type == "bar":
					# Simple logic for grouped bars would go here; this is a stacked/overlap simplification
					plt.bar(x_labels, data, label=label, alpha=0.7, color=color)
				elif chart_type == "line":
					plt.plot(x_labels, data, label=label, marker='o', color=color)
				elif chart_type == "scatter":
					plt.scatter(x_labels, data, label=label, color=color)
			plt.legend()
		
	# --- 2. Heatmaps (Matrix Data) ---
	elif chart_type == "heatmap" and z_matrix:
		# Convert list-of-lists to DataFrame for easier labeling
		df_matrix = pd.DataFrame(z_matrix)
		if x_labels:
			df_matrix.columns = x_labels[:len(df_matrix.columns)]
		
		sns.heatmap(df_matrix, annot=True, cmap="coolwarm", fmt=".1f")

	# --- 3. Spatial Maps (Geo Data) ---
	elif chart_type == "spatial_map" and geo_data:
		# NOTE: In a restricted sandbox without GeoPandas, we plot Lat/Lon as a Scatter Plot.
		lats = geo_data.get("latitudes", [])
		lons = geo_data.get("longitudes", [])
		names = geo_data.get("location_names", [])
		
		plt.scatter(lons, lats, c='red', marker='x', s=100)
		plt.xlabel("Longitude")
		plt.ylabel("Latitude")
		plt.grid(True, linestyle="--", alpha=0.5)
		
		# Annotate points
		for i, name in enumerate(names):
			if i < len(lats) and i < len(lons):
				plt.annotate(name, (lons[i], lats[i]), xytext=(5, 5), textcoords='offset points')

	# --- Output Handling ---
	plt.tight_layout()
	
	# Save to memory buffer
	buf = io.BytesIO()
	plt.savefig(buf, format='png', dpi=100)
	buf.seek(0)
	plt.close()
	
	# Encode to Base64
	img_str = base64.b64encode(buf.read()).decode('utf-8')
	return img_str



# ---------- Tool declarations for Gemini ----------

code_exec_tool = gtypes.Tool(
    code_execution=gtypes.ToolCodeExecution()
)

bq_tool = gtypes.Tool(
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
        ),
        gtypes.FunctionDeclaration(
            name="render_complex_chart",
            description="Generates a visual chart image based on the provided plotting script. Use this to visualize data.",
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    # 1. The Expanded Enum
                    "chart_type": gtypes.Schema(
                        type=gtypes.Type.STRING,
                        enum=[
                            "bar", "line", "scatter", # Basic
                            "heatmap",                       # Matrix
                            "spatial_map",                   # Geo
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

def sanitize_for_json(obj):
	"""
	Recursively converts Decimal objects to float/int so they are JSON serializable.
	"""
	if isinstance(obj, decimal.Decimal):
		# Convert to int if it's a whole number, else float
		return int(obj) if obj % 1 == 0 else float(obj)
	elif isinstance(obj, list):
		return [sanitize_for_json(i) for i in obj]
	elif isinstance(obj, dict):
		return {k: sanitize_for_json(v) for k, v in obj.items()}
	return obj

SYSTEM_PROMPT = """You are a data analyst assistant for BigQuery.
- Prefer SELECT-only SQL.
- When missing a column/table name, use list_datasets/list_tables/get_table_schema.
- For final answer, provide a brief natural-language summary, include the SQL statement you created in a fenced SQL code block, and include the output from the SQL as an HTML <table> in a fenced HTML code block.
- If user asks for a chart to illustrate the data use render_complex_chart.
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
		if name == "render_complex_chart":
			return {'chart': render_complex_chart(**args)}
		return {"error": f"Unknown tool: {name}"}
	except Exception as e:
		return {"error": str(e)}
	"""
	try:
		returned = {}
		if name == "list_datasets":
			returned = {"datasets": list_datasets(**args)}
		elif name == "list_tables":
			returned = {"tables": list_tables(**args)}
		elif name == "get_table_schema":
			returned = {"schema": get_table_schema(**args)}
		elif name == "run_query":
			returned = run_query(**args)
		elif name == "render_complex_chart":
			returned = render_complex_chart(**args)
		else:
			return {"error": f"Unknown tool: {name}"}
		return sanitize_for_json(returned)
	"""


def chat(user_data: list[str],
			history: list[gtypes.Content] | None = None,
			modelTemperature = 0.4,
			modelMode = "AUTO",
			model: str = "gemini-2.5-pro",
			thinking = 1024) -> gtypes.GenerateContentResponse:

	safety_settings = [
		gtypes.SafetySetting(
		    category="HARM_CATEGORY_DANGEROUS_CONTENT",
		    threshold="BLOCK_NONE"
		),
		gtypes.SafetySetting(
		    category="HARM_CATEGORY_HARASSMENT",
		    threshold="BLOCK_NONE"
		),
	]

	# Optional: force the model to use tools when needed (or try AUTO first).
	tool_config_any = gtypes.ToolConfig(
		function_calling_config=gtypes.FunctionCallingConfig(mode="AUTO")
	)

	gen_config = gtypes.GenerateContentConfig(
		tools=[bq_tool],
		tool_config=tool_config_any,
		safety_settings=safety_settings,
		temperature=modelTemperature,
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
		config=gen_config
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
							id=fc.id
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
