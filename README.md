# BQSeine
## Bigquery AI agent and data loader
Seine is a data loader that pushes data in a dictionary to BigQuery in relational normalized form. Seine also has functionality to use Gemini to get data insights from Bigquery.

## Usage

### AI insights
```
from bqseine.agent import chat
response = chat(question, context)
```

In making the call to chat(*question*, *context*, *temperature*, *mode*, *model*), where:
- *context* argument explains to the LLM how the tables in Bigquery and their fields refer to each other and what they mean
- *temperature* (set by default to 0.4) is how creative the model is allowed to get with 2 being max
- *mode* (set by default to "AUTO") is the tool mode
- *model* (set by default to "gemini-2.5-pro") is the AI model to be used

To run, you will need the following environment variables:
- GOOGLE_API_KEY, storing your Google AI Studio API key
- GOOGLE_CLOUD_PROJECT, storing your Google Cloud Project name
- GOOGLE_APPLICATION_CREDENTIALS, that points to a credential with which you can access Bigquery

### Data loading
```
from bqseine.polyp import sync
sourceData = [
	{
		'item': 'Juice',
		'price': 20.0,
		'stock': [
			{
				'batch': '2025-01-20',
				'qty': 300
			},
			{
				'batch': '2025-02-02',
				'qty': 50
			}
		]
	},
	{
		'item': 'Burger',
		'price': 30.0,
		'stock': [
			{
				'batch': '2025-02-10',
				'qty': 200
			}
		]
	}
]
sync('someGoogleProject', sourceData, 'catalog', 'US')
### The arguments above are:
### sync(<Google project name>, <dict>, <main table name>, <BigQuery region>)*
```

The above example will generate the following tables in BigQuery:
#### catalog
| seine_id | seine_parent_id | item | price | injected |
| --- | --- | --- | --- | --- |
| 1 | 0 | 'Juice' | 20.0 | now() |
| 2 | 0 | 'Burger' | 30.0 | now() |

#### catalog_stock
| seine_id | seine_parent_id | batch | qty | injected |
| --- | --- | --- | --- | --- |
| 1 | 1 | '2025-01-20' | 300 | now() |
| 2 | 1 | '2025-02-02' | 50 | now() |
| 3 | 2 | '2025-02-10' | 200 | now() |
