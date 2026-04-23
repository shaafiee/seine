from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from datetime import date, datetime
#from db_lib import *
import json


lastSeineId = {}
tableSchema = {}
tableCurrentSchema = {}
tableCurrentSchemaType = {}
tableReset = {}

#    * EXCEPT(is_generated, generation_expression, is_stored, is_updatable)
tableColumnsQuery = """
SELECT 
    column_name 
  FROM 
    `__dataset__`.INFORMATION_SCHEMA.COLUMNS 
  WHERE 
    table_name = '__table__';
"""


def resolveType(value):
	isDatetime = False
	tempDatetime = None
	try:
		timeDatetime = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
		value = timeDatetime
		isDatetime = True
	except:
		pass

	if isinstance(value, bool):
		return "BOOL"
	elif isinstance(value, int):
		return "INT64"
	elif isinstance(value, float):
		return "FLOAT64"
	elif type(value) == datetime:
		return "DATETIME"
	else:
		return "STRING"


def testValue(value, tableKey, fieldKey):
	global tableCurrentSchemaType
	try:
		timeDatetime = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
		return timeDatetime.strftime('%Y-%m-%dT%H:%M:%S')
	except:
		pass

	fieldType = None
	if fieldKey in tableCurrentSchemaType[tableKey].keys():
		fieldType = tableCurrentSchemaType[tableKey][fieldKey]
	else:
		fieldType = resolveType(value)
		tableCurrentSchemaType[tableKey][fieldKey] = fieldType

	if fieldType == "BOOL":
		if isinstance(value, bool):
			return value
		else:
			return False
	elif fieldType == "INT64":
		if isinstance(value, int):
			return value
		else:
			return 0
	elif fieldType == "FLOAT64":
		if isinstance(value, float):
			return value
		else:
			return 0
	elif fieldType == "STRING":
		if isinstance(value, str):
			return value
		else:
			return ""
	else:
		if isinstance(value, str):
			return value
		else:
			return json.dumps(value)

	return ""


def incrementId(curKey):
	global lastSeineId
	if curKey in lastSeineId.keys():
		lastSeineId[curKey] += 1
	else:
		lastSeineId[curKey] = 1


def dictToBQ(spec: dict) -> list[bigquery.SchemaField]:
    def _field(name: str, node) -> bigquery.SchemaField:
        # REPEATED shorthand: ["TYPE"] or [ {record_spec} ]
        if isinstance(node, list):
            if len(node) != 1:
                raise ValueError(f"{name}: REPEATED shorthand must be a 1-item list, got {node}")
            inner = node[0]
            if isinstance(inner, str):
                return bigquery.SchemaField(name, inner, mode="REPEATED")
            if isinstance(inner, dict):
                # repeated record
                subfields = dictToBQ(inner.get("fields", {}))
                return bigquery.SchemaField(name, "RECORD", mode="REPEATED", fields=subfields)
            raise TypeError(f"{name}: unsupported list inner type: {type(inner)}")

        # Primitive shorthand: "STRING"
        if isinstance(node, str):
            return bigquery.SchemaField(name, node)

        # Full dict form
        if isinstance(node, dict):
            bq_type = node.get("type")
            if not bq_type:
                raise ValueError(f"{name}: missing 'type'")

            mode = node.get("mode", "NULLABLE")
            desc = node.get("description", "")
            sub = node.get("fields")

            if bq_type.upper() == "RECORD":
                if not isinstance(sub, dict):
                    raise ValueError(f"{name}: RECORD requires dict 'fields'")
                subfields = dictToBQ(sub)
                return bigquery.SchemaField(name, "RECORD", mode=mode, fields=subfields, description=desc)

            return bigquery.SchemaField(name, bq_type, mode=mode, description=desc)

        raise TypeError(f"{name}: unsupported node type: {type(node)}")

    return [_field(name, node) for name, node in spec.items()]


def schema(dataset, bqschema):
	bqready = dictToBQ(bqschema)
	client = None
	try:
		client = bigquery.Client(project=myGoogleProject)
	except:
		raise ExceptionType(f"Could not connect to {myGoogleProject}")

	datasetName = "seine"
	try:
		seineDataset = client.get_dataset(myGoogleProject + f".{datasetName}")
	except NotFound:
		seineDataset = bigquery.Dataset(myGoogleProject + f".{datasetName}")
		seineDataset.location = bqRegion
		seineDataset = client.create_dataset(seineDataset, timeout=30)
		print("Created dataset {}".format(seineDataset.dataset_id))

	try:
		table = bigquery.Table(dataset, schema=bqready)
		table = client.create_table(table)
	except:
		pass


def sqlAdd(sqlString, key, value, curTier = 0):
	if isinstance(value, list):
		arrayLit = ''
		if isinstance(, str):
			arrayLit = "'" + "','".join(value) + "'"
		else:
			arrayLit = ",".join(value)
		if curTier == 0:
			sqlString = sqlString + ",[" + arrayLit + "]"
		else:
			sqlString = sqlString + ",[" + arrayLit + "] as " + key
	elif isinstance(value, str):
		if curTier == 0:
			sqlString = f"{sqlString},'{value}'"
		else:
			sqlString = f"{sqlString},'{value}' as {key}"
	else:
		if curTier == 0:
			sqlString = f"{sqlString},{value}"
		else:
			sqlString = f"{sqlString},{value} as {key}"
		
	return sqlString


def sync(myGoogleProject, bqschema, blob, curKey, bqRegion = 'US', firstReset = False, idField = None):
	global lastSeineId
	global tableSchema
	global tableReset
	global tableCurrentSchema
	global tableCurrentSchemaType

	if len(curKey) < 1:
		raise ExceptionType("A default current key (second arg) should be provided")

	initKey = curKey
	stack = []
	curTier = 0
	if isinstance(blob, list):
		for part in blob:
			stack.insert(0, (curKey, part, curKey, 0))
	else:
		stack = [(curKey, blob, curKey, 0)]
		
	tableChecked = False

	client = None
	try:
		client = bigquery.Client(project=myGoogleProject)
	except:
		raise ExceptionType(f"Could not connect to {myGoogleProject}")

	if firstReset:
		lastSeineId = {}
		tableSchema = {}
		tableReset = {}
		tableCurrentSchema = {}
		tableCurrentSchemaType = {}

	keyNotInSchema = {}
	dataToLoad = {}
	seineDataset = False
	dataset = ""
	datasetName = "seine"
	try:
		seineDataset = client.get_dataset(myGoogleProject + f".{datasetName}")
	except NotFound:
		schema(datasetName, bqschema) 
		seineDataset = client.get_dataset(myGoogleProject + f".{datasetName}")
		print("Created dataset {}".format(seineDataset.dataset_id))
	
	valuesArray = []
	currentDepth = 0

	structOpened = 0

	curTier = 0
	lastTier = 0
	sqlString = ''
	keyString = ''

	while stack:
		curKey, curDict, lastKey, curTier = stack.pop()
		currentDepth += 1
		fieldTypes = {}
		fields = []
		fieldsJson = {}
		values = []
		fieldType = {}
		valuePlaceholders = {}
		counter = 0

		#seineDataset = client.get_dataset(myGoogleProject + f".{datasetName}")
		print("-------------------------")
		print(curDict)
		noUpdateNeeded = False

		strTabs = ''
		if curTier <> lastTier:
			tabList = ["\t"] * curTier
			strTabs = ''.join(tabList)
		if curTier > lastTier:
			sqlString = f"{strTabs}{sqlString}STRUCT(\n"
		if curTier < lastTier and curTier != 0:
			sqlString = f"{strTabs}{sqlString})\n"

		lastTier = curTier

		if curTier == 0:
			keyString = f"{keyString},{key}"

		"""
		if curKey not in lastSeineId.keys():
			lastSeineId[curKey] = 1

		if parentId == 0 and curKey == idField and isinstance(curDict, int):
			try:
				curTableName = myGoogleProject + f".{datasetName}." + curKey
				queryJob = client.query(f"select {idField} from {curTableName} where {idField} = {curDict}")
				returned = queryJob.result()
				foundId = False
				for row in returned:
					foundId = True
				if foundId:
					print(f"ID exists {idField} = {curDict}")
					print(returned)
					continue
			except:
				print("================== QUERY FAILED =======================")
				pass

		#if curKey not in keyNotInSchema.keys():
		keyNotInSchema[curKey] = []
		if curKey not in tableCurrentSchema.keys():
			try:
				curTableName = myGoogleProject + f".{datasetName}." + curKey
				queryJob = client.query(f"select max(seine_id) as max_id from {curTableName}")
				returned = queryJob.result()
				print(returned)
				for row in returned:
					if row.max_id is not None and isinstance(row.max_id, int):
						lastSeineId[curKey] = int(row.max_id) + 1
			except:
				pass
			tableCurrentSchema[curKey] = []
			tableCurrentSchemaType[curKey] = {}
		"""
		
		if isinstance(curDict, list):
			if not isinstance(curDict[0], dict):
				"""
				fields.append(curKey)
				fieldTypes[curKey] = resolveType(json.dumps(curDict))
				valuePlaceholders[curKey] = testValue(json.dumps(curDict), curKey, curKey)
				if curKey not in tableCurrentSchema[curKey]:
					tableCurrentSchema[curKey].append(curKey)
					keyNotInSchema[curKey].append(curKey)
					#tableCurrentSchemaType[curKey][curKey] = resolveType(json.dumps(curDict))
				"""
				sqlString = sqlAdd(sqlString, key, value, curTier)
			else:
				noUpdateNeeded = True
				for tempDict in curDict:
					stack.insert(0, (curKey, tempDict, lastKey, curTier + 1))
				continue

		elif isinstance(curDict, dict):
			for key, value in curDict.items():
				if isinstance(value, list):
					if len(value) > 0:
						if not isinstance(value[0], dict) and key not in ["edges"]:
							"""
							fields.append(key)
							fieldTypes[key] = resolveType(json.dumps(value))
							valuePlaceholders[key] = testValue(json.dumps(value), curKey, key)
							if key not in tableCurrentSchema[curKey]:
								tableCurrentSchema[curKey].append(key)
								keyNotInSchema[curKey].append(key)
								#tableCurrentSchemaType[curKey][key] = resolveType(json.dumps(value))
							"""
							sqlString = sqlAdd(sqlString, key, value, curTier)
						elif key in ["edges"]:
							noUpdateNeeded = True
							for part in value:
								stack.insert(0, (curKey, part, lastKey, curTier))
						else:
							for part in value:
								stack.insert(0, (curKey + "_" + key, part, curKey, curTier + 1))
					else:
						"""
						fields.append(key)
						fieldTypes[key] = resolveType(json.dumps(value))
						valuePlaceholders[key] = testValue(json.dumps(value), curKey, key)
						if key not in tableCurrentSchema[curKey]:
							tableCurrentSchema[curKey].append(key)
							keyNotInSchema[curKey].append(key)
							#tableCurrentSchemaType[curKey][key] = resolveType(json.dumps(value))
						"""
						sqlString = sqlAdd(sqlString, key, value, curTier)
				elif isinstance(value, dict):
					if key in ["node"]:
						noUpdateNeeded = True
						stack.insert(0, (curKey, value, lastKey, curTier))
					else:
						stack.insert(0, (curKey + "_" + key, value, curKey, curTier + 1))
						"""
						fields.append(key)
						fieldTypes[key] = resolveType(lastSeineId[curKey])
						valuePlaceholders[key] = testValue(lastSeineId[curKey], curKey, key)
						if key not in tableCurrentSchema[curKey]:
							tableCurrentSchema[curKey].append(key)
							keyNotInSchema[curKey].append(key)
							#tableCurrentSchemaType[curKey][key] = resolveType(json.dumps(value))
						"""
				else:
					"""
					# Add schema
					if key not in fields:
						fields.append(key)
						fieldTypes[key] = resolveType(value)
						valuePlaceholders[key] = testValue(value, curKey, key)
						if key not in tableCurrentSchema[curKey]:
							tableCurrentSchema[curKey].append(key)
							keyNotInSchema[curKey].append(key)
							#tableCurrentSchemaType[curKey][key] = resolveType(value)
					"""
					sqlString = sqlAdd(sqlString, key, value, curTier)

		else:
					sqlString = sqlAdd(sqlString, key, value, curTier)

		"""
		elif isinstance(curDict, str) or isinstance(curDict, int) or isinstance(curDict, float):
			fields.append(curKey)
			fieldTypes[curKey] = resolveType(curDict)
			valuePlaceholders[curKey] = testValue(curDict, curKey, key)
			if curKey not in tableCurrentSchema[curKey]:
				tableCurrentSchema[curKey].append(curKey)
				keyNotInSchema[curKey].append(curKey)
				#tableCurrentSchemaType[curKey][curKey] = resolveType(curDict)

		else:
			fields.append(curKey)
			fieldTypes[curKey] = resolveType(curDict)
			valuePlaceholders[curKey] = testValue(curDict, curKey, key)
			if curKey not in tableCurrentSchema[curKey]:
				tableCurrentSchema[curKey].append(curKey)
				keyNotInSchema[curKey].append(curKey)
				#tableCurrentSchemaType[curKey][curKey] = resolveType(curDict)
		"""

		#if len(fields) < 1 and parentId is None:
		#	continue

		if noUpdateNeeded:
			continue

		"""
		if len(keyNotInSchema[curKey]) > 0 or curKey not in tableReset.keys():
			tableReset[curKey] = True
			curTableName = myGoogleProject + f".{datasetName}." + curKey
			curTable = False
			tableSchema[curKey] = []
			tableSchema[curKey].append(bigquery.SchemaField("seine_id", "INT64"))
			tableSchema[curKey].append(bigquery.SchemaField("seine_parent_id", "INT64"))
			tableSchema[curKey].append(bigquery.SchemaField("injected", "DATETIME"))
			tableCurrentSchema[curKey].append("seine_id");
			tableCurrentSchema[curKey].append("seine_parent_id");
			tableCurrentSchema[curKey].append("injected");
			try:
				curTable = client.get_table(curTableName)
				#colQuery = tableColumnsQuery.replace("__dataset__", seineDataset).replace("__table__", key)
				#queryJob = client.query(colQuery)
				#returned = queryJob.result()
				existingSchema = curTable.schema
				tableSchema[curKey] = existingSchema
				existingColumns = []
				jobCconfig = bigquery.QueryJobConfig(
					destination=curTableName,
					schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
					write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
				)
				for schemaElement in existingSchema: 
					existingColumns.append(schemaElement.name)
				schemaAdjusted = False
				for fidx, field in enumerate(fields):
					if field not in existingColumns:
						if field not in tableCurrentSchema[curKey]:
							tableCurrentSchema[curKey].append(field);
						tableSchema[curKey].append(bigquery.SchemaField(field, fieldTypes[field]))
						existingSchema.append(bigquery.SchemaField(field, fieldTypes[field]))
						if not schemaAdjusted:
							schemaAdjusted = True
				if schemaAdjusted:
					curTable.schema = existingSchema
					try:
						curTable = client.update_table(curTable, ["schema"])
					except:
						pass

				#if returned.total_rows > 0:
				#	counter = 0
				#	for row in returned:
				#		print(row)
				#		counter += 1
				#		if counter < 2:
				#			continue
				#		if row.max_id + 1 > lastSeineId[curKey]:
				#			lastSeineId[curKey] = int(row.max_id) + 1
				#			print(f"{curKey}" + str(lastSeineId[curKey]))
			except NotFound:
				for fidx, field in enumerate(fields):
					tableSchema[curKey].append(bigquery.SchemaField(field, fieldTypes[field]))
					if field not in tableCurrentSchema[curKey]:
						tableCurrentSchema[curKey].append(field);
				curTable = bigquery.Table(curTableName, schema=tableSchema[curKey])
				curTable = client.create_table(curTable)
				lastSeineId[curKey] = 1
			keyNotInSchema[curKey] = []

		if curKey not in dataToLoad.keys():
			dataToLoad[curKey] = []
		tempRow = {
			"seine_id": lastSeineId[curKey],
			"seine_parent_id": parentId,
			"injected": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
		}
		"""

		for idx, field in enumerate(fields):
			tempRow[field] = valuePlaceholders[field]
		print(tempRow)
		dataToLoad[curKey].append(tempRow)
		lastSeineId[curKey] += 1

	for tableName in dataToLoad.keys():
		curTable = client.get_table(myGoogleProject + f".{datasetName}." + tableName)
		errors = client.insert_rows_json(
			curTable, dataToLoad[tableName], row_ids=[None] * len(dataToLoad[tableName])
		)
		if errors == []:
			print("Loaded " + str(len(dataToLoad[tableName])) + " rows into " + tableName)
		else:
			print("FAILED: loading " + str(len(dataToLoad[tableName])) + " rows into " + tableName)
			print(errors)
			
