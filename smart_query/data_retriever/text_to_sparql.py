import time
import re
import sys
import json
import requests
import chromadb
import logging
import os
from dotenv import load_dotenv
import google.generativeai as genai

from json import JSONDecodeError
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from groq import Groq
from datetime import datetime
from openai import OpenAI
from datetime import datetime

load_dotenv()

logging.basicConfig(level=logging.ERROR)
client = chromadb.PersistentClient(path=os.getenv("WENOKN_VECTOR_DB"))
collection = client.get_collection(name="knowledge")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

n_results = 10

# get relevant concepts for the request
def get_relevant_concepts(query_text):
    n_results = 20
    results = collection.query(
        query_embeddings=[embed_model.get_text_embedding(query_text)],
        n_results=n_results,
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return documents, metadatas


# get description of the relevant concepts
def get_description(concepts, metadatas, documents):
    description = ""
    for i in range(len(concepts)):
        if concepts[i]["is_relevant"]:
            try:
                index = documents.index(concepts[i]["entity"])
                description = f"""{description}
               
{ metadatas[index]['def'] }

                """
            except:
                continue
    return description


# create the sparql query request
def sparql_request(query_text, description):

    example = """

# Must include the following PREFIX
PREFIX schema: <https://schema.org/>  
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX unit: <http://www.opengis.net/def/uom/OGC/1.0/>

SELECT ?buildingName ?buildingDescription ?buildingGeometry ?distanceInMeters
WHERE {
  # Get power station with name "dpq5d2851w52"
  ?powerStation schema:additionalType "power";
                schema:name "dpq5d2851w52";
                schema:geo/geo:asWKT ?powerStationGeometry.

  # Find buildings
  ?building schema:additionalType "building";
           schema:name ?buildingName;
           schema:geo/geo:asWKT ?buildingGeometry.
  
  # Calculate distance between building and power station in meters 
  BIND(geof:distance(?buildingGeometry, ?powerStationGeometry, unit:metre) AS ?distanceInMeters)
  
  # Filter buildings within 5 miles (1 mile = 1609.34 meters)
  FILTER(?distanceInMeters <= 5 * 1609.34)
} LIMIT 10
    """

    example2 = """

PREFIX schema: <https://schema.org/>                                        
PREFIX geo: <http://www.opengis.net/ont/geosparql#>                          
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>               
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>                        
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/> 

# Keep the following SELECT variables exactly as they are 
SELECT ?buildingName ?buildingGeometry ?buildingFIPS {    
    
     GRAPH <http://kwg.org> {
        ?county a kwg-ont:AdministrativeRegion_2 ;
                rdfs:label ?countyName ;
                geo:hasGeometry/geo:asWKT ?countyGeometry .

        # use this FILTER for th county name. 
        FILTER(CONTAINS(LCASE(?countyName), LCASE("Ross County"))) .

        # use this FILTER for the state name. Ignore this filter if no state name is provided. 
        FILTER(CONTAINS(LCASE(?countyName), LCASE(", Ohio"))) .
    }
    
    # Calculate the bounding box of ?countyGeometry
    BIND(geof:envelope(?countyGeometry) AS ?envelope) . 
    
    ?building schema:additionalType "building" ;                            
              schema:name ?buildingName ;                                 
              schema:geo/geo:asWKT ?buildingGeometry ;                   
              schema:identifier ?buildingGeoid .             
    ?buildingGeoid schema:name "GEOID" ;                                    
                   schema:value ?buildingFIPS .                                                      

     # Check if the building box contains ?building. This makes the query faster
     FILTER (geof:sfContains(?envelope, ?buildingGeometry)) . 
    
     # Check if the county contains ?building. 
     FILTER (geof:sfContains(?countyGeometry, ?buildingGeometry)) . 
}                                                                                                                                                                             
LIMIT 10    

"""

    return f"""

Use the following prefixes:

PREFIX schema: <https://schema.org/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX unit: <http://www.opengis.net/def/uom/OGC/1.0/>
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX time: <http://www.w3.org/2006/time#>
PREFIX sosa: <http://www.w3.org/ns/sosa/>
PREFIX ext: <http://rdf.useekm.com/ext#>    # Don't miss this
PREFIX hyf: <https://www.opengis.net/def/schema/hy_features/hyf/>

{description}

[ Example 1: distance function ]

The distance function is defined by geof:distance.
The meter unit is defined as unit:metre. 
1 mile = 1609.34 meters .
Here is an example to calculate a distance
    BIND(geof:distance(?location1, ?location2, unit:metre) AS ?distance_m) .
Please make sure that ?location1 and ?location2 are both geometries when use this distance function.
Here is an example how the distance function is used:

{ example }

[ Example 2: sfContains function ]

The sfContains function can be used to determine if a geometry is inside another geometry.
For example, to find 10 buildings in Ross county, you can use this query:

{ example2 }

[ Note 1: BIND ] 

Very Very Important: All BIND statements must be inside SELECT. This is a mistaken you often made.
BIND is illegal Before SELECT. This is a mistaken you often made.
BIND(something AS ?x) CAN ONLY APPEAR ONCE for specific x! The second one is unnecessary.

NEVER USE SELECT * WHERE {...}. Instead, enumerate all the attributes in the SELECT clause.
Don't use any other knowledge more than what I told you. Don't assume anything.

If no attributes of the required entity are mentioned, then return all the attributes of this entity
in SELECT clause. For example, request for buildings without mentioning attributes, then return 
the building names, building geometries and FIPS.

ONLY use simple variables in a SELECT clause.
ONLY use the objects and attributes listed above. 

Make sure SELECT variables always include a variable for geometries. 
For example, if the request is asking for counties, then always include ?countyGeometry in the SELECT variables. 
If the request is asking for buildings, then always include ?buildingGeometry in the SELECT variables.

Use all the provided prefix declarations but remove duplicates.

Please create a SPARQL query using SELECT DISTINCT with PREFIX declarations for this request:

{query_text}

Do your best to use a provided OPTIMIZED query if it can be applied. Must follow the instructions given in the comments of the examples.

Your created SPARQL has to include all PREFIX that are used in the query. It is wrong if your query doesn't have any PREFIX.

Put spatial functions and predicts like geof:sfIntersects, geof:stContains and geof:distance out of SERVICE. 

You must add your comments starting with "# LLM comment:" to your created query. For example, if a name is included in the query, 
then you must have a comment like this:
    # LLM comment: Here we check the name. 
If the request is asking to find objects A inside another object B, then you must have a comment like this: 
    # LLM comment: Here we check A is contained in B
Without a comment starting with "LLM comment:", your created query will be rejected and you have to redo it.

If the user asks for a specific number of entities, then use LIMIT with the number the user asks for.
If the user asks for gages, then use "LIMIT 1000".
If the user asks for all buildings or all counties or all rivers or all dams, etc. then use "LIMIT 200".
If the user's request didn't mention a number or "all",  then use "LIMIT 200".

The provided query instances must be used in preference. 

No square brackets around entities. 
Very Very Important: BIND must be inside the WHERE clause of the SELECT statement.
LIMIT must be outside of the right bracket of SELECT.
Substring checking can use CONTAINS.
Do not use "ORDER BY" if the request did not ask.
Do not include duplicate PREFIX.

    """


safe = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


genai.configure(api_key=os.getenv("GOOGLE_AI_KEY"))

model = genai.GenerativeModel("models/gemini-2.0-flash")


def extract_code_blocks(text):
    # Define the pattern to match text between ```
    pattern = r"```([\s\S]*?)```"
    # Use re.findall to find all matches
    code_blocks = re.findall(pattern, text)
    return code_blocks


# Initialize a counter to keep track of the last used token
counter = 0
tokens = [
    "gsk_w",
    "gsk_T",
    "gsk_U",
]


def get_candidate_concepts(query_text: str):

    documents, metadatas = get_relevant_concepts(query_text)

    check_request = f"""
we got this SPARQL request:

    {query_text}

We are enforced to use the following entities in our GraphDB without any additional knowledge:

    { ', '.join(documents) }

The distance function can only be applied to two geometries and a unit is required. 

An industrial building is a building.

What entities in this list are possible necessary to solve the request?

Please return your answer as a JSON string as a list of the objects with two fields:
a string field "entity" and a boolean field "is_relevant".
"""

    # print('='*70)
    # print('Request to LLM for check relevance')
    # print('-'*70)
    # print(check_request)

    max_tries = 10
    current_try = 0
    while current_try < max_tries:
        try:
            response = model.generate_content(check_request, safety_settings=safe)
            break
        except Exception as e:
            print(e)
            time.sleep(1)
            current_try += 1

    # print('='*70)
    # print('Response from LLM relevance checking')
    # print('-'*70)
    # print(response.text)

    response_text = response.text
    if (
        response_text.startswith("```json")
        or response_text.startswith("```JSON")
        or response_text.startswith("```")
    ):
        json_part = response_text.split("\n", 1)[1].rsplit("\n", 1)[0]
        concepts = json.loads(json_part)
    else:
        concepts = json.loads(response_text)

    description = get_description(concepts, metadatas, documents)
    request_to_sparql = sparql_request(query_text, description)
    # response = model.generate_content(request_to_sparql, safety_settings=safe)

    global counter
    token = tokens[counter % len(tokens)]
    counter += 1

    client2 = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    chat_completion2 = client2.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": request_to_sparql}]
    )
    result = chat_completion2.choices[0].message.content

    try:
        result = extract_code_blocks(result)[0]
    except:
        pass

    if result.startswith("sparql"):
        result = result[6:]

    old_substring = "\\\\"
    new_substring = "\\"
    result = result.replace(old_substring, new_substring)

    result = result.replace(
        'BIND(REPLACE(?string, "^.*\\(\\(.*\\)* \\((.*)\\)\\)$", "$1") AS ?substring) .',
        'BIND(REPLACE(?string, "^.*\\\\(\\\\(.*\\\\)* \\\\((.*)\\\\)\\\\)$", "$1") AS ?substring) .',
    )

    result = result.replace(
        'BIND(REPLACE(?string, "^.*\\(\\(.*\\)*, \\((.*)\\)\\)$", "$1") AS ?substring)',
        'BIND(REPLACE(?string, "^.*\\\\(\\\\(.*\\\\)*, \\\\((.*)\\\\)\\\\)$", "$1") AS ?substring)',
    )

    return result
