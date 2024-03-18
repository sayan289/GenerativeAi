import mysql.connector
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import time
import re
from collections import OrderedDict
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import os
from functools import lru_cache
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import threading
import time
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt[0], question])
    return response.text
def convert_to_human_readable(response):
    # Remove brackets and quotes
    response = re.sub(r"[^\w\s,]", "", response)
    # Split the response by commas and extract the first element
    return response.split(',')

def is_primary_key(field_name):
    # Placeholder logic to check if the given field is a primary key
    # Replace this with your actual logic based on your database schema
    primary_keys = ["id", "post_id", "comment_id","name","email","password",...]  # List of primary key field names
    for field_name in primary_keys:
        return field_name
    # return field_name in primary_keys
def read_sql_query(sql, db):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="sayan@289",
        database="blog_app"
    )
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.commit()
    conn.close()
    formatted_data = []
    for row in rows:
        # Convert each row to a string
        formatted_row = ', '.join(map(str, row))
        formatted_data.append(formatted_row)
    response = '\n'.join(formatted_data)
    return response

def execute_sql_query_with_retry(sql_query):
    max_retries = 3
    retry_delay = 3  # seconds

    for attempt in range(max_retries):
        try:
            data = read_sql_query(sql_query)
            return data
        except mysql.connector.errors.OperationalError as e:
            print(f"Attempt {attempt + 1}: Connection failed. Retrying...")
            time.sleep(retry_delay)
    raise Exception("Failed to execute SQL query after multiple attempts")

app = Flask(__name__)
table_schema = None

# Function to fetch all table descriptions from the database
def fetch_all_table_descriptions():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="sayan@289",
        database="blog_app"
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT table_name, column_name, column_comment FROM information_schema.columns WHERE table_schema = 'blog_app';"
    )
    table_descriptions = {}
    for table, column, comment in cur.fetchall():
        if table not in table_descriptions:
            table_descriptions[table] = {}
        table_descriptions[table][column] = comment
    conn.close()
    return table_descriptions

# Function to construct the prompt using the table descriptions
def construct_prompt():
    # global table_schema
    table_descriptions = fetch_all_table_descriptions()
    # table_schema = table_descriptions  # Update the global variable
    prompt_text = str(table_descriptions)
    prompt_text += """
    You are an expert in converting English questions to SQL queries!
    The SQL database 'blog_app' contains the following tables:
    """
    for table_name, table_comment in table_descriptions.items():
        prompt_text += f"\n{table_name}: {table_comment}"
    prompt_text += """
     For example, suppose you want to retrieve the username of the user who made a particular post:
    Example 1- "What is the username of the user who made post with post_id=2"
    The SQL command will be something like this:
    SELECT users.name
    FROM users
    INNER JOIN posts ON users.id = posts.user_id
    WHERE posts.post_id = 2;
    Example 2-"How many entries of categories are present"
     SQL command will be something like this SELECT COUNT(*) FROM categories;
    Example 3- "show is the username of the user who made comment with id=2"
    The SQL command will be something like this:
    SELECT users.name
    FROM users
    INNER JOIN posts ON users.id = comments.user_id
    WHERE comments.id = 2;
    Example 4-"show the user details of id=2"
    The SQL command will be something like this SELECT * FROM users WHERE id=2;
    Example 5-"show the username of id=1"
    The SQL command will be something like this SELECT name FROM users WHERE id=1;
    Example 6-"show the student name of sid=1"
    The SQL command will be something like this SELECT name FROM student WHERE sid=1;
    Your prompt should include similar examples and instructions for generating SQL queries involving inner joins.
    also the sql command should not have ```in the beginning or end and ssql word in the output
    """
    # print("Updated prompt is :",prompt_text)  # Debug message
    return prompt_text

def refresh_schema_and_prompt(interval):
    def refresh():
        while True:
            prompt[0] = construct_prompt()
            fetch_all_table_descriptions()
            time.sleep(interval)

    prompt = [construct_prompt()]
    thread = threading.Thread(target=refresh)
    thread.daemon = True
    thread.start()
    return prompt

@app.route('/')
def index():
    global cached_embeddings
    if cached_embeddings is None:
        # Embed the PDF only if the embeddings are not cached
        os.environ["OPENAI_API_KEY"] = ""
        reader = PdfReader('/home/cbnits-51/Desktop/sayan/sample_project/data/Apache Spark.pdf')
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        # The entire document of the pdf store in raw_text
        text_splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        # CharacterTextSplitter instance is configured to split a longer text into smaller
        # chunks, each separated by a newline character, with a maximum size of 1000 characters per
        # chunk and an overlap of 200 characters between adjacent chunks
        texts = text_splitter.split_text(raw_text)
        # texts consists the chunks of raw text
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        cached_embeddings = docsearch
    return render_template('index.html')
cached_embeddings = None
@app.route('/query', methods=['POST'])
def query():
    try:
        question = request.form['text']
        disallowed_words = ["delete", "update", "create"]
        if any(word in question.lower() for word in disallowed_words):
            return jsonify({"response": "Sorry, performing this operations is not allowed."})
        for operation in ["sum", "average", "maximum", "minimum"]:
            if operation in question.lower():
                field_name = question.split(operation)[1].split()[0].strip()
                if is_primary_key(field_name):
                    return jsonify(
                        {"response": f"Sorry, performing {operation} operation  is not allowed."})
        response = get_gemini_response(question, refresh_schema_and_prompt(2))
        print("Response is ",response)
        data = read_sql_query(response, "blog_app")
        if len(data) == 0:
            chain = load_qa_chain(OpenAI(), chain_type="stuff")
            query = question
            docs = cached_embeddings.similarity_search(query)
            return jsonify({"response": chain.run(input_documents=docs, question=query)})
        else:
            human_readable_response = convert_to_human_readable(str(data))
            return jsonify({"response": human_readable_response})
    except Exception as e:
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        query = question
        docs = cached_embeddings.similarity_search(query)
        return jsonify({"response": chain.run(input_documents=docs, question=query)})
if __name__ == '__main__':
    app.run(debug=True)
