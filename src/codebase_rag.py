from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from git import Repo
import os
import tempfile
from openai import OpenAI
from langchain.schema import Document
import subprocess
from dotenv import load_dotenv
import time

load_dotenv()

class CodebaseRAG:

    SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.java', '.ts'}

    IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git', '__pycache__', '.next', '.vscode', 'vendor'}

    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        os.environ['PINECONE_API_KEY'] = self.pinecone_api_key

        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.pinecone_index = self.pc.Index("codebase-rag")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
    def clone_repository(self, repo_url):
        repo_name = repo_url.split('/')[-1]
        repo_path = f'/Users/zahyoo/Desktop/Headstarter/SecureAgent/content/{repo_name}'
        Repo.clone_from(repo_url, str(repo_path))
        return str(repo_path)

    def get_file_content(self, file_path, repo_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            rel_path = os.path.relpath(file_path, repo_path)

            return {
                "name": rel_path, 
                "content": content
            }
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return None

    def get_main_files_content(self, repo_path):
        files_content = []

        try:
            for root, _, files in os.walk(repo_path):
                if any(ignored_dir in root for ignored_dir in CodebaseRAG.IGNORED_DIRS):
                    continue

                for file in files:
                    file_path = os.path.join(root, file)

                    if os.path.splitext(file)[1] in CodebaseRAG.SUPPORTED_EXTENSIONS:
                        file_content = self.get_file_content(file_path, repo_path)
                        if file_content:
                            files_content.append(file_content)

        except Exception as e:
            print(f"Error reading repository: {str(e)}")

        return files_content

    def get_huggingface_embeddings(self, text, model_name="sentence-transformers/all-mpnet-base-v2"):
        model = SentenceTransformer(model_name)
        return model.encode(text)

    def perform_rag(self, query):
        raw_query_embedding = self.get_huggingface_embeddings(query)

        top_matches = self.pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace="https://github.com/desire04/SecureAgent")

        contexts = [item['metadata']['text'] for item in top_matches['matches']]

        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

        system_prompt = f"""You are a Senior Software Engineer, specializing in TypeScript. 

        Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
        """

        llm_response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ], 
            stream=True
        )

        return llm_response
