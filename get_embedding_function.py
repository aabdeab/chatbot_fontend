from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import boto3


def get_embedding_function():
    #bedrock_client = boto3.client(service_name='bedrock-runtime', 
                             # region_name='us-east-1')
    #embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                      # client=bedrock_client)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
