# Documents Q&A PoC

## Overview

This project is a conversational AI system built using Python and streamlit. It leverages OpenAI's GPT-3.5-turbo model for language understanding and generation, and
Pinecone, a vector database for machine learning.                                                                                                                       

The system is designed to process and understand queries from users, retrieve relevant information from a database, and generate appropriate responses. It is capable of
handling complex queries and returning concise, relevant answers.                                                                                                       

The project is structured into several key components:                                                                                                                  

 1. `app.py`: This is the main entry point of the application. It handles user input and initializes the application.                                                      
 2. `auto_trace.py`: This is a utility module for tracing function calls, which can be useful for debugging or performance monitoring.                                     
 3. `src/custom_object_retriever.py`: This module defines classes for retrieving custom objects, possibly from a database or an external service.                          
 4. `src/document_processor.py`: This module is responsible for processing documents, possibly to extract information or prepare them for further processing.              
 5. `src/pincone_manager.py`: This module manages interactions with Pinecone, a vector database for machine learning.                                                      
 6. `src/query_manager.py`: This module manages queries to the system, interacting with the document processor and the Pinecone manager.                                   

## Deployment

### Docker Commands for `gptserve` Container Registry

This document provides instructions on how to log into the `gptserve` Azure Container Registry (ACR) and interact with the `chatgpt_gxp` repository.

### Resources

- Container Registry: `gptserve`
- Repository: `chatgpt_gxp`
- WebApp: `gptgxp`
- Resource Group: `gptserve-rg`

### Logging into the Registry

You can log into the ACR using either the Azure CLI or Docker. Here are the commands for both:

- Azure CLI: `az acr login --name gptserve`
- Docker: `docker login gptserve.azurecr.io`

You will need to use the username and password associated with the `gptserve` ACR.

### Retrieving Credentials

#### Via Azure Portal

1. Navigate to your Azure Container Registry resource.
2. Click on "Access keys" in the left-hand menu.
3. Here you'll find the "Login server", "Username", and two passwords (password and password2). You can use either of the two passwords.

#### Via Azure CLI

If you have the Azure CLI installed, you can retrieve the credentials using the following command:

`az acr credential show --name gptserve`