from openai import OpenAI

from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

if openai_api_key:
    print(openai_api_key[:10])

DATA_FILE_PATH = input("Enter the full path to the file you want to chat with: ")

loader = TextLoader(DATA_FILE_PATH, encoding="utf-8")

raw_documents = loader.load()

print(f"Document Loaded: {len(raw_documents)}")

print(raw_documents[0].page_content[:500] + "...")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 150,)

try:
    documents = text_splitter.split_documents(raw_documents)
    print(f"Document split into: {len(documents)} chunks")
except Exception as e:
    raise ValueError("Error splitting document: {e}")

# Created vector store and input documents
vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)

vector_count = vector_store._collection.count()
print(f"Chroma vectorstore created with {vector_count} items.")

# Retrieve the first chunk of stored data from the vector store
stored_data = vector_store.get(include=["embeddings", "documents"], limit=1)

print("First chunk:\n", stored_data['documents'][0])
print("\nEmbedding vector\n", stored_data['embeddings'][0])
print(f"\nFull embedding dimensions: {len(stored_data['embeddings'][0])}")

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    verbose=True
)

def ask_elevenmadison_assistant(user_query):
    """
    Process the user query using the RAG chain and return formatted results
    """
    print(f"\nProcessing Gradio query: '{user_query}")
    if not user_query or user_query.strip() == "":
        print("--> Empty query recieved.")
        return "Please enter a question.", "" # Handle empty input gracefully
    
    try:
        result = qa_chain.invoke({"question": user_query})

        # Extract the answer and sources
        answer = result.get("answer", "Sorry, I couldn't find an answer in the provided documents.")
        sources = result.get("sources", "No specific sources identified.")

        if sources == DATA_FILE_PATH:
            sources = f"Retrieved from: {DATA_FILE_PATH}"
        elif isinstance(sources, list): # Handle potential list of sources
            sources = ", ".join(list(set(sources))) # Unique, comma-separated

        print(f"--> Answer generated: {answer[:100].strip()}...")
        print(f"--> Sources identified: {sources}")

        # Return the answer and sources to be displayed with Gradio output compontents
        return answer.strip(), sources

    except Exception as e:
        error_message = f"An error occurred: {e}"
        print("--> Error during chain execution: {error_message}")
        return error_message, "Error occurred"
    
# Create the gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Eleven Madison Park Q&A Assistant") as demo:
    gr.Markdown(
    """
    # Eleven Madison Park - AI Q&A Assistant
    Ask questions about the restaurant based on its website data.
    The AI provides answers and cites the source document.
    *(Examples: What are the menu prices?  Who is the chef? Is it plant-based?)*
    """
    )

    # Input  component for the user's question
    question_input = gr.Textbox(
        label = "Your Question:",
        placeholder = "e.g., What are the operating hours on Saturday?",
        lines = 2, # Allow space for longer questions
    )

    # Row layout for the output
    with gr.Row():
        # Output component for the generated answer (read only)
        answer_output = gr.Textbox(label="Answer:", interactive=False, lines=6) # User cannot edit this
        # Output component for the souce (read only)
        sources_output = gr.Textbox(label="Sources:", interactive=False, lines=2)

    # Row for buttons
    with gr.Row():
        # Button for submitting question
        submit_button = gr.Button("Ask RAG Q&A Chat App", variant="primary")
        # Clear button to reset input and output values
        clear_button = gr.ClearButton(components=[question_input, answer_output, sources_output], value="Clear All")

    # Add some example questions for users to try
    gr.Examples(
        examples=[
            "What are the different menu options and prices?",
            "Who is the head chef?",
            "What is Magic Farms?"
        ],
        inputs = question_input, # Clicking example will load this input
        cache_examples = False, # Don't pre-compute results for examples for simplicity
    )

    # Connect the submit button to the Function
    submit_button.click(fn=ask_elevenmadison_assistant, inputs=question_input, outputs=[answer_output, sources_output])

print("Gradio interface defined")

# Launch the Gradio app
demo.launch()

