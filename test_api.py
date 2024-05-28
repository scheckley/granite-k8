# Import the libraries
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

# Create a FastAPI app
app = FastAPI()

# Create a class for the input data
class InputData(BaseModel):
    prompt: str

# Create a class for the output data
class OutputData(BaseModel):
    response: str

# Load a local LLM using Hugging Face Transformers with PyTorch backend
# You can change the model name and the task according to your needs
# For example, you can use “t5-base” for summarization or “bert-base-cased” for question answering
model = pipeline("text-generation", model="gpt2", framework="pt")

# Create a route for the web application
@app.post("/generate", response_model=OutputData)
def generate(input_data: InputData):
    # Get the prompt from the input data
    prompt = input_data.prompt

    # Generate a response from the local LLM using the prompt
    response = model(prompt)[0]["generated_text"]

    # Return the response as output data
    return OutputData(response=response)

# If running locally, ensure that this script is executed as the main program
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)