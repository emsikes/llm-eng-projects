from openai import OpenAI
import gradio as gr

import os
from dotenv import load_dotenv
from IPython.display import display, Markdown


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

openai = OpenAI(api_key = openai_api_key)


def print_markdown(text):
    """Better output format"""
    display(Markdown(text))


# define a map of explaination levels
explanation_levels = {
    1: "very simple, as if I am a 5 year old",
    2: "simple, as if I am 10 years old",
    3: "high school level equivalent",
    4: "college student level equivalent",
    5: "expert level in this field",
    6: "PHD level knowledge in this field"
}


def stream_ai_tutor_with_level(user_question, explanation_level_value):
    """
    Sends a question to the OpenAI API and streams a response as a generator

    Args:
        user_question (str): The question asked  by the user
        explanation_level_value (int): The value from the slider bar (1-5)

    Yields:
        str: Chunks of the AI's response 
    """

    # Get the descriptive text from the chosen level, default to clearly and concisely
    level_description = explanation_levels.get(
        explanation_level_value, "clearly and concisely" 
    )

    # Construct the system prompt based on the user question and the explanation level
    system_prompt = f"You are a helpful AI Totor.  Explain the following concept {level_description}"

    print(f"DEUBUG: Using System Prompt: '{system_prompt}'")

    try:
        stream = openai.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}],
            temperature = 0.7,
            stream = True,
        )

        # Empty list we will use to store the full response
        full_response = ""

        # Iterate through each response chunk as its recieved
        for chunk in stream:
            # Verify contents
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                # extract the text from current chunk
                text_chunk = chunk.choices[0].delta.content
                # Append this chunk to the overall response
                full_response += text_chunk
                # Send current state of the response to Gradio UI as it arrives
                yield full_response

    except Exception as e:
        print(f"An error occurred during streaming: {e}")
        yield f"Encoutnered an error: {e}"


# New Gradiop UI with Textbox and Slider Bar
ai_tutor_interface_slider = gr.Interface(
    fn = stream_ai_tutor_with_level,
    inputs = [
        gr.Textbox(lines=5, placeholder="Ask the AI Tutor a question...", label="Your Question"),
        gr.Slider(
            minimum=1,
            maximum=6,
            step=1, # allow only whole number increments
            value=3,
            label="Explanation Level"
        ),
    ],
    outputs = gr.Markdown(label="AI Tutor's Answer", container=True, height=500),
    title = "Advanced AI Tutor",
    description = "Ask a question, and select the desired level of explaination using the slider bar",
    flagging_mode = "never"
)

print("Lanching Addvaced AI Tutor Interface...")
ai_tutor_interface_slider.launch()