from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model='llama3.2')
template = """
 You are an expert on pizza dining

Here are some relevant review: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Question and answer loop
while True:
    question = input ("Enter your question:")
    if question == 'q':
        break

    result = chain.invoke(
        {
            "reviews": [],
            "question": question
        }
    )

    print(result)
