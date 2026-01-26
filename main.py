from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model='llama3.2')
template = """
 You are an expert on pizza dining

Here are some relevant review: {reviews}

Here is the question to answer: {question}
"""

def restaurant_llm():
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    # Question and answer loop
    while True:
        question = input ("Enter your question:")
        if question == 'q':
            break

        reviews = retriever.invoke(question)
        result = chain.invoke(
            {
                "reviews": reviews,
                "question": question
            }
        )

        print(result)

def phb_llm():
    phb_template = """
    You are an expert on the Dungeons & Dragons Player's Handbook.
    The user is often incorrect in their assumptions of how rules interact in Dungeons and Dragons,
    and so you are required to evaluate the validity of any assumptions made in a question. If there
    is nothing wrong, then proceed to answer without any caveats or commentary. If something is incorrect,
    then explain the problem with the assumption and provide the correct information.
    Here are some relevant excerpts from the PHB: {excerpts}
    Here is the question to answer: {question}    
    """
    from vectorise_phb import PHB_DB_LOCATION, embed_file
    retriever = embed_file()
    phb_prompt = ChatPromptTemplate.from_template(phb_template)
    chain = phb_prompt | model
    # Question and answer loop
    while True:
        question = input ("Enter your question:")
        if question == 'q':
            break

        excerpts = retriever.invoke(question)
        result = chain.invoke(
            {
                "excerpts": excerpts,
                "question": question
            }
        )

        print(result)



if __name__ == "__main__":
    # restaurant_llm()
    phb_llm()