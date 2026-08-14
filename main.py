from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main() -> None:
    information = """
    The 2 facts: im cool your not. Plus your not real.
    """
    summary_prompt = """ 
    Given the following information: {information}\nPlease summerize the following, then give two interesting facts about the text.
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_prompt
    )

    model = ChatOllama(
        model="gemma3:270m",
        temperature=0.5,
    )

    chain = summary_prompt_template | model
    response = chain.invoke(input={"information": information})

    print(response.content)


if __name__ == "__main__":
    main()
