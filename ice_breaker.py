from third_parties.linkedin import scrape_linkedin_profile
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import os

if __name__ == "__main__":

    print("Hello Langchain")

    prompt_template_str = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    prompt_template = PromptTemplate(input_variables=["information"], template=prompt_template_str)

    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm = OllamaLLM(model="llama3")

    # LLMChain class is deprecated currently.
    # chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    # print(chain.run(information=information))

    # instead we use  RunnableSequence, e.g., prompt | llm
    chain = prompt_template | llm | StrOutputParser()


    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/yuanliang-li/", # Yuanliang Li's Linkedin profile
        # linkedin_profile_url="https://www.linkedin.com/in/jun-yan-9b2a173a/", # Jun Yan's Linkedin profile
        mock=True
    )

    res = chain.invoke(input={"information": linkedin_data})
    print(res)
