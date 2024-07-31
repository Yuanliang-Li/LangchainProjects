# LangChain Learning Note

## Tutorial: Udemy - Develop LLM powered applications with LangChain
	
### Section 2: The GIST of LangChain- Get started by with your "Hello World" chain


#### 5. Project Setup (Pycharm) recommend)
- How to configure pipenv in Pycharm for MacOS:
    - https://www.jetbrains.com/help/pycharm/pipenv.html
    - We need make sure global pipenv executable can be found, which means we should install pipenv using pip globally.


#### 8. Your First LangChain application - Chaining a simple prompt
- code
    ```python
    from langchain_core.output_parsers import StrOutputParser
    from langchain.prompts.prompt import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain.chains import LLMChain

    if __name__ == "__main__":
        # load_dotenv() # load environmental variables from .env
        print("Hello Langchain")

        information = """
            Elon Reeve Musk (/ˈiːlɒn/; EE-lon; born June 28, 1971) is a businessman and investor. He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect and former chairman of Tesla, Inc.; owner, chairman and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. He is the wealthiest person in the world, with an estimated net worth of US$232 billion as of December 2023, according to the Bloomberg Billionaires Index, and $254 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.[5][6]
        """

        summary_template = """
        given the information {information} about a person I want you to create:
        1. A short summary
        2. two interesting facts about them
        """

        summary_prompt_template = PromptTemplate(
            input_variables=["information"], template=summary_template
        )

        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

        # LLMChain class is deprecated.
        # chain = LLMChain(llm=llm, prompt=summary_prompt_template)
        # print(chain.run(information=information))

        # instead we use  RunnableSequence, e.g., prompt | llm
        chain = summary_prompt_template | llm | StrOutputParser()
        res = chain.invoke(input={"information": information})
        print(res)
    ```

#### Learned API
```python
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
```

