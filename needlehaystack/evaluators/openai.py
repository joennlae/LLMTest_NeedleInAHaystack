import os

from .evaluator import Evaluator

from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI
from openai import AsyncOpenAI

class OpenAIEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens = 300, temperature=0.01)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo-0125",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = None,
                 question_asked: str = None,
                 base_url: str | None = None):

        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked

        api_key = os.getenv('NIAH_EVALUATOR_API_KEY')
        if (not api_key):
            raise ValueError("NIAH_EVALUATOR_API_KEY must be in env for using openai evaluator.")

        self.api_key = api_key
        self.is_mixtral = False
        
        if base_url is not None:
            self.model = AsyncOpenAI(api_key=self.api_key, base_url=base_url)
            self.is_mixtral = True
        else:
            self.evaluator = ChatOpenAI(model=self.model_name,
                                        openai_api_key=self.api_key,
                                        **self.model_kwargs)

    async def do_request(self, prompt: str) -> str:
        """
        Evaluates a given prompt using the OpenAI model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """
        response = await self.model.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                **self.model_kwargs
            )
        return response.choices[0].message.content

    async def evaluate_response(self, response: str) -> int:
        
        if not self.is_mixtral:
            evaluator = load_evaluator(
                "labeled_score_string",
                criteria=self.CRITERIA,
                llm=self.evaluator,
            )

            eval_result = evaluator.evaluate_strings(
                # The models response
                prediction=response,

                # The actual answer
                reference=self.true_answer,

                # The question asked
                input=self.question_asked,
            )

            return int(eval_result['score'])

        else:
            prediction = response
            reference = self.true_answer
            input = self.question_asked
            generated_prompt = [{
                "role": "user",
                "content": f"""
You are a helpful AI bot that answers questions for a user.
## Instructions
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.
Ground truth:
```
{reference}
```

Begin your evaluationby providing a short explanation. Be as objective as possible.
After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
If the ground truth is in the response, please rate it a 10.
So you must provide a rating between 1 and 10, inclusive, in double brackets. 
With the following format:
```
Rating: [[5]]
```

Question:
```
{input}
```


Assistant\'s Answer:
```
{prediction}
```

Rating:
"""
            }]

            rating_text = await self.do_request(generated_prompt)

            import re
            _FIND_DOUBLE_BRACKETS = re.compile(r"\[\[(.*?)\]\]")
            match = _FIND_DOUBLE_BRACKETS.search(rating_text)

            if match:
                verdict = match.group(1)

            if not match or verdict not in list("123456789") + ["10"]:
                print(
                    f"Invalid output: {rating_text}. "
                    "Output must contain a double bracketed string\
                    with the verdict between 1 and 10."
                )
                return 1

            return int(verdict)
