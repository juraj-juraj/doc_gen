import ast
import openai

_api_key = "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"


def gpt_docstring_generator(api_key: str) -> str | None:
    """
    _summary_

    Args:
        k (int): _description
        node (DLL_node_t, optional): _description_. Defaults to None.
    Returns:
        DLL_node_t: _description

    """
    gpt_client = openai.OpenAI(api_key=api_key)

    def wrapper(prompt: ast.stmt) -> str | None:
        """
        _summary_

        Args:
            k (int): _description
            node (DLL_node_t, optional): _description_. Defaults to None.
        Returns:
            DLL_node_t: _description

        """
        prompt = ast.unparse(prompt)
        completion = gpt_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant to create google-style docstrings for python functions.",
                },
                {
                    "role": "user",
                    "content": f'Create google-style dosctring for function: \n "{prompt}" \n Return only docstring without quotes.',
                },
            ],
            n=1,
        )
        return completion.choices[0].message.content

    return wrapper


def constant_docstring_generator(*args, **kwargs) -> str:
    """
    _summary_

    Args:
        k (int): _description
        node (DLL_node_t, optional): _description_. Defaults to None.
    Returns:
        DLL_node_t: _description

    """
    return "_summary_\n\nArgs:\n    k (int): _description\n    node (DLL_node_t, optional): _description_. Defaults to None.\nReturns:\n    DLL_node_t: _description\n"
