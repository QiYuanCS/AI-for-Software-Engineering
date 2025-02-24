
def quick_chat_system_prompt() -> str:
    return f"""
            Forget all previous instructions.
        You are a chatbot named Fred. You are assisting a software developer
        with their software development tasks.
        Each time the user converses with you, make sure the context is about
        * software development,
        * or coding,
        * or debugging,
        * or code reviewing,
        and that you are providing a helpful response.

        If the user asks you to do something that is not
        concerning one of those topics, you should refuse to respond.
        """

def system_learning_prompt() -> str:
    return """
    You are assisting a user with their general software development tasks.
Each time the user converses with you, make sure the context is generally about software development,
or creating a course syllabus about software development,
and that you are providing a helpful response.
If the user asks you to do something that is not concerning software
in the least, you should refuse to respond.
"""


def learning_prompt(learner_level: str, answer_type: str, topic: str) -> str:
    return f"""
Please disregard any previous context.

The topic at hand is ```{topic}```.
Analyze the sentiment of the topic.
If it does not concern software development or creating an online course syllabus about software development,
you should refuse to respond.

You are now assuming the role of a highly acclaimed software engineer specializing in the topic
 at a prestigious software company.  You are assisting a fellow software engineer with
 their software development tasks.
You have an esteemed reputation for presenting complex ideas in an accessible manner.
Your colleague wants to hear your answers at the level of a {learner_level}.

Please develop a detailed, comprehensive {answer_type} to teach me the topic as a {learner_level}.
The {answer_type} should include high level advice, key learning outcomes,
detailed examples, step-by-step walkthroughs if applicable,
and major concepts and pitfalls people associate with the topic.

Make sure your response is formatted in markdown format.
Ensure that embedded formulae are quoted for good display.
"""

############################################################################################################
# Requirements prompts
############################################################################################################

def system_requirements_prompt(product_name, product_description):
    """
    Generate a system requirements prompt based on the product name and description
    Args:
        product_name: The name of a product described in a system prompt
        product_description: A description of the product

    Returns:
        A prompt to use as a system prompt for making requirements documents for the product name and description.

    """
    return f"""
    Forget all previous instructions and context.

    You are an expert in requirements engineering.

    Your job is to learn and understand the following text about a product named {product_name}.
    ```
    {product_description}
    ```
    """

def requirements_prompt(product_name, requirement_type):
    """
    Generate a requirements prompt based on the requirement type and product name.
    Args:
        product_name: the name of a product described in a system prompt
        requirement_type: ["Business Problem Statement", "Vision Statement", "Ecosystem map", "RACI Matrix"]

    Returns:
        A prompt to use to generate a requirements document
        for the requirement type and product name.
    """
    if requirement_type not in ["Business Problem Statement", "Vision Statement", "Ecosystem map", "RACI Matrix"]:
        raise ValueError(f"Invalid requirement type.")
    if requirement_type == "Business Problem Statement":
        return business_problem_prompt(product_name)
    if requirement_type == "Vision Statement":
        return vision_statement_prompt(product_name)
    if requirement_type == "Ecosystem map":
        return ecosystem_map_prompt(product_name)
    if requirement_type == "RACI Matrix":
        return responsibility_matrix_prompt(product_name)

def business_problem_prompt(product_name):
    # TODO: Implement this function
    raise NotImplementedError(f"<derive prompt for {product_name} from textbook>")

def vision_statement_prompt(product_name):
    # TODO: Implement this function
    raise NotImplementedError(f"<derive prompt for {product_name} from textbook>")

def ecosystem_map_prompt(product_name):
    # TODO: Implement this function
    raise NotImplementedError(f"<derive prompt for {product_name} from textbook>")

def responsibility_matrix_prompt(product_name):
    # TODO: Implement this function
    raise NotImplementedError(f"<derive prompt for {product_name} from textbook>")



############################################################################################################
# Code Generation prompts
############################################################################################################

def review_prompt(existing_code: str) -> str:
    # TODO: Implement this function
    raise NotImplementedError(f"<write your own prompt>")


def modify_code_prompt(user_prompt: str, existing_code: str) -> str:
    # TODO: Implement this function
    raise NotImplementedError(f"<write your own prompt>")


def debug_prompt(debug_error_string: str, existing_code: str) -> str:
    # TODO: Implement this function
    raise NotImplementedError(f"<write your own prompt>")




