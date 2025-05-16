# backend/scripts/test_gemini_templating.py
import asyncio
import logging

# --- Path Setup ---
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))
# --- End Path Setup ---

from langchain_core.prompts import PromptTemplate
# We won't even use the LLM yet, just test PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini_template_test_ULTRA_MINIMAL")


async def main():
    print("*************************************ULTRA MINIMAL TEST - PROMPT TEMPLATE")
    print("*************************************")
    print("*************************************")
    subject = "LangChain"
    topic = "templating"
    test_string = f"This is a test for {subject} about {topic}."
    print(f"BASIC PYTHON F-STRING TEST: {test_string}")
    print("*************************************")
    print("*************************************")
    
    
    logger.info(f"<<< ULTRA MINIMAL PromptTemplate TEST >>>")

    # 1. Define an extremely simple template string with obvious placeholders
    #    Use DIFFERENT variable names than before, just in case.
    minimal_template_str = "This is a test for {{subject}} about {{topic}}."
    logger.info(f"Minimal Template String: '{minimal_template_str}'")

    # 2. Create PromptTemplate
    try:
        prompt = PromptTemplate.from_template(minimal_template_str)
        logger.info(
            f"PromptTemplate input variables: {prompt.input_variables}"
        )  # <<< CRITICAL LOG
    except Exception as e:
        logger.error(f"Error creating PromptTemplate: {e}", exc_info=True)
        return

    # 3. Define a sample payload MATCHING the new variable names
    payload = {"subject": "LangChain", "topic": "templating"}
    logger.info(f"Test Payload: {payload}")

    # 4. Attempt to format (invoke the prompt template)
    if prompt.input_variables:  # Only try if variables were recognized
        try:
            formatted_prompt_value = prompt.invoke(payload)
            rendered_string = formatted_prompt_value.to_string()

            logger.info(
                f"LangChain `prompt.invoke(payload).to_string()` output:\n------\n{rendered_string}\n------"
            )

            if "{{subject}}" in rendered_string or "{{topic}}" in rendered_string:
                logger.error(
                    "ULTRA MINIMAL TEST - SUBSTITUTION FAILURE: Placeholders are still present!"
                )
            else:
                logger.info(
                    "ULTRA MINIMAL TEST - SUBSTITUTION SUCCESS: Placeholders replaced."
                )
        except Exception as e_invoke:
            logger.error(
                f"Error during prompt.invoke(payload) in ultra minimal test: {e_invoke}",
                exc_info=True,
            )
    else:
        logger.error(
            "ULTRA MINIMAL TEST - NO INPUT VARIABLES RECOGNIZED BY PROMPTTEMPLATE. Substitution will not occur."
        )


if __name__ == "__main__":
    asyncio.run(main())
