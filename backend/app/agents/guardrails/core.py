from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from app.config.prompts import GUARDRAILS_INPUT_CHECK_PROMPT, GUARDRAILS_OUTPUT_CHECK_PROMPT
import logging
from google.api_core.exceptions import ResourceExhausted

logger = logging.getLogger(__name__)

# LangChain Guardrails
class Guardrails:
    """Guardrails implementation using purely local components with LangChain."""

    def __init__(self, llm):
        """Initialize guardrails with the provided LLM."""
        self.llm = llm

        # Input guardrails prompt
        self.input_check_prompt = PromptTemplate.from_template(GUARDRAILS_INPUT_CHECK_PROMPT)

        # Output guardrails prompt
        self.output_check_prompt = PromptTemplate.from_template(GUARDRAILS_OUTPUT_CHECK_PROMPT)

        # Create the input guardrails chain
        self.input_guardrail_chain = (
            self.input_check_prompt
            | self.llm
            | StrOutputParser()
        )

        # Create the output guardrails chain
        self.output_guardrail_chain = (
            self.output_check_prompt
            | self.llm
            | StrOutputParser()
        )

    def check_input(self, user_input: str) -> tuple[bool, str]:
        """
        Check if user input passes safety filters.

        Args:
            user_input: The raw user input text

        Returns:
            Tuple of (is_allowed, message)
        """
        try:
            result = self.input_guardrail_chain.invoke({"input": user_input})

            if result.startswith("UNSAFE"):
                reason = result.split(":", 1)[1].strip() if ":" in result else "Content policy violation"
                return False, AIMessage(content = f"I cannot process this request. Reason: {reason}")

            return True, user_input
        except ResourceExhausted as e:
            logger.warning(f"API rate limit exceeded during input check: {str(e)}")
            return True, user_input  # Allow the input to pass through when we can't check it
        except Exception as e:
            logger.exception(f"Error in input guardrails: {str(e)}")
            return True, user_input  # Fall back to allowing the input

    def needs_redaction(self, text: str, user_input: str = "") -> bool:
        """
        Only flag self-harm, illegal or PII - NOT generic advice.

        Args:
            text: The output text to check
            user_input: The original user query (for context)

        Returns:
            Boolean indicating if the text needs redaction
        """
        if not text:
            return False

        try:
            result = self.output_guardrail_chain.invoke({
                "output": text,
                "user_input": user_input
            })

            # Only redact if explicitly marked as UNSAFE
            return result.strip().upper().startswith("UNSAFE")
        except ResourceExhausted as e:
            logger.warning(f"API rate limit exceeded during output check: {str(e)}")
            return False  # Assume content is safe when we can't check it
        except Exception as e:
            logger.exception(f"Error checking if output needs redaction: {str(e)}")
            return False  # Assume content is safe on error

    def safe_output(self, text: str) -> str:
        """
        Return a safe alternative when content must be redacted.

        Args:
            text: The original text (not used, just for signature)

        Returns:
            A safe alternative message
        """
        return "I'm sorry, I can't help with that request due to content safety guidelines."

    def check_output(self, output: str, user_input: str = "") -> str:
        """
        Process the model's output through safety filters.

        Args:
            output: The raw output from the model
            user_input: The original user query (for context)

        Returns:
            Sanitized/modified output
        """
        if not output:
            return output

        # Convert AIMessage to string if necessary
        output_text = output if isinstance(output, str) else output.content

        try:
            # Check if redaction is needed
            if self.needs_redaction(output_text, user_input):
                return self.safe_output(output_text)

            # Otherwise return the original text untouched
            return output_text
        except ResourceExhausted as e:
            logger.warning(f"API rate limit exceeded during output check: {str(e)}")
            # Provide a useful message to the user about rate limits
            return "I'm currently experiencing high demand and have reached my usage limits. Please try again in a few minutes."
        except Exception as e:
            logger.exception(f"Error in output guardrails: {str(e)}")
            # Return a graceful error message
            return "I apologize, but I'm having trouble processing your request right now. Please try again shortly."
