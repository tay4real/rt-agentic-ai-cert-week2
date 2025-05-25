import sys
from pathlib import Path
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Add the parent directory to the path so we can import utils and paths
sys.path.append(str(Path(__file__).parent.parent))

from utils import load_publication, load_yaml_config, load_env, save_text_to_file
from paths import PROMPT_CONFIG_FPATH, OUTPUTS_DIR, APP_CONFIG_FPATH
from prompt_builder import build_prompt_from_config


def invoke_llm(prompt, model="gpt-4o-mini", temperature=0.0):
    """
    Simple LLM invocation function.

    Args:
        prompt (str): The prompt to send to the LLM
        model (str): The model to use
        temperature (float): Temperature for response generation

    Returns:
        str: The LLM's response content
    """
    try:
        llm = ChatOpenAI(
            model=model, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY")
        )
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


def run_prompt_example(
    all_prompts_config, prompt_config_key, publication_content, model_name, app_config
):
    """
    Run a single prompt example.

    Args:
        all_prompts_config (dict): Configuration for all prompts
        prompt_config_key (str): Key for the prompt configuration
        publication_content (str): The publication text
        model_name (str): The LLM model to use
        app_config (dict): Application configuration including reasoning strategies
    """
    # Build the prompt
    if prompt_config_key not in all_prompts_config:
        print(f"Config key '{prompt_config_key}' not found in configuration")
        return

    prompt_config = all_prompts_config[prompt_config_key]

    # Pass app_config to build_prompt_from_config
    prompt = build_prompt_from_config(prompt_config, publication_content, app_config)
    save_text_to_file(
        prompt,
        os.path.join(OUTPUTS_DIR, f"{prompt_config_key}_prompt.md"),
        header=f"Prompt Generated From Config: {prompt_config_key}",
    )

    # Get LLM response
    llm_response = invoke_llm(prompt, model=model_name)
    if llm_response:
        save_text_to_file(
            llm_response,
            os.path.join(OUTPUTS_DIR, f"{prompt_config_key}_llm_response.md"),
            header=f"LLM Response for Prompt: {prompt_config_key}",
        )
    else:
        print("✗ LLM response was empty or failed.")


def main(prompt_config_key:str="linkedin_post_prompt_cfg"):
    """
    Main function demonstrating modular prompt engineering.
    """
    print("=" * 80)
    print("LESSON 1: THE MODULAR APPROACH TO PROMPT ENGINEERING")
    print("=" * 80)
    print("Demonstrating how to build prompts systematically using modular components")

    try:
        # Load environment variables
        print("\nLoading environment variables...")
        load_env()
        print("✓ OpenAI API key loaded")

        # Load the publication content
        print("Loading publication content...")
        publication_content = load_publication()
        print(f"✓ Publication loaded ({len(publication_content)} characters)")

        print("Loading application configuration...")
        app_config = load_yaml_config(APP_CONFIG_FPATH)
        model_name = app_config.get("llm", "gpt-4o-mini")
        print(f"✓ Model set to: {model_name}")

        # Load the prompt configuration
        print(f"Loading prompt config from: {PROMPT_CONFIG_FPATH}")
        all_prompts_config = load_yaml_config(PROMPT_CONFIG_FPATH)
        print(f"✓ Config loaded with prompt keys: {list(all_prompts_config.keys())}")

        if prompt_config_key not in all_prompts_config:
            print(f"Error: Prompt config key '{prompt_config_key}' not found.")
            return
        
        print(f"\nRunning prompt example: {prompt_config_key}")
        run_prompt_example(
            all_prompts_config=all_prompts_config,
            prompt_config_key=prompt_config_key,
            publication_content=publication_content,
            model_name=model_name,
            app_config=app_config  # Pass app_config here
        )

        print(f"\n{'='*80}")
        print("LESSON COMPLETE!")
        print("You've seen how modular prompt components can be systematically")
        print("combined to create more effective and reliable prompts.")
        print("=" * 80)

    except Exception as e:
        print(f"Error in lesson execution: {e}")
        return None


if __name__ == "__main__":

    prompt_cfg_key = "linkedin_post_prompt_cfg"

    main(prompt_config_key=prompt_cfg_key)