import json
import re
from typing import Dict, List, Optional, Union

from jinja2 import Template


class PromptManager:
    def __init__(self):
        self.templates: Dict[str, Template] = {}
        self.templates_source: Dict[str, str] = {}
        self.models: Dict[str, Dict[str, str]] = {}
        self.postprocessors: Dict[str, callable] = {
            "json": self._process_json,
            "logical_form": self._process_logical_form,
            "raw": self._process_raw,
        }
        self.language_examples: Dict[str, List[Dict[str, str]]] = {}

    def add_language_example(self, language: str, example: List[Dict[str, str]]):
        """Add a new language example to the prompt manager."""
        self.language_examples[language] = example

    def add_template(self, name: str, template: str):
        """Add a new template to the prompt manager."""
        template = template.replace("        ", "")
        self.templates[name] = Template(template)
        self.templates_source[name] = template

    def add_model(self, model_name: str, prompts: Dict[str, str]):
        """Add a new model with its associated prompts."""
        self.models[model_name] = prompts

    def generate_prompt(
        self,
        template_name: str,
        model_name: str,
        shot_count: int = 0,
        language: str = "eng",
        **kwargs,
    ) -> str:
        """Generate a prompt based on the template, model, and parameters."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found.")
        # if model_name not in self.models:
        #     raise ValueError(f"Model '{model_name}' not found.")

        template = self.templates[template_name]
        # model_prompt = self.models[model_name].get(template_name, "")

        context = {
            "shot_count": shot_count,
            "language": language,
            "model_prompt": "",
            "examples": self.language_examples[language],
            **kwargs,
        }

        return template.render(context)

    def process_output(self, output: str, processor: str) -> Union[Dict, List, str]:
        """Process the output using the specified postprocessor."""
        if processor not in self.postprocessors:
            raise ValueError(f"Postprocessor '{processor}' not found.")

        return self.postprocessors[processor](output)

    def save(self, path: str):
        """Save the prompt manager to a file."""
        with open(path, "w") as f:
            json.dump(
                {
                    "templates": {
                        name: template
                        for name, template in self.templates_source.items()
                    },
                    "models": self.models,
                },
                f,
            )

    @staticmethod
    def _process_json(output: str) -> Dict:
        """Process JSON output."""
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON output.")

    @staticmethod
    def _process_logical_form(output: str) -> List[Dict]:
        """Process logical form output."""
        pattern = r"\[(\w+):(\w+)\s+([^\]]+)\]"
        matches = re.findall(pattern, output)
        return [
            {"type": match[0], "subtype": match[1], "content": match[2]}
            for match in matches
        ]

    @staticmethod
    def _process_raw(output: str) -> str:
        """Process raw output."""
        return output.strip().strip().replace("```text","").replace("```", "").strip().strip()# .replace("\n", "")

    def __repr__(self):
        return f"PromptManager(\ntemplates={list(self.templates.keys())}, \nmodels={list(self.models.keys())})"

    # https://simmering.dev/blog/structured_output/


if __name__ == "__main__":
    pm = PromptManager()

    # Add templates
    pm.add_template(
        "translation",
        "Translate the following {{ shot_count }}-shot prompt from {{ source_language }} to {{ target_language }}:\n\n{{ text }}\n\n{{ model_prompt }}",
    )
    pm.add_template(
        "sentiment",
        "Analyze the sentiment of the following {{ shot_count }}-shot prompt in {{ language }}:\n\n{{ text }}\n\n{{ model_prompt }}",
    )

    # Add models
    pm.add_model(
        "gpt-3",
        {
            "translation": "Translate accurately, preserving the original meaning and tone.",
            "sentiment": "Provide a sentiment analysis with a score from -1 (very negative) to 1 (very positive).",
        },
    )
    pm.add_model(
        "bert",
        {
            "translation": "Perform a direct translation, focusing on literal meaning.",
            "sentiment": "Classify the sentiment as positive, negative, or neutral.",
        },
    )

    # Generate prompts
    translation_prompt = pm.generate_prompt(
        "translation",
        "gpt-3",
        shot_count=2,
        source_language="English",
        target_language="French",
        text="Hello, how are you?",
    )
    print(translation_prompt)

    sentiment_prompt = pm.generate_prompt(
        "sentiment",
        "bert",
        shot_count=0,
        language="English",
        text="I love this product!",
    )
    print(sentiment_prompt)

    # Process outputs
    json_output = pm.process_output('{"sentiment": "positive", "score": 0.8}', "json")
    print(json_output)

    logical_form_output = pm.process_output(
        "[IN:translate [SL:TIME ŋdi me] [SL:LANGUAGE_NAME ncam] ]", "logical_form"
    )
    print(logical_form_output)
