from mlds.data_loader import (
    INTENTS,
    SLOTS_MERGED,
)

PROMPTS = {
    "intent-detection-five-round": """{% if round <= 1 -%}
Classify the given sentence by identifying its intent and selecting the most appropriate category from the provided list.

# Steps
1. Analyze the sentence to understand its primary intention or purpose.
2. Compare the identified intention against the possible intent categories.
3. Select the category that best matches the sentence's intent.

# Output Format
- Return the only one matching intent category from the list above. 
- No additional text or punctuation should be included in the output. 
{% endif %}{% if round == 2 -%}
Identify the intent of the provided text by selecting the most suitable category from the list of available options.

# Steps
1. Analyze the sentence to determine its primary purpose or intention.
2. Match the identified intention with the available intent categories.
3. Choose the category that best aligns with the sentence's intent.

# Output Format
- Return the selected intent category from the list above.
- Do not include any additional text or punctuation in the response.
{% endif %}{% if round == 3 -%}
Determine the intent of the provided text by selecting the most appropriate category from the given options.

# Steps
1. **Read the Text**: Carefully read the provided text to understand the context and main message.
2. **Identify Key Elements**: Identify the main action, subject, and any relevant details that indicate the overall purpose of the text.
3. **Consider Categories**: Review the list of available categories and consider which category best matches the text's intent.
4. **Reasoning**: Consider why you believe the text fits a certain category by assessing how the identified key elements align with the category's definition.
5. **Selection**: Select the category that most accurately represents the intent of the text.

# Output Format
- Provide the selected category as a plain text response. 
- Don't include any justification.
{% endif %}{% if round == 4 -%}
Identify the intent of the provided text by selecting the most suitable category from the list of available options.

# Steps

1. Analyze the text to understand its primary purpose and context.
2. Consider the range of possible intents that the text might express, such as inquiry, statement, request, etc.
3. Match the text with the most appropriate category based on its content and purpose.

# Output Format
Provide the resulting intent category as a short, concise phrase or word that best represents the text's purpose from the available options.

# Notes
- Carefully evaluate any subtleties in the language to determine the intent accurately.
- Consider edge cases where texts might have multiple overlapping intents, and choose the most dominant one.
{% endif %}{% if round == 5 -%}
Identify the intent of the provided text by selecting the most suitable category from the list of available options.

Consider the subtleties in language and any overlapping intents to determine the most dominant intent category.

# Steps

1. **Analyze the Text**: Thoroughly read and understand the text to grasp its primary purpose and context.
2. **Consider Possible Intents**: Reflect on the range of potential intents the text could express, such as inquiry, statement, or request.
3. **Match with Category**: Align the text with the most appropriate category based on content, language subtleties, and dominant purpose.

# Output Format

Provide the resulting intent category as a short, concise phrase or word.

# Notes

- Pay attention to context and subtleties in the text.
- Evaluate texts with multiple intents, prioritizing the most dominant one.
{% endif %}\n"""
+ f"# Intent Categories\n{', '.join(INTENTS)}\n\n"
+ """{% if shot_count == 0 -%}
# Format Example:
Sentence: Can you tell me the weather forecast for today?
Output: weather
{% else %}{% for example in examples -%}
Sentence: {{ example.text }}
Output: {{ example.intent }}

{% endfor %}Based on the example, consider the following:{% endif %}
Sentence: {{ text }}
Output: """,
    "slot-filling-five-round": """{% if round <= 1 -%}
Identify all named entities in the sentence provided according to the available entity types. Use `$$` as a separator between each pair of identified named entity types and corresponding content from the sentence. Only return the listed named entities without providing any additional commentary.

# Output Format
- List all the named entities found in the passage provided by the user. 
- Separate the paired named entities types and text using a `$$` symbol.
- Only return the entity list, without any prefix or explanation.
{% endif %}
{% if round == 2 -%}
Identify and extract named entities from the provided sentence. Each identified entity pair (including entity type and content from the sentence) should be separated from their content using the "$$" delimiter.

# Steps
1. Analyze the sentence to identify named entities.
2. Extract each identified named entity and its content.
3. Concatenate the named entity type and its content with space as one pair.
4. Join all pairs of named entities using "$$" as a delimiter.
{% endif %}{% if round == 3 -%}
Extract named entities from the provided text and format the output by placing $$ between each entity type and its respective content. Ensure the output contains only the extracted entities and their labels, with no additional commentary or information.

# Steps
1. Analyze the provided text and identify named entities.
2. Categorize each identified entity by its correct type, careful to match the entity with the appropriate label.
3. Format the output by placing the entity type and its corresponding content, separated by $$.
{% endif %}{% if round == 4 -%}
Identify named entities from the provided text. Format each entity and its content using $$ as a separator. 

# Steps
1. Parse the input text to identify all named entities. This includes proper nouns like names of people, places, organizations, dates, etc.
2. For each identified entity, extract the specific text corresponding to the entity.
3. Concatenate the name of the entity type and the associated text using space. 
4. Compile these formatted entries into a list with the $$ as a separator.

# Output Format
- A string joined by a " $$ " for each pair of the entity type and content, formatted as `EntityType EntityContent`.
{% endif %}{% if round == 5 -%}
Detect named entities in the supplied sentence. Use $$ as a separator between entities and their corresponding parts of the sentence. Limit the response strictly to the formatted list.

# Output Format
- Entities and their parts separated by $$
- Return a plain list with no additional context
- If no entities are present, return `$$`
{% endif %}\n"""
+ f"# Named Entities Types to Identify\n{', '.join(SLOTS_MERGED)}\n\n"
+ """{% if shot_count == 0 -%}
Please ensure that the entities match the listed types and that unstated entities should not be included in the response if no entities are found, return `$$` only.

# Format Example:
Sentence: John went to Paris and paid 100 dollars at an Awater restaurant.
Output: PERSONAL_NAME John $$ CITY_OR_PROVINCE Paris $$ MONEY 100 $$ RESTAURANT_NAME Awater
{% else %}
# Output Examples (Do not include in the response):
{% for example in examples -%}
Sentence: {{ example.text }}
Output: {{ example.slot }}

{% endfor %}Based on the example, consider the following:{% endif %}
Sentence: {{ text }}
Output: """
}