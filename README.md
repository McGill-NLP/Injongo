# Injongo Dataset
    A Multicultural Intent Detection and Slot-filling Dataset for 16 African Languages

```python
language = [
    "amh", "ewe", "hau", "ibo", "kin", 
    "lin", "lug", "orm", "sna", "sot", 
    "swa", "twi", "wol", "xho", "yor", "zul"
]
```

## Published Data

### *data/cleand-dataset*: xtream-up format dataset

Path: `data/cleaned-dataset/{language}/{train|dev|test}.jsonl`
Item example:
    ```json
    {"intent":"alarm","text":"ለ10፡00 ሰአት እና ሌላ አንድ ለ 3፡00 ማንቂያ ድውል ሙላልኝ","spans":[{"start_byte":1,"limit_byte":10,"label":"TIME"},{"start_byte":17,"limit_byte":20,"label":"NUMBER"},{"start_byte":23,"limit_byte":27,"label":"TIME"}],"target":"TIME: 10፡00 ሰአት $$ NUMBER: አንድ $$ TIME: 3፡00","example_id":"dev-00000060"}
    ```

<!-- ### *data/json*: raw annotation of slot filling [Not Publish]

Path: `data/json/{language}_{reviewed|unreviewed}.json` -->

### *data/output*: csv format for the dataset, including logical_form and spans

Item Example:
    ```
    split,domain,intent,text,spans,logical_form
    test,banking,balance,በ አባይ ባንክ አካውንት ለሶፋ የሚሆን ገንዘብ አለኝ,"2:9:SL:BANK_NAME,17:19:SL:SHOPPING_ITEM",[IN:balance [SL:BANK_NAME አባይ ባንክ] [SL:SHOPPING_ITEM ሶፋ] ]
    ```
## Package Structure

```bash
pip install -e .
```

Additional Dependencies:
- vllm: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html
- tgi: https://huggingface.co/docs/text-generation-inference/installation_nvidia

## Environment Variables (.env file)

```bash
OPENAI_API_KEY=sk-proj-
GEMINI_API_KEY=ABCD
```
