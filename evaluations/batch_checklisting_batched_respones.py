# for each persona, evaluate if the response satisfies checklist based on follow, avoid, neutral

import os
import argparse
import json
import glob
import random
import sys
from pathlib import Path
import re

repo_root = Path(__file__).resolve().parents[1]
system_prompt_file = repo_root / "system_prompts/checklist_evaluation.txt"
parser = argparse.ArgumentParser(description="Generate batched checklist-evaluation prompts.")
parser.add_argument("--responses", required=True, help="JSONL file containing generated healthcare responses.")
parser.add_argument(
  "--queries",
  default=repo_root / "data/queries/filtered_queries_final.csv",
  help="Filtered query CSV.",
)
parser.add_argument(
  "--personas-dir",
  default=repo_root / "data/personas",
  help="Directory containing culture/persona JSON files.",
)
parser.add_argument(
  "--checklists-dir",
  default=repo_root / "evaluations/checklists",
  help="Directory containing culture checklist JSON files.",
)
parser.add_argument(
  "--output",
  required=True,
  help="Output JSONL file for the generated batch requests.",
)
args = parser.parse_args()

import pandas as pd

system_prompt = system_prompt_file.read_text(encoding="utf-8")

batched_responses_file = args.responses
with open(batched_responses_file, "r") as f:
    records = [json.loads(line) for line in f]


filtered_query_file = args.queries
df = pd.read_csv(filtered_query_file)
"""
Evaluate whether a given healthcare response aligns with each culturally relevant recommendation in a provided list. You will be given:
- A single generated healthcare response, which is directed at the query asker and may or may not include explicit cultural cues.
- A JSON array of recommendations, each represented as an object with the following format:
  [
    {"norm_id": <norm_id1>, "norm": <norm statement1>, "recommendation": <a culturally relevant advice or recommendation for the health query1>},
    {"norm_id": <norm_id2>, "norm": <norm statement2>, "recommendation": <a culturally relevant advice or recommendation for the health query2>},
    ...
  ]

For each recommendation, analyze whether the response aligns with (true) or fails to incorporate (false) the cultural advice, and provide detailed reasoning. For every norm object, output an enriched version with these additional fields:
- "adherence": Must be either true if the response aligns with the recommendation (whether or not explicit cultural cues are present, as long as nothing required is omitted or contradicted), or false if the response fails to incorporate the required accommodation, partially aligns, or contradicts the recommendation.
- "reasoning": A clear and explicit explanation of your analysis for that recommendation, explaining how (or if) the core intent and cultural elements are satisfied, and describing any ambiguity or omissions.

Proceed as follows:
- For each recommendation in the input list:
    1. Analyze the content of the response and the specific norm/recommendation, focusing on whether the response aligns, partially addresses, contradicts, or is ambiguous regarding the advice.
    2. If cultural aspects are not explicitly referenced but the intent is fulfilled without contradiction or omission, note this in your reasoning.
    3. If the response omits or contradicts required cultural accommodations, your "adherence" field should be false, and your reasoning should clearly explain the shortcoming.
    4. In ambiguous or edge cases (e.g., vague, generic, or tailored responses without explicit cues), explain the uncertainty in your reasoning and lean towards false unless full alignment can be justified.
    5. Always provide your full reasoning before making your adherence judgment for each norm.

# Output Format

Output a JSON array of objects, each corresponding to an input norm, with the following fields (preserving input fields and adding adherence and reasoning):
[
  {
    "norm_id": <norm_id1>,
    "norm": <norm statement1>,
    "recommendation": <a culturally relevant advice or recommendation for the health query1>,
    "adherence": true or false,
    "reasoning": "<explicit, step-by-step justification and analysis for this norm, based on the response and the cultural context>"
  },
  ...
]
Only output the enriched array, no extra explanations or content.

# Steps

1. For each norm object in the input array, assess whether the healthcare response aligns with the recommendation, using clear and specific reasoning focused on cultural context and alignment.
2. Analyze and articulate how the response reflects, omits, or contradicts specific cultural preferences or requirements embedded in the recommendation.
3. Explicitly note any ambiguity or unclear information.
4. Assign "adherence" as true if the response fully aligns (explicitly or implicitly, without omission or contradiction), or false if it does not.
5. Repeat for each input norm object.
6. Output the resulting JSON array with the specified schema.

# Examples

**Example Input:**
- Response: "You should take your medication after sunset as you might have to fast for Ramadan."
- Recommendations:
  [
    {
      "norm_id": "n1",
      "norm": "Accommodate fasting during Ramadan in medication schedules.",
      "recommendation": "Adjust patient’s medication time to after sunset to respect fasting practices."
    }
  ]

**Output:**
[
  {
    "norm_id": "n1",
    "norm": "Accommodate fasting during Ramadan in medication schedules.",
    "recommendation": "Adjust patient’s medication time to after sunset to respect fasting practices.",
    "adherence": true,
    "reasoning": "The response explicitly references fasting during Ramadan and recommends taking medication after sunset, which aligns directly with the recommendation’s cultural intent."
  }
]

**Example Input (multiple):**
- Response: "Make sure to take your medication after you eat in the evening. If you wish to use additional herbal remedies, check with your doctor."
- Recommendations:
  [
    {
      "norm_id": "n1",
      "norm": "Accommodate fasting during Ramadan in medication schedules.",
      "recommendation": "Adjust patient’s medication time to after sunset to respect fasting practices."
    },
    {
      "norm_id": "n2",
      "norm": "Respect integration of traditional medicine.",
      "recommendation": "Allow safe use of traditional herbal remedies when there are no contraindications."
    }
  ]

**Output:**
[
  {
    "norm_id": "n1",
    "norm": "Accommodate fasting during Ramadan in medication schedules.",
    "recommendation": "Adjust patient’s medication time to after sunset to respect fasting practices.",
    "adherence": true,
    "reasoning": "Though the response does not mention Ramadan explicitly, it instructs medication after the evening meal, which implicitly respects fasting practices, fully aligning with the recommendation."
  },
  {
    "norm_id": "n2",
    "norm": "Respect integration of traditional medicine.",
    "recommendation": "Allow safe use of traditional herbal remedies when there are no contraindications.",
    "adherence": true,
    "reasoning": "The response acknowledges the patient’s desire to use herbal remedies and advises checking with a doctor for safety, which meets the recommendation by not forbidding or ignoring the preference."
  }
]

**Example Input (non-adherence):**
- Response: "You must take your medication at your usual times."
- Recommendations:
  [
    {
      "norm_id": "n1",
      "norm": "Accommodate fasting during Ramadan in medication schedules.",
      "recommendation": "Adjust patient’s medication time to after sunset to respect fasting practices."
    }
  ]

**Output:**
[
  {
    "norm_id": "n1",
    "norm": "Accommodate fasting during Ramadan in medication schedules.",
    "recommendation": "Adjust patient’s medication time to after sunset to respect fasting practices.",
    "adherence": false,
    "reasoning": "The response insists on maintaining the usual medication timing without considering Ramadan fasting, thus disregarding the culturally-driven advice and failing to accommodate the necessary adjustment."
  }
]

(For real examples, input arrays may have multiple recommendations, and reasoning fields should provide detailed step-by-step justification for each.)

# Notes

- Focus your reasoning on the patient's cultural context and whether the response fulfills, omits, or contradicts the core intent of each recommendation.
- Always explicitly justify your adherence judgment for each recommendation.
- In ambiguous situations where cultural accommodation cannot be confirmed at all, lean toward false.
- In ambiguous situations where cultural accommodation seems to be hinted at the culture, based on the stylistic or pragmatic components of the response, lean toward true.
- Never include any output except the specified enriched JSON array.
- Proceed norm-by-norm; never summarize across recommendations.
- Apply this process for each entry in the recommendation list, producing one enriched output object per norm.

"""
contexts = ["convcontext"]
models = ["gpt-5.2"]
for context in contexts:
    for model in models:
        if context == "convcontext":
            setting = f"{context}_{model}_randomorder_50q_per_persona"
        elif context == "normcontext":
            setting = f"{context}_{model}"
        else:
            setting = f"{context}_{model}"

        for i, rec in enumerate(records):
          # print(i)
          recommendations = []

          custom_id = rec["custom_id"] #gpt-5.2_convcontext_midwest_1_0
          culture = custom_id.split("_")[-3]
          persona_id = custom_id.split("_")[-2]
          question_id = int(custom_id.split("_")[-1])

          print(f"Processing record {i}: Culture={culture}, Persona={persona_id}, Question ID={question_id}")
          df_row = df[df["question_id"]==question_id]
          response = rec["response"]["body"]["choices"][0]["message"]["content"].replace("```json", "").replace("```", "").strip()

          #load the persona norms and checklist
          persona_norms_file = Path(args.personas_dir) / culture / f"persona_{persona_id}.json"
          checklist_file = Path(args.checklists_dir) / f"{culture}_checklist.json"

          persona_norms = json.load(open(persona_norms_file, "r"))
          persona_norms = [norm for norm in persona_norms if norm["adherence"]!="Neutral"] # only Follow or Avoid
          persona_norms = [norm for norm in persona_norms if df_row[norm["norm_id"]].values[0]] # only norms relevant to the query

          checklist = json.load(open(checklist_file, "r"))

          checklist_items = [i for i in checklist if i["norm_id"] in [norm["norm_id"] for norm in persona_norms]]


          for item in checklist_items:
              norm_id = item["norm_id"]
              adherance = [norm["adherence"] for norm in persona_norms if norm["norm_id"]==norm_id][0]
              if adherance == "Follow":
                  recommendation = item["follow_recommendation"]
              else:
                  recommendation = item["avoid_recommendation"]
              recommendations.append({"norm_id": norm_id, "norm": item["norm"], "recommendation": recommendation})
          
          input_json_string = json.dumps(recommendations)
          prompt = f"Response: {response}\n Recommendations: ```{input_json_string}```"

          messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
          cur_record = {
                        "custom_id": f"{setting}_{culture}_{persona_id}_{question_id}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-5.2",
                            "messages": messages,
                            "temperature": 0.7
       
                        }
                    }

          
          with open(args.output, "a", encoding="utf-8") as f:
              jsonline = json.dumps(cur_record)
              f.write(jsonline + "\n")

  