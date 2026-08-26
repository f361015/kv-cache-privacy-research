# Data policy and manifest

The initial dataset must contain synthetic secrets only. Do not commit real personally identifiable, medical, financial, organizational, or confidential data.

The pilot target is at least 150 prompts, distributed across public controls and the four sensitive categories. Each prompt must have a stable `prompt_id`, category, synthetic secret span, attribute label, template version, and split.

Raw and processed datasets are ignored by Git. Small JSONL manifests may be committed after manual inspection confirms that they contain no real secrets.

Suggested record:

```json
{
  "prompt_id": "medical_0001",
  "category": "medical",
  "prompt": "Synthetic example only: Patient R-104 reports condition Zeta-7.",
  "secret_span": "condition Zeta-7",
  "attribute_label": "medical",
  "template_version": "v1",
  "split": "pilot"
}
```
