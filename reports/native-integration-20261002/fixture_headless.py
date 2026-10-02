"""Deterministic transport fixture: emits text, never invokes a model."""
import json
import sys

if "--check" in sys.argv:
    print(json.dumps({"fixture": True, "inference_calls": 0}))
else:
    print('''<NAME>transport_fixture</NAME>
<DESCRIPTION>Fixed diagnostic mutation; this is not an LLM proposal.</DESCRIPTION>
<CODE>
```python
# EVOLVE-BLOCK-START
def get_ga_config():
    return {"sigma": 0.4, "elite_ratio": 0.5}
# EVOLVE-BLOCK-END
```
</CODE>''')
    print(json.dumps({"usage": {"input_tokens": 0, "output_tokens": 0, "total_cost": 0}}))
