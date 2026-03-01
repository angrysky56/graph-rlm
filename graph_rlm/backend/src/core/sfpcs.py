import json
from typing import Any, Dict, List


def build_simulation_framework(
    name: str, domain: str, focus: str, scale: str, tension: str, rules_markdown: str
) -> str:
    """
    Step 1 & 2: Generate the overarching environment definition for SFPCS v2.0.
    """
    return f"""<simulation_framework id="{name}" domain="{domain}">
  <context>
    <focus>{focus}</focus>
    <scale>{scale}</scale>
    <tension>{tension}</tension>
  </context>
  <environment_rules format="markdown">
{rules_markdown}
  </environment_rules>
</simulation_framework>
"""


def build_entity_prompt(
    name: str,
    archetype: str,
    mandate: str,
    tendencies_markdown: str,
    initial_state: Dict[str, Any],
    attention_gate: str,
    throughput_k: int,
    skills: List[Dict[str, str]],
) -> str:
    """
    Step 3: Generate the Skill-Based Agent entity prompt.
    Uses XML for nesting, Markdown for human-readable rules, and JSON for strict state tracking.
    """
    state_json = json.dumps(initial_state, indent=2)

    skills_xml = ""
    for skill in skills:
        s_id = skill.get("id", "unknown_skill")
        s_version = skill.get("version", "1.0")
        s_status = skill.get("status", "active")
        s_prof = skill.get("proficiency", "1.0")
        s_logic = skill.get("logic", "")
        skills_xml += f'    <skill id="{s_id}" version="{s_version}" status="{s_status}" proficiency="{s_prof}">\n      {s_logic}\n    </skill>\n'

    return f"""<entity id="{name}" archetype="{archetype}">
  <instructions>
    <mandate>{mandate}</mandate>
    <behavioral_tendencies format="markdown">
{tendencies_markdown}
    </behavioral_tendencies>
  </instructions>

  <state format="json">
{state_json}
  </state>

  <metacognitive_control>
    <attention_gate>{attention_gate}</attention_gate>
    <throughput_k>{throughput_k}</throughput_k>
  </metacognitive_control>

  <skills_repository>
{skills_xml.rstrip()}
  </skills_repository>
</entity>
"""


def build_simulation_cycle_prompt(step: int, input_signal: str) -> str:
    """
    Step 4: Generate the `<event>` and `<thinking>` prompt interface.
    Forces the agent to output a `<thinking>` block before the final event JSON.
    """
    return f"""<simulation_cycle step="{step}">
  <input>
{input_signal}
  </input>

  <thinking>
    1. [Attention filter evaluation]
    2. [World model update based on input]
    3. [Skill selection and application]
    4. [Prediction of outcome]
  </thinking>

  <event id="[Event_ID]" initiator="[Entity_Name]" type="[Action_Type]">
    <impact format="json">
      {{
        "Target": "[Target_Entity]",
        "Attribute_Changes": {{"[Attribute]": "[Change_Value]"}},
        "Generated_Prediction_Error": 0.0
      }}
    </impact>
  </event>
</simulation_cycle>
"""


def build_system_tracking_prompt(
    workflow_id: str, sequence_markdown: str, metrics: List[Dict[str, Any]]
) -> str:
    """
    Step 5: Generate workflow and metrics tracking prompt.
    """
    metrics_json = json.dumps(metrics, indent=2)
    return f"""<system_tracking>
  <workflow id="{workflow_id}">
    <sequence format="markdown">
{sequence_markdown}
    </sequence>
  </workflow>

  <metrics format="json">
{metrics_json}
  </metrics>
</system_tracking>
"""
