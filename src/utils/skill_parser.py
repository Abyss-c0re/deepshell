import re
from typing import Dict, List, Any, Optional, Callable
import os

def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Simple YAML parser without external libraries. Handles basic dicts, lists, indentation."""
    result = {}
    lines = text.split('\n')
    stack: List[Any] = [result]
    indents = [0]
    for line in lines:
        if not line.strip() or line.strip().startswith('#'):
            continue
        indent = len(line) - len(line.lstrip())
        while indent < indents[-1]:
            stack.pop()
            indents.pop()
        stripped = line.strip()
        if ':' in stripped:
            key, value_part = stripped.split(':', 1)
            key = key.strip()
            value_part = value_part.strip()
            if value_part.startswith('{') or value_part.startswith('['):  # Treat as string if complex
                value = value_part
            elif value_part:
                try:
                    value = int(value_part)
                except ValueError:
                    try:
                        value = float(value_part)
                    except ValueError:
                        if value_part.lower() == 'true':
                            value = True
                        elif value_part.lower() == 'false':
                            value = False
                        elif value_part.lower() == 'null':
                            value = None
                        else:
                            value = value_part.strip('"').strip("'")
            else:
                value = {}  # Sub dict
            stack[-1][key] = value
            if isinstance(value, dict):
                stack.append(value)
                indents.append(indent + 2)  # Assume 2-space indent
        elif stripped.startswith('- '):
            item = stripped[2:].strip()
            if not isinstance(stack[-1], list):
                stack[-1] = []  # Convert to list if needed
            try:
                item = int(item)
            except ValueError:
                try:
                    item = float(item)
                except ValueError:
                    if item.lower() == 'true':
                        item = True
                    elif item.lower() == 'false':
                        item = False
                    elif item.lower() == 'null':
                        item = None
                    else:
                        item = item.strip('"').strip("'")
            stack[-1].append(item)
        else:
            # Continuation or error
            pass  # Skip for simplicity
    return result

def load_skill_md(skill_path: str) -> Tool:
    """
    Load a Claude-style SKILL.md file and convert it to a Tool object.
    Assumes the file has YAML frontmatter with 'name', 'description', 'parameters' (dict), 'required' (list).
    Looks for a '## Executor' section with a ```python code block for the executor function.
    The code should define a function named 'executor' that takes **kwargs and returns the result.
    """
    if os.path.isdir(skill_path):
        skill_path = os.path.join(skill_path, 'SKILL.md')
    if not os.path.exists(skill_path):
        raise FileNotFoundError(f"SKILL.md not found at {skill_path}")

    with open(skill_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract frontmatter
    frontmatter_match = re.match(r'---\n(.*?)\n---\n', content, re.DOTALL)
    if not frontmatter_match:
        raise ValueError("No YAML frontmatter found in SKILL.md")

    yaml_text = frontmatter_match.group(1)
    body = content[frontmatter_match.end():]

    # Parse YAML
    fm = parse_simple_yaml(yaml_text)

    name = fm.get('name')
    description = fm.get('description')
    parameters = fm.get('parameters', {})
    required = fm.get('required', [])

    if not name or not description:
        raise ValueError("SKILL.md must have 'name' and 'description' in frontmatter")

    # Extract executor code from body
    executor_match = re.search(r'## Executor\n```python\n(.*?)\n```', body, re.DOTALL)
    if not executor_match:
        # Default dummy executor if no code provided
        code = "def executor(**kwargs):\n    return 'Skill executed with args: ' + str(kwargs)"
    else:
        code = executor_match.group(1)

    # Define the executor function safely
    loc: Dict[str, Any] = {}
    try:
        exec(code, {"__builtins__": {}}, loc)  # Restricted globals for safety
    except Exception as e:
        raise ValueError(f"Error compiling executor code: {e}")

    executor_func = loc.get('executor')
    if not callable(executor_func):
        raise ValueError("Executor code must define a callable 'executor' function")

    # Create and return the Tool
    tool = Tool(
        name=name,
        description=description,
        parameters=parameters,
        executor=executor_func,
        required=required,
    )
    return tool

# Example usage in your app
if __name__ == "__main__":
    # Assume you have a skill folder or file, e.g., 'my_skill/SKILL.md'
    skill_tool = load_skill_md('path/to/my_skill')
    tool_set = ToolSet("My Skills")
    tool_set.add(skill_tool)
    print(tool_set)

    # To use: executor = tool_set.get_executor('skill_name')
    # result = executor(arg1=value1, ...)