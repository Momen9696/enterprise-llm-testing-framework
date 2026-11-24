"""Safe prompt-template rendering.

Values are substituted by literal replacement. ``str.format`` is intentionally not used,
so values that contain braces cannot change the template structure.
"""

from __future__ import annotations

import re

from llm_eval.models import PromptTemplateSpec

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class TemplateError(ValueError):
    """Raised when a template cannot be rendered with the supplied variables."""


def render_template(spec: PromptTemplateSpec) -> str:
    missing = [name for name in spec.input_variables if name not in spec.variables]
    if missing:
        raise TemplateError(f"Missing template variables: {', '.join(missing)}")

    unknown = sorted(set(spec.variables) - set(spec.input_variables))
    if unknown:
        raise TemplateError(f"Unexpected template variables: {', '.join(unknown)}")

    placeholders = set(_PLACEHOLDER_RE.findall(spec.template))
    undeclared = sorted(placeholders - set(spec.input_variables))
    if undeclared:
        raise TemplateError(
            f"Template '{spec.name}' has placeholders that are not input variables: {', '.join(undeclared)}"
        )

    rendered = spec.template
    for name in spec.input_variables:
        placeholder = "{" + name + "}"
        if placeholder not in rendered:
            raise TemplateError(f"Template '{spec.name}' does not contain {{{name}}}")
        rendered = rendered.replace(placeholder, spec.variables[name])
    return rendered
