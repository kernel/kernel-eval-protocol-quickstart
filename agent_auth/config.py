"""
Configuration for Agent Auth (login discovery).

Provides system prompts and settings for login discovery agents.
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from core import build_system_prompt

from .actions import AGENT_AUTH_ACTIONS

# Evaluation criteria for authentication/login discovery tasks.
AGENT_AUTH_EVALUATION_CRITERIA = """1. The agent must have navigated to an authentication page (login, sign-up, register, create account).
2. The agent must have identified input fields that are actually visible on the final page.
3. Many sites use progressive disclosure (showing email first, then password on the next step) - this is valid and should be considered successful.
4. The reported fields should match what is visible in the screenshots.
5. Do not penalize for "missing" fields that would only appear in later steps of a multi-step auth flow.
6. If the task asks for "first" input fields, only the initially visible fields need to be reported.
7. The agent must not fill in, submit, or otherwise interact with credential fields - only identify them. Do not penalize the agent for leaving forms empty.
8. If a site only offers SSO buttons (e.g., "Sign in with Google") with no traditional input fields, returning an empty fields list with an explanation is acceptable."""

# System prompt for login discovery agents
LOGIN_DISCOVERY_PROMPT = """Your task is to find the login page for a website and identify what input fields are required.

Instructions:
1. Navigate the website to find the login/sign-in page
2. Once you find a login form, identify all input fields (username, email, password, etc.)
3. Use the found_inputs action to report the required fields (this completes the task)

Do not click on or interact with cookie consent buttons unless they are blocking your task.
Do not attempt to fill in any credentials - just identify what fields exist."""


def get_agent_auth_system_prompt() -> str:
    """Get the system prompt for Agent Auth (login discovery)."""
    return build_system_prompt(
        base_prompt=LOGIN_DISCOVERY_PROMPT,
        extra_actions=AGENT_AUTH_ACTIONS,
        exclude_actions={"terminate"},
    )


def make_agent_auth_task(domain: str) -> str:
    """
    Create an agent-auth task string for a domain.

    Args:
        domain: The domain name (e.g., "github.com")

    Returns:
        Standardized task instruction
    """
    return (
        f"Navigate to {domain} and find the login or registration page. "
        f"Identify the first input field(s) required to begin the login or registration process."
    )
