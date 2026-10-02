from typing import Any
import pytest

from backend.app.agents.base import (
    AgentError,
    AgentExecutionError,
    AgentInputError,
    AgentResult,
    BaseAgent,
)


class DummySuccessAgent(BaseAgent):
    """Dummy agent for testing successful execution."""

    @property
    def name(self) -> str:
        return "DummySuccessAgent"

    @property
    def description(self) -> str:
        return "Test agent that returns transformed input."

    def execute(self, text: str) -> str:
        if not text:
            raise AgentInputError("text cannot be empty.")
        return text.upper()


class DummyFailingAgent(BaseAgent):
    """Dummy agent for testing error wrapping."""

    @property
    def name(self) -> str:
        return "DummyFailingAgent"

    @property
    def description(self) -> str:
        return "Test agent that raises an error."

    def execute(self, *args, **kwargs) -> Any:
        raise AgentExecutionError("Internal failure simulation.")


def test_base_agent_successful_run():
    """Verify BaseAgent.run wraps successful execution in AgentResult."""
    agent = DummySuccessAgent()
    assert agent.name == "DummySuccessAgent"
    assert "Test agent" in agent.description

    # Direct execute
    assert agent.execute("hello") == "HELLO"

    # Safe run
    res = agent.run("hello")
    assert isinstance(res, AgentResult)
    assert res.agent_name == "DummySuccessAgent"
    assert res.success is True
    assert res.data == "HELLO"
    assert res.error_message is None
    assert res.latency_seconds >= 0.0

    d = res.to_dict()
    assert d["agent_name"] == "DummySuccessAgent"
    assert d["data"] == "HELLO"


def test_base_agent_failing_run():
    """Verify BaseAgent.run catches exceptions and sets success=False with error_message."""
    agent = DummyFailingAgent()
    res = agent.run()

    assert isinstance(res, AgentResult)
    assert res.success is False
    assert res.data is None
    assert "Internal failure simulation" in res.error_message


def test_agent_cannot_instantiate_without_abstract_methods():
    """Verify BaseAgent cannot be instantiated without implementing abstract methods."""
    class IncompleteAgent(BaseAgent):
        pass

    with pytest.raises(TypeError):
        IncompleteAgent()  # type: ignore
