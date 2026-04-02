# SPDX-License-Identifier: Apache-2.0
"""Tests for agent_config.py and serialization.py agent-config paths.

Covers the uncovered branches identified in issue #646:
- The extra=="allow" branch in set_agent_config()
- reset_agent_config() edge cases
- Flow loading with agent blocks (serialization._agent_config_set path)
"""

# Standard
from pathlib import Path
from unittest.mock import Mock, patch
import logging
import tempfile

# First Party
from sdg_hub import Flow, FlowMetadata
from sdg_hub.core.flow.metadata import RecommendedModels

# Third Party
import yaml

from tests.flow.conftest import MockBlock


class TestAgentConfigCoverage:
    """Tests targeting uncovered branches in agent_config.py."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_metadata = FlowMetadata(
            name="Test Flow",
            description="A test flow",
            version="1.0.0",
            author="Test Author",
            recommended_models=RecommendedModels(
                default="test-model", compatible=["alt-model"], experimental=[]
            ),
            tags=["test"],
        )

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_mock_block(self, name="test_block"):
        return MockBlock(block_name=name, input_cols=["input"], output_cols=["output"])

    def _create_mock_agent_block(self, name="agent_block"):
        block = MockBlock(block_name=name, input_cols=["input"], output_cols=["output"])
        block.block_type = "agent"
        block.agent_framework = "langflow"
        block.agent_url = "http://localhost:7860"
        block.agent_api_key = None
        return block

    # --- Gap 1: extra == "allow" branch in set_agent_config ---

    def test_set_agent_config_extra_allow_branch(self):
        """Passing an unknown kwarg triggers the elif extra=='allow' branch."""
        agent_block = self._create_mock_agent_block("agent1")
        flow = Flow(blocks=[agent_block], metadata=self.test_metadata)

        assert not hasattr(agent_block, "custom_timeout")

        flow.set_agent_config(
            agent_url="http://new:7860",
            custom_timeout=60,
        )

        # agent_url hit the hasattr branch (already exists)
        assert flow.blocks[0].agent_url == "http://new:7860"
        # custom_timeout hit the elif extra=="allow" branch
        assert flow.blocks[0].custom_timeout == 60

    def test_set_agent_config_extra_allow_sensitive_redaction(self, caplog):
        """Sensitive params via the extra=='allow' branch are redacted in logs."""
        agent_block = self._create_mock_agent_block("agent1")
        flow = Flow(blocks=[agent_block], metadata=self.test_metadata)

        assert not hasattr(agent_block, "secret")

        with caplog.at_level(logging.DEBUG, logger="sdg_hub.core.flow.agent_config"):
            flow.set_agent_config(
                agent_url="http://new:7860",
                secret="super-secret-value",
            )

        assert flow.blocks[0].secret == "super-secret-value"

        # The secret value must not appear in any log message
        all_log_text = " ".join(record.message for record in caplog.records)
        assert "super-secret-value" not in all_log_text

    # --- Gap 2: reset_agent_config() edge cases ---

    def test_reset_agent_config_no_agent_blocks(self):
        """Reset on a flow with no agent blocks is a silent no-op."""
        regular_block = self._create_mock_block("regular")
        flow = Flow(blocks=[regular_block], metadata=self.test_metadata)

        assert flow.is_agent_config_set()

        flow.reset_agent_config()

        # Should still report as set (no agent blocks = nothing to reset)
        assert flow.is_agent_config_set()

    def test_reset_agent_config_never_configured(self):
        """Reset on a flow with agent blocks that was never configured."""
        agent_block = self._create_mock_agent_block("agent1")
        flow = Flow(blocks=[agent_block], metadata=self.test_metadata)

        assert not flow.is_agent_config_set()

        flow.reset_agent_config()

        # Should remain not set
        assert not flow.is_agent_config_set()

    # --- Gap 3: serialization._agent_config_set path ---

    def test_from_yaml_with_agent_blocks_sets_flag_false(self):
        """Loading a YAML with agent blocks sets _agent_config_set=False."""
        flow_config = {
            "metadata": {
                "name": "Agent Flow",
                "description": "Flow with agent block",
                "version": "1.0.0",
                "recommended_models": {
                    "default": "test-model",
                    "compatible": [],
                    "experimental": [],
                },
            },
            "blocks": [
                {
                    "block_type": "AgentBlock",
                    "block_config": {
                        "block_name": "my_agent",
                        "input_cols": "input",
                        "output_cols": "output",
                    },
                }
            ],
        }

        yaml_path = Path(self.temp_dir) / "agent_flow.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(flow_config, f)

        with patch("sdg_hub.core.flow.serialization.BlockRegistry._get") as mock_get:
            mock_block_class = Mock()
            mock_instance = self._create_mock_agent_block("my_agent")
            mock_block_class.return_value = mock_instance
            mock_get.return_value = mock_block_class

            flow = Flow.from_yaml(str(yaml_path))

        assert flow._agent_config_set is False
        assert not flow.is_agent_config_set()

    def test_from_yaml_without_agent_blocks_sets_flag_true(self):
        """Loading a YAML with only regular blocks sets _agent_config_set=True."""
        flow_config = {
            "metadata": {
                "name": "Regular Flow",
                "description": "Flow without agent blocks",
                "version": "1.0.0",
                "recommended_models": {
                    "default": "test-model",
                    "compatible": [],
                    "experimental": [],
                },
            },
            "blocks": [
                {
                    "block_type": "MockBlock",
                    "block_config": {
                        "block_name": "regular_block",
                        "input_cols": "input",
                        "output_cols": "output",
                    },
                }
            ],
        }

        yaml_path = Path(self.temp_dir) / "regular_flow.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(flow_config, f)

        with patch("sdg_hub.core.flow.serialization.BlockRegistry._get") as mock_get:
            mock_block_class = Mock()
            mock_instance = self._create_mock_block("regular_block")
            mock_block_class.return_value = mock_instance
            mock_get.return_value = mock_block_class

            flow = Flow.from_yaml(str(yaml_path))

        assert flow._agent_config_set is True
        assert flow.is_agent_config_set()
