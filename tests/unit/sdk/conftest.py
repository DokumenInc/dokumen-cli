import pytest

from dokumen.sdk.query_runner import MockQueryRunner
from dokumen.sdk.testing import make_init, make_assistant, make_result


@pytest.fixture
def mock_runner():
    """A MockQueryRunner with a simple success sequence."""
    return MockQueryRunner(
        [make_init(), make_assistant("test response"), make_result("test response")]
    )
