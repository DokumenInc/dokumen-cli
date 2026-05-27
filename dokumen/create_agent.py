"""
Create Agent module for generating test scaffolds from natural language goals.

Generates valid test scaffolds by:
1. Exploring project knowledge to discover relevant files
2. Analyzing source material
3. Generating executor prompts and judge criteria
4. Outputting valid YAML scaffold format
"""
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import yaml

from .explore_agent import ExploreAgent, ExploreResult

if TYPE_CHECKING:
    from .agent_object import Provider
    from .tools_object import ToolDefinition

logger = logging.getLogger(__name__)


# =============================================================================
# System Prompts
# =============================================================================


CREATE_ANALYSIS_PROMPT = """You are a skill-test analyzer. Given a goal and source files, provide a brief analysis of what the source material covers.

## Instructions
1. Summarize what the source material describes
2. Identify key concepts, rules, or procedures
3. Note any specific values, thresholds, or requirements

Keep your analysis under 300 words."""


CREATE_SCAFFOLD_PROMPT = """You are a test scaffold generator for the Dokumen skill testing framework.

## Task
Generate a valid test scaffold YAML that validates whether an agent can perform the user's goal from grounded project knowledge.

## CRITICAL: Single Judge Rule
**Generate exactly ONE judge per test.** Keep tests focused and simple.
- Each test should validate ONE specific agent skill or user workflow
- Users can request additional judges in follow-up conversations
- Do NOT create multiple judges even if the documentation has many sections

## Test Scaffold Format
```yaml
name: kebab-case-test-name  # 3-5 words, max 40 chars, COMPLETE phrase
reason: |
  Brief description of what this test validates.

files:
  - path: relative/path/to/file.md

executor:
  system_prompt: "@prompts/documentation-validation.txt"
  user_prompt: |
    Describe WHAT to validate (e.g., "Verify the refund policy is accurate").
    Do NOT mention file paths - the executor discovers files automatically.
  tools:
    - read_file

judges:
  - name: accuracy
    system_prompt: |
      Evaluation criteria for this test.

      Return: {"verdict": "PASS|FAIL", "reason": "..."}

timeout: 120
```

## Guidelines
1. Use kebab-case for the test name (3-5 words, max 40 chars)
   - MUST be a complete phrase, not a sentence fragment
   - Good: `refund-policy-accuracy`, `api-token-expiry`
   - Bad: `verify-that-refund-policy-handles` (incomplete)
2. The executor user_prompt should be specific and actionable
3. **ONE judge only** - focused on the core validation goal
4. Judge criteria should be measurable and concise
5. Include relevant files discovered during exploration
6. Use @prompts/documentation-validation.txt for system_prompt
7. CRITICAL: Never include file paths in user_prompt. The executor discovers files via explore phase.

Output ONLY the YAML scaffold in a ```yaml code block."""


CREATE_BROWSER_SCAFFOLD_PROMPT = """You are a test scaffold generator for the Dokumen skill testing framework.

## Task
Generate a valid **browser test** scaffold YAML that validates web UI behavior.

## CRITICAL: Single Judge Rule
**Generate exactly ONE judge per test.** Keep tests focused and simple.

## Browser Test Scaffold Format
```yaml
name: kebab-case-test-name  # 3-5 words, max 40 chars, COMPLETE phrase
type: browser
reason: |
  Brief description of what this browser test validates.

files:
  - path: relative/path/to/credentials-or-fixture.txt

browser:
  headless: false
  save_video: "1920x1080"

executor:
  system_prompt: "@prompts/browser-testing.txt"
  user_prompt: |
    Navigate to https://example.com and verify the login flow.
    Read docs/credentials/pat.txt for the PAT token.
  tools:
    - browser_navigate
    - browser_click
    - browser_type
    - browser_take_screenshot
    - browser_snapshot
    - read_file
    - browser_wait

judges:
  - name: ui-check
    system_prompt: |
      Evaluate the browser test results.
      Return: {"verdict": "PASS|FAIL", "reason": "..."}
    tools:
      - browser_snapshot
      - browser_take_screenshot

timeout: 120
```

## Guidelines
1. Use kebab-case for the test name (3-5 words, max 40 chars)
   - MUST be a complete phrase, not a sentence fragment
   - Good: `login-flow-validation`, `dashboard-page-load`
   - Bad: `verify-that-login-page` (incomplete)
2. Always include `type: browser` and a `browser:` section
3. The executor user_prompt should include URLs to navigate to
4. File paths in user_prompt are allowed (for credentials, fixtures)
5. **ONE judge only** - focused on the core UI validation goal
6. Use @prompts/browser-testing.txt for system_prompt
7. Browser tools: browser_navigate, browser_click, browser_type, browser_take_screenshot, browser_snapshot, read_file, browser_wait
8. Judge tools can include browser_snapshot and browser_take_screenshot for visual verification

Output ONLY the YAML scaffold in a ```yaml code block."""


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class CreateResult:
    """Result from CreateAgent.create() method."""
    success: bool
    scaffold_yaml: str
    scaffold_dict: Dict[str, Any]
    name: str
    discovered_files: List[str]
    duration: float
    error: Optional[str] = None
    test_type: str = "standard"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "scaffold_yaml": self.scaffold_yaml,
            "scaffold_dict": self.scaffold_dict,
            "name": self.name,
            "discovered_files": self.discovered_files,
            "duration": self.duration,
            "error": self.error,
            "test_type": self.test_type,
        }


# =============================================================================
# CreateAgent Class
# =============================================================================


class CreateAgent:
    """Agent that generates test scaffolds from natural language goals."""

    def __init__(
        self,
        provider: "Provider",
        base_dir: str = ".",
        timeout: float = 120.0,
        tools: Optional[List["ToolDefinition"]] = None,
    ):
        """Initialize the CreateAgent.

        Args:
            provider: LLM provider for generating responses.
            base_dir: Base directory for file operations.
            timeout: Maximum time for the create operation.
            tools: List of tools available to the agent.
        """
        self.provider = provider
        self.base_dir = base_dir
        self.timeout = timeout
        self.tools = tools or []

        logger.debug(
            "create_agent.init",
            extra={
                "base_dir": base_dir,
                "timeout": timeout,
                "tools_count": len(self.tools),
            }
        )

    async def create(
        self,
        goal: str,
        files: Optional[List[str]] = None,
        existing_tests: Optional[List[str]] = None,
        on_progress: Optional[Callable[[str, Dict], None]] = None,
        test_type: str = "standard",
    ) -> CreateResult:
        """Generate a test scaffold from a natural language goal.

        Args:
            goal: What the test should validate.
            files: Optional list of files to test (auto-discovered if not provided).
            existing_tests: List of existing test names to avoid conflicts.
            on_progress: Optional callback for progress events.
            test_type: Type of test to generate ('standard' or 'browser').

        Returns:
            CreateResult with the generated scaffold.
        """
        start_time = time.time()

        # Validate test_type early
        if test_type not in ("standard", "browser"):
            duration = time.time() - start_time
            error_msg = f"Invalid test_type: '{test_type}'. Must be 'standard' or 'browser'"
            logger.error(f"[CREATE] {error_msg}")
            return CreateResult(
                success=False,
                scaffold_yaml="",
                scaffold_dict={},
                name="",
                discovered_files=[],
                duration=duration,
                error=error_msg,
                test_type=test_type,
            )

        logger.info(f"[CREATE] Starting scaffold generation for: {goal[:100]} (type={test_type})")

        # Emit start event
        if on_progress:
            on_progress("create_start", {"goal": goal})

        try:
            # Step 1: Discover files if not provided
            discovered_files = files or []
            if not discovered_files:
                logger.info("[CREATE] Step 1: Exploring documentation...")
                if on_progress:
                    on_progress("explore_start", {"goal": goal})

                explore_result = await self._run_explore(goal, on_progress)
                discovered_files = [f.path for f in explore_result.files]

                if on_progress:
                    on_progress("explore_complete", {
                        "files_found": len(discovered_files),
                        "files": discovered_files,
                    })

                logger.info(f"[CREATE] Found {len(discovered_files)} files")
            else:
                logger.info(f"[CREATE] Using {len(discovered_files)} provided files")

            # Step 2: Analyze documentation
            logger.info("[CREATE] Step 2: Analyzing documentation...")
            doc_analysis = await self._analyze_docs(discovered_files, goal)

            # Step 3: Generate scaffold (LLM picks the name)
            logger.info("[CREATE] Step 3: Generating scaffold...")
            temp_name = self._goal_to_kebab(goal)
            if on_progress:
                on_progress("scaffold_generating", {"name": temp_name})

            scaffold_yaml = await self._generate_scaffold(goal, temp_name, discovered_files, doc_analysis, test_type=test_type)

            # Step 4: Parse and extract name
            scaffold_dict = yaml.safe_load(scaffold_yaml)

            # Use LLM-generated name if valid, fall back to goal-based
            name = self._extract_name_from_scaffold(scaffold_dict, goal, existing_tests or [])
            logger.info(f"[CREATE] Final name: {name}")
            scaffold_dict["name"] = name

            # Sanitize type/browser fields based on test_type
            if test_type == "browser":
                scaffold_dict["type"] = "browser"
                if not scaffold_dict.get("browser"):
                    scaffold_dict["browser"] = {"headless": False, "save_video": "1920x1080"}
            else:
                scaffold_dict.pop("type", None)
                scaffold_dict.pop("browser", None)

            scaffold_yaml = yaml.dump(scaffold_dict, default_flow_style=False, sort_keys=False)

            duration = time.time() - start_time

            if on_progress:
                on_progress("scaffold_generated", {
                    "name": name,
                    "files": discovered_files,
                })

            logger.info(f"[CREATE] Scaffold generated successfully in {duration:.2f}s")

            return CreateResult(
                success=True,
                scaffold_yaml=scaffold_yaml,
                scaffold_dict=scaffold_dict,
                name=name,
                discovered_files=discovered_files,
                duration=duration,
                error=None,
                test_type=test_type,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[CREATE] Failed: {e}", exc_info=True)

            if on_progress:
                on_progress("create_error", {"error": str(e)})

            return CreateResult(
                success=False,
                scaffold_yaml="",
                scaffold_dict={},
                name="",
                discovered_files=[],
                duration=duration,
                error=str(e),
            )

    async def _run_explore(
        self,
        goal: str,
        on_progress: Optional[Callable] = None,
    ) -> ExploreResult:
        """Run the explore phase to find relevant documentation.

        Args:
            goal: The goal to explore documentation for.
            on_progress: Optional progress callback.

        Returns:
            ExploreResult with discovered files.
        """
        explore_agent = ExploreAgent(
            provider=self.provider,
            base_dir=self.base_dir,
            max_files=20,
            max_iterations=50,
            timeout=60.0,
            tools=self.tools,
        )

        return await explore_agent.explore(goal, on_progress)

    async def _analyze_docs(
        self,
        files: List[str],
        goal: str,
    ) -> str:
        """Analyze documentation content.

        Args:
            files: List of file paths to analyze.
            goal: The test goal for context.

        Returns:
            Analysis summary string.
        """
        # Build context with file list
        file_list = "\n".join(f"- {f}" for f in files)
        user_prompt = f"""Goal: {goal}

Documentation files to analyze:
{file_list}

Provide a brief analysis of what these documents likely cover based on their paths and the goal."""

        messages = [
            {"role": "system", "content": CREATE_ANALYSIS_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.provider.complete(messages)
        return response.get("content", "")

    async def _generate_scaffold(
        self,
        goal: str,
        name: str,
        files: List[str],
        doc_analysis: str,
        test_type: str = "standard",
    ) -> str:
        """Generate the test scaffold YAML.

        Args:
            goal: The test goal.
            name: The generated test name.
            files: Discovered files.
            doc_analysis: Documentation analysis.
            test_type: Type of test ('standard' or 'browser').

        Returns:
            YAML scaffold string.
        """
        # Select prompt based on test type
        prompt = CREATE_BROWSER_SCAFFOLD_PROMPT if test_type == "browser" else CREATE_SCAFFOLD_PROMPT

        file_list = "\n".join(f"- {f}" for f in files)
        user_prompt = f"""Generate a test scaffold for the following:

**Goal:** {goal}

**Test Name:** {name}

**Documentation Files:**
{file_list}

**Documentation Analysis:**
{doc_analysis}

Generate a complete, valid test scaffold YAML."""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.provider.complete(messages)
        content = response.get("content", "")

        return self._extract_yaml_from_response(content)

    def _extract_yaml_from_response(self, response: str) -> str:
        """Extract YAML content from LLM response.

        Args:
            response: The full LLM response.

        Returns:
            Extracted YAML string.
        """
        # Try to extract from code block
        yaml_match = re.search(r"```(?:yaml)?\s*\n(.*?)\n```", response, re.DOTALL)
        if yaml_match:
            return yaml_match.group(1).strip()

        # If no code block, assume the whole response is YAML
        # Strip any leading/trailing non-YAML text
        lines = response.strip().split("\n")

        # Find where YAML starts (look for 'name:' at start of line)
        yaml_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("name:"):
                yaml_start = i
                break

        return "\n".join(lines[yaml_start:]).strip()

    def _is_valid_name(self, name: str) -> bool:
        """Check if a name is valid for use as a test name.

        Args:
            name: The candidate test name.

        Returns:
            True if the name is valid kebab-case, ≤40 chars, and at least 2 words.
        """
        if not name or not isinstance(name, str):
            return False
        # Must be kebab-case (lowercase alphanumeric + hyphens, no leading/trailing)
        if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', name):
            return False
        # Max 40 chars
        if len(name) > 40:
            return False
        # At least 2 words (single word is too vague)
        if name.count('-') < 1:
            return False
        return True

    def _extract_name_from_scaffold(
        self,
        scaffold_dict: dict,
        goal: str,
        existing_tests: List[str],
    ) -> str:
        """Extract and validate the name from LLM-generated scaffold.

        Uses the LLM's name if valid (kebab-case, ≤40 chars, not a truncated fragment).
        Falls back to _goal_to_kebab() if the LLM name is missing or invalid.

        Args:
            scaffold_dict: Parsed scaffold dictionary from LLM.
            goal: The original goal string (for fallback).
            existing_tests: List of existing test names to avoid conflicts.

        Returns:
            A valid, unique test name.
        """
        llm_name = scaffold_dict.get("name", "")

        if self._is_valid_name(llm_name):
            # Ensure uniqueness
            if llm_name not in existing_tests:
                logger.info(f"[CREATE] Using LLM-generated name: {llm_name}")
                return llm_name
            # Add suffix for uniqueness
            for i in range(2, 100):
                candidate = f"{llm_name}-{i}"
                if candidate not in existing_tests:
                    logger.info(f"[CREATE] Using LLM name with suffix: {candidate}")
                    return candidate

        # Fallback to goal-based generation
        logger.info(f"[CREATE] LLM name invalid or missing ('{llm_name}'), falling back to goal-based name")
        return self._generate_unique_name(goal, existing_tests)

    def _goal_to_kebab(self, goal: str) -> str:
        """Convert a goal string to kebab-case name.

        Args:
            goal: The goal description.

        Returns:
            Kebab-case name string.
        """
        # Convert to lowercase
        name = goal.lower()

        # Remove special characters except spaces and hyphens
        name = re.sub(r"[^a-z0-9\s-]", "", name)

        # Replace spaces with hyphens
        name = re.sub(r"\s+", "-", name)

        # Remove multiple consecutive hyphens
        name = re.sub(r"-+", "-", name)

        # Strip leading/trailing hyphens
        name = name.strip("-")

        # Truncate to reasonable length (max 30 chars)
        if len(name) > 30:
            # Try to cut at a word boundary
            name = name[:30]
            last_hyphen = name.rfind("-")
            if last_hyphen > 15:
                name = name[:last_hyphen]

        return name

    def _generate_unique_name(
        self,
        goal: str,
        existing_tests: List[str],
    ) -> str:
        """Generate a unique kebab-case name from goal.

        Args:
            goal: The goal description.
            existing_tests: List of existing test names to avoid.

        Returns:
            Unique kebab-case name.
        """
        base_name = self._goal_to_kebab(goal)

        if base_name not in existing_tests:
            return base_name

        # Add numeric suffix to avoid conflict
        for i in range(2, 100):
            candidate = f"{base_name}-{i}"
            if candidate not in existing_tests:
                return candidate

        # Fallback (should never happen)
        raise ValueError(f"Could not generate unique name for: {goal}")
