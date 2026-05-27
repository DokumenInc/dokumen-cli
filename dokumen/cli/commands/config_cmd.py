"""
config command — interactive TUI for dokumen.yaml.

dokumen config          # opens interactive config editor (tabbed: status / config)
dokumen config init     # create starter dokumen.yaml
"""
import curses
import logging
import os
from importlib.metadata import version as get_version
from typing import Any, Dict, List, Optional, Tuple

import click
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "dokumen.yaml"

# ── config schema: sections → fields with types and descriptions ──
# types: bool, int, float, str, choice:a,b,c

SCHEMA: List[Tuple[str, List[Tuple[str, str, Any, str]]]] = [
    ("provider", [
        ("name", "choice:anthropic", "anthropic", "LLM provider"),
        ("model", "choice:claude-opus-4-6,claude-sonnet-4-6,claude-haiku-4-5-20251001", "claude-opus-4-6", "default model"),
    ]),
    ("compaction", [
        ("enabled", "bool", True, "auto-compact conversations"),
        ("token_threshold", "choice:0.7,0.8,0.9,0.95", 0.9, "compact at this % of budget"),
        ("token_budget", "choice:200000,400000,1000000", 1000000, "max token budget"),
        ("keep_recent_turns", "choice:5,10,15,20,30", 20, "turns to keep after compaction"),
        ("micro_compact_enabled", "bool", True, "trim stale tool results"),
        ("micro_compact_age_seconds", "choice:300,600,1800,3600", 3600, "seconds before trimming tool output"),
        ("micro_compact_max_chars", "choice:500,1000,2000,5000", 2000, "max chars per trimmed result"),
    ]),
    ("coordinator", [
        ("enabled", "bool", False, "[experimental] multi-agent orchestration"),
        ("max_workers", "choice:3,5,8,10", 5, "parallel worker agents"),
        ("synthesis_strategy", "choice:merge,vote,chain", "merge", "merge=concat  vote=majority  chain=sequential"),
        ("worker_timeout", "choice:900.0,1800.0,3600.0,7200.0", 2700.0, "per-worker timeout (sec)"),
        ("worker_model", "choice:,claude-opus-4-6,claude-sonnet-4-6,claude-haiku-4-5-20251001", "", "worker model override"),
        ("decompose_timeout", "choice:10.0,30.0,60.0,120.0,300.0", 60.0, "auto-decompose timeout (sec)"),
        ("decompose_model", "choice:,claude-opus-4-6,claude-sonnet-4-6,claude-haiku-4-5-20251001", "", "decompose planner model"),
        ("executor_mode", "choice:api,sdk", "api", "api=direct calls for subagents  sdk=claude code cli (legacy)"),
    ]),
    ("tasks", [
        ("enabled", "bool", False, "agent subtask tracking"),
        ("persist_to_disk", "bool", True, "save tasks to .dokumen-cache/"),
        ("max_tasks", "choice:100,200,500", 200, "max subtasks per run"),
    ]),
    ("skills", [
        ("enabled", "bool", True, "prompt-based skill injection"),
        ("dir", "choice:skills/,./skills,", "skills/", "custom skills folder"),
        ("include_system", "bool", True, "include system skills (qa-check, etc.)"),
        ("max_skills_per_prompt", "choice:5,10,15,20", 10, "skills injected per agent turn"),
    ]),
    ("mimick", [
        ("max_turns", "choice:10,20,30,50,80,100", 50, "max agent turns"),
        ("timeout", "choice:900.0,1800.0,3600.0,7200.0", 3600.0, "timeout (sec)"),
        ("model", "choice:,claude-opus-4-6,claude-sonnet-4-6,claude-haiku-4-5-20251001", "", "model override"),
        ("build_max_turns", "choice:10,30,50,80,120", 80, "build phase max turns"),
    ]),
    ("execution", [
        ("timeout", "choice:900,1800,3600,7200", 3600, "executor timeout (sec)"),
        ("retries", "choice:0,1,2,3", 2, "retry count on failure"),
    ]),
    ("explore", [
        ("enabled", "bool", True, "scan codebase before execution"),
        ("max_files", "choice:50,100,200,500", 100, "max files to discover"),
    ]),
]

# model → max context window
MODEL_CONTEXT = {
    "claude-opus-4-6": 1000000,
    "claude-sonnet-4-6": 1000000,
    "claude-haiku-4-5-20251001": 200000,
}


def _find_config() -> str:
    current = os.getcwd()
    for _ in range(10):
        path = os.path.join(current, DEFAULT_CONFIG_PATH)
        if os.path.exists(path):
            return path
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.join(os.getcwd(), DEFAULT_CONFIG_PATH)


def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: str, data: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _get(data: dict, section: str, key: str, default: Any) -> Any:
    return data.get(section, {}).get(key, default)


def _set(data: dict, section: str, key: str, value: Any) -> None:
    if section not in data:
        data[section] = {}
    data[section][key] = value


def _build_items(data: dict) -> List[dict]:
    items = []
    for section, fields in SCHEMA:
        items.append({"type": "header", "label": section})
        for key, ftype, default, desc in fields:
            value = _get(data, section, key, default)
            items.append({
                "type": "field",
                "section": section,
                "key": key,
                "ftype": ftype,
                "value": value,
                "default": default,
                "desc": desc,
            })
    return items


def _toggle_value(item: dict) -> Any:
    ftype = item["ftype"]
    value = item["value"]
    if ftype == "bool":
        return not value
    elif ftype.startswith("choice:"):
        choices = ftype.split(":", 1)[1].split(",")
        str_val = str(value)
        try:
            idx = choices.index(str_val)
            return choices[(idx + 1) % len(choices)]
        except ValueError:
            return choices[0]
    return value


def _auto_set_budget(data: dict, items: List[dict], model: str) -> None:
    budget = MODEL_CONTEXT.get(model)
    if budget is None:
        return
    _set(data, "compaction", "token_budget", budget)
    for item in items:
        if item.get("key") == "token_budget" and item.get("section") == "compaction":
            item["value"] = budget
            break


def _coerce_value(item: dict, raw: str) -> Any:
    orig = item["default"]
    if isinstance(orig, int):
        try:
            return int(raw)
        except ValueError:
            return orig
    elif isinstance(orig, float):
        try:
            return float(raw)
        except ValueError:
            return orig
    return raw


def _format_value(item: dict, data: dict = None) -> str:
    v = item["value"]
    if item["ftype"] == "bool":
        return "on" if v else "off"
    # for model overrides, show the main model when blank
    if item.get("key") in ("worker_model", "model", "decompose_model") and item.get("section") in ("coordinator", "mimick") and (v == "" or v is None):
        if data:
            main_model = _get(data, "provider", "model", "claude-opus-4-6")
            return main_model
        return "claude-opus-4-6"
    if v == "" or v is None:
        return str(item["default"]) if item["default"] else ""
    return str(v)


def _get_status_lines(data: dict, config_path: str) -> List[Tuple[str, str, int]]:
    lines = []

    try:
        ver = get_version("dokumen")
    except Exception:
        ver = "unknown"
    lines.append(("Version", ver, 6))
    lines.append(("Config", config_path, 8))
    lines.append(("cwd", os.getcwd(), 8))

    lines.append(("", "", 0))

    provider_name = _get(data, "provider", "name", "anthropic")
    model = _get(data, "provider", "model", "claude-opus-4-6")
    lines.append(("Provider", provider_name, 7))
    lines.append(("Model", model, 7))

    lines.append(("", "", 0))

    features = [
        ("compaction", "Compaction"),
        ("coordinator", "Coordinator"),
        ("tasks", "Tasks"),
        ("skills", "Skills"),
        ("explore", "Explore"),
    ]
    # look up schema defaults for "enabled" per section
    schema_defaults = {}
    for section, fields in SCHEMA:
        for key, ftype, default, desc in fields:
            if key == "enabled":
                schema_defaults[section] = default

    for section, label in features:
        section_data = data.get(section, {})
        enabled = section_data.get("enabled", schema_defaults.get(section, False))
        if enabled:
            lines.append((label, "enabled", 3))
        else:
            lines.append((label, "disabled", 4))

    lines.append(("", "", 0))

    # show key settings for enabled features
    comp_data = data.get("compaction", {})
    if comp_data.get("enabled", False):
        budget = comp_data.get("token_budget", 1000000)
        threshold = comp_data.get("token_threshold", 0.9)
        lines.append(("  token budget", str(budget), 5))
        lines.append(("  compact at", f"{float(threshold)*100:.0f}%", 5))

    coord_data = data.get("coordinator", {})
    if coord_data.get("enabled", False):
        workers = coord_data.get("max_workers", 5)
        strategy = coord_data.get("synthesis_strategy", "merge")
        lines.append(("  workers", str(workers), 5))
        lines.append(("  strategy", strategy, 5))

    mimick_data = data.get("mimick", {})
    if mimick_data:
        turns = mimick_data.get("max_turns", 50)
        m_model = mimick_data.get("model", "") or _get(data, "provider", "model", "claude-opus-4-6")
        lines.append(("Mimick", f"{turns} turns, {m_model}", 7))

    return lines


# ── TUI ──

TABS = ["Status", "Config"]


def _run_tui(stdscr, config_path: str) -> bool:
    data = _load_yaml(config_path)
    items = _build_items(data)

    cursor = 0
    for i, item in enumerate(items):
        if item["type"] == "field":
            cursor = i
            break

    scroll_offset = 0
    search = ""
    editing = False
    edit_buf = ""
    message = ""
    saved = False
    active_tab = 0

    curses.start_color()
    curses.use_default_colors()
    # 1: headers (purple)
    curses.init_pair(1, curses.COLOR_MAGENTA, -1)
    # 2: not used for bg anymore — we use A_REVERSE
    curses.init_pair(2, curses.COLOR_WHITE, -1)
    # 3: on/enabled — bright green
    curses.init_pair(3, curses.COLOR_GREEN, -1)
    # 4: off/disabled — magenta/pink
    curses.init_pair(4, curses.COLOR_MAGENTA, -1)
    # 5: dim info text — cyan
    curses.init_pair(5, curses.COLOR_CYAN, -1)
    # 6: search text / accents — yellow
    curses.init_pair(6, curses.COLOR_YELLOW, -1)
    # 7: purple for model/provider values
    curses.init_pair(7, curses.COLOR_MAGENTA, -1)
    # 8: normal white
    curses.init_pair(8, curses.COLOR_WHITE, -1)
    # 9: active tab
    curses.init_pair(9, curses.COLOR_WHITE, curses.COLOR_MAGENTA)
    # 10: inactive tab
    curses.init_pair(10, curses.COLOR_WHITE, -1)

    curses.curs_set(0)
    stdscr.keypad(True)

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        # ── tab bar ──
        col = 2
        for i, tab_name in enumerate(TABS):
            label = f" {tab_name} "
            try:
                if i == active_tab:
                    stdscr.addstr(0, col, label, curses.color_pair(9) | curses.A_BOLD)
                else:
                    stdscr.addstr(0, col, label, curses.color_pair(10) | curses.A_DIM)
            except curses.error:
                pass
            col += len(label) + 2

        try:
            stdscr.addstr(1, 1, "─" * (w - 2), curses.color_pair(7))
        except curses.error:
            pass

        if active_tab == 0:
            # ── STATUS TAB ──
            status_lines = _get_status_lines(data, config_path)
            row = 3
            for label, value, color in status_lines:
                if row >= h - 2:
                    break
                if not label and not value:
                    row += 1
                    continue
                try:
                    stdscr.addstr(row, 3, f"{label}:", curses.color_pair(8) | curses.A_BOLD)
                    stdscr.addstr(row, 3 + len(label) + 2, value, curses.color_pair(color))
                except curses.error:
                    pass
                row += 1

            try:
                stdscr.addstr(h - 2, 1, "─" * (w - 2), curses.color_pair(7))
                stdscr.addstr(h - 1, 2, "  tab switch  q quit", curses.color_pair(5) | curses.A_DIM)
            except curses.error:
                pass

        else:
            # ── CONFIG TAB ──
            if search:
                visible = []
                for item in items:
                    if item["type"] == "header":
                        visible.append(item)
                    elif search.lower() in item["key"].lower() or search.lower() in item["desc"].lower() or search.lower() in item["section"].lower():
                        visible.append(item)
                cleaned = []
                for i, item in enumerate(visible):
                    if item["type"] == "header":
                        has_field = any(v["type"] == "field" for v in visible[i+1:i+20] if v.get("type") != "header")
                        if has_field:
                            cleaned.append(item)
                    else:
                        cleaned.append(item)
                visible = cleaned
            else:
                visible = items

            if visible:
                while cursor < len(visible) and visible[cursor]["type"] != "field":
                    cursor += 1
                if cursor >= len(visible):
                    cursor = max(0, len(visible) - 1)
                    while cursor > 0 and visible[cursor]["type"] != "field":
                        cursor -= 1

            search_row = 2
            list_start = 5
            list_height = h - list_start - 3
            footer_row = h - 2

            # search bar
            try:
                stdscr.addstr(search_row, 1, "╭" + "─" * (w - 4) + "╮", curses.color_pair(7))
                stdscr.addstr(search_row + 1, 1, "│", curses.color_pair(7))
                if search:
                    stdscr.addstr(search_row + 1, 3, "/ " + search, curses.color_pair(6))
                else:
                    stdscr.addstr(search_row + 1, 3, "/ search settings...", curses.color_pair(5) | curses.A_DIM)
                stdscr.addstr(search_row + 1, w - 2, "│", curses.color_pair(7))
            except curses.error:
                pass

            # scrolling
            if cursor - scroll_offset >= list_height:
                scroll_offset = cursor - list_height + 1
            if cursor - scroll_offset < 0:
                scroll_offset = cursor

            for idx in range(scroll_offset, min(len(visible), scroll_offset + list_height)):
                item = visible[idx]
                row = list_start + (idx - scroll_offset)
                if row >= h - 3:
                    break

                if item["type"] == "header":
                    label = f"  [{item['label'].upper()}]"
                    try:
                        stdscr.addstr(row, 1, label, curses.color_pair(1) | curses.A_BOLD)
                    except curses.error:
                        pass
                else:
                    is_selected = idx == cursor
                    value_str = _format_value(item, data)
                    val_col = max(50, w - 35)

                    # value colors
                    if item["ftype"] == "bool":
                        val_color = curses.color_pair(3) | curses.A_BOLD if item["value"] else curses.color_pair(4)
                    elif item["key"] in ("model", "worker_model", "name"):
                        val_color = curses.color_pair(7)  # purple for model/provider
                    else:
                        val_color = curses.color_pair(8)

                    if is_selected:
                        # inverted text for selection — much more readable
                        try:
                            stdscr.addstr(row, 1, " " * (w - 2), curses.A_REVERSE)
                            stdscr.addstr(row, 2, f"  {item['desc']}", curses.A_REVERSE | curses.A_BOLD)
                            if editing:
                                stdscr.addstr(row, val_col, edit_buf + "▌", curses.A_REVERSE | curses.A_BOLD)
                            else:
                                stdscr.addstr(row, val_col, value_str, curses.A_REVERSE | curses.A_BOLD)
                        except curses.error:
                            pass
                    else:
                        try:
                            stdscr.addstr(row, 2, f"  {item['desc']}", curses.color_pair(8))
                            stdscr.addstr(row, val_col, value_str, val_color)
                        except curses.error:
                            pass

            # scroll indicator
            if len(visible) > list_height:
                below = len(visible) - scroll_offset - list_height
                if below > 0:
                    try:
                        stdscr.addstr(h - 4, 2, f"  ↓ {below} more", curses.color_pair(5) | curses.A_DIM)
                    except curses.error:
                        pass

            # footer
            try:
                stdscr.addstr(footer_row, 1, "─" * (w - 2), curses.color_pair(7))
                if editing:
                    hint = "  type value  enter=confirm  esc=cancel"
                else:
                    hint = "  ↑↓ navigate  enter/space=toggle  /=search  s=save  tab=switch  q=quit"
                stdscr.addstr(footer_row + 1, 1, hint, curses.color_pair(5) | curses.A_DIM)
                if message:
                    stdscr.addstr(footer_row + 1, w - len(message) - 3, message, curses.color_pair(3) | curses.A_BOLD)
            except curses.error:
                pass

        stdscr.refresh()

        try:
            key = stdscr.getch()
        except curses.error:
            continue

        # editing mode (config tab, str fields only)
        if editing and active_tab == 1:
            if key == 27:
                editing = False
                edit_buf = ""
            elif key in (10, 13):
                editing = False
                item = visible[cursor]
                item["value"] = edit_buf if edit_buf else item["default"]
                _set(data, item["section"], item["key"], item["value"])
                _save_yaml(config_path, data)
                saved = True
                message = "saved ✓"
                edit_buf = ""
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                edit_buf = edit_buf[:-1]
            elif 32 <= key < 127:
                edit_buf += chr(key)
            continue

        # global keys
        if key == ord("q") or key == 27:
            # auto-save on quit if anything changed
            current = yaml.dump(data, default_flow_style=False)
            original = yaml.dump(_load_yaml(config_path), default_flow_style=False)
            if current != original:
                _save_yaml(config_path, data)
                saved = True
            break
        elif key == ord("\t") or key == curses.KEY_RIGHT:
            if active_tab == 0:
                active_tab = 1
                message = ""
        elif key == curses.KEY_LEFT:
            if active_tab == 1:
                active_tab = 0
                message = ""

        # config tab keys
        elif active_tab == 1:
            if key == ord("s"):
                _save_yaml(config_path, data)
                message = "saved ✓"
                saved = True
            elif key == ord("/"):
                search = ""
                curses.curs_set(1)
                while True:
                    stdscr.erase()
                    h2, w2 = stdscr.getmaxyx()
                    try:
                        stdscr.addstr(2, 1, "╭" + "─" * (w2 - 4) + "╮", curses.color_pair(7))
                        stdscr.addstr(3, 1, "│", curses.color_pair(7))
                        stdscr.addstr(3, 3, "/ " + search, curses.color_pair(6))
                        stdscr.addstr(3, w2 - 2, "│", curses.color_pair(7))
                        stdscr.addstr(h2 - 1, 1, "  type to filter  enter=confirm  esc=clear", curses.color_pair(5) | curses.A_DIM)
                    except curses.error:
                        pass
                    stdscr.refresh()
                    sk = stdscr.getch()
                    if sk in (10, 13):
                        break
                    elif sk == 27:
                        search = ""
                        break
                    elif sk in (curses.KEY_BACKSPACE, 127, 8):
                        search = search[:-1]
                    elif 32 <= sk < 127:
                        search += chr(sk)
                curses.curs_set(0)
                cursor = 0
                scroll_offset = 0
            elif key == curses.KEY_UP:
                cursor -= 1
                while cursor > 0 and visible[cursor]["type"] != "field":
                    cursor -= 1
                if cursor < 0:
                    cursor = 0
                message = ""
            elif key == curses.KEY_DOWN:
                cursor += 1
                while cursor < len(visible) - 1 and visible[cursor]["type"] != "field":
                    cursor += 1
                if cursor >= len(visible):
                    cursor = len(visible) - 1
                message = ""
            elif key in (10, 13, ord(" ")):
                if cursor < len(visible) and visible[cursor]["type"] == "field":
                    item = visible[cursor]
                    if item["ftype"] == "bool" or item["ftype"].startswith("choice:"):
                        new_val = _toggle_value(item)
                        item["value"] = _coerce_value(item, new_val) if isinstance(new_val, str) else new_val
                        _set(data, item["section"], item["key"], item["value"])
                        if item["key"] == "model" and item["section"] == "provider":
                            _auto_set_budget(data, items, str(item["value"]))
                        _save_yaml(config_path, data)
                        saved = True
                        message = "saved ✓"
                    elif item["ftype"] == "str":
                        editing = True
                        edit_buf = str(item["value"]) if item["value"] else ""
                        message = ""

    return saved


@click.group("config", invoke_without_command=True)
@click.pass_context
def config(ctx):
    """interactive config editor for dokumen.yaml.

    opens a tabbed TUI with status overview and config editor.
    use arrow keys to navigate, enter/space to toggle, s to save.
    """
    if ctx.invoked_subcommand is None:
        config_path = _find_config()
        if not os.path.exists(config_path):
            click.echo(f"no dokumen.yaml found — run 'dokumen config init' first")
            return

        try:
            saved = curses.wrapper(_run_tui, config_path)
            if saved:
                click.echo(f"config saved to {config_path}")
        except Exception as e:
            click.echo(f"error: {e}", err=True)


@config.command("init")
def config_init():
    """create a starter dokumen.yaml with all feature sections."""
    path = os.path.join(os.getcwd(), DEFAULT_CONFIG_PATH)
    if os.path.exists(path):
        click.echo(f"{path} already exists — run 'dokumen config' to edit")
        return

    starter = {}
    for section, fields in SCHEMA:
        starter[section] = {}
        for key, ftype, default, desc in fields:
            starter[section][key] = default

    _save_yaml(path, starter)
    click.echo(f"created {path}")
