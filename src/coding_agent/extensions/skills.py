from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .types import LoadedExtensions, RegisteredCommand, SkillSpec


def discover_skill_paths(workspace_dir: str | Path, configured_paths: list[str] | None = None) -> list[Path]:
    workspace = Path(workspace_dir)
    paths: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        paths.append(resolved)

    def _collect_skill_files(root: Path) -> None:
        # AgentSkills style: <root>/<skill-name>/SKILL.md
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            for filename in ("SKILL.md", "skill.md"):
                target = child / filename
                if target.exists() and target.is_file():
                    _add(target)
                    break
        # Backward compatible legacy style: <root>/*.md
        for path in sorted(root.glob("*.md")):
            _add(path)

    home = Path.home()
    # OpenClaw precedence (high -> low), with LiaoClaw compatibility roots inserted near top.
    default_dirs = [
        workspace / "skills",
        workspace / ".liaoclaw" / "skills",
        workspace / ".Liaoclaw" / "skills",
        workspace / ".agents" / "skills",
        home / ".agents" / "skills",
        home / ".openclaw" / "skills",
    ]
    for root in default_dirs:
        if root.exists() and root.is_dir():
            _collect_skill_files(root)

    # Extra dirs are lowest precedence.
    for raw in configured_paths or []:
        target = Path(raw)
        if not target.is_absolute():
            target = workspace / raw
        if target.exists() and target.is_dir():
            _collect_skill_files(target)
        elif target.exists() and target.is_file() and target.suffix.lower() == ".md":
            _add(target)

    return paths


def load_skills(
    workspace_dir: str | Path,
    configured_paths: list[str] | None = None,
    allowed_skill_names: set[str] | None = None,
) -> LoadedExtensions:
    result = LoadedExtensions()
    seen_names: dict[str, str] = {}
    seen_cmds: dict[str, str] = {}
    config_ctx = _load_openclaw_compat_config(Path(workspace_dir))

    for path in discover_skill_paths(workspace_dir, configured_paths=configured_paths):
        try:
            raw_text = path.read_text(encoding="utf-8").lstrip("\ufeff").strip()
            if not raw_text:
                continue
            meta, text = _parse_skill_frontmatter(raw_text)

            openclaw_meta = _extract_openclaw_metadata(meta)
            eligible, reason = _is_skill_eligible(openclaw_meta, config_ctx)
            if not eligible:
                result.diagnostics.append(f"skill skipped: {path} ({reason})")
                continue

            fallback_name = path.parent.name if path.stem.lower() == "skill" else path.stem
            title = str(meta.get("name") or _extract_title(text) or fallback_name).strip()
            if not title:
                title = fallback_name

            if allowed_skill_names is not None and title not in allowed_skill_names:
                result.diagnostics.append(f"skill skipped by allowlist: {title} from {path}")
                continue

            if title in seen_names:
                result.diagnostics.append(f"skill name conflict: {title} from {path} ignored; winner={seen_names[title]}")
                continue

            cmd_default = _default_command_name(meta.get("name"), title, fallback_name)
            cmd = str(meta.get("command") or cmd_default).strip().lstrip("/")
            if not cmd:
                cmd = f"skill:{_slugify(fallback_name)}"
            desc = str(meta.get("description") or f"执行技能：{title}").strip()
            user_invocable = _as_bool(meta.get("user-invocable"), default=True)
            disable_model_invocation = _as_bool(meta.get("disable-model-invocation"), default=False)
            command_dispatch = str(meta.get("command-dispatch") or "").strip().lower()
            command_tool = str(meta.get("command-tool") or "").strip()
            command_arg_mode = str(meta.get("command-arg-mode") or "raw").strip().lower()

            skill = SkillSpec(
                name=title,
                command_name=cmd,
                description=desc,
                content=text,
                source_path=str(path),
            )
            seen_names[title] = str(path)

            result.skills.append(skill)
            if not disable_model_invocation:
                result.prompt_guidelines.append(f"技能约束（{title}）：按该技能流程执行。")
                result.append_prompts.append(f"## Skill: {title}\n{text}")

            if user_invocable:
                if cmd in seen_cmds:
                    result.diagnostics.append(
                        f"skill command conflict: /{cmd} from {path} ignored; winner={seen_cmds[cmd]}"
                    )
                else:
                    seen_cmds[cmd] = str(path)
                    if command_dispatch == "tool" and command_tool:
                        async def _tool_handler(
                            ctx,
                            _skill=skill,
                            _tool=command_tool,
                            _arg_mode=command_arg_mode,
                        ) -> str:
                            return await _dispatch_skill_command_to_tool(
                                ctx,
                                skill=_skill,
                                tool_name=_tool,
                                arg_mode=_arg_mode,
                            )

                        handler = _tool_handler
                    else:
                        handler = lambda ctx, _skill=skill: _render_skill_prompt(_skill, ctx.raw_text)

                    result.commands[cmd] = RegisteredCommand(
                        name=cmd,
                        description=desc,
                        source="skill",
                        handler=handler,
                    )

            result.loaded_paths.append(str(path))
        except Exception as exc:
            result.errors.append(f"{path}: {exc}")
    return result


def _extract_title(text: str) -> str | None:
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip()
    return None


def _parse_skill_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip().lstrip("\ufeff") != "---":
        return {}, text
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end < 0:
        return {}, text
    head = "\n".join(lines[1:end]).strip()
    meta: dict[str, Any] = {}
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(head)
        if isinstance(loaded, dict):
            meta = dict(loaded)
    except Exception:
        meta = _fallback_parse_frontmatter(head)

    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def _fallback_parse_frontmatter(text: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        val = v.strip().strip("'").strip('"')
        if key and val:
            meta[key] = val
    return meta


def _default_command_name(meta_name: Any, title: str, fallback_name: str) -> str:
    candidate = str(meta_name or "").strip()
    if candidate and re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
        return candidate
    if title and re.fullmatch(r"[A-Za-z0-9_-]+", title):
        return title
    return f"skill:{_slugify(fallback_name)}"


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _to_list_of_str(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _load_openclaw_compat_config(workspace: Path) -> dict[str, Any]:
    candidates = [
        workspace / "openclaw.json",
        workspace / ".openclaw" / "openclaw.json",
        workspace / ".liaoclaw" / "settings.json",
        workspace / ".Liaoclaw" / "settings.json",
        Path.home() / ".openclaw" / "openclaw.json",
    ]
    merged: dict[str, Any] = {}
    for path in candidates:
        try:
            if not path.exists() or not path.is_file():
                continue
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                _deep_merge(merged, loaded)
        except Exception:
            continue
    return merged


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _extract_openclaw_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    metadata = meta.get("metadata")
    parsed_metadata: dict[str, Any] = {}
    if isinstance(metadata, dict):
        parsed_metadata = dict(metadata)
    elif isinstance(metadata, str):
        candidate = metadata.strip()
        if candidate:
            try:
                loaded = json.loads(candidate)
                if isinstance(loaded, dict):
                    parsed_metadata = loaded
            except Exception:
                try:
                    import yaml  # type: ignore

                    loaded_yaml = yaml.safe_load(candidate)
                    if isinstance(loaded_yaml, dict):
                        parsed_metadata = loaded_yaml
                except Exception:
                    parsed_metadata = {}

    # Compatibility with flattened keys like metadata.openclaw.os
    flat_openclaw: dict[str, Any] = {}
    for key, value in meta.items():
        if not isinstance(key, str) or not key.startswith("metadata.openclaw."):
            continue
        suffix = key[len("metadata.openclaw.") :]
        _set_by_path(flat_openclaw, suffix.split("."), value)
    if flat_openclaw:
        parsed_metadata.setdefault("openclaw", {})
        if isinstance(parsed_metadata["openclaw"], dict):
            _deep_merge(parsed_metadata["openclaw"], flat_openclaw)

    openclaw_meta = parsed_metadata.get("openclaw")
    if isinstance(openclaw_meta, dict):
        return openclaw_meta
    return {}


def _set_by_path(root: dict[str, Any], keys: list[str], value: Any) -> None:
    cur: dict[str, Any] = root
    for key in keys[:-1]:
        child = cur.get(key)
        if not isinstance(child, dict):
            child = {}
            cur[key] = child
        cur = child
    cur[keys[-1]] = value


def _config_truthy(config: dict[str, Any], key_path: str) -> bool:
    parts = [part for part in key_path.split(".") if part]
    if not parts:
        return False
    cur: Any = config
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return bool(cur)


def _is_skill_eligible(openclaw_meta: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    if _as_bool(openclaw_meta.get("always"), default=False):
        return True, "always=true"

    allowed_os = [os_name.lower() for os_name in _to_list_of_str(openclaw_meta.get("os"))]
    if allowed_os:
        current_os = sys.platform.lower()
        if current_os not in allowed_os:
            return False, f"os mismatch: requires {allowed_os}, current={current_os}"

    requires = openclaw_meta.get("requires")
    if not isinstance(requires, dict):
        requires = {}

    bins = _to_list_of_str(requires.get("bins"))
    for name in bins:
        if shutil.which(name) is None:
            return False, f"missing binary: {name}"

    any_bins = _to_list_of_str(requires.get("anyBins"))
    if any_bins and not any(shutil.which(name) for name in any_bins):
        return False, f"missing any binary in: {any_bins}"

    envs = _to_list_of_str(requires.get("env"))
    for name in envs:
        if not os.getenv(name) and not _config_truthy(config, name):
            return False, f"missing env/config: {name}"

    config_keys = _to_list_of_str(requires.get("config"))
    for key in config_keys:
        if not _config_truthy(config, key):
            return False, f"missing config: {key}"

    return True, "eligible"


def _extract_result_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts).strip()
    return ""


async def _dispatch_skill_command_to_tool(ctx, *, skill: SkillSpec, tool_name: str, arg_mode: str) -> str:
    session = getattr(ctx, "session", None)
    agent = getattr(session, "agent", None)
    state = getattr(agent, "state", None)
    tools = list(getattr(state, "tools", []) or [])
    tool = next((candidate for candidate in tools if getattr(candidate, "name", "") == tool_name), None)
    if tool is None:
        return f"技能 `{skill.name}` 需要工具 `{tool_name}`，但当前会话未启用该工具。"

    raw_args = " ".join(getattr(ctx, "args", []) or [])
    if arg_mode != "raw":
        raw_args = " ".join(getattr(ctx, "args", []) or [])
    params = {
        "command": raw_args,
        "commandName": f"/{getattr(ctx, 'name', skill.command_name)}",
        "skillName": skill.name,
    }
    try:
        value = tool.execute(f"skillcmd-{int(time.time() * 1000)}", params)
        if inspect.isawaitable(value):
            value = await value
        text = _extract_result_text(value)
        return text or f"已将 `/{ctx.name}` 分发到工具 `{tool_name}`，但工具未返回文本内容。"
    except Exception as exc:
        return f"技能 `{skill.name}` 分发工具 `{tool_name}` 失败：{exc}"


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower())
    normalized = normalized.strip("-")
    return normalized or "skill"


def _render_skill_prompt(skill: SkillSpec, raw_text: str) -> str:
    cmd_text = raw_text.strip() if raw_text else f"/{skill.command_name}"
    return (
        f"已应用技能 `{skill.name}`（命令：`{cmd_text}`）。\n"
        "请严格按照下述技能内容执行，并给出可执行结果：\n\n"
        f"{skill.content}"
    )
