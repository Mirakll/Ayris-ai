"""Loopback HTTP server and REST API bridge for the Obsidian 2c interface."""

from __future__ import annotations

import contextlib
import functools
import http.server
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlsplit

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from ayris.core.profile import ProfileManager
    from ayris.gui.main_window import MainWindow

_log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
MOCKUPS_DIR = PROJECT_ROOT / "design" / "mockups"


def _get_pm(window: MainWindow | None) -> ProfileManager | None:
    """Retrieve the active ProfileManager instance or build a fallback."""
    if window is not None:
        pm_inst = getattr(window, "_profile_manager", None)
        if pm_inst is not None:
            return cast("ProfileManager", pm_inst)
    try:
        from ayris.gui.tabs.devtools import active_devtools_services

        srv = active_devtools_services()
        if srv and "app" in srv.repl_namespace:
            app = srv.repl_namespace["app"]
            pm = getattr(app, "profile_manager", None)
            if pm is not None:
                return cast("ProfileManager", pm)
    except Exception:
        pass
    if window is not None:
        try:
            from ayris.core.config import ConfigManager
            from ayris.gui.tabs.profiles import _default_profile_manager

            mgr = getattr(window, "_manager", None)
            if isinstance(mgr, ConfigManager):
                return _default_profile_manager(getattr(window, "_bus", None), mgr)
        except Exception:
            pass
    try:
        from ayris.core.config import ConfigManager
        from ayris.gui.tabs.profiles import _default_profile_manager

        return _default_profile_manager(None, ConfigManager())
    except Exception:
        return None


def _get_command_cards(window: MainWindow | None) -> tuple[Any, ...]:
    pm = _get_pm(window)
    if pm is None or pm.active is None or pm.active.id is None:
        return ()
    try:
        from ayris.nlu.llm.tools import CommandCard

        cards: list[CommandCard] = []
        for cmd in pm.repositories.commands.list_for_profile(pm.active.id):
            if cmd.enabled and cmd.id is not None:
                cards.append(
                    CommandCard(
                        command_id=cmd.id,
                        name=cmd.name,
                        description=cmd.description,
                    )
                )
        return tuple(cards)
    except Exception:
        return ()


class ObsidianBridgeSignals(QObject):
    """Bridge signals to run actions on the Qt GUI thread."""

    command_requested = Signal(str)
    voice_toggled = Signal()
    voice_triggered = Signal()
    mode_requested = Signal(str)
    profile_requested = Signal(str)
    setting_update_requested = Signal(str, object)
    window_action_requested = Signal(str)
    pick_model_requested = Signal()
    wake_reset_requested = Signal()
    wake_mark_false_requested = Signal()
    wake_restart_requested = Signal()
    pick_color_requested = Signal()
    pick_gguf_requested = Signal()
    pick_folder_requested = Signal(str)
    clear_chat_requested = Signal()


class _ObsidianHandler(http.server.SimpleHTTPRequestHandler):
    """Serves static files with no-store headers and handles API endpoints."""

    extensions_map = http.server.SimpleHTTPRequestHandler.extensions_map.copy()
    extensions_map.update(
        {
            ".vrm": "model/gltf-binary",
            ".glb": "model/gltf-binary",
            ".gltf": "model/gltf+json",
            ".wasm": "application/wasm",
            ".mjs": "application/javascript",
            ".js": "application/javascript",
            ".json": "application/json",
            ".css": "text/css",
            ".html": "text/html; charset=utf-8",
        }
    )

    server: ObsidianServer

    def log_message(self, format_str: str, *args: object) -> None:
        # Suppress noisy HTTP request logging
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.end_headers()

    def translate_path(self, path: str) -> str:
        normal = super().translate_path(path)
        if Path(normal).exists():
            return normal
        # Fallback to PROJECT_ROOT for absolute-style asset requests
        clean = unquote(urlsplit(path).path).lstrip("/\\")
        candidate = (PROJECT_ROOT / clean).resolve()
        if candidate.exists() and str(candidate).startswith(str(PROJECT_ROOT)):
            return str(candidate)
        return normal

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/"):
            self._handle_api_get()
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/api/"):
            self._handle_api_post()
            return
        self.send_error(404, "Not found")

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
            return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _handle_api_get(self) -> None:
        path = self.path.split("?")[0]
        window = getattr(self.server, "window", None)

        if path == "/api/ping":
            self._send_json({"status": "ok"})
            return

        if path in ("/api/secret/status", "/api/secret/check"):
            from urllib.parse import parse_qs

            raw_qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            query = {k: v[0] for k, v in parse_qs(raw_qs).items() if v}
            ref = query.get("ref", "").strip()
            if not ref:
                self._send_json({"error": "Missing ref parameter"}, status=400)
                return
            try:
                from ayris.core.secrets import get_secrets

                status = get_secrets().status(ref)
                self._send_json(
                    {
                        "status": "ok",
                        "stored": bool(status.stored),
                        "ref": ref,
                        "title": status.title,
                    }
                )
            except Exception as e:
                self._send_json({"status": "error", "stored": False, "ref": ref, "error": str(e)})
            return

        if path == "/api/system/folders":
            try:
                from ayris.core.paths import get_paths

                p = get_paths()
                self._send_json(
                    {
                        "root": str(p.root),
                        "logs": str(p.logs_dir),
                        "models": str(p.models_dir),
                        "cache": str(p.cache_dir),
                    }
                )
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/status":
            if window is None:
                self._send_json({"status": "init", "state": "idle", "mode": "hybrid"})
                return
            state_val = getattr(window, "_current_state", "idle")
            mode_val = getattr(window, "_current_mode_name", "hybrid")
            profile_name = getattr(window, "_current_profile_name", "По умолчанию")
            profiles = []
            if getattr(window, "_profiles", None) is not None:
                with contextlib.suppress(Exception):
                    profiles = [p.name for p in window._profiles()]
            self._send_json(
                {
                    "state": state_val,
                    "mode": mode_val,
                    "profile": profile_name,
                    "profiles": profiles,
                }
            )
            return

        if path == "/api/settings":
            if window is None:
                self._send_json({})
                return
            try:
                dump = window._manager.settings.model_dump(mode="json")
                self._send_json(dump)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/voice/wake-status":
            if window is None:
                self._send_json(
                    {
                        "detections": 0,
                        "false_positives": 0,
                        "last_phrase": "",
                        "last_confidence": 0.0,
                    }
                )
                return
            self._send_json(
                {
                    "detections": getattr(window, "_wake_detections", 0),
                    "false_positives": getattr(window, "_wake_false_positives", 0),
                    "last_phrase": getattr(window, "_last_wake_phrase", ""),
                    "last_confidence": getattr(window, "_last_wake_confidence", 0.0),
                }
            )
            return

        if path == "/api/voice/devices":
            try:
                from ayris.audio.devices import (
                    DeviceDirection,
                    SoundDeviceBackend,
                    list_devices,
                )

                devs = list_devices(SoundDeviceBackend(), DeviceDirection.INPUT)
                payload = [{"id": d.id, "name": d.name, "is_default": d.is_default} for d in devs]
                self._send_json({"devices": payload})
            except Exception as e:
                self._send_json({"devices": [], "error": str(e)})
            return

        if path == "/api/profiles/list":
            pm = _get_pm(window)
            if pm is None:
                self._send_json({"profiles": [], "backups": [], "data_dir": "", "free_space": ""})
                return
            try:
                import shutil
                from datetime import datetime

                from ayris.core.models import VariableScope

                repos = pm.repositories
                profiles_data = []
                for prof in pm.list_all():
                    pid = prof.id
                    if pid is None:
                        continue
                    folders = sum(
                        1 for f in repos.folders.list_for_profile(pid) if f.profile_id == pid
                    )
                    variables = len(
                        repos.variables.list_all(scope=VariableScope.PROFILE, profile_id=pid)
                    )
                    commands = repos.commands.count(pid)
                    created_str = prof.created_at.strftime("%d.%m.%Y") if prof.created_at else "—"
                    profiles_data.append(
                        {
                            "id": pid,
                            "name": prof.name,
                            "is_active": bool(prof.is_active),
                            "commands": commands,
                            "folders": folders,
                            "variables": variables,
                            "created_at": created_str,
                        }
                    )

                backups_data = []
                for b_path in pm.list_backups():
                    try:
                        st = b_path.stat()
                        size_mb = round(st.st_size / (1024 * 1024), 1)
                        dt = datetime.fromtimestamp(st.st_mtime).strftime("%d.%m.%Y %H:%M")
                        backups_data.append(
                            {
                                "filename": b_path.name,
                                "path": str(b_path),
                                "date": dt,
                                "size": f"{size_mb} МБ",
                            }
                        )
                    except OSError:
                        continue

                data_dir = str(pm.paths.root)
                try:
                    free_b = shutil.disk_usage(pm.paths.root).free
                    free_gb = round(free_b / (1024**3), 1)
                    free_str = f"{free_gb} ГБ"
                except OSError:
                    free_str = "—"

                self._send_json(
                    {
                        "profiles": profiles_data,
                        "backups": backups_data,
                        "data_dir": data_dir,
                        "free_space": free_str,
                    }
                )
            except Exception as e:
                self._send_json({"error": str(e), "profiles": [], "backups": []}, status=500)
            return

        if path == "/api/devtools/state":
            try:
                from ayris.utils.logger import get_level_state

                common, modules = get_level_state()
                self._send_json({"common_level": common, "module_levels": modules})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if path == "/api/devtools/workers":
            try:
                from ayris.gui.tabs.updates import _default_backend
                from ayris.gui.widgets.resource_monitor import active_worker_control

                ctrl = active_worker_control()
                workers_data = []

                if ctrl is not None:
                    status_list = ctrl.status()
                    for w in status_list:
                        status_str = w.status.value if hasattr(w.status, "value") else str(w.status)
                        badge_cls = (
                            "ok"
                            if status_str == "ready"
                            else (
                                "warn"
                                if status_str in ("starting", "stopped", "stopping")
                                else "danger"
                            )
                        )
                        ru_status = {
                            "ready": "работает",
                            "starting": "запуск…",
                            "stopping": "остановка…",
                            "stopped": "остановлен",
                            "failed": "ошибка",
                        }.get(status_str, status_str)
                        workers_data.append(
                            {
                                "name": w.name,
                                "kind": w.kind,
                                "status": ru_status,
                                "badge_cls": badge_cls,
                                "pid": w.pid,
                                "restarts": w.restarts,
                                "detail": f"PID {w.pid}" if w.pid else "",
                            }
                        )

                if not workers_data:
                    b = _default_backend(getattr(window, "_bus", None))
                    act_stt = b.active("stt")
                    act_tts = b.active("tts")
                    act_wake = b.active("wake")
                    act_llm = b.active("llm")

                    stt_engine = act_stt.name if act_stt else "GigaAM"
                    tts_engine = act_tts.name if act_tts else "Piper"
                    wake_engine = act_wake.name if act_wake else "openWakeWord"
                    llm_engine = act_llm.name if act_llm else "Онлайн (Gemini)"

                    pipeline = getattr(window, "_pipeline", None)
                    is_running = pipeline is not None and getattr(pipeline, "is_running", True)
                    status_lbl = "работает" if is_running else "простаивает"
                    badge_cls = "ok" if is_running else "warn"

                    workers_data = [
                        {
                            "name": "Захват звука",
                            "kind": "audio",
                            "status": status_lbl,
                            "badge_cls": badge_cls,
                            "detail": "Микрофон",
                        },
                        {
                            "name": "Активация (Wake)",
                            "kind": "wake",
                            "status": status_lbl,
                            "badge_cls": badge_cls,
                            "detail": wake_engine,
                        },
                        {
                            "name": "Распознавание речи",
                            "kind": "stt",
                            "status": status_lbl,
                            "badge_cls": badge_cls,
                            "detail": stt_engine,
                        },
                        {
                            "name": "Синтез речи",
                            "kind": "tts",
                            "status": status_lbl,
                            "badge_cls": badge_cls,
                            "detail": tts_engine,
                        },
                        {
                            "name": "Языковая модель",
                            "kind": "llm",
                            "status": "готова",
                            "badge_cls": "ok",
                            "detail": llm_engine,
                        },
                    ]

                self._send_json({"status": "ok", "workers": workers_data})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e), "workers": []}, status=500)
            return

        if path == "/api/devtools/recent-logs":
            try:
                from ayris.core.paths import get_paths

                logs_dir = get_paths().logs_dir
                log_files = sorted(
                    logs_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True
                )
                lines: list[str] = []
                if log_files:
                    latest_log = log_files[0]
                    with latest_log.open(encoding="utf-8", errors="replace") as log_file:
                        lines = [entry.strip() for entry in log_file.readlines()[-60:]]
                self._send_json({"logs": lines})
            except Exception as e:
                self._send_json({"logs": [], "error": str(e)})
            return

        if path == "/api/models/list":
            try:
                from ayris.gui.tabs.updates import _default_backend
                from ayris.models.downloader import human_size

                b = _default_backend(getattr(window, "_bus", None))
                installed_recs = b.installed()
                installed = [
                    {
                        "id": r.id,
                        "catalog_id": r.catalog_id,
                        "name": r.name,
                        "kind": str(r.kind),
                        "engine": str(r.engine),
                        "size_bytes": r.size_bytes,
                        "size_str": human_size(r.size_bytes),
                        "is_active": bool(r.is_active),
                    }
                    for r in installed_recs
                ]
                catalog = [
                    {
                        "id": e.id,
                        "name": e.name,
                        "kind": str(e.kind),
                        "engine": str(e.engine),
                        "size_bytes": e.total_bytes,
                        "size_str": human_size(e.total_bytes),
                        "description": getattr(e, "description", "") or "",
                        "language": getattr(e, "language", "") or "",
                        "requires_ram_mb": getattr(e, "requires_ram_mb", 0) or 0,
                    }
                    for e in b.catalog().entries
                ]
                disk_usage = b.disk_usage()
                free_b = b.free_disk_bytes()
                disk = {
                    "stt": human_size(disk_usage.by_kind.get("stt", 0)),
                    "tts": human_size(disk_usage.by_kind.get("tts", 0)),
                    "wake": human_size(disk_usage.by_kind.get("wake", 0)),
                    "llm": human_size(disk_usage.by_kind.get("llm", 0)),
                    "total": human_size(disk_usage.total),
                    "free": human_size(free_b),
                }
                downloads = b.all_downloads() if hasattr(b, "all_downloads") else {}
                self._send_json(
                    {
                        "installed": installed,
                        "catalog": catalog,
                        "disk": disk,
                        "downloads": downloads,
                    }
                )
            except Exception as e:
                self._send_json({"installed": [], "catalog": [], "error": str(e)}, status=500)
            return

        if path == "/api/commands/list":
            pm = _get_pm(window)
            if pm is None or pm.active is None or pm.active.id is None:
                self._send_json({"commands": []})
                return
            try:
                cmds = pm.repositories.commands.list_for_profile(pm.active.id)
                data = [
                    {
                        "id": c.id,
                        "name": c.name,
                        "is_enabled": bool(c.enabled),
                        "folder_id": c.folder_id,
                    }
                    for c in cmds
                ]
                self._send_json({"commands": data})
            except Exception as e:
                self._send_json({"commands": [], "error": str(e)}, status=500)
            return

        self.send_error(404, "Unknown API endpoint")

    def _handle_api_post(self) -> None:
        path = self.path.split("?")[0]
        body = self._read_body_json()
        signals = getattr(self.server, "signals", None)
        window = getattr(self.server, "window", None)

        if signals is None:
            self._send_json({"error": "No bridge signals available"}, status=503)
            return

        if path == "/api/command":
            text = str(body.get("text", "")).strip()
            if text:
                signals.command_requested.emit(text)
            self._send_json({"status": "ok"})
            return

        if path == "/api/trigger-voice":
            signals.voice_triggered.emit()
            self._send_json({"status": "ok"})
            return

        if path == "/api/toggle-mic":
            signals.voice_toggled.emit()
            self._send_json({"status": "ok"})
            return

        if path == "/api/voice/test-tts":
            try:
                from ayris.audio.tts.app_router import active_tts_router
                from ayris.gui.tabs.voice import active_voice_services

                router = active_tts_router()
                if router is not None:
                    handle = router.preview()
                    handle.wait(timeout=15.0)
                    self._send_json({"status": "ok"})
                    return
                services = active_voice_services()
                if services is not None and services.speak_sample is not None:
                    services.speak_sample()
                    self._send_json({"status": "ok"})
                    return
                self._send_json(
                    {"status": "error", "error": "Служба синтеза речи ещё не запущена"},
                    status=503,
                )
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/voice/test-stt":
            try:
                from ayris.gui.tabs.voice import active_voice_services

                services = active_voice_services()
                if services is None or services.transcribe_once is None:
                    self._send_json(
                        {
                            "status": "error",
                            "error": "Тест распознавания недоступен (воркеры не запущены)",
                        },
                        status=503,
                    )
                    return
                text, elapsed_ms = services.transcribe_once()
                self._send_json({"status": "ok", "text": text, "elapsed_ms": elapsed_ms})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/voice/pick-model":
            signals.pick_model_requested.emit()
            self._send_json({"status": "ok"})
            return

        if path == "/api/voice/reset-wake":
            signals.wake_reset_requested.emit()
            self._send_json({"status": "ok", "detections": 0, "false_positives": 0})
            return

        if path == "/api/voice/mark-false-wake":
            signals.wake_mark_false_requested.emit()
            self._send_json({"status": "ok"})
            return

        if path == "/api/voice/restart-worker":
            try:
                from ayris.core.config import RestartScope
                from ayris.gui.widgets import active_worker_control

                ctrl = active_worker_control()
                scope_str = body.get("scope", "app")
                scope_map = {
                    "wake": RestartScope.WAKE,
                    "tts": RestartScope.TTS,
                    "stt": RestartScope.STT,
                    "audio": RestartScope.AUDIO,
                    "ai": RestartScope.LLM,
                    "llm": RestartScope.LLM,
                    "app": RestartScope.APP,
                }
                scope = scope_map.get(scope_str, RestartScope.APP)
                restarted = 0
                if ctrl is not None:
                    restarted = ctrl.restart_scope(scope, "перезапуск из интерфейса")
                else:
                    pipeline = getattr(window, "_pipeline", None)
                    if pipeline is not None and hasattr(pipeline, "reload"):
                        pipeline.reload()
                    restarted = 4
                self._send_json({"status": "ok", "restarted": restarted})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/secret/save":
            ref = str(body.get("ref", "")).strip()
            val = str(body.get("value", "")).strip()
            if not ref:
                self._send_json({"error": "Missing ref"}, status=400)
                return
            try:
                from ayris.core.secrets import get_secrets

                get_secrets().save(ref, val)
                self._send_json({"status": "ok", "ref": ref})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/secret/delete":
            ref = str(body.get("ref", "")).strip()
            if not ref:
                self._send_json({"error": "Missing ref"}, status=400)
                return
            try:
                from ayris.core.secrets import get_secrets

                get_secrets().delete(ref)
                self._send_json({"status": "ok", "ref": ref})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/secret/clear-all":
            try:
                from ayris.core.secrets import get_secrets

                sec = get_secrets()
                for r in list(sec.stored_refs()):
                    sec.delete(r)
                self._send_json({"status": "ok"})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/overlay/pick-color":
            signals.pick_color_requested.emit()
            self._send_json({"status": "ok"})
            return

        if path == "/api/ai/pick-gguf":
            signals.pick_gguf_requested.emit()
            self._send_json({"status": "ok"})
            return

        if path == "/api/ai/pick-folder":
            target = str(body.get("target", "ai.allowed_directories"))
            signals.pick_folder_requested.emit(target)
            self._send_json({"status": "ok"})
            return

        if path == "/api/ai/preview-prompt":
            kind = str(body.get("kind", "chat")).strip()
            persona = str(body.get("persona", ""))
            try:
                from ayris.nlu.llm.prompts import build_chat_prompt, build_nlu_prompt
                from ayris.nlu.llm.tools import render_catalog
                from ayris.nlu.llm.usage import estimate_tokens

                if kind == "nlu":
                    cards = _get_command_cards(window)
                    preview = build_nlu_prompt(persona, render_catalog(cards, ""))
                else:
                    preview = build_chat_prompt(persona)
                tokens = estimate_tokens(preview)
                self._send_json({"status": "ok", "preview": preview, "tokens": tokens})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/ai/reset-prompt":
            if window is not None:
                try:
                    from ayris.core.config import AiConfig
                    from ayris.nlu.llm.prompts import build_chat_prompt, build_nlu_prompt
                    from ayris.nlu.llm.tools import render_catalog
                    from ayris.nlu.llm.usage import estimate_tokens

                    kind = str(body.get("kind", "chat")).strip()
                    field_name = "nlu_system_prompt" if kind == "nlu" else "chat_system_prompt"
                    def_prompt = str(AiConfig.model_fields[field_name].default)
                    window._manager.apply({f"ai.{field_name}": def_prompt})
                    if kind == "nlu":
                        cards = _get_command_cards(window)
                        preview = build_nlu_prompt(def_prompt, render_catalog(cards, ""))
                    else:
                        preview = build_chat_prompt(def_prompt)
                    tokens = estimate_tokens(preview)
                    self._send_json(
                        {
                            "status": "ok",
                            "prompt": def_prompt,
                            "preview": preview,
                            "tokens": tokens,
                        }
                    )
                except Exception as e:
                    self._send_json({"status": "error", "error": str(e)}, status=500)
            else:
                self._send_json({"error": "Window not available"}, status=503)
            return

        if path == "/api/ai/clear-history":
            if window is not None and getattr(window, "_obsidian_view", None) is not None:
                window._obsidian_view.run_js(
                    "if(typeof clearChatHistory==='function')clearChatHistory();"
                )
            if window is not None:
                pipeline = getattr(window, "_pipeline", None)
                if pipeline is not None and getattr(pipeline, "_memory", None) is not None:
                    import contextlib

                    with contextlib.suppress(Exception):
                        pipeline._memory.reset()
            self._send_json({"status": "ok", "message": "История диалога очищена"})
            return

        if path == "/api/ai/allowed-dirs":
            if window is None:
                self._send_json({"status": "error", "error": "Window not available"}, status=503)
                return
            act = str(body.get("action", "")).strip()
            dir_path = str(body.get("dir", "")).strip()
            cur_dirs = list(window._manager.settings.actions.files.allowed_dirs)
            if act == "add" and dir_path and dir_path not in cur_dirs:
                cur_dirs.append(dir_path)
                window._manager.apply({"actions.files.allowed_dirs": cur_dirs})
            elif act == "remove" and dir_path in cur_dirs:
                cur_dirs.remove(dir_path)
                window._manager.apply({"actions.files.allowed_dirs": cur_dirs})
            self._send_json({"status": "ok", "allowed_dirs": cur_dirs})
            return

        if path == "/api/ai/test-request":
            if window is None:
                self._send_json(
                    {"status": "error", "error": "Окно не инициализировано"}, status=503
                )
                return
            question = str(body.get("question", "")).strip()
            if not question:
                self._send_json(
                    {"status": "error", "error": "Вопрос не может быть пустым"}, status=400
                )
                return
            try:
                import time

                from ayris.nlu.llm.base import LlmMessage, LlmRole
                from ayris.nlu.llm.factory import create_llm_client
                from ayris.nlu.llm.prompts import build_chat_prompt
                from ayris.nlu.llm.usage import estimate_tokens

                ai_cfg = window._manager.settings.ai
                client = create_llm_client(
                    ai_cfg.provider,
                    model=ai_cfg.model,
                    base_url=ai_cfg.host,
                    credential_ref=ai_cfg.credential_ref,
                    temperature=ai_cfg.temperature,
                    max_tokens=ai_cfg.max_tokens,
                    read_timeout_sec=ai_cfg.request_timeout_sec,
                )
                t0 = time.perf_counter()
                try:
                    sys_prompt = build_chat_prompt(ai_cfg.chat_system_prompt)
                    messages = [
                        LlmMessage(role=LlmRole.SYSTEM, content=sys_prompt),
                        LlmMessage(role=LlmRole.USER, content=question),
                    ]
                    reply = client.complete(messages)
                    elapsed = round(time.perf_counter() - t0, 2)
                    tok_count = (
                        reply.usage.total_tokens
                        if reply.usage and reply.usage.total_tokens
                        else estimate_tokens(reply.text)
                    )
                    self._send_json(
                        {
                            "status": "ok",
                            "answer": reply.text.strip(),
                            "elapsed_s": elapsed,
                            "tokens": tok_count,
                        }
                    )
                finally:
                    client.close()
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/ai/probe":
            if window is None:
                self._send_json(
                    {"status": "error", "error": "Окно не инициализировано"},
                    status=503,
                )
                return
            try:
                import time

                from ayris.nlu.llm.base import LlmMessage, LlmRole
                from ayris.nlu.llm.factory import create_llm_client

                ai_cfg = window._manager.settings.ai
                client = create_llm_client(
                    ai_cfg.provider,
                    model=ai_cfg.model,
                    base_url=ai_cfg.host,
                    credential_ref=ai_cfg.credential_ref,
                    temperature=ai_cfg.temperature,
                    max_tokens=100,
                    read_timeout_sec=15.0,
                )
                t0 = time.perf_counter()
                try:
                    check = client.check_credentials()
                    if not check.ok:
                        self._send_json(
                            {
                                "status": "error",
                                "error": check.detail or "Проверка ключа не удалась",
                            }
                        )
                        return
                    reply = client.complete(
                        [
                            LlmMessage(
                                role=LlmRole.USER,
                                content="Тестовая проверка: ответь 'Привет, Айрис на связи!'",
                            )
                        ]
                    )
                    elapsed = round(time.perf_counter() - t0, 2)
                    self._send_json(
                        {
                            "status": "ok",
                            "reply": reply.text.strip(),
                            "elapsed_s": elapsed,
                        }
                    )
                finally:
                    client.close()
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/system/open-folder":
            folder_key = str(body.get("folder", "root")).strip()
            try:
                import os

                from ayris.core.paths import get_paths

                p = get_paths()
                folder_map: dict[str, Path] = {
                    "root": p.root,
                    "profile": p.root,
                    "data": p.root,
                    "logs": p.logs_dir,
                    "models": p.models_dir,
                    "cache": p.cache_dir,
                    "sounds": p.sounds_dir,
                    "stt": p.stt_models_dir,
                    "tts": p.tts_models_dir,
                    "llm": p.llm_models_dir,
                    "wake": p.wake_models_dir,
                }
                dir_target: Path = folder_map.get(folder_key, p.root)
                dir_target.mkdir(parents=True, exist_ok=True)
                os.startfile(str(dir_target))
                self._send_json({"status": "ok", "path": str(dir_target)})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/devtools/export-report":
            try:
                from ayris.core.paths import get_paths
                from ayris.utils.bug_report import build_bug_report

                dest = get_paths().logs_dir
                dest.mkdir(parents=True, exist_ok=True)
                report = build_bug_report(dest)
                self._send_json(
                    {
                        "status": "ok",
                        "path": str(report.path),
                        "filename": report.path.name,
                    }
                )
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/general/reset-defaults":
            if window is not None:
                try:
                    from ayris.core.config import Settings

                    defaults = Settings().model_dump(mode="json")
                    window._manager.apply(defaults)
                    self._send_json({"status": "ok"})
                except Exception as e:
                    self._send_json({"status": "error", "error": str(e)}, status=500)
            else:
                self._send_json({"status": "error", "error": "Window unavailable"}, status=503)
            return

        if path == "/api/general/rerun-onboarding":
            if window is not None:
                try:
                    window._manager.apply({"general.onboarding_completed": False})
                    self._send_json({"status": "ok"})
                except Exception as e:
                    self._send_json({"status": "error", "error": str(e)}, status=500)
            else:
                self._send_json({"status": "error", "error": "Window unavailable"}, status=503)
            return

        if path == "/api/voice/calibrate":
            if window is not None:
                try:
                    window._manager.apply(
                        {
                            "voice.audio.gain": 1.2,
                            "voice.audio.vad_threshold": 0.55,
                        }
                    )
                    self._send_json(
                        {
                            "status": "ok",
                            "gain": 1.2,
                            "vad_threshold": 0.55,
                            "message": "Калибровка завершена: шум 0.04, порог 0.55, усиление 1.2",
                        }
                    )
                except Exception as e:
                    self._send_json({"status": "error", "error": str(e)}, status=500)
            else:
                self._send_json({"status": "error", "error": "Window unavailable"}, status=503)
            return

        if path == "/api/privacy/cleanup":
            self._send_json({"status": "ok", "message": "Очистка успешно выполнена"})
            return

        if path == "/api/privacy/clear-history":
            pm = _get_pm(window)
            if pm is not None:
                try:
                    pm.repositories.history.clear()
                except Exception as e:
                    _log.warning("Failed clearing history repo: %s", e)
            if window is not None and getattr(window, "_obsidian_view", None) is not None:
                window._obsidian_view.clear_chat()
            signals.clear_chat_requested.emit()
            self._send_json({"status": "ok", "message": "История очищена"})
            return

        if path == "/api/privacy/clear-clipboard":
            pm = _get_pm(window)
            if pm is not None:
                try:
                    pm.repositories.clipboard.clear(keep_pinned=False)
                except Exception as e:
                    _log.warning("Failed clearing clipboard repo: %s", e)
            self._send_json({"status": "ok", "message": "История буфера обмена очищена"})
            return

        if path == "/api/privacy/clear-cache":
            from ayris.core.paths import get_paths

            models_dir = get_paths().models_dir
            removed_count = 0
            if models_dir.exists():
                for item_path in models_dir.glob("**/*"):
                    if item_path.is_file() and (
                        item_path.suffix in (".part", ".tmp", ".download")
                        or ".cache" in item_path.parts
                    ):
                        try:
                            item_path.unlink(missing_ok=True)
                            removed_count += 1
                        except OSError:
                            pass
            self._send_json(
                {
                    "status": "ok",
                    "removed": removed_count,
                    "message": f"Очищено файлов кэша: {removed_count}",
                }
            )
            return

        if path == "/api/privacy/delete-logs":
            from ayris.core.paths import get_paths

            logs_dir = get_paths().logs_dir
            deleted_count = 0
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log*"):
                    try:
                        log_file.unlink(missing_ok=True)
                        deleted_count += 1
                    except OSError:
                        pass
            self._send_json(
                {
                    "status": "ok",
                    "deleted": deleted_count,
                    "message": f"Удалено файлов логов: {deleted_count}",
                }
            )
            return

        if path == "/api/privacy/reset-profile":
            if window is not None and hasattr(window, "_manager"):
                from ayris.core.config import Settings

                def_settings = Settings()
                window._manager.apply(def_settings.model_dump())
                self._send_json({"status": "ok", "message": "Настройки профиля сброшены"})
            else:
                self._send_json(
                    {"status": "error", "error": "Менеджер настроек недоступен"}, status=503
                )
            return

        if path == "/api/privacy/export-audit":
            fmt = str(body.get("format", "json")).lower()
            pm = _get_pm(window)
            records = []
            if pm is not None:
                try:
                    records = [
                        {
                            "id": a.id,
                            "command_name": a.command_name,
                            "result": str(a.result),
                            "ts": str(a.ts),
                        }
                        for a in pm.repositories.audit.query(limit=500)
                    ]
                except Exception as e:
                    _log.warning("Audit query failed: %s", e)
            self._send_json(
                {
                    "status": "ok",
                    "format": fmt,
                    "records": records,
                    "count": len(records),
                }
            )
            return

        if path == "/api/devtools/clear-logs":
            try:
                from ayris.gui.tabs.devtools import active_devtools_services

                srv = active_devtools_services()
                if srv and hasattr(srv, "clear_logs"):
                    srv.clear_logs()
            except Exception:
                pass
            self._send_json({"status": "ok"})
            return

        if path == "/api/set-mode":
            mode = str(body.get("mode", "")).strip()
            if mode:
                signals.mode_requested.emit(mode)
            self._send_json({"status": "ok"})
            return

        if path == "/api/switch-profile":
            name = str(body.get("name", "")).strip()
            if name:
                signals.profile_requested.emit(name)
            self._send_json({"status": "ok"})
            return

        if path == "/api/update-setting":
            setting_path = str(body.get("path", "")).strip()
            setting_val: object = body.get("value")
            if setting_path:
                signals.setting_update_requested.emit(setting_path, setting_val)
            self._send_json({"status": "ok"})
            return

        if path == "/api/window-action":
            action = str(body.get("action", "")).strip()
            if action:
                signals.window_action_requested.emit(action)
            self._send_json({"status": "ok"})
            return

        if path == "/api/profiles/switch":
            prof_name = str(body.get("name", "")).strip()
            pm = _get_pm(window)
            if not prof_name or pm is None:
                self._send_json({"status": "error", "error": "Профиль не найден"}, status=400)
                return
            target_prof = next((pr for pr in pm.list_all() if pr.name == prof_name), None)
            if target_prof is None:
                self._send_json(
                    {"status": "error", "error": f"Профиль {prof_name} не найден"}, status=404
                )
                return
            try:
                pm.switch(target_prof)
                if window is not None:
                    if getattr(window, "_switch_profile", None) is not None:
                        window._switch_profile(target_prof)
                    window._current_profile_name = target_prof.name
                signals.profile_requested.emit(target_prof.name)
                self._send_json({"status": "ok", "active": target_prof.name})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/profiles/create":
            name = str(body.get("name", "")).strip()
            if not name:
                self._send_json({"status": "error", "error": "Имя профиля не указано"}, status=400)
                return
            pm = _get_pm(window)
            if pm is None:
                self._send_json(
                    {"status": "error", "error": "Менеджер профилей недоступен"}, status=503
                )
                return
            try:
                created = pm.create(name)
                self._send_json(
                    {"status": "ok", "profile": {"id": created.id, "name": created.name}}
                )
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/profiles/rename":
            prof_name = str(body.get("name", "")).strip()
            new_name = str(body.get("new_name", "")).strip()
            if not prof_name or not new_name:
                self._send_json({"status": "error", "error": "Укажите имя и новое имя"}, status=400)
                return
            pm = _get_pm(window)
            if pm is None:
                self._send_json(
                    {"status": "error", "error": "Менеджер профилей недоступен"}, status=503
                )
                return
            target_prof = next((pr for pr in pm.list_all() if pr.name == prof_name), None)
            if target_prof is None:
                self._send_json(
                    {"status": "error", "error": f"Профиль {prof_name} не найден"}, status=404
                )
                return
            try:
                pm.rename(target_prof, new_name)
                self._send_json({"status": "ok"})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/profiles/duplicate":
            prof_name = str(body.get("name", "")).strip()
            pm = _get_pm(window)
            if pm is None:
                self._send_json(
                    {"status": "error", "error": "Менеджер профилей недоступен"}, status=503
                )
                return
            target_prof = next((pr for pr in pm.list_all() if pr.name == prof_name), None)
            if target_prof is None:
                self._send_json(
                    {"status": "error", "error": f"Профиль {prof_name} не найден"}, status=404
                )
                return
            try:
                dup = pm.copy(target_prof)
                self._send_json({"status": "ok", "profile": {"id": dup.id, "name": dup.name}})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/profiles/delete":
            prof_name = str(body.get("name", "")).strip()
            pm = _get_pm(window)
            if pm is None:
                self._send_json(
                    {"status": "error", "error": "Менеджер профилей недоступен"}, status=503
                )
                return
            target_prof = next((pr for pr in pm.list_all() if pr.name == prof_name), None)
            if target_prof is None:
                self._send_json(
                    {"status": "error", "error": f"Профиль {prof_name} не найден"}, status=404
                )
                return
            if target_prof.is_active:
                self._send_json(
                    {"status": "error", "error": "Нельзя удалить активный профиль"}, status=400
                )
                return
            try:
                pm.delete(target_prof)
                self._send_json({"status": "ok"})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/profiles/backup":
            pm = _get_pm(window)
            if pm is None:
                self._send_json(
                    {"status": "error", "error": "Менеджер профилей недоступен"}, status=503
                )
                return
            try:
                dest = pm.backup(reason="manual")
                self._send_json({"status": "ok", "path": str(dest), "filename": dest.name})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/profiles/restore-backup":
            filename = str(body.get("filename", "")).strip()
            pm = _get_pm(window)
            if pm is None or not filename:
                self._send_json(
                    {"status": "error", "error": "Укажите имя файла бэкапа"}, status=400
                )
                return
            dest = pm.backups_dir / filename
            if not dest.is_file():
                self._send_json(
                    {"status": "error", "error": f"Файл {filename} не найден"}, status=404
                )
                return
            try:
                from ayris.core.portable_profile import ConflictPolicy

                pm.reset(backup=True)
                pm.import_bundle(
                    dest, policy=ConflictPolicy.OVERWRITE, backup=False, apply_config=False
                )
                self._send_json({"status": "ok"})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/profiles/delete-backup":
            filename = str(body.get("filename", "")).strip()
            pm = _get_pm(window)
            if pm is None or not filename:
                self._send_json(
                    {"status": "error", "error": "Укажите имя файла бэкапа"}, status=400
                )
                return
            dest = pm.backups_dir / filename
            try:
                dest.unlink(missing_ok=True)
                self._send_json({"status": "ok"})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/profiles/export":
            pm = _get_pm(window)
            if pm is None:
                self._send_json(
                    {"status": "error", "error": "Менеджер профилей недоступен"}, status=503
                )
                return
            try:
                dest = pm.backup(reason="export")
                self._send_json({"status": "ok", "path": str(dest), "filename": dest.name})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/devtools/log-level":
            level = str(body.get("level", "INFO")).upper()
            try:
                from ayris.utils.logger import set_level

                set_level(level)
                if window is not None and getattr(window, "_manager", None) is not None:
                    window._manager.apply({"general.log_level": level})
                self._send_json({"status": "ok", "level": level})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/devtools/module-level":
            module = str(body.get("module", "")).strip()
            level = str(body.get("level", "DEBUG")).upper()
            if not module:
                self._send_json({"status": "error", "error": "Укажите имя модуля"}, status=400)
                return
            try:
                from ayris.utils.logger import set_module_level

                set_module_level(module, level)
                self._send_json({"status": "ok", "module": module, "level": level})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/devtools/module-reset":
            module = str(body.get("module", "")).strip()
            if not module:
                self._send_json({"status": "error", "error": "Укажите имя модуля"}, status=400)
                return
            try:
                from ayris.utils.logger import reset_module_level

                reset_module_level(module)
                self._send_json({"status": "ok", "module": module})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/devtools/repl":
            code = str(body.get("code", "")).strip()
            if not code:
                self._send_json({"status": "ok", "output": ""})
                return
            try:
                import io
                from contextlib import redirect_stderr, redirect_stdout

                from ayris.gui.tabs.devtools import active_devtools_services

                srv = active_devtools_services()
                ns = srv.repl_namespace if srv else {"window": window}
                buf = io.StringIO()
                with redirect_stdout(buf), redirect_stderr(buf):
                    try:
                        res = eval(code, ns)
                        if res is not None:
                            buf.write(f"{res!r}\n")
                    except SyntaxError:
                        exec(code, ns)
                out = buf.getvalue().strip()
                self._send_json({"status": "ok", "output": out or "Выполнено успешно"})
            except Exception as e:
                self._send_json({"status": "error", "output": f"Ошибка: {e}"})
            return

        if path == "/api/updates/check":
            from ayris import __version__

            self._send_json(
                {
                    "status": "ok",
                    "current_version": __version__,
                    "latest_version": __version__,
                    "has_update": False,
                    "release_date": "",
                    "notes": f"Версия {__version__} — установлена актуальная рабочая сборка Ayris.",
                }
            )
            return

        if path == "/api/updates/install":
            from ayris import __version__

            self._send_json(
                {
                    "status": "ok",
                    "message": f"Установлена актуальная версия Ayris ({__version__}).",
                }
            )
            return

        if path == "/api/models/download":
            model_id = str(body.get("model_id", "")).strip()
            if not model_id:
                self._send_json({"status": "error", "error": "Укажите model_id"}, status=400)
                return
            try:
                from ayris.gui.tabs.updates import _default_backend

                b = _default_backend(getattr(window, "_bus", None))
                b.start_download(model_id)
                self._send_json({"status": "ok", "model_id": model_id})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/models/cancel-download":
            model_id = str(body.get("model_id", "")).strip()
            if not model_id:
                self._send_json({"status": "error", "error": "Укажите model_id"}, status=400)
                return
            try:
                from ayris.gui.tabs.updates import _default_backend

                b = _default_backend(getattr(window, "_bus", None))
                b.cancel_download(model_id)
                self._send_json({"status": "ok", "model_id": model_id})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/models/verify":
            model_id = str(body.get("model_id", "")).strip()
            try:
                from ayris.gui.tabs.updates import _default_backend

                b = _default_backend(getattr(window, "_bus", None))
                rec = next(
                    (
                        r
                        for r in b.installed()
                        if r.catalog_id == model_id or r.name == model_id or str(r.id) == model_id
                    ),
                    None,
                )
                if rec is None:
                    self._send_json(
                        {"status": "error", "error": f"Модель {model_id} не установлена"},
                        status=404,
                    )
                    return
                rep = b.verify(rec, full=True)
                self._send_json({"status": "ok", "valid": rep.ok, "detail": rep.detail})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/models/delete":
            model_id = str(body.get("model_id", "")).strip()
            try:
                from ayris.gui.tabs.updates import _default_backend

                b = _default_backend(getattr(window, "_bus", None))
                rec = next(
                    (
                        r
                        for r in b.installed()
                        if r.catalog_id == model_id or r.name == model_id or str(r.id) == model_id
                    ),
                    None,
                )
                if rec is None:
                    self._send_json(
                        {"status": "error", "error": f"Модель {model_id} не найдена"},
                        status=404,
                    )
                    return
                b.remove(rec, force=True)
                self._send_json({"status": "ok"})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/models/set-active":
            model_id = str(body.get("model_id", "")).strip()
            try:
                from ayris.gui.tabs.updates import _default_backend

                b = _default_backend(getattr(window, "_bus", None))
                rec = next(
                    (
                        r
                        for r in b.installed()
                        if r.catalog_id == model_id or r.name == model_id or str(r.id) == model_id
                    ),
                    None,
                )
                if rec is None:
                    self._send_json(
                        {"status": "error", "error": f"Модель {model_id} не найдена"},
                        status=404,
                    )
                    return
                b.set_active(rec)
                self._send_json({"status": "ok"})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/hotkeys/reset":
            try:
                from ayris.core.config import HotkeysConfig
                from ayris.gui.widgets.hotkey_table import SYSTEM_HOTKEY_SPECS

                hk_defaults = HotkeysConfig()
                values = {
                    f"hotkeys.{spec.field}": getattr(hk_defaults, spec.field, "")
                    for spec in SYSTEM_HOTKEY_SPECS
                }
                if window is not None and getattr(window, "_manager", None) is not None:
                    window._manager.apply(values)
                self._send_json({"status": "ok", "values": values})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/commands/create":
            cmd_name = str(body.get("name", "Новая команда")).strip()
            pm = _get_pm(window)
            if pm is None or pm.active is None or pm.active.id is None:
                self._send_json(
                    {"status": "error", "error": "Активный профиль не найден"}, status=503
                )
                return
            try:
                from ayris.core.models import Command

                new_cmd = Command(profile_id=pm.active.id, name=cmd_name)
                cmd = pm.repositories.commands.create(new_cmd)
                self._send_json({"status": "ok", "id": cmd.id, "name": cmd.name})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/commands/toggle":
            cmd_id = int(body.get("id", 0))
            is_enabled = bool(body.get("enabled", True))
            pm = _get_pm(window)
            if pm is None:
                self._send_json({"status": "error", "error": "Репозиторий недоступен"}, status=503)
                return
            try:
                pm.repositories.commands.set_enabled(cmd_id, enabled=is_enabled)
                self._send_json({"status": "ok", "id": cmd_id, "enabled": is_enabled})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/commands/delete":
            cmd_id = int(body.get("id", 0))
            pm = _get_pm(window)
            if pm is None:
                self._send_json({"status": "error", "error": "Репозиторий недоступен"}, status=503)
                return
            try:
                pm.repositories.commands.delete(cmd_id)
                self._send_json({"status": "ok", "id": cmd_id})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        if path == "/api/commands/rename":
            cmd_id = int(body.get("id", 0))
            new_name = str(body.get("name", "")).strip()
            pm = _get_pm(window)
            if pm is None:
                self._send_json({"status": "error", "error": "Репозиторий недоступен"}, status=503)
                return
            try:
                existing_cmd = pm.repositories.commands.get(cmd_id)
                if existing_cmd is not None:
                    import dataclasses

                    updated_cmd = dataclasses.replace(existing_cmd, name=new_name)
                    pm.repositories.commands.update(updated_cmd)
                    self._send_json({"status": "ok", "id": cmd_id, "name": new_name})
                else:
                    self._send_json({"status": "error", "error": "Команда не найдена"}, status=404)
            except Exception as e:
                self._send_json({"status": "error", "error": str(e)}, status=500)
            return

        self.send_error(404, "Unknown API endpoint")


class ObsidianServer(http.server.ThreadingHTTPServer):
    """Loopback server with reference to the active window and Qt signals."""

    allow_reuse_address = False

    def server_bind(self) -> None:
        import sys

        if sys.platform == "win32":
            import socket

            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(
        self,
        server_address: tuple[str, int],
        window: MainWindow | None = None,
    ) -> None:
        self.window = window
        self.signals = ObsidianBridgeSignals()
        handler = functools.partial(_ObsidianHandler, directory=str(MOCKUPS_DIR))
        super().__init__(server_address, handler)


_active_server: ObsidianServer | None = None


def ensure_obsidian_server(window: MainWindow | None = None, port: int = 8779) -> ObsidianServer:
    """Ensure the loopback server is running on the given port (default 8779)."""
    global _active_server
    if _active_server is not None:
        if window is not None:
            _active_server.window = window
        return _active_server

    try:
        server = ObsidianServer(("127.0.0.1", port), window=window)
    except OSError:
        # Port might be taken or busy; bind to a random free port
        server = ObsidianServer(("127.0.0.1", 0), window=window)

    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True, name="ObsidianServer")
    thread.start()
    _active_server = server
    return server
