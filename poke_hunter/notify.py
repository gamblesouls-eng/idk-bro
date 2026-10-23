"""Discord webhook notifications (Section 11).

Security rules implemented here:
  - the URL is never logged (only delivery status),
  - missing/invalid config disables notifications gracefully (no crash),
  - delivery is bounded by timeout + retries + Retry-After (429),
  - duplicate messages are suppressed via a short-term dedupe window.

Delivery is decoupled from the state machine: it never raises into game logic
and never causes an already-completed catch to be redone.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import DiscordConfig
from .logging_setup import get_logger

log = get_logger("notify")


class Notifier:
    def __init__(self, dc: DiscordConfig, instance_name: str = ""):
        self.cfg = dc
        self.url: Optional[str] = None
        self.source = "none"
        from .config import resolve_webhook
        self.url, self.source = resolve_webhook(dc)
        if self.url and not self.url.startswith(("http://", "https://")):
            log.error("Configured webhook is not a valid URL; notifications disabled")
            self.url = None
        if self.url:
            log.info("Discord webhook loaded from %s (URL not logged)", self.source)
        else:
            log.warning(
                "Discord webhook not configured (%s env var or .webhook file) - "
                "notifications disabled", dc.webhook_env_var,
            )
        self.instance_name = instance_name
        self._recent: dict = {}

    @property
    def enabled(self) -> bool:
        return self.url is not None

    def _dedupe_ok(self, key: str, window_s: float = 30.0) -> bool:
        now = time.monotonic()
        last = self._recent.get(key)
        self._recent[key] = now
        return last is None or (now - last) > window_s

    def send(self, message: str, dedupe_key: Optional[str] = None,
             condition: bool = True) -> bool:
        """Best-effort delivery. Returns success; never raises."""
        return self.send_embed(description=message,
                               dedupe_key=dedupe_key, condition=condition)

    def send_embed(self, description: str, title: str = "",
                   color: int = 0x5865F2, dedupe_key: Optional[str] = None,
                   condition: bool = True, image_path: Optional[str] = None,
                   fields: Optional[list] = None) -> bool:
        """Deliver a rich embed (optionally with a screenshot attachment).

        Best-effort: never raises. Returns success."""
        if not (self.enabled and condition and description):
            return False
        if dedupe_key and not self._dedupe_ok(dedupe_key):
            log.info("Suppressed duplicate notification (key=%s)", dedupe_key)
            return False
        if not self.url.startswith("https://discord.com/api/"):
            log.error("Webhook URL does not look like a Discord webhook; skipping send")
            return False
        import json

        import requests

        embed: dict = {
            "description": description,
            "color": color,
            "footer": {
                "text": f"poke-hunter · {self.instance_name or 'hunter'}"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if title:
            embed["title"] = title
        if fields:
            embed["fields"] = [
                {"name": name, "value": value, "inline": inline}
                for name, value, inline in fields
            ]
        payload = {"embeds": [embed]}
        files = None
        if image_path:
            try:
                path = Path(image_path)
                data = path.read_bytes()
                # light sanity cap (webhooks accept 8MB; stay well under)
                if len(data) <= 6_000_000:
                    embed["image"] = {"url": f"attachment://{path.name}"}
                    files = {"files[0]": (path.name, data, "image/png")}
            except OSError as exc:
                log.warning("screenshot attachment skipped: %s", exc)
        for attempt in range(1, self.cfg.max_attempts + 1):
            try:
                if files is not None:
                    resp = requests.post(
                        self.url,
                        data={"payload_json": json.dumps(payload)},
                        files=files,
                        timeout=self.cfg.timeout_s,
                        headers={"User-Agent": "poke-hunter/1.0"},
                    )
                else:
                    resp = requests.post(
                        self.url, json=payload, timeout=self.cfg.timeout_s,
                        headers={"User-Agent": "poke-hunter/1.0"},
                    )
            except requests.RequestException as exc:
                log.warning("Discord delivery attempt %d failed (network): %s",
                            attempt, type(exc).__name__)
                time.sleep(2.0)
                continue
            if resp.status_code in (200, 204):
                log.info("Discord notification delivered (embed)%s",
                         " + screenshot" if files else "")
                return True
            if resp.status_code == 429:
                retry_after = 2.0
                try:
                    retry_after = float(resp.json().get("retry_after", 2.0))
                except Exception:  # noqa: BLE001
                    pass
                log.warning("Discord rate limited; waiting %.1fs", retry_after)
                time.sleep(retry_after / 1000.0 if retry_after > 10 else retry_after)
                continue
            log.warning("Discord delivery failed: HTTP %s (no URL logged)", resp.status_code)
            time.sleep(1.5)
        return False

    # Configurable message builders ---------------------------------------
    _SUBJECT_SPLIT = " | "

    @staticmethod
    def _split_detected(detected_text: str) -> tuple:
        """Split the detected-text into a subject + inline aspect fields.

        `as_notification()` emits `Subject | Reskin: x | Gradient: y | ...`;
        each aspect becomes its own clean embed field instead of a
        run-on string."""
        chunks = (detected_text or "").split(Notifier._SUBJECT_SPLIT)
        subject = chunks[0].strip() or "Unknown Pokemon"
        fields = []
        for chunk in chunks[1:]:
            if ":" in chunk:
                name, _, value = chunk.partition(":")
                fields.append((name.strip(), f"**{value.strip() or '—'}**", True))
        return subject, fields

    def _encounter_screenshot(self, encounter_id: str) -> Optional[str]:
        """Newest encounter debug frame belonging to this encounter's
        timestamp (files are `instance_encounter_<stamp>_full.png`)."""
        try:
            stamp = encounter_id.split("-", 1)[1].rsplit("-", 1)[0]
        except (IndexError, AttributeError):
            return None
        candidates = sorted(
            Path("logs").glob(f"*encounter_{stamp}*_full.png"))
        return str(candidates[-1]) if candidates else None

    def target_detected(self, detected_text: str, encounter_id: str = "",
                        matched_target_ids: Optional[list[str]] = None) -> bool:
        subject, fields = self._split_detected(detected_text)
        if matched_target_ids:
            fields.append(("Criteria", ", ".join(matched_target_ids), True))
        return self.send_embed(
            title="Match found",
            color=0x57F287,
            description=f"**{subject}** is a target "
                        "— weakening to 1 HP, then Repeat Ball",
            fields=fields,
            dedupe_key=f"target:{encounter_id or detected_text}"
                       f"{'matched' if matched_target_ids else ''}",
            condition=bool(matched_target_ids))

    def captured(self, detected_text: str, encounter_id: str = "",
                 matched_target_ids: Optional[list[str]] = None,
                 save_status: Optional[str] = None) -> bool:
        subject, fields = self._split_detected(detected_text)
        if save_status:
            fields.append(("Save", save_status, True))
        return self.send_embed(
            title="Captured",
            color=0xFEE75C,
            description=f"**{subject}** caught!",
            fields=fields,
            image_path=self._encounter_screenshot(encounter_id),
            dedupe_key=f"captured:{encounter_id or detected_text}"
                       f"{'matched' if matched_target_ids else ''}",
            condition=bool(matched_target_ids))

    def save_result(self, detected_text: str, encounter_id: str, save_status: str,
                    matched_target_ids: Optional[list[str]] = None) -> bool:
        subject, fields = self._split_detected(detected_text)
        ok = "success" in save_status.lower() or "saved" in save_status.lower()
        return self.send_embed(
            title="Game saved" if ok else "Save NOT verified",
            color=0x57F287 if ok else 0xED4245,
            description=f"**{subject}** — {save_status}",
            fields=fields,
            dedupe_key=f"save:{encounter_id}:{save_status}")

    def capturing(self, detected_text: str,
                  strategy: Optional[str] = None) -> bool:
        subject, fields = self._split_detected(detected_text)
        default = strategy or "False Swipe to red HP, then Repeat Ball"
        description = f"**{subject}** — {default}"
        return self.send_embed(
            title="Weakening target",
            color=0x5865F2,
            description=description,
            fields=fields,
            condition=self.cfg.notify_capturing,
            dedupe_key=f"capturing:{detected_text}")

    def recovery_used(self, detail: str) -> bool:
        return self.send_embed(
            title="Recovery used", color=0xE67E22,
            description=detail,
            condition=self.cfg.notify_recovery,
            dedupe_key=f"recovery:{detail}")

    def failure(self, detail: str, key: Optional[str] = None) -> bool:
        return self.send_embed(
            title="Workflow failure", color=0xED4245,
            description=detail,
            condition=self.cfg.notify_failures,
            dedupe_key=key or f"fail:{detail}")


def target_text(detected_text: str) -> str:
    return detected_text
