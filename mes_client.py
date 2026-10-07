"""
mes_client.py
-------------
Generalized MES API client.

Differs from the originally-shared snippet in two ways:
1. All connection / stage / product parameters are passed in via a config
   dict at construction time (nothing is hard-coded as a module-level
   constant), so the same client can be pointed at any MES server / stage /
   product from the UI's Settings screen.
2. `create_stage_report()` follows the "Create Stage Report API"
   (/api/StageReport/create_stageReport) documented in
   Client-MES_API_Integration-FAIL-PASS.pdf, which reports a SINGLE serial
   number per call (no parent/child pair). The old create_marrying_data /
   child_qr flow from the reference snippet has been removed.

Sequence used by the UI (per PDF, page 1):
    1. login()
    2. previous_stage_check(serial_no)
    3. upload_file(file_path, serial_no)
    4. create_stage_report(serial_no, status, ...)
    5. logout()
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Optional

import requests


class MesClient:
    """
    Handles all MES API calls for a single-barcode test stage.

    Call login() once at startup.
    Call set_serial_log(serial_no) before each new unit's MES calls so all
    interactions for that unit are written to the correct log file (this
    satisfies the PDF's "Mandatory Tasks" logging requirement: Time, URL,
    Payload, Response for every API interaction).
    """

    def __init__(self, config: dict) -> None:
        """
        config keys (see mes_config.json for defaults):
            mes_ip, username, password, login_stage_name,
            stage_id, stage_name, product_id, product_name,
            operator_id, created_by, machine_no,
            results_dir, file_format
        """
        self.cfg = config
        self._token: Optional[str] = None
        self._base = f"http://{config['mes_ip']}"
        self._serial_log_path: Optional[str] = None

        os.makedirs(config.get("results_dir", "mes_logs"), exist_ok=True)

    # ---------------------------------------------------------------- utils

    def update_config(self, config: dict) -> None:
        """Allow the UI to push a fresh set of parameters at runtime."""
        self.cfg = config
        self._base = f"http://{config['mes_ip']}"
        os.makedirs(config.get("results_dir", "mes_logs"), exist_ok=True)

    def set_serial_log(self, serial_no: str) -> None:
        """Set the log file path for the current unit under test."""
        safe = serial_no.replace("/", "").replace("\\", "")
        results_dir = self.cfg.get("results_dir", "mes_logs")
        self._serial_log_path = os.path.join(results_dir, f"{safe}_mes_log.txt")

    def _log(self, entry: dict) -> None:
        """Append one MES API interaction entry to the per-serial log file."""
        if not self._serial_log_path:
            return
        try:
            with open(self._serial_log_path, "a", encoding="utf-8") as f:
                f.write("=" * 70 + "\n")
                f.write(f"Time     : {entry.get('time', '')}\n")
                f.write(f"URL      : {entry.get('url', '')}\n")
                payload = entry.get("payload")
                f.write("Payload  :\n")
                if payload is not None:
                    f.write(json.dumps(payload, indent=2) + "\n")
                else:
                    f.write("  (none)\n")
                f.write("Response :\n")
                resp = entry.get("response")
                if isinstance(resp, dict):
                    f.write(json.dumps(resp, indent=2) + "\n")
                else:
                    f.write(str(resp) + "\n")
                f.write("\n")
        except Exception as e:
            print(f"[MES-Log] Write error: {e}")

    @staticmethod
    def _now() -> str:
        return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _safe_json(resp) -> dict:
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}

    def _auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"}

    def is_logged_in(self) -> bool:
        return bool(self._token)

    # ---------------------------------------------------------------- login

    def login(self) -> tuple[bool, str]:
        """Login to MES. Returns (ok, message)."""
        url = f"{self._base}/user/user_login/"
        payload = {
            "username": self.cfg["username"],
            "password": self.cfg["password"],
            "stage_name": self.cfg["login_stage_name"],
        }
        ts = self._now()
        try:
            resp = requests.post(url, json=payload, timeout=15)
            self._log({
                "time": ts, "url": url, "payload": {**payload, "password": "****"},
                "response": {"status_code": resp.status_code, "body": self._safe_json(resp)},
            })
            if resp.status_code == 200:
                self._token = (resp.headers.get("Authorization") or
                               resp.headers.get("authorization", ""))
                if not self._token:
                    return False, "Login OK but Authorization header missing."
                return True, "Login successful."
            return False, f"Login failed: HTTP {resp.status_code}"
        except Exception as e:
            self._log({"time": ts, "url": url, "payload": payload, "response": str(e)})
            return False, f"Login exception: {e}"

    # ---------------------------------------------------------- stage check

    def previous_stage_check(self, serial_no: str) -> tuple[bool, str]:
        """
        Backwards-compatible wrapper around previous_stage_check_ex().
        Returns (ok, message) where ok=True means testing may proceed.
        """
        state, msg = self.previous_stage_check_ex(serial_no)
        return state in ("pass", "all_passed"), msg

    def previous_stage_check_ex(self, serial_no: str) -> tuple[str, str]:
        """
        Returns (state, message) where state is one of:
            "pass"        -> unit passed its previous stage, proceed.
            "all_passed"  -> MES says the board already cleared every stage.
                             This is NOT a failure: testing may proceed.
            "fail"        -> previous stage not passed, block per policy.
            "error"       -> could not be verified (network / auth / HTTP).
        """
        url = (f"{self._base}/api/StageReport/previous_stage_result_pts/"
               f"{serial_no}/{self.cfg['stage_id']}")
        ts = self._now()
        try:
            resp = requests.get(url, headers=self._auth_header(), timeout=15)
            body = self._safe_json(resp)
            self._log({"time": ts, "url": url, "payload": None,
                       "response": {"status_code": resp.status_code, "body": body}})

            if resp.status_code == 204:
                msg = (body.get("msg") if isinstance(body, dict) else None) or \
                    f"{serial_no}: Board has passed all stages already."
                return "all_passed", msg

            if resp.status_code != 200:
                err = (body.get("error", "") if isinstance(body, dict) else str(body)) \
                    or f"HTTP {resp.status_code}"
                return "error", f"{serial_no}: {err}"

            data = body.get("data", {}) if isinstance(body, dict) else {}
            status = data.get("status", False)
            msg = (body.get("msg") or body.get("error") or "") if isinstance(body, dict) else ""
            if status:
                return "pass", msg or "Previous stage validated."

            # Some deployments answer 200 + status:false for a board that has
            # already cleared every stage — treat that the same as 204.
            if "all stage" in str(msg).lower() or "all the stage" in str(msg).lower():
                return "all_passed", msg

            return "fail", msg or f"Previous stage not passed for {serial_no}."

        except Exception as e:
            msg = f"{serial_no}: Connection error - {e}"
            self._log({"time": ts, "url": url, "payload": None, "response": str(e)})
            return "error", msg

    # ---------------------------------------------------------------- upload

    def upload_file(self, file_path: str, serial_no: str, status_label: str) -> tuple[bool, str]:
        """
        Upload a file to MES. Renames a local copy to the mandatory MES
        naming convention first:
            <serial_number>_<stage_id>_<status>_<utc-date_time_seconds>.<file_format>
        Returns (ok, download_link_or_error).
        """
        url = f"{self._base}/uploads/files"
        ts = self._now()

        ext = self.cfg.get("file_format") or os.path.splitext(file_path)[1].lstrip(".") or "txt"
        stamp = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
        mes_filename = f"{serial_no}_{self.cfg['stage_id']}_{status_label}_{stamp}.{ext}"

        try:
            with open(file_path, "rb") as fh:
                resp = requests.post(
                    url,
                    files={"file": (mes_filename, fh)},
                    data={"product_id": self.cfg["product_id"], "serial_no": serial_no},
                    timeout=60,
                )
            body = self._safe_json(resp)
            self._log({
                "time": ts, "url": url,
                "payload": {"product_id": self.cfg["product_id"], "serial_no": serial_no,
                            "file": mes_filename},
                "response": {"status_code": resp.status_code, "body": body},
            })
            if resp.status_code == 201:
                link = body.get("download_link", "") if isinstance(body, dict) else ""
                return True, link
            err = body.get("error", str(body)) if isinstance(body, dict) else str(body)
            return False, err or f"HTTP {resp.status_code}"
        except Exception as e:
            self._log({"time": ts, "url": url, "payload": {"file": mes_filename},
                       "response": str(e)})
            return False, str(e)

    # ------------------------------------------------------------ report

    def create_stage_report(self, serial_no: str, status: bool,
                             download_link: str,
                             defects: Optional[list] = None,
                             extra_details: Optional[dict] = None) -> tuple[bool, str]:
        """
        Call the Create Stage Report API for a SINGLE serial number.
        Returns (ok, message).

        defects: list of {"defectLocation", "defectType", "description"}
                 used only when status is False.
        extra_details: optional extra key/values merged into "details"
                       for the PASS case (e.g. {"version": "1.1"}).
        """
        url = f"{self._base}/api/StageReport/create_stageReport"
        ts = self._now()

        if status:
            details = {"mac": "", "download_link": download_link}
            if extra_details:
                details.update(extra_details)
        else:
            details = {
                "download_link": download_link,
                "fieldsNew": defects or [],
                "child_serial_number": [],
            }

        payload = {
            "stage_id": self.cfg["stage_id"],
            "stage_name": self.cfg["stage_name"],
            "product_id": self.cfg["product_id"],
            "product_name": self.cfg["product_name"],
            "serial_no": serial_no,
            "status": bool(status),
            "operator_id": self.cfg["operator_id"],
            "created_by": self.cfg["created_by"],
            "line_name": None,
            "work_order": None,
            "mac": "",
            "machine_no": self.cfg["machine_no"],
            "details": details,
            "ct_label": "",
            "serial_check": False,
        }

        try:
            resp = requests.post(
                url, json=payload,
                headers={**self._auth_header(), "Content-Type": "application/json"},
                timeout=30,
            )
            body = self._safe_json(resp)
            self._log({"time": ts, "url": url, "payload": payload,
                       "response": {"status_code": resp.status_code, "body": body}})

            if resp.status_code in (200, 201):
                if isinstance(body, dict) and body.get("error"):
                    # HTTP 200 but MES reports a logical error (e.g. retry
                    # limit / TRC-IN messages described in the PDF).
                    return False, body["error"]
                msg = body.get("msg", "Stage report saved.") if isinstance(body, dict) else "Stage report saved."
                return True, msg

            err = body.get("error", str(body)) if isinstance(body, dict) else str(body)
            return False, err or f"HTTP {resp.status_code}"
        except Exception as e:
            self._log({"time": ts, "url": url, "payload": payload, "response": str(e)})
            return False, str(e)

    # ---------------------------------------------------------------- logout

    def logout(self) -> tuple[bool, str]:
        if not self._token:
            return True, "Not logged in."
        url = f"{self._base}/user/user_logout/"
        ts = self._now()
        try:
            resp = requests.post(url, headers=self._auth_header(), timeout=15)
            body = self._safe_json(resp)
            self._log({"time": ts, "url": url, "payload": None,
                       "response": {"status_code": resp.status_code, "body": body}})
            ok = resp.status_code == 200
            return ok, (body.get("message", "Logged out.") if isinstance(body, dict) else "Logged out.")
        except Exception as e:
            return False, str(e)
        finally:
            self._token = None
