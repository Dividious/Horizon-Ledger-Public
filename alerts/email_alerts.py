"""
Horizon Ledger — Email Alert & Newsletter Distribution System

Setup (one-time):
  1. Create secrets/email_config.json:
     {
       "smtp_host": "smtp.gmail.com",
       "smtp_port": 587,
       "smtp_user": "your.email@gmail.com",
       "smtp_pass": "your-app-password",
       "alert_email": "your.email@gmail.com",
       "newsletter_recipients": ["subscriber1@example.com"]
     }

  2. Gmail App Password setup:
     - Google Account → Security → 2-Step Verification → App passwords
     - Create app password for "Horizon Ledger"
     - Use that 16-char password (not your regular Gmail password)

Functions:
  send_alert(subject, body, level) — Send error/warning alert to yourself
  send_newsletter(pdf_path, recipients) — Distribute weekly PDF newsletter
  send_test_email() — Verify SMTP config works
"""

import json
import logging
import smtplib
import ssl
from datetime import date
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from config import SECRETS_DIR

log = logging.getLogger(__name__)

_EMAIL_CONFIG_PATH = SECRETS_DIR / "email_config.json"
_TEMPLATE_CONFIG_PATH = SECRETS_DIR / "email_config.json.template"

# ── Config loading ─────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load email config from secrets/email_config.json."""
    if not _EMAIL_CONFIG_PATH.exists():
        # Write a template if config doesn't exist
        template = {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "",
            "smtp_pass": "",
            "alert_email": "",
            "newsletter_recipients": [],
        }
        _TEMPLATE_CONFIG_PATH.write_text(
            json.dumps(template, indent=2) +
            "\n\n# Rename this file to email_config.json and fill in your details.\n"
        )
        log.warning(
            "Email config not found. Template written to %s. "
            "Email alerts will be disabled until configured.",
            _TEMPLATE_CONFIG_PATH,
        )
        return {}
    try:
        return json.loads(_EMAIL_CONFIG_PATH.read_text())
    except Exception as e:
        log.error("Failed to load email config: %s", e)
        return {}


def _is_configured() -> bool:
    """Return True if email config exists and has required fields."""
    cfg = _load_config()
    return bool(cfg.get("smtp_user") and cfg.get("smtp_pass") and cfg.get("alert_email"))


# ── Core send function ─────────────────────────────────────────────────────────

def _send_email(
    to: list[str],
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
    attachments: Optional[list[Path]] = None,
) -> bool:
    """
    Send an email via Gmail SMTP with TLS.
    Returns True on success, False on failure.
    """
    cfg = _load_config()
    if not cfg.get("smtp_user") or not cfg.get("smtp_pass"):
        log.warning("Email not configured — skipping send. See secrets/email_config.json.template")
        return False

    try:
        msg = MIMEMultipart("mixed")
        msg["From"]    = cfg["smtp_user"]
        msg["To"]      = ", ".join(to)
        msg["Subject"] = subject

        # Body (prefer HTML, fall back to text)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(body_text, "plain"))
        if body_html:
            alt.attach(MIMEText(body_html, "html"))
        msg.attach(alt)

        # Attachments
        for path in (attachments or []):
            path = Path(path)
            if not path.exists():
                log.warning("Attachment not found: %s", path)
                continue
            with open(path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{path.name}"',
            )
            msg.attach(part)

        ctx = ssl.create_default_context()
        with smtplib.SMTP(cfg["smtp_host"], cfg.get("smtp_port", 587)) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.login(cfg["smtp_user"], cfg["smtp_pass"])
            server.sendmail(cfg["smtp_user"], to, msg.as_string())

        log.info("Email sent to %s — %s", to, subject)
        return True

    except Exception as e:
        log.error("Email send failed: %s", e)
        return False


# ── Public API ─────────────────────────────────────────────────────────────────

def send_alert(
    subject: str,
    body: str,
    level: str = "ERROR",
) -> bool:
    """
    Send a system alert email to the configured alert_email address.

    Usage in scripts (wrap around risky steps):
        try:
            run_something()
        except Exception as e:
            import traceback
            from alerts.email_alerts import send_alert
            send_alert(f"[Horizon Ledger] StepN failed", traceback.format_exc())

    Args:
        subject: Email subject (will be prefixed with level emoji)
        body: Plain text body — typically traceback.format_exc()
        level: "ERROR", "WARNING", or "INFO"
    """
    cfg = _load_config()
    alert_email = cfg.get("alert_email", "")
    if not alert_email:
        log.warning("No alert_email configured — skipping alert: %s", subject)
        return False

    emoji = {"ERROR": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(level, "⚪")
    full_subject = f"{emoji} {subject}"

    html = f"""
    <html><body>
    <h2 style="color:{'red' if level=='ERROR' else 'orange' if level=='WARNING' else 'blue'}">
        {emoji} {subject}
    </h2>
    <p><strong>Date:</strong> {date.today().isoformat()}</p>
    <p><strong>Level:</strong> {level}</p>
    <hr>
    <pre style="background:#f5f5f5;padding:12px;font-size:12px">{body}</pre>
    <hr>
    <p style="color:#666;font-size:11px">
        Horizon Ledger automated alert. Check logs at data/*.log for full details.
    </p>
    </body></html>
    """

    return _send_email(
        to=[alert_email],
        subject=full_subject,
        body_text=f"{subject}\n\nDate: {date.today().isoformat()}\n\n{body}",
        body_html=html,
    )


def send_newsletter(
    pdf_path: Path,
    recipients: Optional[list[str]] = None,
    issue_date: Optional[str] = None,
    issue_number: Optional[int] = None,
) -> bool:
    """
    Send the weekly newsletter PDF to the mailing list.

    Args:
        pdf_path: Path to the generated PDF file.
        recipients: List of email addresses. If None, uses newsletter_recipients from config.
        issue_date: Issue date string for subject line.
        issue_number: Issue number for subject line.

    Returns:
        True on success, False on failure.
    """
    cfg = _load_config()
    if recipients is None:
        recipients = cfg.get("newsletter_recipients", [])

    if not recipients:
        log.warning("No newsletter recipients configured — skipping send.")
        return False

    pdf_path = Path(pdf_path)
    date_str  = issue_date or date.today().isoformat()
    issue_str = f"#{issue_number}" if issue_number else ""

    subject = f"📊 Horizon Ledger Weekly {issue_str} — {date_str}"

    html = f"""
    <html><body style="font-family:Georgia,serif;max-width:600px;margin:0 auto">
    <div style="background:#1a2744;padding:24px;text-align:center">
        <h1 style="color:#fff;margin:0;font-size:28px">HORIZON LEDGER</h1>
        <p style="color:#a8bbd4;margin:8px 0 0 0;font-size:14px">Personal Investment Research</p>
    </div>
    <div style="padding:24px;background:#f8f9fa">
        <h2 style="color:#1a2744">Weekly Newsletter — {date_str}</h2>
        <p>Your Horizon Ledger weekly newsletter is attached as a PDF.</p>
        <p>This week's issue covers:</p>
        <ul>
            <li>Market health score and pulse indicators</li>
            <li>Conservative &amp; Aggressive index updates</li>
            <li>Top 10 stock viability scores</li>
            <li>Risk flags and market conditions</li>
            <li>The Almanac: historical context and upcoming events</li>
            <li>Simulated portfolio performance update</li>
        </ul>
        <p style="color:#999;font-size:11px;margin-top:24px">
            <em>This newsletter is generated automatically by Horizon Ledger using quantitative
            models. It is not financial advice. All projections are illustrative.
            Past performance does not predict future results.</em>
        </p>
    </div>
    </body></html>
    """

    text = (
        f"Horizon Ledger Weekly Newsletter — {date_str}\n\n"
        "Your newsletter is attached as a PDF.\n\n"
        "NOT FINANCIAL ADVICE. Personal research tool only."
    )

    success = _send_email(
        to=recipients,
        subject=subject,
        body_text=text,
        body_html=html,
        attachments=[pdf_path] if pdf_path.exists() else None,
    )

    # Update newsletter_history DB status
    if success:
        try:
            from db.schema import get_connection
            conn = get_connection()
            from datetime import datetime
            conn.execute(
                """UPDATE newsletter_history
                   SET sent_to=?, sent_at=?, status='sent'
                   WHERE issue_date=?""",
                (len(recipients), datetime.now().isoformat(), date_str),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("Could not update newsletter_history: %s", e)

    return success


def send_test_email() -> bool:
    """
    Send a test email to verify SMTP configuration.
    Prints result to console.
    """
    cfg = _load_config()
    alert_email = cfg.get("alert_email", "")

    if not alert_email:
        print("❌ No alert_email configured in secrets/email_config.json")
        return False

    print(f"Sending test email to {alert_email}...")
    success = send_alert(
        subject="[Horizon Ledger] Test — Email alerts are working",
        body=(
            "This is a test message from Horizon Ledger.\n\n"
            "If you received this, your email alert system is configured correctly.\n\n"
            "You will receive alerts when:\n"
            "  - run_daily.py encounters an error\n"
            "  - run_weekly.py encounters an error\n"
            "  - run_quarterly.py encounters an error\n"
            "  - Data fetches fail\n\n"
            "You will receive the newsletter PDF each Saturday morning."
        ),
        level="INFO",
    )

    if success:
        print(f"✅ Test email sent to {alert_email}")
    else:
        print("❌ Test email failed — check SMTP credentials in secrets/email_config.json")
    return success


def add_newsletter_recipient(email: str) -> None:
    """Add an email address to the newsletter mailing list."""
    cfg = _load_config()
    recipients = cfg.get("newsletter_recipients", [])
    if email not in recipients:
        recipients.append(email)
        cfg["newsletter_recipients"] = recipients
        _EMAIL_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        log.info("Added newsletter recipient: %s", email)


def remove_newsletter_recipient(email: str) -> None:
    """Remove an email address from the newsletter mailing list."""
    cfg = _load_config()
    recipients = cfg.get("newsletter_recipients", [])
    if email in recipients:
        recipients.remove(email)
        cfg["newsletter_recipients"] = recipients
        _EMAIL_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        log.info("Removed newsletter recipient: %s", email)


def get_recipients() -> list[str]:
    """Return current newsletter recipient list."""
    cfg = _load_config()
    return cfg.get("newsletter_recipients", [])
