"""Horizon Ledger — Newsletter Preview & Management Dashboard Page"""

import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from config import DISCLAIMER, NEWSLETTER_DIR
from db.schema import get_connection


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _most_recent_saturday() -> date:
    """Return the most recent Saturday on or before today."""
    today = date.today()
    days_since_saturday = (today.weekday() - 5) % 7
    return today - timedelta(days=days_since_saturday)


def _get_next_issue_number(conn) -> int:
    """Return the next sequential issue number from the DB."""
    try:
        row = conn.execute(
            "SELECT MAX(issue_number) FROM newsletter_history"
        ).fetchone()
        max_num = row[0] if row and row[0] is not None else 0
        return max_num + 1
    except Exception:
        return 1


def _get_newsletter_history(conn) -> pd.DataFrame:
    """Return the newsletter_history table as a DataFrame."""
    try:
        df = pd.read_sql(
            """SELECT issue_date, issue_number, status,
                      sent_to, sent_at, pdf_path
               FROM newsletter_history
               ORDER BY issue_date DESC""",
            conn,
        )
        return df
    except Exception:
        return pd.DataFrame()


def _pdf_exists_for_issue(issue_date: str) -> Path | None:
    """Return the PDF path for an issue if the file exists, else None."""
    try:
        row = get_connection().execute(
            "SELECT pdf_path FROM newsletter_history WHERE issue_date=?",
            (issue_date,),
        ).fetchone()
        if row and row["pdf_path"]:
            p = Path(row["pdf_path"])
            if p.exists():
                return p
            # Try .html fallback
            h = p.with_suffix(".html")
            if h.exists():
                return h
    except Exception:
        pass
    return None


# ─── Tab 1: Preview & Generate ────────────────────────────────────────────────

def _tab_generate(conn) -> None:
    st.subheader("Generate Newsletter")

    # Issue date selector
    default_saturday = _most_recent_saturday()
    issue_date_input = st.date_input(
        "Issue date (typically a Saturday):",
        value=default_saturday,
        max_value=date.today(),
    )
    issue_date_str = issue_date_input.isoformat()

    # Determine issue number
    next_num = _get_next_issue_number(conn)
    issue_number = st.number_input(
        "Issue number:",
        min_value=1,
        value=next_num,
        step=1,
    )

    st.divider()

    # Generate button
    if st.button("📄 Generate Newsletter", type="primary", use_container_width=False):
        with st.spinner("Assembling newsletter — this may take 10–30 seconds..."):
            try:
                from newsletter.generator import generate_newsletter
                pdf_path = generate_newsletter(
                    issue_date=issue_date_str,
                    issue_number=int(issue_number),
                )
                st.success(f"Newsletter generated: `{pdf_path}`")

                # Show open-in-browser link
                uri = pdf_path.as_uri()
                suffix = pdf_path.suffix.upper()
                st.markdown(
                    f'<a href="{uri}" target="_blank">🔗 Open {suffix} in browser</a>',
                    unsafe_allow_html=True,
                )

                # Offer download
                with open(pdf_path, "rb") as fh:
                    file_bytes = fh.read()
                mime = "application/pdf" if pdf_path.suffix == ".pdf" else "text/html"
                st.download_button(
                    label=f"⬇️ Download {pdf_path.name}",
                    data=file_bytes,
                    file_name=pdf_path.name,
                    mime=mime,
                )

            except Exception as exc:
                st.error(f"Newsletter generation failed: {exc}")
                st.exception(exc)
    else:
        # If a PDF already exists for this date, show a preview notice
        existing = _pdf_exists_for_issue(issue_date_str)
        if existing:
            st.info(f"A newsletter already exists for {issue_date_str}: `{existing.name}`")
            uri = existing.as_uri()
            st.markdown(
                f'<a href="{uri}" target="_blank">🔗 Open existing file in browser</a>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.caption(
        "**Tip:** Run `scripts/run_weekly.py` before generating to ensure "
        "scores, market digest, and paper portfolio are up-to-date."
    )


# ─── Tab 2: Send ─────────────────────────────────────────────────────────────

def _tab_send(conn) -> None:
    st.subheader("Distribute Newsletter")

    from alerts.email_alerts import (
        get_recipients,
        add_newsletter_recipient,
        remove_newsletter_recipient,
        send_newsletter,
    )

    recipients = get_recipients()

    # ── Recipient list ─────────────────────────────────────────────────────────
    st.markdown("**Current mailing list:**")
    if recipients:
        for r in recipients:
            st.markdown(f"- {r}")
    else:
        st.info("No recipients configured yet. Add one below.")

    st.divider()

    # ── Add recipient ─────────────────────────────────────────────────────────
    with st.expander("Add recipient", expanded=False):
        new_email = st.text_input(
            "Email address to add:",
            placeholder="subscriber@example.com",
            key="add_recipient_input",
        )
        if st.button("➕ Add", key="add_recipient_btn"):
            new_email = new_email.strip()
            if new_email and re.match(r"[^@]+@[^@]+\.[^@]+", new_email):
                add_newsletter_recipient(new_email)
                st.success(f"Added: {new_email}")
                st.rerun()
            elif new_email:
                st.error("Please enter a valid email address.")

    # ── Remove recipient ──────────────────────────────────────────────────────
    with st.expander("Remove recipient", expanded=False):
        if recipients:
            to_remove = st.selectbox(
                "Select recipient to remove:",
                options=recipients,
                key="remove_recipient_select",
            )
            if st.button("🗑️ Remove", key="remove_recipient_btn"):
                remove_newsletter_recipient(to_remove)
                st.success(f"Removed: {to_remove}")
                st.rerun()
        else:
            st.info("No recipients to remove.")

    st.divider()

    # ── Send newsletter ────────────────────────────────────────────────────────
    st.markdown("**Send the most recent generated newsletter:**")

    # Find the most recently generated PDF
    latest_pdf: Path | None = None
    try:
        row = conn.execute(
            """SELECT pdf_path, issue_date, issue_number
               FROM newsletter_history
               WHERE status IN ('generated', 'sent')
               ORDER BY issue_date DESC LIMIT 1"""
        ).fetchone()
        if row and row["pdf_path"]:
            p = Path(row["pdf_path"])
            if p.exists():
                latest_pdf = p
                latest_issue_date = row["issue_date"]
                latest_issue_num  = row["issue_number"]
    except Exception:
        pass

    if latest_pdf:
        st.info(
            f"Ready to send: **{latest_pdf.name}**  "
            f"(Issue #{latest_issue_num}, {latest_issue_date})"
        )

        send_disabled = not bool(recipients)
        if send_disabled:
            st.warning("Add at least one recipient before sending.")

        if st.button(
            "📬 Send Newsletter",
            type="primary",
            disabled=send_disabled,
            key="send_newsletter_btn",
        ):
            # Confirmation via session state
            st.session_state["confirm_send"] = True

        if st.session_state.get("confirm_send"):
            st.warning(
                f"This will email the newsletter to **{len(recipients)} recipient(s)**. "
                "Confirm?"
            )
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Yes, send it", key="confirm_yes"):
                    with st.spinner("Sending..."):
                        ok = send_newsletter(
                            pdf_path=latest_pdf,
                            recipients=recipients,
                            issue_date=latest_issue_date,
                            issue_number=latest_issue_num,
                        )
                    if ok:
                        st.success(
                            f"Newsletter sent to {len(recipients)} recipient(s)!"
                        )
                    else:
                        st.error(
                            "Send failed. Check SMTP configuration in "
                            "`secrets/email_config.json`."
                        )
                    st.session_state.pop("confirm_send", None)
            with col_no:
                if st.button("❌ Cancel", key="confirm_no"):
                    st.session_state.pop("confirm_send", None)
                    st.rerun()
    else:
        st.warning(
            "No generated newsletter found. "
            "Go to the **Preview & Generate** tab to create one first."
        )

    st.divider()
    st.caption(
        "Email is sent via SMTP using the settings in "
        "`secrets/email_config.json`. "
        "Run **Send Test Email** from the sidebar to verify your SMTP setup."
    )


# ─── Tab 3: Archive ───────────────────────────────────────────────────────────

def _tab_archive(conn) -> None:
    st.subheader("Newsletter Archive")

    history_df = _get_newsletter_history(conn)

    if history_df.empty:
        st.info("No newsletters have been generated yet.")
        return

    # Display overview table (without pdf_path column — it's just a path string)
    overview = history_df.drop(columns=["pdf_path"], errors="ignore").copy()
    st.dataframe(overview, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("**Download a past issue:**")

    # Per-row download buttons
    for _, row in history_df.iterrows():
        col_info, col_btn = st.columns([3, 1])
        label = (
            f"Issue #{int(row['issue_number'])} — {row['issue_date']}  "
            f"({row.get('status', 'unknown')})"
        )
        with col_info:
            st.markdown(label)
        with col_btn:
            pdf_path_str = row.get("pdf_path")
            if pdf_path_str:
                p = Path(pdf_path_str)
                # Try PDF then HTML
                if not p.exists():
                    p = p.with_suffix(".html")

                if p.exists():
                    with open(p, "rb") as fh:
                        file_bytes = fh.read()
                    mime = "application/pdf" if p.suffix == ".pdf" else "text/html"
                    st.download_button(
                        label="⬇️ Download",
                        data=file_bytes,
                        file_name=p.name,
                        mime=mime,
                        key=f"dl_{row['issue_date']}",
                    )
                else:
                    st.caption("File missing")
            else:
                st.caption("No file")


# ─── Main page ────────────────────────────────────────────────────────────────

def show() -> None:
    st.title("📰 Newsletter")
    st.caption("Generate, preview, distribute, and archive your weekly Horizon Ledger PDF.")
    st.caption(DISCLAIMER)

    conn = get_connection()

    tab_preview, tab_send, tab_archive = st.tabs(
        ["📋 Preview & Generate", "📬 Send", "📚 Archive"]
    )

    with tab_preview:
        _tab_generate(conn)

    with tab_send:
        _tab_send(conn)

    with tab_archive:
        _tab_archive(conn)

    conn.close()
