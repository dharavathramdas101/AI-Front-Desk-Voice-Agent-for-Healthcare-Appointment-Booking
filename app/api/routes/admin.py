from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import joinedload

from app.models.db import Appointment, CallLog, Slot, get_session

router = APIRouter(prefix="/admin")


# ── helpers ───────────────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    colors = {"booked": "#16a34a", "cancelled": "#dc2626", "rescheduled": "#d97706"}
    c = colors.get(status, "#6b7280")
    return f'<span class="badge" style="background:{c}">{status}</span>'


def _source_badge(source: str | None) -> str:
    if source == "phone":
        return '<span class="badge" style="background:#7c3aed">📞 phone</span>'
    return '<span class="badge" style="background:#2563eb">🌐 browser</span>'


def _intent_badge(intent: str | None) -> str:
    colors = {
        "BOOK": "#0891b2", "RESCHEDULE": "#d97706", "CANCEL": "#dc2626",
        "FAQ": "#16a34a", "SMALL_TALK": "#6b7280", "ESCALATE": "#9333ea",
    }
    c = colors.get(intent or "", "#6b7280")
    return f'<span class="badge" style="background:{c}">{intent or "—"}</span>'


# ── main page ─────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def admin_page(
    search: str = "",
    source: str = "all",
    intent: str = "all",
    date: str = "",
    autorefresh: str = "off",
) -> HTMLResponse:
    db = get_session()
    try:
        # --- appointments query ---
        appt_q = (
            db.query(Appointment)
            .options(joinedload(Appointment.doctor), joinedload(Appointment.slot))
            .order_by(Appointment.created_at.desc())
        )
        if search:
            appt_q = appt_q.filter(
                or_(
                    Appointment.patient_name.ilike(f"%{search}%"),
                    Appointment.phone.ilike(f"%{search}%"),
                )
            )
        appointments = appt_q.limit(100).all()

        # --- call logs query ---
        log_q = db.query(CallLog).order_by(CallLog.timestamp.desc())
        if source != "all":
            log_q = log_q.filter(CallLog.source == source)
        if intent != "all":
            log_q = log_q.filter(CallLog.intent == intent)
        if date:
            try:
                d = datetime.strptime(date, "%Y-%m-%d")
                log_q = log_q.filter(
                    CallLog.timestamp >= d.replace(hour=0, minute=0),
                    CallLog.timestamp <= d.replace(hour=23, minute=59, second=59),
                )
            except ValueError:
                pass
        if search:
            log_q = log_q.filter(CallLog.transcript_snippet.ilike(f"%{search}%"))
        logs = log_q.limit(200).all()

        # --- available slots for reschedule modal ---
        slots = (
            db.query(Slot)
            .options(joinedload(Slot.doctor))
            .filter(Slot.is_available == True, Slot.slot_datetime > datetime.now())
            .order_by(Slot.slot_datetime)
            .limit(60)
            .all()
        )
    finally:
        db.close()

    # ── build appointment rows ────────────────────────────────────────────────
    appt_rows = ""
    for a in appointments:
        dt = a.slot.slot_datetime.strftime("%d %b %Y %I:%M %p") if a.slot else "—"
        created = a.created_at.strftime("%d %b %H:%M") if a.created_at else "—"
        doc = a.doctor.name if a.doctor else "—"
        dept = a.doctor.department if a.doctor else "—"
        cancel_btn = ""
        reschedule_btn = ""
        if a.status == "booked":
            cancel_btn = f'''
              <form method="post" action="/admin/appointments/{a.id}/cancel"
                    style="display:inline" onsubmit="return confirm('Cancel this appointment?')">
                <button class="btn btn-danger">Cancel</button>
              </form>'''
            reschedule_btn = f'<button class="btn btn-warn" onclick="openReschedule({a.id}, \'{doc}\')">Reschedule</button>'
        appt_rows += f"""
        <tr>
          <td>{a.id}</td>
          <td><strong>{a.patient_name}</strong></td>
          <td>{a.phone}</td>
          <td>{doc}</td>
          <td>{dept}</td>
          <td>{dt}</td>
          <td>{_status_badge(a.status)}</td>
          <td style="color:#6b7280;font-size:12px">{created}</td>
          <td style="white-space:nowrap">{cancel_btn} {reschedule_btn}</td>
        </tr>"""

    if not appt_rows:
        appt_rows = '<tr><td colspan="9" class="empty">No appointments found</td></tr>'

    # ── build log rows ────────────────────────────────────────────────────────
    log_rows = ""
    for l in logs:
        ts = l.timestamp.strftime("%d %b %H:%M:%S") if l.timestamp else "—"
        total = f"{l.latency_total_ms:.0f}ms" if l.latency_total_ms else "—"
        stt = f"{l.latency_stt_ms:.0f}" if l.latency_stt_ms else "—"
        llm = f"{l.latency_llm_ms:.0f}" if l.latency_llm_ms else "—"
        tts = f"{l.latency_tts_ms:.0f}" if l.latency_tts_ms else "—"
        transcript = (l.transcript_snippet or "")[:70]
        log_rows += f"""
        <tr>
          <td style="color:#6b7280;font-size:12px">{ts}</td>
          <td>{_source_badge(l.source)}</td>
          <td>{_intent_badge(l.intent)}</td>
          <td style="color:#374151;font-size:13px">{transcript}</td>
          <td style="font-size:12px;color:#6b7280">{stt} / {llm} / {tts}</td>
          <td style="font-weight:600;font-size:13px">{total}</td>
        </tr>"""

    if not log_rows:
        log_rows = '<tr><td colspan="6" class="empty">No logs found</td></tr>'

    # ── slot options for reschedule modal ─────────────────────────────────────
    slot_options = "\n".join(
        f'<option value="{s.id}">[{s.doctor.name}] {s.slot_datetime.strftime("%d %b %Y %I:%M %p")}</option>'
        for s in slots
    )

    # ── intent options ────────────────────────────────────────────────────────
    intents = ["all", "BOOK", "RESCHEDULE", "CANCEL", "FAQ", "SMALL_TALK", "ESCALATE"]
    intent_opts = "\n".join(
        f'<option value="{i}" {"selected" if i == intent else ""}>{i}</option>'
        for i in intents
    )

    autorefresh_checked = "checked" if autorefresh == "on" else ""
    params = f"search={search}&source={source}&intent={intent}&date={date}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Admin — AI Front Desk</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f1f5f9;color:#111827;font-size:14px}}
    header{{background:#0f172a;color:#fff;padding:14px 28px;display:flex;align-items:center;gap:12px}}
    header h1{{font-size:17px;font-weight:600}}
    .container{{max-width:1300px;margin:24px auto;padding:0 20px;display:flex;flex-direction:column;gap:24px}}
    .card{{background:#fff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.1);overflow:hidden}}
    .card-header{{padding:14px 20px;border-bottom:1px solid #e5e7eb;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}}
    .card-header h2{{font-size:14px;font-weight:600}}
    .count{{font-size:12px;color:#6b7280;background:#f3f4f6;padding:2px 10px;border-radius:9999px}}
    .toolbar{{padding:12px 20px;border-bottom:1px solid #f3f4f6;display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:#fafafa}}
    input,select{{padding:6px 10px;border:1px solid #d1d5db;border-radius:6px;font-size:13px;outline:none}}
    input:focus,select:focus{{border-color:#2563eb;box-shadow:0 0 0 2px rgba(37,99,235,.15)}}
    table{{width:100%;border-collapse:collapse}}
    th{{text-align:left;padding:9px 14px;font-size:11px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;background:#f9fafb;border-bottom:1px solid #e5e7eb}}
    td{{padding:11px 14px;border-bottom:1px solid #f3f4f6;vertical-align:middle}}
    tr:last-child td{{border-bottom:none}}
    tr:hover td{{background:#f9fafb}}
    .badge{{color:#fff;padding:2px 9px;border-radius:9999px;font-size:11px;font-weight:500;white-space:nowrap}}
    .btn{{padding:4px 10px;border-radius:6px;border:none;font-size:12px;cursor:pointer;font-weight:500}}
    .btn-danger{{background:#fee2e2;color:#dc2626}}
    .btn-danger:hover{{background:#fecaca}}
    .btn-warn{{background:#fef3c7;color:#d97706}}
    .btn-warn:hover{{background:#fde68a}}
    .btn-primary{{background:#2563eb;color:#fff;padding:6px 14px;font-size:13px}}
    .btn-primary:hover{{background:#1d4ed8}}
    .btn-export{{background:#f0fdf4;color:#16a34a;padding:5px 12px;font-size:12px;border:1px solid #bbf7d0;border-radius:6px;text-decoration:none;font-weight:500}}
    .btn-export:hover{{background:#dcfce7}}
    .empty{{padding:32px;text-align:center;color:#9ca3af}}
    .modal{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:100;align-items:center;justify-content:center}}
    .modal.open{{display:flex}}
    .modal-box{{background:#fff;border-radius:12px;padding:24px;width:440px;max-width:95vw;box-shadow:0 20px 60px rgba(0,0,0,.3)}}
    .modal-box h3{{font-size:15px;font-weight:600;margin-bottom:16px}}
    .modal-box select{{width:100%;margin-bottom:16px}}
    .modal-actions{{display:flex;gap:8px;justify-content:flex-end}}
    .latency-bar{{display:flex;gap:4px;font-size:11px}}
    .latency-bar span{{padding:1px 5px;border-radius:3px}}
    #refresh-countdown{{font-size:12px;color:#94a3b8;margin-left:8px}}
    label{{font-size:13px;color:#374151}}
  </style>
</head>
<body>

<header>
  <div style="width:30px;height:30px;background:#2563eb;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px">+</div>
  <h1>AI Front Desk — Admin</h1>
  <div style="margin-left:auto;display:flex;align-items:center;gap:16px">
    <label style="color:#94a3b8;display:flex;align-items:center;gap:6px;cursor:pointer">
      <input type="checkbox" id="ar-toggle" {autorefresh_checked} onchange="toggleAutoRefresh(this)">
      Auto-refresh
      <span id="refresh-countdown"></span>
    </label>
    <a href="/admin/export/appointments.csv" class="btn-export">⬇ Appointments CSV</a>
    <a href="/admin/export/logs.csv?{params}" class="btn-export">⬇ Logs CSV</a>
  </div>
</header>

<div class="container">

  <!-- SEARCH BAR -->
  <form method="get" action="/admin" style="display:flex;gap:8px;align-items:center">
    <input name="search" value="{search}" placeholder="Search patient name or phone…" style="width:280px"/>
    <button type="submit" class="btn btn-primary">Search</button>
    {"<a href='/admin' style='font-size:13px;color:#6b7280;text-decoration:none'>✕ Clear</a>" if search else ""}
    <input type="hidden" name="autorefresh" value="{autorefresh}"/>
  </form>

  <!-- APPOINTMENTS -->
  <div class="card">
    <div class="card-header">
      <h2>Appointments</h2>
      <span class="count">{len(appointments)} shown</span>
    </div>
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>#</th><th>Patient</th><th>Phone</th><th>Doctor</th><th>Dept</th>
        <th>Slot</th><th>Status</th><th>Booked At</th><th>Actions</th>
      </tr></thead>
      <tbody>{appt_rows}</tbody>
    </table>
    </div>
  </div>

  <!-- CALL LOGS -->
  <div class="card">
    <div class="card-header">
      <h2>Call Logs</h2>
      <span class="count">{len(logs)} shown</span>
    </div>
    <form method="get" action="/admin" class="toolbar">
      <select name="source" onchange="this.form.submit()">
        <option value="all" {"selected" if source=="all" else ""}>All sources</option>
        <option value="phone" {"selected" if source=="phone" else ""}>📞 Phone</option>
        <option value="browser" {"selected" if source=="browser" else ""}>🌐 Browser</option>
      </select>
      <select name="intent" onchange="this.form.submit()">{intent_opts}</select>
      <input type="date" name="date" value="{date}" onchange="this.form.submit()" title="Filter by date"/>
      <input type="hidden" name="search" value="{search}"/>
      <input type="hidden" name="autorefresh" value="{autorefresh}"/>
      {"<a href='/admin' style='font-size:13px;color:#6b7280;text-decoration:none'>✕ Reset filters</a>" if (source != "all" or intent != "all" or date) else ""}
    </form>
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>Time</th><th>Source</th><th>Intent</th><th>Transcript</th>
        <th>STT/LLM/TTS (ms)</th><th>Total</th>
      </tr></thead>
      <tbody>{log_rows}</tbody>
    </table>
    </div>
  </div>

</div>

<!-- RESCHEDULE MODAL -->
<div class="modal" id="reschedule-modal">
  <div class="modal-box">
    <h3>Reschedule Appointment</h3>
    <p style="font-size:13px;color:#6b7280;margin-bottom:12px" id="reschedule-label">Appointment #<span id="reschedule-appt-id"></span></p>
    <select id="reschedule-slot" style="width:100%;margin-bottom:16px">
      {slot_options}
    </select>
    <div class="modal-actions">
      <button class="btn" onclick="closeReschedule()" style="background:#f3f4f6;color:#374151">Cancel</button>
      <button class="btn btn-primary" onclick="submitReschedule()">Confirm Reschedule</button>
    </div>
  </div>
</div>

<script>
// ── auto-refresh ──────────────────────────────────────────────────────────────
let arTimer = null, arSeconds = 10;

function toggleAutoRefresh(cb) {{
  const url = new URL(window.location);
  url.searchParams.set('autorefresh', cb.checked ? 'on' : 'off');
  window.location = url;
}}

function startCountdown() {{
  const el = document.getElementById('refresh-countdown');
  arTimer = setInterval(() => {{
    arSeconds--;
    el.textContent = '(' + arSeconds + 's)';
    if (arSeconds <= 0) {{ clearInterval(arTimer); window.location.reload(); }}
  }}, 1000);
}}

if (document.getElementById('ar-toggle').checked) {{
  document.getElementById('refresh-countdown').textContent = '(10s)';
  startCountdown();
}}

// ── reschedule modal ──────────────────────────────────────────────────────────
let currentApptId = null;

function openReschedule(apptId, doctorName) {{
  currentApptId = apptId;
  document.getElementById('reschedule-appt-id').textContent = apptId;
  document.getElementById('reschedule-modal').classList.add('open');
}}

function closeReschedule() {{
  document.getElementById('reschedule-modal').classList.remove('open');
  currentApptId = null;
}}

async function submitReschedule() {{
  const slotId = document.getElementById('reschedule-slot').value;
  if (!slotId || !currentApptId) return;
  const resp = await fetch('/admin/appointments/' + currentApptId + '/reschedule', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
    body: 'slot_id=' + slotId,
  }});
  if (resp.ok) {{ closeReschedule(); window.location.reload(); }}
  else {{ alert('Reschedule failed. Slot may already be taken.'); }}
}}

// Close modal on backdrop click
document.getElementById('reschedule-modal').addEventListener('click', function(e) {{
  if (e.target === this) closeReschedule();
}});
</script>
</body>
</html>"""

    return HTMLResponse(content=html)


# ── cancel endpoint ───────────────────────────────────────────────────────────

@router.post("/appointments/{appt_id}/cancel")
async def cancel_appointment_admin(appt_id: int):
    db = get_session()
    try:
        appt = db.query(Appointment).filter(Appointment.id == appt_id).first()
        if appt and appt.status == "booked":
            slot = db.query(Slot).filter(Slot.id == appt.slot_id).first()
            appt.status = "cancelled"
            if slot:
                slot.is_available = True
            db.commit()
    finally:
        db.close()
    return RedirectResponse(url="/admin", status_code=303)


# ── reschedule endpoint ───────────────────────────────────────────────────────

@router.post("/appointments/{appt_id}/reschedule")
async def reschedule_appointment_admin(appt_id: int, slot_id: int = Form(...)):
    db = get_session()
    try:
        appt = db.query(Appointment).filter(Appointment.id == appt_id).first()
        new_slot = db.query(Slot).filter(Slot.id == slot_id).first()
        if not appt or not new_slot or not new_slot.is_available:
            from fastapi.responses import Response
            return Response(content="Slot unavailable", status_code=400)
        old_slot = db.query(Slot).filter(Slot.id == appt.slot_id).first()
        if old_slot:
            old_slot.is_available = True
        new_slot.is_available = False
        appt.slot_id = new_slot.id
        appt.status = "rescheduled"
        db.commit()
    finally:
        db.close()
    return RedirectResponse(url="/admin", status_code=303)


# ── export CSV ────────────────────────────────────────────────────────────────

@router.get("/export/appointments.csv")
async def export_appointments():
    db = get_session()
    try:
        rows = (
            db.query(Appointment)
            .options(joinedload(Appointment.doctor), joinedload(Appointment.slot))
            .order_by(Appointment.created_at.desc())
            .all()
        )
    finally:
        db.close()

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "patient_name", "phone", "doctor", "department", "slot_datetime", "status", "created_at"])
    for a in rows:
        w.writerow([
            a.id, a.patient_name, a.phone,
            a.doctor.name if a.doctor else "",
            a.doctor.department if a.doctor else "",
            a.slot.slot_datetime if a.slot else "",
            a.status,
            a.created_at,
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=appointments.csv"},
    )


@router.get("/export/logs.csv")
async def export_logs(source: str = "all", intent: str = "all", date: str = ""):
    db = get_session()
    try:
        q = db.query(CallLog).order_by(CallLog.timestamp.desc())
        if source != "all":
            q = q.filter(CallLog.source == source)
        if intent != "all":
            q = q.filter(CallLog.intent == intent)
        if date:
            try:
                d = datetime.strptime(date, "%Y-%m-%d")
                q = q.filter(
                    CallLog.timestamp >= d.replace(hour=0, minute=0),
                    CallLog.timestamp <= d.replace(hour=23, minute=59, second=59),
                )
            except ValueError:
                pass
        rows = q.all()
    finally:
        db.close()

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "session_id", "timestamp", "source", "intent", "transcript", "outcome",
                "stt_ms", "retrieval_ms", "llm_ms", "tts_ms", "total_ms"])
    for l in rows:
        w.writerow([
            l.id, l.session_id, l.timestamp, l.source, l.intent,
            l.transcript_snippet, l.outcome,
            l.latency_stt_ms, l.latency_retrieval_ms, l.latency_llm_ms,
            l.latency_tts_ms, l.latency_total_ms,
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=call_logs.csv"},
    )
