"""
VisionGate X — Challan Generator
Generates a PDF e-challan using ReportLab and optionally sends
SMS/email notifications.
"""

from __future__ import annotations
import os
import uuid
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.base      import MIMEBase
from email               import encoders

from reportlab.lib.pagesizes import A5
from reportlab.lib.units     import mm
from reportlab.lib           import colors
from reportlab.platypus      import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums     import TA_CENTER, TA_LEFT

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.violation import ViolationEvent


def _new_challan_id() -> str:
    return "VGX-" + uuid.uuid4().hex[:8].upper()


# ── PDF builder ────────────────────────────────────────────────────────────────

def _build_pdf(
    challan_id: str,
    plate_number: str,
    violation_type: str,
    description: str,
    fine_inr: float,
    issued_at: str,
    snapshot_path: str = "",
    owner_name: str = "",
    owner_contact: str = "",
) -> str:
    """Build the challan PDF and return its file path."""

    filename = os.path.join(config.CHALLAN_DIR, f"{challan_id}.pdf")
    doc      = SimpleDocTemplate(
        filename,
        pagesize    = A5,
        leftMargin  = 15 * mm,
        rightMargin = 15 * mm,
        topMargin   = 15 * mm,
        bottomMargin= 15 * mm,
    )

    styles     = getSampleStyleSheet()
    RED        = colors.HexColor("#C0392B")
    DARK_GRAY  = colors.HexColor("#2C3E50")
    LIGHT_GRAY = colors.HexColor("#F5F5F5")

    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=16, textColor=RED, alignment=TA_CENTER, spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=9, textColor=DARK_GRAY, alignment=TA_CENTER, spaceAfter=2,
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"],
        fontSize=9, textColor=colors.grey,
    )
    value_style = ParagraphStyle(
        "Value", parent=styles["Normal"],
        fontSize=10, textColor=DARK_GRAY,
    )
    fine_style = ParagraphStyle(
        "Fine", parent=styles["Heading2"],
        fontSize=18, textColor=RED, alignment=TA_CENTER, spaceBefore=6,
    )
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontSize=7, textColor=colors.grey, alignment=TA_CENTER,
    )

    story = []

    # Header
    story.append(Paragraph("VisionGate X", title_style))
    story.append(Paragraph("Automated Traffic Enforcement System", sub_style))
    story.append(Paragraph("e-Challan / Traffic Violation Notice", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=RED, spaceAfter=8))

    # Details table
    data = [
        [Paragraph("Challan ID", label_style),    Paragraph(challan_id, value_style)],
        [Paragraph("Issued At",  label_style),    Paragraph(issued_at, value_style)],
        [Paragraph("Vehicle No.", label_style),   Paragraph(plate_number or "Unknown", value_style)],
        [Paragraph("Violation",  label_style),    Paragraph(description, value_style)],
        [Paragraph("Type Code",  label_style),    Paragraph(violation_type, value_style)],
    ]
    if owner_name:
        data.append([Paragraph("Owner", label_style), Paragraph(owner_name, value_style)])
    if owner_contact:
        data.append([Paragraph("Contact", label_style), Paragraph(owner_contact, value_style)])

    tbl = Table(data, colWidths=[45 * mm, 85 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_GRAY),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)

    # Fine amount
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(f"Fine Amount: ₹{int(fine_inr)}", fine_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceBefore=8))

    # Snapshot note
    if snapshot_path and os.path.exists(snapshot_path):
        from reportlab.platypus import Image as RLImage
        try:
            img = RLImage(snapshot_path, width=100 * mm, height=55 * mm, kind="proportional")
            story.append(Spacer(1, 4 * mm))
            story.append(img)
        except Exception:
            pass

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "This is an auto-generated challan by VisionGate X AI System. "
        "Pay within 30 days to avoid further penalty. "
        "For disputes, contact your local RTO.",
        footer_style,
    ))

    doc.build(story)
    return filename


# ── Notification helpers ───────────────────────────────────────────────────────

def _send_email(
    to_addr: str,
    challan_id: str,
    plate: str,
    fine_inr: float,
    pdf_path: str,
):
    if not config.SMTP_USER or not config.SMTP_PASS or not to_addr:
        return
    try:
        msg = MIMEMultipart()
        msg["From"]    = config.SMTP_USER
        msg["To"]      = to_addr
        msg["Subject"] = f"e-Challan Issued: {challan_id}"

        body = (
            f"Dear Vehicle Owner,\n\n"
            f"An e-Challan has been issued for vehicle {plate}.\n"
            f"Challan ID : {challan_id}\n"
            f"Fine Amount: ₹{int(fine_inr)}\n\n"
            f"Please find the challan PDF attached.\n\n"
            f"— VisionGate X Automated Enforcement System"
        )
        msg.attach(MIMEText(body, "plain"))

        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(pdf_path)}")
            msg.attach(part)

        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASS)
            server.sendmail(config.SMTP_USER, to_addr, msg.as_string())
    except Exception as e:
        print(f"[ChallanGen] Email send failed: {e}")


def _send_sms(phone: str, challan_id: str, plate: str, fine_inr: float):
    if not config.TWILIO_SID or not phone:
        return
    try:
        from twilio.rest import Client
        client = Client(config.TWILIO_SID, config.TWILIO_TOKEN)
        client.messages.create(
            body=(
                f"VisionGate X Alert: Challan {challan_id} issued for {plate}. "
                f"Fine: Rs.{int(fine_inr)}. Pay within 30 days."
            ),
            from_=config.TWILIO_FROM,
            to=phone,
        )
    except Exception as e:
        print(f"[ChallanGen] SMS send failed: {e}")


# ── Public class ───────────────────────────────────────────────────────────────

class ChallanGenerator:

    def generate(
        self,
        violation: ViolationEvent,
        snapshot_path: str = "",
        owner_name: str = "",
        owner_contact: str = "",
        notify_email: str = "",
        notify_phone: str = "",
    ) -> tuple[str, str]:
        """
        Build a challan PDF for a ViolationEvent.
        Returns (challan_id, pdf_path).
        """
        challan_id = _new_challan_id()
        issued_at  = datetime.now().strftime("%d %b %Y  %I:%M %p")

        pdf_path = _build_pdf(
            challan_id    = challan_id,
            plate_number  = violation.plate_number,
            violation_type= violation.violation_type,
            description   = violation.description,
            fine_inr      = violation.fine_inr,
            issued_at     = issued_at,
            snapshot_path = snapshot_path,
            owner_name    = owner_name,
            owner_contact = owner_contact,
        )

        # Notifications (fire-and-forget; failures logged, not raised)
        if notify_email:
            _send_email(notify_email, challan_id, violation.plate_number,
                        violation.fine_inr, pdf_path)
        if notify_phone:
            _send_sms(notify_phone, challan_id, violation.plate_number, violation.fine_inr)

        return challan_id, pdf_path
