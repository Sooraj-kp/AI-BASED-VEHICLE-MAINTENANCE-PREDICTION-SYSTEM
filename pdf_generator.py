"""
PDF Diagnostic Report Generator — VehicleAI
White/light theme, DejaVu Sans for full Unicode support (₹ ✓ ✗ ⚠).
"""
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Register DejaVu Sans (Unicode: ₹ ✓ ✗ ⚠ ≈) ──────────────
# Fonts are bundled inside static/fonts/ so this works on Windows, Mac & Linux
import os as _os
_FD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'static', 'fonts')
pdfmetrics.registerFont(TTFont('DV',      _os.path.join(_FD, 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DV-B',    _os.path.join(_FD, 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('DV-I',    _os.path.join(_FD, 'DejaVuSans-Oblique.ttf')))
pdfmetrics.registerFont(TTFont('DV-Mono', _os.path.join(_FD, 'DejaVuSansMono.ttf')))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DV-B', italic='DV-I')

W, H = A4

# ── Colours (light theme — fully visible in all PDF viewers) ──
C_WHITE   = colors.white
C_NAVY    = colors.HexColor('#0d1b2a')
C_BLUE    = colors.HexColor('#1a6ef5')
C_BLUE_LT = colors.HexColor('#e8f0fe')
C_TEXT    = colors.HexColor('#111827')
C_MUTED   = colors.HexColor('#6b7280')
C_LIGHT   = colors.HexColor('#f3f4f6')
C_ALT     = colors.HexColor('#f9fafb')
C_BORDER  = colors.HexColor('#d1d5db')
C_GREEN   = colors.HexColor('#166534')
C_GREEN_L = colors.HexColor('#dcfce7')
C_RED     = colors.HexColor('#991b1b')
C_RED_L   = colors.HexColor('#fee2e2')
C_AMBER   = colors.HexColor('#92400e')
C_AMBER_L = colors.HexColor('#fef3c7')
C_ORANGE  = colors.HexColor('#9a3412')
C_ORANGE_L= colors.HexColor('#ffedd5')


# ── Page template (header + footer) ──────────────────────────
class ReportCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        self._meta = kwargs.pop('meta', {})
        super().__init__(*args, **kwargs)
        self._pages = []

    def showPage(self):
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._pages)
        for p in self._pages:
            self.__dict__.update(p)
            self._paint(n)
            super().showPage()
        super().save()

    def _paint(self, total):
        self.saveState()
        # Header bar
        self.setFillColor(C_NAVY)
        self.rect(0, H - 26*mm, W, 26*mm, fill=1, stroke=0)
        self.setFillColor(C_BLUE)
        self.rect(0, H - 26*mm - 2.5, W, 2.5, fill=1, stroke=0)
        # Brand
        self.setFont('DV-B', 17)
        self.setFillColor(C_WHITE)
        self.drawString(15*mm, H - 15*mm, 'VEHICLE')
        ow = self.stringWidth('VEHICLE', 'DV-B', 17)
        self.setFillColor(colors.HexColor('#60a5fa'))
        self.drawString(15*mm + ow + 2, H - 15*mm, 'AI')
        self.setFont('DV', 7.5)
        self.setFillColor(colors.HexColor('#94a3b8'))
        self.drawString(15*mm, H - 21.5*mm, 'AI-Based Predictive Maintenance System')
        # Right metadata
        self.setFont('DV-B', 9.5)
        self.setFillColor(C_WHITE)
        self.drawRightString(W - 15*mm, H - 15*mm, self._meta.get('type', 'Diagnostic Report'))
        self.setFont('DV', 7.5)
        self.setFillColor(colors.HexColor('#94a3b8'))
        self.drawRightString(W - 15*mm, H - 21.5*mm, self._meta.get('ts', ''))
        # Footer
        self.setFillColor(C_LIGHT)
        self.rect(0, 0, W, 14*mm, fill=1, stroke=0)
        self.setStrokeColor(C_BORDER); self.setLineWidth(0.5)
        self.line(0, 14*mm, W, 14*mm)
        self.setFont('DV', 7)
        self.setFillColor(C_MUTED)
        self.drawString(15*mm, 5.5*mm,
            'VehicleAI Report — For informational purposes only. '
            'Consult a certified technician before any maintenance decision.')
        self.drawRightString(W - 15*mm, 5.5*mm, f'Page {self._pageNumber} of {total}')
        self.restoreState()


# ── Style helpers ─────────────────────────────────────────────
def S(name, font='DV', size=9.5, color=None, align=TA_LEFT,
       leading=None, bold=False, **kw):
    return ParagraphStyle(
        name,
        fontName='DV-B' if bold else font,
        fontSize=size,
        textColor=color or C_TEXT,
        alignment=align,
        leading=leading if leading is not None else size * 1.5,
        **kw
    )


def section_label(title):
    return [
        Spacer(1, 5*mm),
        HRFlowable(width='100%', thickness=0.8, color=C_BORDER, spaceAfter=3),
        Paragraph(
            title.upper(),
            ParagraphStyle('_sl', fontName='DV-B', fontSize=8, textColor=C_BLUE,
                           leading=12, spaceAfter=4, letterSpacing=1.5)
        ),
    ]


def alert_banner(level, message):
    cfg = {
        'NORMAL':   (C_GREEN,  C_GREEN_L,  '✓'),
        'CAUTION':  (C_ORANGE, C_ORANGE_L, '⚠'),
        'WARNING':  (C_AMBER,  C_AMBER_L,  '⚠'),
        'CRITICAL': (C_RED,    C_RED_L,    '✕'),
    }
    fg, bg, icon = cfg.get(level, (C_BLUE, C_BLUE_LT, 'ℹ'))

    lbl = ParagraphStyle('_al', fontName='DV-B', fontSize=12, textColor=fg,
                          alignment=TA_CENTER, leading=18)
    hd  = ParagraphStyle('_ah', fontName='DV-B', fontSize=9.5, textColor=C_TEXT, leading=14)
    sub = ParagraphStyle('_as', fontName='DV',   fontSize=8.5, textColor=C_MUTED, leading=13)

    t = Table([
        [Paragraph(f'{icon}  {level}', lbl),
         [Paragraph(f'Alert Status: {level}', hd), Paragraph(message, sub)]]
    ], colWidths=[30*mm, 146*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(0,0), bg),
        ('BACKGROUND',    (1,0),(1,0), C_WHITE),
        ('BOX',           (0,0),(-1,-1), 1.2, fg),
        ('LINEAFTER',     (0,0),(0,0),  1.2, fg),
        ('TOPPADDING',    (0,0),(-1,-1), 10),
        ('BOTTOMPADDING', (0,0),(-1,-1), 10),
        ('LEFTPADDING',   (0,0),(0,0),   4),
        ('LEFTPADDING',   (1,0),(1,0),  12),
        ('RIGHTPADDING',  (0,0),(-1,-1), 8),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
    ]))
    return t


def kpi_tiles(items, n_cols=3):
    """White KPI cards with coloured large value text."""
    items = list(items)
    while len(items) % n_cols:
        items.append(('', '', None))

    vp = ParagraphStyle('_kv', fontName='DV-B', fontSize=16,
                         alignment=TA_CENTER, leading=20)
    lp = ParagraphStyle('_kl', fontName='DV',   fontSize=7.5,
                         textColor=C_MUTED, alignment=TA_CENTER, leading=11)

    rows = []
    for i in range(0, len(items), n_cols):
        row = []
        for j in range(n_cols):
            lbl, val, hex_c = items[i+j]
            col = colors.HexColor(hex_c) if hex_c else C_BLUE
            inner = Table([
                [Paragraph(f'<font color="{col.hexval()}">{val}</font>', vp)],
                [Paragraph(lbl, lp)],
            ], colWidths=[52*mm])
            inner.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,-1), C_WHITE),
                ('BOX',           (0,0),(-1,-1), 0.8, C_BORDER),
                ('TOPPADDING',    (0,0),(-1,0),  8),
                ('BOTTOMPADDING', (0,0),(-1,0),  1),
                ('TOPPADDING',    (0,1),(-1,1),  1),
                ('BOTTOMPADDING', (0,1),(-1,1),  8),
                ('LEFTPADDING',   (0,0),(-1,-1), 4),
                ('RIGHTPADDING',  (0,0),(-1,-1), 4),
            ]))
            row.append(inner)
        rows.append(row)

    outer = Table(rows, colWidths=[54*mm]*n_cols, rowHeights=[22*mm]*len(rows))
    outer.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), C_LIGHT),
        ('LEFTPADDING',   (0,0),(-1,-1), 2),
        ('RIGHTPADDING',  (0,0),(-1,-1), 2),
        ('TOPPADDING',    (0,0),(-1,-1), 2),
        ('BOTTOMPADDING', (0,0),(-1,-1), 2),
    ]))
    return outer


def params_table(params):
    REF = {
        'Engine rpm':       ('RPM', '700 – 3,000', 700,  3000),
        'Lub oil pressure': ('bar', '2.0 – 6.0',   2.0,  6.0),
        'Fuel pressure':    ('bar', '8.0 – 22.0',  8.0,  22.0),
        'Coolant pressure': ('bar', '1.0 – 4.0',   1.0,  4.0),
        'lub oil temp':     ('°C',  '60 – 100',    60,   100),
        'Coolant temp':     ('°C',  '70 – 95',     70,   95),
    }
    DISP = {
        'Engine rpm': 'Engine RPM', 'Lub oil pressure': 'Lub Oil Pressure',
        'Fuel pressure': 'Fuel Pressure', 'Coolant pressure': 'Coolant Pressure',
        'lub oil temp': 'Lub Oil Temp', 'Coolant temp': 'Coolant Temp',
    }
    hp  = S('_ph', bold=True,  size=8,   color=C_MUTED)
    bp  = S('_pb',             size=9,   color=C_TEXT)
    okp = S('_po', bold=True,  size=9,   color=C_GREEN)
    bdp = S('_pd', bold=True,  size=9,   color=C_RED)
    rp  = S('_pr',             size=8.5, color=C_MUTED)

    rows = [[Paragraph(h, hp) for h in
             ['PARAMETER', 'MEASURED VALUE', 'HEALTHY RANGE', 'STATUS']]]

    for feat, (unit, rng, lo, hi) in REF.items():
        raw = params.get(feat)
        try:
            v = float(raw)
            ok = lo <= v <= hi
            rows.append([
                Paragraph(DISP[feat], bp),
                Paragraph(f'{v:g} {unit}', okp if ok else bdp),
                Paragraph(f'{rng} {unit}', rp),
                Paragraph('✓  In Range' if ok else '✗  Out of Range', okp if ok else bdp),
            ])
        except Exception:
            rows.append([Paragraph(DISP[feat], bp), Paragraph('—', rp),
                         Paragraph(f'{rng} {unit}', rp), Paragraph('—', rp)])

    t = Table(rows, colWidths=[56*mm, 42*mm, 46*mm, 36*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND',     (0,0),(-1,0),  C_LIGHT),
        ('LINEBELOW',      (0,0),(-1,0),  1, C_BORDER),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [C_WHITE, C_ALT]),
        ('TOPPADDING',     (0,0),(-1,-1), 7),
        ('BOTTOMPADDING',  (0,0),(-1,-1), 7),
        ('LEFTPADDING',    (0,0),(-1,-1), 8),
        ('RIGHTPADDING',   (0,0),(-1,-1), 6),
        ('BOX',            (0,0),(-1,-1), 0.8, C_BORDER),
        ('INNERGRID',      (0,0),(-1,-1), 0.4, C_BORDER),
        ('VALIGN',         (0,0),(-1,-1), 'MIDDLE'),
    ]))
    return t


def maintenance_table(needed, ok_items):
    hp  = S('_ih',  bold=True, size=8,   color=C_MUTED)
    bp  = S('_ib',             size=9,   color=C_TEXT)
    okp = S('_iok', bold=True, size=9,   color=C_GREEN, align=TA_CENTER)
    bdp = S('_ibd', bold=True, size=9,   color=C_RED,   align=TA_CENTER)
    cp  = S('_ic',             size=8.5, color=C_MUTED, align=TA_RIGHT)

    rows = [[Paragraph(h, hp) for h in ['MAINTENANCE ITEM', 'DECISION', 'CONFIDENCE']]]
    for item in needed:
        rows.append([Paragraph(item.get('label', '—'), bp),
                     Paragraph('REPLACE', bdp),
                     Paragraph(f"{item.get('probability', 0):.1f}%", cp)])
    for item in ok_items:
        rows.append([Paragraph(item.get('label', '—'), bp),
                     Paragraph('✓  OK', okp),
                     Paragraph(f"{item.get('probability', 0):.1f}%", cp)])

    t = Table(rows, colWidths=[106*mm, 36*mm, 38*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND',     (0,0),(-1,0),  C_LIGHT),
        ('LINEBELOW',      (0,0),(-1,0),  1, C_BORDER),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [C_WHITE, C_ALT]),
        ('TOPPADDING',     (0,0),(-1,-1), 7),
        ('BOTTOMPADDING',  (0,0),(-1,-1), 7),
        ('LEFTPADDING',    (0,0),(-1,-1), 8),
        ('RIGHTPADDING',   (0,0),(-1,-1), 8),
        ('BOX',            (0,0),(-1,-1), 0.8, C_BORDER),
        ('INNERGRID',      (0,0),(-1,-1), 0.3, C_BORDER),
        ('VALIGN',         (0,0),(-1,-1), 'MIDDLE'),
    ]))
    return t


# ── Entry point ──────────────────────────────────────────────
def generate_pdf(engine_data: dict, service_data: dict) -> bytes:
    buf = io.BytesIO()
    ts  = datetime.now().strftime('%d %b %Y  %H:%M')
    meta = {'type': 'Engine Diagnostic Report', 'ts': ts}

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=32*mm, bottomMargin=20*mm,
        leftMargin=15*mm, rightMargin=15*mm,
    )
    story = []

    # ═══ PAGE 1 — ENGINE ════════════════════════════════════
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph('Vehicle Diagnostic Report',
                            S('_t', bold=True, size=22, leading=28)))
    story.append(Paragraph(
        f'Generated on {ts}  ·  VehicleAI Machine Learning System',
        S('_sub', size=10, color=C_MUTED, leading=15, spaceAfter=6)))
    story.append(HRFlowable(width='100%', thickness=2, color=C_BLUE, spaceAfter=6))
    story.append(Spacer(1, 2*mm))

    level   = engine_data.get('alert_level', 'NORMAL')
    message = engine_data.get('message', 'Engine operating within normal parameters.')
    story.append(alert_banner(level, message))
    story.append(Spacer(1, 5*mm))

    story += section_label('Engine Health Summary')
    story.append(Spacer(1, 2*mm))
    story.append(kpi_tiles([
        ('Health Score',      f"{engine_data.get('health_score',0):.1f}%",       '#166534'),
        ('Fault Probability', f"{engine_data.get('fault_probability',0):.1f}%",  '#991b1b'),
        ('Risk Score',        f"{engine_data.get('risk_score','—')}/100",         '#92400e'),
        ('RUL Estimate',      f"{int(engine_data.get('rul_km',0)/1000)}k km",    '#1a6ef5'),
        ('Anomaly Status',    engine_data.get('anomaly_flag', 'Normal'),          '#9a3412'),
        ('Condition',         engine_data.get('label', '—'),                      '#111827'),
    ]))
    story.append(Spacer(1, 5*mm))

    story += section_label('Engine Parameters Analysed')
    story.append(Spacer(1, 2*mm))
    story.append(params_table(engine_data.get('parameters', {})))
    story.append(Spacer(1, 5*mm))

    story += section_label('Recommendations')
    recs = engine_data.get('recommendations',
        ['Continue regular maintenance schedule.',
         'Monitor engine parameters at next service.'])
    for i, rec in enumerate(recs, 1):
        story.append(Paragraph(f'<b>{i}.</b>  {rec}',
                                S('_rc', size=9.5, leftIndent=10, leading=15, spaceAfter=3)))
        story.append(Spacer(1, 1*mm))

    # ═══ PAGE 2 — SERVICE (optional) ════════════════════════
    if service_data and service_data.get('success'):
        story.append(PageBreak())
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph('Service Cost Prediction',
                                S('_t2', bold=True, size=22, leading=28)))
        story.append(Paragraph(
            'Maintenance cost estimate based on vehicle profile and mileage.',
            S('_sub2', size=10, color=C_MUTED, leading=15, spaceAfter=6)))
        story.append(HRFlowable(width='100%', thickness=2, color=C_BLUE, spaceAfter=6))

        cost   = service_data.get('predicted_cost', 0)
        ci_low = service_data.get('ci_low', 0)
        ci_hi  = service_data.get('ci_high', 0)
        std    = service_data.get('std_dev', 0)
        urg    = service_data.get('urgency', '—')
        n_need = service_data.get('total_items_needed', 0)

        story += section_label('Cost Estimate')
        story.append(Spacer(1, 2*mm))
        story.append(kpi_tiles([
            ('Predicted Cost',   f'Rs. {int(cost):,}',   '#1a6ef5'),
            ('Lower 90% CI',     f'Rs. {int(ci_low):,}', '#166534'),
            ('Upper 90% CI',     f'Rs. {int(ci_hi):,}',  '#9a3412'),
            ('Urgency',          urg,                      '#92400e'),
            ('Items to Replace', str(n_need),              '#991b1b'),
            ('Items OK',         str(14 - n_need),         '#166534'),
        ]))
        story.append(Spacer(1, 4*mm))

        ci_box = Table([[
            Paragraph(
                f'<b>90% Confidence Interval:</b>  '
                f'Rs. {int(ci_low):,}  —  Rs. {int(ci_hi):,}  '
                f'(std dev = Rs. {int(std):,})',
                S('_ci', size=9.5, color=C_BLUE, leading=15))
        ]], colWidths=[180*mm])
        ci_box.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), C_BLUE_LT),
            ('BOX',           (0,0),(-1,-1), 1.2, C_BLUE),
            ('TOPPADDING',    (0,0),(-1,-1), 9),
            ('BOTTOMPADDING', (0,0),(-1,-1), 9),
            ('LEFTPADDING',   (0,0),(-1,-1), 14),
        ]))
        story.append(ci_box)
        story.append(Spacer(1, 5*mm))

        story += section_label('Maintenance Items Assessment (14 Items)')
        story.append(Spacer(1, 2*mm))
        story.append(maintenance_table(
            service_data.get('items_needed', []),
            service_data.get('items_not_needed', [])
        ))

    # ═══ PAGE 3 — MODELS & DISCLAIMER ═══════════════════════
    story.append(PageBreak())
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph('Model Information & Disclaimer',
                            S('_t3', bold=True, size=22, leading=28)))
    story.append(HRFlowable(width='100%', thickness=2, color=C_BLUE, spaceAfter=6))

    story += section_label('ML Models Used in This Report')
    story.append(Spacer(1, 2*mm))

    hp = S('_mh', bold=True, size=8,  color=C_MUTED)
    bp = S('_mb',            size=9,  color=C_TEXT)
    ap = S('_ma', bold=True, size=9,  color=C_BLUE, align=TA_CENTER)

    mrows = [[Paragraph(h, hp) for h in
              ['MODEL', 'ALGORITHM', 'TRAINING DATA', 'PERFORMANCE']]]
    for a, b, c, d in [
        ('Engine Condition Classifier', 'Gradient Boosting',     '19,535 records', '66.3% accuracy'),
        ('Service Cost Predictor',      'Random Forest · 200T',  '1,139 records',  'R² = 0.868'),
        ('Maintenance Items × 14',      'Random Forest × 14',    '1,139 records',  '97.3% avg accuracy'),
        ('Anomaly Detector',            'Isolation Forest',       '7,218 normal',   'Calibrated 5%'),
        ('RUL Predictor (km + days)',   'Random Forest · 200T',  '19,535 records', 'R² ≈ 0.91'),
    ]:
        mrows.append([Paragraph(a,bp),Paragraph(b,bp),Paragraph(c,bp),Paragraph(d,ap)])

    mt = Table(mrows, colWidths=[52*mm, 50*mm, 46*mm, 38*mm])
    mt.setStyle(TableStyle([
        ('BACKGROUND',     (0,0),(-1,0),  C_LIGHT),
        ('LINEBELOW',      (0,0),(-1,0),  1, C_BORDER),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [C_WHITE, C_ALT]),
        ('TOPPADDING',     (0,0),(-1,-1), 7),
        ('BOTTOMPADDING',  (0,0),(-1,-1), 7),
        ('LEFTPADDING',    (0,0),(-1,-1), 8),
        ('BOX',            (0,0),(-1,-1), 0.8, C_BORDER),
        ('INNERGRID',      (0,0),(-1,-1), 0.3, C_BORDER),
        ('VALIGN',         (0,0),(-1,-1), 'MIDDLE'),
    ]))
    story.append(mt)

    story += section_label('Disclaimer')
    story.append(Paragraph(
        'This report is generated by an AI machine learning system trained on historical vehicle '
        'sensor and service data. All predictions are for <b>informational purposes only</b>. '
        'Always consult a certified automotive technician before any maintenance decision. '
        'RUL estimates use engineered degradation labels and should be treated as relative '
        'health indicators. VehicleAI accepts no liability for actions taken solely on '
        'the basis of this report.',
        S('_disc', size=9.5, color=C_TEXT, leading=15)
    ))
    story.append(Spacer(1, 6*mm))

    fn = Table([[
        Paragraph(
            f'Generated by VehicleAI  ·  {ts}  ·  '
            'Python · Flask · Scikit-learn · ReportLab',
            S('_fn', size=8, color=C_MUTED, align=TA_CENTER))
    ]], colWidths=[180*mm])
    fn.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), C_LIGHT),
        ('BOX',           (0,0),(-1,-1), 0.5, C_BORDER),
        ('TOPPADDING',    (0,0),(-1,-1), 7),
        ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ('LEFTPADDING',   (0,0),(-1,-1), 8),
    ]))
    story.append(fn)

    def _canvas(filename, doc=None, **kw):
        return ReportCanvas(filename, pagesize=A4, meta=meta)

    doc.build(story, canvasmaker=_canvas)
    return buf.getvalue()
