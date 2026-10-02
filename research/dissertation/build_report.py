"""Build an evidence-bound standalone LaTeX source and PDF reading edition."""
from __future__ import annotations

import hashlib
import html
import json
import re
from functools import partial
from pathlib import Path

from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak,
                              PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from build_evidence import HERE, verify

TITLE = 'Resource-Sensitive GPU Attention'
SUBTITLE = 'Confidence-Gated Tactic Selection and Native-Transparent Execution'
OUT = HERE / 'output' / 'pdf'
FONT_ROOT = Path('/Users/kevin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype')
BLUE = colors.HexColor('#17324d')
TEAL = colors.HexColor('#147b78')
LIGHT = colors.HexColor('#edf2f6')
INK = colors.HexColor('#24313c')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex_escape(text):
    table = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
             '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
             '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    parts = re.split(r'(https?://\S+)', text)
    return ''.join(r'\url{' + p + '}' if p.startswith(('https://', 'http://')) else
                   ''.join('/\\allowbreak{}' if c == '/' else table.get(c, c) for c in p) for p in parts)


def pdf_text(text):
    return re.sub(r'(https?://[^\s<]+)', lambda m: '<link href="' + m.group(0) + '">' + m.group(0) + '</link>',
                  html.escape(text))


def data_tables(v):
    intervention = [['GPU / workload', 'Native', '48 / 48', '64 / 48', '64 / 64']]
    for gpu in ('4090', '5090'):
        for layout, name in [('paged1_cached_prefix', 'long prefix'), ('ragged_projection_suffix', 'short suffix')]:
            prefix = 'intervention_' + gpu + '_' + layout + '_'
            arms = ('native', 'attribute49152_launch49152', 'attribute65536_launch49152', 'attribute65536_launch65536')
            intervention.append([gpu + ' / ' + name] + [f'{v[prefix + arm]:.4f}' for arm in arms])
    versions = [['Source / stage', 'GPU', 'Selected / total', 'Selected geo.', 'Native LCB', 'Verdict']]
    for family, name in [('static', 'v3.2.2 fresh release'), ('v411', 'v4.1.1 exposed dev'), ('dev', 'v4.2 b769e7c dev'), ('canary', 'v4.2 b769e7c canary')]:
        for gpu in ('4090', '5090'):
            p = family + '_' + gpu + '_'
            if family == 'static':
                selected, total, gain, native = v[p + 'count'], 720, v[p + 'ratio'], 'not shown'
            else:
                selected, total = v[p + 'selected_records'], v[p + 'fold_records']
                gain, native = v[p + 'selected_geomean'], f'{v[p + "native_overlay_joint_min_lcb95"]:.6f}'
            versions.append([name, gpu, f'{selected}/{total}', f'{gain:.6f}', native, 'PASS' if family == 'dev' else 'HOLD'])
    versions.append(['c6879fb supplement', '5090', '108/144', f'{v["public5090_selected_geomean"]:.6f}',
                     f'{v["public5090_native_pristine_joint_min_lcb95"]:.6f}', 'scoped PASS'])
    pending = [['Milestone', 'State at cutoff', 'Evidence needed to close']]
    pending.extend([
        ['Complete exposed repetition', 'RUNNING: four jobs', 'All roles/cells; Slurm completion; dual gates; SASS; full archive'],
        ['Formal smoke and dev', 'NOT STARTED', 'Archived dual public-path PASS; exact new source and manifest'],
        ['New ten-case canary', 'NOT GENERATED / STARTED', 'One frozen fresh manifest; dual-card numerical and performance gates'],
        ['Original release 48 / stress 12', 'UNTOUCHED / NOT STARTED', 'All preceding stages PASS; complete strict qualification'],
        ['Full-model HTTP', 'PREPARED / NOT STARTED', 'All kernel stages archived PASS; 72 allocations; 24 scoped verdicts'],
        ['Resource Graph updates', 'UNQUALIFIED', 'Real metadata transitions, physical KV identity, exact outputs, fallback'],
        ['Historical 2/432', 'UNRESOLVED', 'Reproduced bound failure; first-state attribution; exact original parity'],
        ['Upstream patches', 'LOCAL / UNPUBLISHED', 'Final reviewable diffs, evidence, contribution requirements']
    ])
    evidence = [['Evidence label', 'Repository location / interpretation']]
    evidence.extend([
        ['E-authority', 'research/selector_v4/STATUS.json; frozen authority.json; original canary HOLD'],
        ['E-static', 'research/selector_v32/release_v322/evidence/final/; both release summary files and FINAL_VERDICT.md'],
        ['E-v411', 'research/selector_v4/evidence/v411-exact-dev/analysis/; exact v4.1.1 development summaries'],
        ['E-dev', 'research/selector_v4/evidence/v42-exact-dev/; exposed b769e7c summaries'],
        ['E-intervention', 'research/selector_v4/evidence/v42-attribute-launch-dual-controls/; receipt and raw diagnostic archive'],
        ['E-public', 'evidence/v42-public-path-original-complete-hold/; complete original archive receipts and verification'],
        ['E-public5090', 'evidence/v42-public-path-5090-supplement/; separate arithmetic-and-column analyzer, summary and source'],
        ['E-history', 'research/parity_remediation/; forced-history-v3/summary.json, README.md and REPORT_CORRECTIONS.md'],
        ['E-functional', 'research/selector_v4/evidence/; source, persistence, lease, epoch and packed-layout receipts; see CURRENT_PROGRESS_ZH.md'],
        ['E-protocol', 'research/selector_v4/PROTOCOL.md and COMPLETE_CAMPAIGN.md; frozen formal/public contracts and source ledgers'],
        ['E-serving', 'research/selector_v4/serving/; training, router, client, analysis, pair and pipeline sources with bound receipts'],
        ['E-ownership', 'research/parity_remediation/ plus reviewed SGLang 3825c5a source; latest remote ownership campaign receipt'],
        ['E-live', 'evidence/v42-complete-workflow-cpu-c34d5d9/ and evidence/v42-complete-repetition-cpu-0a5c735/; CPU receipts and initial dispatch snapshot']
    ])
    return {'intervention': (intervention, 'Diagnostic CUPTI median kernel times in microseconds. Arm labels are attribute ceiling / actual launch reservation in KiB. Each cell has sixteen traced launches; no qualification confidence interval is implied.'),
            'versions': (versions, 'Versioned evidence, not a paired improvement trend. Native LCB means candidate Native versus pristine where available. Static release has different control estimands. PASS is restricted to the stated population.'),
            'pending': (pending, 'Pending results are not measurements. Original qualification and both promotion flags remain HOLD / false.'),
            'evidence': (evidence, 'Paths are relative to the research repository. Exact frozen JSON source hashes and pointers are in claim_ledger.json; other receipts remain separately scoped.')}


def chart(v):
    drawing = Drawing(490, 308)
    x0, x1 = 152, 475
    max_us = 2250
    colors_by_arm = [BLUE, colors.HexColor('#657f98'), colors.HexColor('#9aabba'), TEAL]
    labels = ['Native', '48 / 48', '64 / 48', '64 / 64']
    arms = ['native', 'attribute49152_launch49152', 'attribute65536_launch49152', 'attribute65536_launch65536']
    drawing.add(String(0, 290, 'Long-prefix diagnostic: actual reservation changes latency', fontName='Body-Bold', fontSize=11, fillColor=BLUE))
    for tick in (0, 500, 1000, 1500, 2000):
        x = x0 + (x1 - x0) * tick / max_us
        drawing.add(Line(x, 42, x, 271, strokeColor=colors.HexColor('#dbe3e9'), strokeWidth=.5))
        drawing.add(String(x, 28, str(tick), textAnchor='middle', fontName='Body', fontSize=8, fillColor=INK))
    for card_index, gpu in enumerate(('4090', '5090')):
        top = 247 - 117 * card_index
        drawing.add(String(0, top + 18, 'RTX ' + gpu, fontName='Body-Bold', fontSize=9, fillColor=BLUE))
        for i, (arm, label) in enumerate(zip(arms, labels)):
            value = v['intervention_' + gpu + '_paged1_cached_prefix_' + arm]
            y = top - 21 * i
            drawing.add(String(2, y + 3, label, fontName='Body', fontSize=9, fillColor=INK))
            drawing.add(Rect(x0, y, (x1-x0)*value/max_us, 13, fillColor=colors_by_arm[i], strokeColor=None))
            drawing.add(String(x0+(x1-x0)*value/max_us+4, y+3, f'{value:.1f}', fontName='Body', fontSize=8, fillColor=INK))
    drawing.add(String((x0+x1)/2, 10, 'CUPTI median kernel time (microseconds); zero-based axis', textAnchor='middle', fontName='Body', fontSize=8, fillColor=INK))
    return drawing


def latex_chart(v):
    lines = [r'\begin{figure}[htbp]\centering', r'\setlength{\unitlength}{1pt}', r'\begin{picture}(460,280)',
             r'\put(0,264){\small\bfseries Long-prefix diagnostic: median kernel time}',
             r'\put(0,250){\scriptsize Arm labels: attribute / launch in KiB; times in microseconds}']
    x0, width = 105, 288
    for tick in (0, 500, 1000, 1500, 2000):
        x = x0 + width * tick / 2250
        lines.append(r'\put(%.2f,28){\line(0,1){208}}' % x)
        lines.append(r'\put(%.2f,16){\makebox(0,0){\scriptsize %d}}' % (x, tick))
    arms = ('native', 'attribute49152_launch49152', 'attribute65536_launch49152', 'attribute65536_launch65536')
    labels = ('Native', '48 / 48', '64 / 48', '64 / 64')
    for j, gpu in enumerate(('4090', '5090')):
        top = 211 - 102*j
        lines.append(r'\put(0,%d){\small\bfseries RTX %s}' % (top+18, gpu))
        for i, (arm, label) in enumerate(zip(arms, labels)):
            value = v['intervention_' + gpu + '_paged1_cached_prefix_' + arm]
            y = top - i*20
            lines.append(r'\put(0,%d){\small %s}' % (y+2, label))
            colour = 'teal' if i == 3 else 'blue!55!black'
            lines.append(r'\put(%d,%d){{\color{%s}\rule{%.2fpt}{11pt}}}' % (x0, y, colour, width*value/2250))
            lines.append(r'\put(%.2f,%d){\scriptsize %.1f}' % (x0+width*value/2250+4, y+2, value))
    lines.extend([r'\end{picture}', r'\caption{Actual 64 KiB reservation reduces long-prefix diagnostic medians. Ceiling-only changes do not show the same reduction. Sixteen traced launches per cell; not a qualification interval.}', r'\end{figure}'])
    return '\n'.join(lines)


class ResearchDoc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == 'Chapter' and flowable.getPlainText() != 'Contents':
            title = flowable.getPlainText()
            key = 'chapter-' + str(self.seq.nextf('chapter'))
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, 0)
            self.notify('TOCEntry', (0, title, self.page, key))


def footer(canvas, doc):
    canvas.saveState()
    width, height = doc.pagesize
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor('#d3dce4'))
        canvas.line(23*mm, height-17*mm, width-23*mm, height-17*mm)
        canvas.setFont('Body', 8)
        canvas.setFillColor(BLUE)
        canvas.drawString(23*mm, height-13*mm, 'RESOURCE-SENSITIVE GPU ATTENTION')
        canvas.drawRightString(width-23*mm, height-13*mm, 'RESEARCH DRAFT  /  02 OCT 2026')
    canvas.setFillColor(colors.HexColor('#647480'))
    canvas.setFont('Body', 8)
    canvas.drawString(23*mm, 13*mm, 'Canary HOLD  |  Default OFF  |  Serving OFF  |  Historical parity unresolved')
    canvas.drawRightString(width-23*mm, 13*mm, str(doc.page))
    canvas.restoreState()


def pdf_table(rows, caption, name, styles, width):
    fractions = {'intervention': [.32,.17,.17,.17,.17], 'versions': [.28,.08,.17,.16,.16,.15],
                 'pending': [.25,.23,.52], 'evidence': [.22,.78]}
    widths = [width*f for f in fractions[name]]
    cell_style = styles['TableCell']
    head_style = styles['TableHead']
    cells = [[Paragraph(html.escape(str(c)), head_style if i == 0 else cell_style) for c in row]
             for i, row in enumerate(rows)]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BLUE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 6), ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,0), (-1,0), .6, BLUE),
    ]))
    group = [t, Spacer(1,6), Paragraph(pdf_text(caption), styles['Caption']), Spacer(1,10)]
    return [KeepTogether(group)] if name != 'evidence' else group


def latex_table(rows, caption, name):
    specs = {'intervention': [4.6,2.3,2.3,2.3,2.3], 'versions': [4.1,1.1,2.3,2.2,2.2,2.3],
             'pending': [4,3.5,7], 'evidence': [3.2,11.5]}
    spec = '@{}' + ''.join('p{%.1fcm}' % w for w in specs[name]) + '@{}'
    header = ' & '.join(r'\textbf{' + tex_escape(str(c)) + '}' for c in rows[0]) + r' \\'
    lines = [r'\begin{small}', r'\begin{longtable}{' + spec + '}', r'\toprule', header, r'\midrule\endfirsthead',
             r'\toprule', header, r'\midrule\endhead']
    for row in rows[1:]:
        lines.append(' & '.join(tex_escape(str(c)) for c in row) + r' \\')
    lines.extend([r'\bottomrule', r'\end{longtable}', r'\end{small}', r'\noindent\small ' + tex_escape(caption) + r'\normalsize\par\medskip'])
    return '\n'.join(lines)


def build():
    ledger, values = verify()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, filename in [('Body','DejaVuSerif.ttf'),('Body-Bold','DejaVuSerif-Bold.ttf'),('Body-Italic','DejaVuSerif-Italic.ttf'),
                           ('Sans','DejaVuSans.ttf'),('Sans-Bold','DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(FONT_ROOT/filename)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body-Italic', boldItalic='Body-Bold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('ResearchBody', fontName='Body', fontSize=10.1, leading=14.2,
                              textColor=INK, spaceAfter=8, splitLongWords=True))
    styles.add(ParagraphStyle('Chapter', fontName='Sans-Bold', fontSize=17, leading=22, textColor=BLUE,
                              spaceBefore=12, spaceAfter=14, keepWithNext=True))
    styles.add(ParagraphStyle('Subsection', fontName='Sans-Bold', fontSize=11.5, leading=16, textColor=TEAL,
                              spaceBefore=14, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle('TitleLarge', fontName='Sans-Bold', fontSize=28, leading=34, textColor=BLUE, spaceAfter=20))
    styles.add(ParagraphStyle('Subtitle', fontName='Sans', fontSize=15, leading=21, textColor=TEAL, spaceAfter=28))
    styles.add(ParagraphStyle('CoverNote', fontName='Body', fontSize=11, leading=17, textColor=INK, spaceAfter=14))
    styles.add(ParagraphStyle('TableCell', fontName='Body', fontSize=8, leading=11, textColor=INK, splitLongWords=True))
    styles.add(ParagraphStyle('TableHead', fontName='Sans-Bold', fontSize=8, leading=11, textColor=colors.white))
    styles.add(ParagraphStyle('Caption', fontName='Body-Italic', fontSize=8.5, leading=12, textColor=INK, spaceAfter=9))
    tables = data_tables(values)
    source = (HERE/'manuscript.txt').read_text()
    resolved = re.sub(r'\{\{fact:([^:}]+):([^}]+)\}\}', lambda m: format(values[m.group(1)], m.group(2)), source)
    chunks = resolved.split('\n\n')
    tex = [r'\documentclass[11pt,a4paper]{article}',
           r'\usepackage[T1]{fontenc}', r'\usepackage[utf8]{inputenc}',
           r'\usepackage[margin=24mm]{geometry}', r'\usepackage{booktabs,longtable,array,xcolor,hyperref}',
           r'\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue}',
           r'\setlength{\parindent}{0pt}', r'\setlength{\parskip}{7pt}', r'\setlength{\emergencystretch}{3em}',
           r'\renewcommand{\arraystretch}{1.18}', r'\setlength{\tabcolsep}{4pt}', r'\begin{document}', r'\begin{titlepage}',
           r'\vspace*{2cm}{\Huge\bfseries '+tex_escape(TITLE)+r'\par}\vspace{0.7cm}',
           r'{\Large '+tex_escape(SUBTITLE)+r'\par}\vspace{1.4cm}',
           r'{\large English research draft\\Evidence cutoff: 2 October 2026\par}\vspace{1cm}',
           r'\textbf{Authoritative state: dual-card canary HOLD.}\\Default promotion: false. Serving promotion: false. Historical 2/432 token divergence: unresolved.\par\vspace{1cm}',
           r'Prepared for review as a BSc Computer Science final-year project research draft. Cohort-specific rubric, word limit, author and supervisor details have not been supplied. No grade forecast or submission-readiness certificate is implied.\par',
           r'\vfill Source-bound evidence, retained failures, explicit pending results.\end{titlepage}',
           r'\tableofcontents\clearpage']
    pdf = [Spacer(1,34*mm), Paragraph(TITLE, styles['TitleLarge']), Paragraph(SUBTITLE, styles['Subtitle']),
           Paragraph('ENGLISH RESEARCH DRAFT<br/>Evidence cutoff: 2 October 2026',styles['CoverNote']),
           Paragraph('<b>Authoritative state: dual-card canary HOLD</b><br/>Default promotion: false<br/>Serving promotion: false<br/>Historical 2/432 token divergence: unresolved',styles['CoverNote']),
           Spacer(1,10*mm), Paragraph('Prepared for review as a BSc Computer Science final-year project research draft. Cohort-specific rubric, word limit, author and supervisor details have not been supplied.',styles['CoverNote']),
           Paragraph('Source-bound evidence. Retained failures. Explicit pending results.',styles['CoverNote']), PageBreak()]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle('ContentsEntry', fontName='Body', fontSize=10, leading=17, textColor=INK, spaceAfter=4)]
    pdf.extend([Paragraph('Contents', styles['Chapter']), toc, PageBreak()])
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith('@@@ '):
            title = chunk[4:]
            tex.append(r'\subsection*{'+tex_escape(title)+'}')
            pdf.append(Paragraph(pdf_text(title),styles['Subsection']))
        elif chunk.startswith('@@ '):
            title = chunk[3:]
            tex.extend([r'\clearpage', r'\section*{'+tex_escape(title)+'}',
                        r'\addcontentsline{toc}{section}{'+tex_escape(title)+'}'])
            if pdf and not isinstance(pdf[-1], PageBreak):
                pdf.append(PageBreak())
            pdf.append(Paragraph(pdf_text(title), styles['Chapter']))
        elif chunk.startswith('{{table:'):
            name = re.fullmatch(r'\{\{table:([^}]+)\}\}',chunk).group(1)
            rows, caption = tables[name]
            tex.append(latex_table(rows,caption,name))
            pdf.extend(pdf_table(rows,caption,name,styles,164*mm))
        elif chunk == '{{figure:mechanism}}':
            tex.append(latex_chart(values))
            plot = chart(values)
            plot.scale((164*mm)/490, (164*mm)/490)
            plot.width, plot.height = 164*mm, 308*(164*mm)/490
            pdf.extend([plot, Paragraph('Zero-based, common microsecond axis. Diagnostic traced medians; no uncertainty intervals or serving claim. The suffix results are shown separately in the preceding table.',styles['Caption'])])
        else:
            assert '{{' not in chunk, ('unresolved_template',chunk[:80])
            tex.append(tex_escape(chunk)+'\n')
            pdf.append(Paragraph(pdf_text(chunk),styles['ResearchBody']))
    tex.append(r'\end{document}')
    tex_path = HERE/'resource_sensitive_attention.tex'
    tex_path.write_text('\n\n'.join(tex)+'\n')
    pdf_path = OUT/'resource_sensitive_attention_research_draft.pdf'
    doc = ResearchDoc(str(pdf_path), pagesize=(210*mm,297*mm), leftMargin=23*mm, rightMargin=23*mm,
                      topMargin=25*mm, bottomMargin=23*mm, title=TITLE+': '+SUBTITLE,
                      author='Research draft; author details pending', invariant=1)
    frame = Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id='body',leftPadding=0,rightPadding=0)
    doc.addPageTemplates([PageTemplate(id='research',frames=[frame],onPage=footer)])
    doc.multiBuild(pdf, canvasmaker=partial(Canvas, invariant=1))
    renderSVG.drawToFile(chart(values),str(HERE/'resource_intervention.svg'))
    (HERE/'resolved_manuscript.txt').write_text(resolved)
    inventory = {'manuscript_word_count': len(resolved.split()),
                 'ledger_sha256': sha(HERE/'claim_ledger.json'),
                 'manuscript_sha256': sha(HERE/'manuscript.txt'),
                 'outputs': {str(p.relative_to(HERE)): {'sha256':sha(p),'bytes':p.stat().st_size}
                             for p in [tex_path,pdf_path,HERE/'resource_intervention.svg',HERE/'resolved_manuscript.txt']},
                 'pdf_layout': 'Separate ReportLab edition of the same resolved manuscript and generated tables; native LaTeX compilation is independently checked.',
                 'qualifications': {'old_dual_canary':'HOLD','default_promotion':False,'serving_promotion':False,'historical_closure':False}}
    (HERE/'build_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(inventory,indent=2))


if __name__ == '__main__':
    build()
