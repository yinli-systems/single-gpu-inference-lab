"""Verify the actual exported document, not the report renderer's implementation.

Visual inspection of rendered pages is separate and recorded only after it occurs.
"""
import hashlib
import json
import re

import pdfplumber
from pypdf import PdfReader

from build_evidence import HERE


def compact(text):
    return ''.join(text.split())


def main():
    pdf = HERE / 'output/pdf/resource_sensitive_attention_research_draft.pdf'
    inventory = json.loads((HERE / 'build_inventory.json').read_text())
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert digest == inventory['outputs'][str(pdf.relative_to(HERE))]['sha256']
    reader = PdfReader(pdf)
    text = '\n'.join(p.extract_text() for p in reader.pages)
    with pdfplumber.open(pdf) as document:
        body_text = '\n'.join(page.crop((0, 65, page.width, page.height - 60)).extract_text() or ''
                              for page in document.pages)
    normalized = compact(body_text)
    manuscript = (HERE / 'resolved_manuscript.txt').read_text()
    chunks = manuscript.split('\n\n')
    checked = 0
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or chunk.startswith('{{'):
            continue
        if chunk.startswith('@@'):
            chunk = re.sub(r'^@@@? ', '', chunk)
        assert compact(chunk) in normalized, ('missing_manuscript_passage', chunk[:100])
        checked += 1
    outside = []
    page_metrics = []
    with pdfplumber.open(pdf) as document:
        for number, page in enumerate(document.pages, 1):
            bad = [c for c in page.chars if c['x0'] < 0 or c['x1'] > page.width + .5
                   or c['top'] < 0 or c['bottom'] > page.height + .5]
            outside.extend((number, c['text']) for c in bad)
            assert len(page.chars) > 150, ('unexpected_nearly_empty_page', number)
            body = [c for c in page.chars if 65 < c['top'] < page.height - 60]
            page_metrics.append({'page':number, 'characters':len(page.chars),
                                 'body_bottom_pt':max((c['bottom'] for c in body),default=None),
                                 'outside_page_characters':len(bad)})
    assert not outside, ('clipped_text', outside[:5])
    assert '\u25a0' not in text and '\ufffd' not in text, 'replacement_glyph'
    assert '{{fact:' not in text and '{{table:' not in text, 'unresolved_template'
    assert compact(text).count('CanaryHOLD|DefaultOFF|ServingOFF|Historicalparityunresolved') == len(reader.pages)
    assert '0.042468' in text, 'regret_percent_unit'
    assert '1.331737' in text and '0.926743' in text, 'scoped_positive_and_failure_results'
    report = {'pass':True, 'pdf_sha256':digest, 'pages':len(reader.pages),
              'manuscript_word_count':inventory['manuscript_word_count'],
              'checked_manuscript_passages':checked, 'outside_page_characters':0,
              'page_metrics':page_metrics,
              'visual_review':{'state':'pending', 'method':'Render every page, inspect all contact sheets and full-resolution critical table/figure/reference pages.'},
              'limits':'This checks content presence and physical page bounds. It does not substitute for visual layout review or establish experimental qualification.'}
    (HERE / 'pdf_qa.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'page_metrics'},indent=2))


if __name__ == '__main__':
    main()
