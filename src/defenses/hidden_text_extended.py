# Detect CSS-based concealment beyond v1 (display:none/visibility:hidden/etc.).


from __future__ import annotations

# JS snippet that returns a list of strings: text content of elements that
# are concealed by CSS but contain non-empty text.
EXTENDED_HIDDEN_JS = r"""
() => {
  const hidden = [];
  const elements = Array.from(document.querySelectorAll('body *'));
  for (const el of elements) {
    const txt = (el.innerText || '').trim();
    if (!txt) continue;
    const style = window.getComputedStyle(el);
    let concealed = false;
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0' || el.getAttribute('aria-hidden') === 'true') concealed = true;
    if (!concealed) {
      const fontSize = parseFloat(style.fontSize || '16');
      if (!isNaN(fontSize) && fontSize > 0 && fontSize < 4) concealed = true;
    }
    if (!concealed) {
      const pos = style.position;
      if (pos === 'absolute' || pos === 'fixed') {
        const left = parseFloat(style.left || '0');
        const top = parseFloat(style.top || '0');
        if (left < -1000 || top < -1000) concealed = true;
      }
    }
    if (!concealed) {
      const fg = style.color;
      const bg = style.backgroundColor;
      if (fg && bg && fg !== 'rgba(0, 0, 0, 0)' && fg === bg) concealed = true;
    }
    if (!concealed) {
      const clip = style.clipPath || '';
      if (clip.includes('inset(100%)')) concealed = true;
    }
    if (!concealed) {
      const tr = style.transform || '';
      if (tr.includes('matrix(0') || tr.includes('scale(0)')) concealed = true;
    }
    if (concealed) hidden.push(txt);
    if (hidden.length >= 30) break;
  }
  return hidden;
}
"""
