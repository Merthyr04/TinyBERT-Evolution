"""Build the second-annotator sheet for the 119 OD-100K entries.

Two artefacts, from the same source:

  data/od100k_secondpass_sheet.html   a self-contained click-through page: one
                                      card per entry, twelve class buttons,
                                      the first label hidden until asked for,
                                      keyboard shortcuts, and a button that
                                      downloads the finished CSV.  Open it in a
                                      browser; nothing is installed and nothing
                                      is sent anywhere.

  data/od100k_secondpass_sheet.csv    the same sheet as a plain CSV, for anyone
                                      who would rather type in a spreadsheet.

The first label is hidden by default on purpose: a second annotator who can see
the first one is not a second annotator.

`od100k_read_sheet.py` turns the finished CSV into Cohen's kappa.
"""
from __future__ import annotations

import csv
import html
import json
import os

import organic_agreement as OA

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SAMPLE = os.path.join(DATA, "od100k_secondpass_sample.json")

SHORT = {
    "art_music": "art and music",
    "commerce_industry": "commerce and industry",
    "geography_places": "geography and places",
    "history_politics": "history and politics",
    "law_education_institutions": "law and educational institutions",
    "literature_writing": "literature and writing",
    "myth_legend": "myth and legend",
    "peoples_languages": "peoples and languages",
    "philosophy_scholarship": "philosophy and scholarship",
    "religion_theology": "religion and theology",
    "science_nature": "science and nature",
    "war_military": "war and military",
}
ORDER = ["art_music", "commerce_industry", "geography_places",
         "history_politics", "law_education_institutions",
         "literature_writing", "myth_legend", "peoples_languages",
         "philosophy_scholarship", "religion_theology", "science_nature",
         "war_military"]


def load():
    with open(SAMPLE, encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows):
    path = os.path.join(DATA, "od100k_secondpass_sheet.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "idx", "first_label", "second_label", "text"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["idx"], r["label_1"], "", r["text"]])
    return path


def write_html(rows):
    cards = []
    for i, r in enumerate(rows, 1):
        btns = "".join(
            '<button class="cls" data-c="%s">%s</button>' % (c, html.escape(SHORT[c]))
            for c in ORDER)
        cards.append(
            '<li class="card" data-n="%d" data-idx="%d" data-first="%s">'
            '<div class="meta"><span class="num">%d</span>'
            '<span class="idx">idx %d</span>'
            '<span class="rev">first label: <b class="first">%s</b></span></div>'
            '<p class="txt">%s</p><div class="btns">%s</div></li>'
            % (i, r["idx"], html.escape(r["label_1"]), i, r["idx"],
               html.escape(SHORT[r["label_1"]]), html.escape(r["text"]), btns))

    legend = "".join('<li><b>%s</b> %s</li>' % (c, html.escape(SHORT[c]))
                     for c in ORDER)
    doc = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>OD-100K second annotation sheet</title>
<style>
:root{--bg:#f7f7f5;--fg:#1b1b1b;--mut:#6b6b6b;--card:#fff;--line:#e2e2dd;
--pick:#1f6feb;--picksoft:#e7f0ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
padding:14px 20px;z-index:5}
h1{font-size:18px;margin:0 0 6px}
.rules{color:var(--mut);font-size:13px;margin:0 0 8px;max-width:900px}
.legend{list-style:none;display:flex;flex-wrap:wrap;gap:6px 14px;padding:0;margin:0 0 10px;
font-size:12.5px;color:var(--mut)}
.bar{display:flex;align-items:center;gap:12px;font-size:13px}
.prog{height:6px;background:var(--line);border-radius:3px;flex:0 0 260px;overflow:hidden}
.prog i{display:block;height:100%;width:0;background:var(--pick)}
button{font:inherit;cursor:pointer}
.tools button{font-size:13px;padding:4px 10px;border:1px solid var(--line);
background:#fff;border-radius:6px}
ul.cards{list-style:none;margin:0;padding:12px 20px 60px;display:grid;
grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card.done{border-color:var(--pick)}
.meta{display:flex;gap:10px;align-items:baseline;font-size:12px;color:var(--mut);margin-bottom:6px}
.num{font-weight:700;color:var(--fg)}
.rev{margin-left:auto}
.first{display:none}
.card.show .first{display:inline}
.txt{margin:0 0 10px;font-size:14.5px}
.btns{display:flex;flex-wrap:wrap;gap:5px}
.cls{font-size:12px;padding:3px 8px;border:1px solid var(--line);background:#fff;
border-radius:999px;color:var(--fg)}
.cls:hover{border-color:var(--pick)}
.cls.on{background:var(--picksoft);border-color:var(--pick);color:var(--pick);font-weight:600}
kbd{font:inherit;font-size:11px;border:1px solid var(--line);border-radius:4px;
padding:0 4px;color:var(--mut)}
</style></head><body>
<header>
<h1>OD-100K second annotation</h1>
<p class="rules">Pick the one class that fits each entry. The first annotator's
label is hidden while you work and can be shown afterwards with <kbd>r</kbd>;
seeing it first would make this a review rather than a second annotation.
Keys: <kbd>1</kbd>-<kbd>9</kbd> and <kbd>q</kbd> <kbd>w</kbd> <kbd>e</kbd> choose
a class for the focused card, <kbd>&uarr;</kbd>/<kbd>&darr;</kbd> move,
<kbd>r</kbd> reveals the first label, <kbd>d</kbd> downloads the CSV.</p>
<ul class="legend">__LEGEND__</ul>
<div class="bar"><div class="prog"><i id="p"></i></div>
<span id="n">0 / __N__ annotated</span>
<span class="tools"><button id="dl">Download CSV</button>
<button id="clr">Clear</button></span></div>
</header>
<ul class="cards">__CARDS__</ul>
<script>
var ORDER = __ORDER__;
var cards = [].slice.call(document.querySelectorAll('.card'));
var state = {};
var cur = 0;
function refresh(){
  var done = 0;
  cards.forEach(function(c){
    var n = c.dataset.n, v = state[n];
    var on = c.querySelector('.cls.on');
    if(on) on.classList.remove('on');
    if(v){ done++; c.classList.add('done');
      var b = c.querySelector('.cls[data-c="'+v+'"]');
      if(b) b.classList.add('on');
    } else { c.classList.remove('done'); }
  });
  document.getElementById('n').textContent = done + ' / ' + cards.length + ' annotated';
  document.getElementById('p').style.width = (100*done/cards.length) + '%';
}
function focusCard(i){
  if(i < 0 || i >= cards.length) return;
  cur = i; cards[i].scrollIntoView({block:'center',behavior:'smooth'});
}
cards.forEach(function(c,i){
  c.setAttribute('tabindex','0');
  c.addEventListener('focus', function(){ cur = i; });
  c.querySelectorAll('.cls').forEach(function(b){
    b.addEventListener('click', function(){
      state[c.dataset.n] = b.dataset.c; refresh(); focusCard(i+1);
    });
  });
});
document.addEventListener('keydown', function(e){
  if(e.metaKey || e.ctrlKey || e.altKey) return;
  var c = cards[cur]; if(!c) return;
  var k = e.key;
  if(k >= '1' && k <= '9'){ state[c.dataset.n] = ORDER[+k-1]; refresh(); focusCard(cur+1); }
  else if(k === 'q'){ state[c.dataset.n] = ORDER[9];  refresh(); focusCard(cur+1); }
  else if(k === 'w'){ state[c.dataset.n] = ORDER[10]; refresh(); focusCard(cur+1); }
  else if(k === 'e'){ state[c.dataset.n] = ORDER[11]; refresh(); focusCard(cur+1); }
  else if(k === 'ArrowDown'){ focusCard(cur+1); }
  else if(k === 'ArrowUp'){ focusCard(cur-1); }
  else if(k === 'r'){ cards.forEach(function(x){ x.classList.toggle('show'); }); }
  else if(k === 'd'){ download(); }
  else return;
  e.preventDefault();
});
function download(){
  var rows = [['n','idx','first_label','second_label','text']];
  cards.forEach(function(c){
    var v = state[c.dataset.n] || '';
    rows.push([c.dataset.n, c.dataset.idx, c.dataset.first, v,
               c.querySelector('.txt').textContent]);
  });
  var csv = rows.map(function(r){
    return r.map(function(x){ return '"' + String(x).replace(/"/g,'""') + '"'; }).join(',');
  }).join('\\n');
  var a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {type:'text/csv'}));
  a.download = 'od100k_secondpass_filled.csv';
  a.click();
}
document.getElementById('dl').addEventListener('click', download);
document.getElementById('clr').addEventListener('click', function(){
  state = {}; refresh();
});
var saved = localStorage.getItem('od100k_second');
if(saved){ try { state = JSON.parse(saved); } catch(e) { state = {}; } }
refresh(); focusCard(0);
setInterval(function(){ localStorage.setItem('od100k_second', JSON.stringify(state)); }, 2000);
</script></body></html>
"""
    doc = (doc.replace("__LEGEND__", legend)
              .replace("__CARDS__", "".join(cards))
              .replace("__N__", str(len(rows)))
              .replace("__ORDER__", json.dumps(ORDER)))
    path = os.path.join(DATA, "od100k_secondpass_sheet.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


def main():
    rows = load()
    unknown = sorted({r["label_1"] for r in rows} - set(OA.CLASSES))
    if unknown:
        raise SystemExit("make_annotation_sheet.py: unknown class %r" % unknown)
    a = write_csv(rows)
    b = write_html(rows)
    print("wrote %s" % a)
    print("wrote %s" % b)
    print("%d entries, %d classes" % (len(rows), len(OA.CLASSES)))


if __name__ == "__main__":
    main()
