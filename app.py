import feedparser
import requests
import io
import re
from PyPDF2 import PdfReader
from flask import Flask, render_template, redirect, url_for, request, jsonify
from transformers import pipeline
from urllib.parse import quote, unquote_plus
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # Requires Python 3.9+
import time  # for timestamp conversion in feeds
from bs4 import BeautifulSoup
import html
from flask_caching import Cache
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

app.config['CACHE_TYPE'] = 'simple'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['CACHE_DEFAULT_TIMEOUT'] = 0
cache = Cache(app)

papers = []

DESIRED_TOPICS = [
    "reinforcement learning",
    "digital twin",
    "agent coordination",
    "multi-agent systems",
    "transformers",
    "explainable ai",
    "self-supervised learning",
    "federated learning"
]

SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)

def build_query(selected_tag="all", max_results=100, start=0):
    base_categories = "(cat:cs.AI+OR+cat:cs.LG+OR+cat:stat.ML)"
    if selected_tag.lower() == "all":
        topics_query = "+OR+".join([f'all:"{topic}"' for topic in DESIRED_TOPICS])
    else:
        topics_query = f'all:"{selected_tag}"'
    full_query = f"{base_categories}+AND+({topics_query})"
    encoded_query = quote(full_query, safe='()+OR:"')
    query_url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={encoded_query}&start={start}&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    return query_url

def get_tags(combined_text):
    tags = []
    for keyword in DESIRED_TOPICS:
        if keyword in combined_text:
            tags.append(keyword)
    if re.search(r'\btransformer(s)?\b', combined_text):
        tags.append("transformers")
    return tags

def fetch_papers(selected_tag="all", max_results=100, start=0):
    global papers
    query_url = build_query(selected_tag, max_results, start)
    feed = feedparser.parse(query_url)
    new_papers = []
    for idx, entry in enumerate(feed.entries):
        pdf_link = None
        for link in entry.get("links", []):
            if link.get("type") == "application/pdf":
                pdf_link = link.get("href")
                break
        pdf_link = pdf_link or entry.link

        published = entry.get("published", "No date available")
        try:
            dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            dt_pst = dt.astimezone(ZoneInfo("America/Los_Angeles"))
            published_str = dt_pst.strftime("%b %d, %Y %H:%M")
        except Exception:
            published_str = published

        combined_text = (entry.title + " " + entry.summary).lower()
        tags = get_tags(combined_text)

        new_papers.append({
            'id': idx,
            'title': entry.title,
            'link': entry.link,
            'pdf_link': pdf_link,
            'summary': entry.summary,
            'tags': tags,
            'published': published_str
        })
    papers = new_papers

@app.route('/')
def index():
    selected_tag = request.args.get("tag", "all")
    try:
        page = int(request.args.get("page", "1"))
    except ValueError:
        page = 1
    max_results = 100
    start = (page - 1) * max_results
    fetch_papers(selected_tag, max_results, start)

    grouped_papers = defaultdict(list)
    for paper in papers:
        pub = paper.get("published", "No date available")
        try:
            dt = datetime.strptime(pub, "%b %d, %Y %H:%M")
            day_key = dt.strftime("%b %d, %Y")
        except Exception as e:
            print(f"⚠️ Date parsing failed for: {pub} → {e}")
            day_key = pub
        grouped_papers[day_key].append(paper)

    return render_template('index.html', grouped_papers=grouped_papers,
                           selected_tag=selected_tag, page=page, papers=papers)

@app.route('/news')
def news():
    selected_source = unquote_plus(request.args.get("source", "").strip())
    print(f"🔎 Selected Source: {selected_source}")

    feed_sources = [
        {"url": "https://techcrunch.com/category/artificial-intelligence/feed/", "source": "TechCrunch"},
        {"url": "https://venturebeat.com/category/ai/feed/", "source": "VentureBeat"},
        {"url": "https://www.wired.com/feed/tag/ai/latest/rss", "source": "Wired"},
        {"url": "https://www.zdnet.com/topic/artificial-intelligence/rss.xml", "source": "ZDNet"},
        {"url": "https://syncedreview.com/feed/", "source": "Synced Review"}
    ]

    news_items = []
    for src in feed_sources:
        if selected_source and src["source"] != selected_source:
            continue

        parsed = feedparser.parse(src["url"])
        for entry in parsed.entries:
            summary_html = entry.get("summary") or entry.get("description", "")
            soup = BeautifulSoup(summary_html, "html.parser")
            text_only = soup.get_text()
            summary = html.unescape(text_only)

            published_parsed = entry.get("published_parsed")
            if published_parsed:
                dt_utc = datetime(*published_parsed[:6], tzinfo=timezone.utc)
                dt_pst = dt_utc.astimezone(ZoneInfo("America/Los_Angeles"))
                formatted_time = dt_pst.strftime("%H:%M")
                published_str = dt_pst.strftime("%b %d, %Y %H:%M")
                day_key = dt_pst.strftime("%b %d, %Y")
                timestamp = dt_pst.timestamp()
            else:
                published_str = entry.get("published", "No date")
                formatted_time = published_str
                day_key = published_str
                timestamp = 0

            news_items.append({
                "title": entry.title,
                "link": entry.link,
                "summary": summary,
                "published": published_str,
                "formatted_time": formatted_time,
                "source": src["source"],
                "timestamp": timestamp,
                "day_key": day_key
            })

    grouped_unsorted = defaultdict(list)
    for item in news_items:
        grouped_unsorted[item["day_key"]].append(item)

    for day in grouped_unsorted:
        grouped_unsorted[day].sort(key=lambda x: x["timestamp"], reverse=True)

    grouped_news = dict(
        sorted(
            grouped_unsorted.items(),
            key=lambda x: datetime.strptime(x[0], "%b %d, %Y"),
            reverse=True
        )
    )

    return render_template(
        "news.html",
        grouped_news=grouped_news,
        sources=[src["source"] for src in feed_sources],
        selected_source=selected_source
    )

@app.route('/summarize/<int:paper_id>')
def summarize_paper(paper_id):
    try:
        paper = next(p for p in papers if p['id'] == paper_id)
    except StopIteration:
        return jsonify({'error': 'Paper not found'}), 404
    pdf_url = paper.get("pdf_link")
    if not pdf_url:
        return jsonify({'error': 'No PDF URL available for this paper.'}), 404
    pdf_text = extract_pdf_text(pdf_url)
    if not pdf_text:
        return jsonify({'error': 'Could not download or extract text from PDF.'}), 404
    truncated_text = pdf_text[:5000]
    try:
        min_length = int(request.args.get("min_length", 80))
        max_length = int(request.args.get("max_length", 300))
    except ValueError:
        min_length = 80
        max_length = 300
    model_name = request.args.get("model", SUMMARIZER_MODEL)
    custom_prompt = request.args.get("prompt", None)

    ai_summary = generate_summary(truncated_text, min_length, max_length, model_name, custom_prompt)

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'summary': ai_summary})
    else:
        return render_template("tweet_summary.html", paper=paper, tweet_summary=ai_summary)

@app.route('/open_pdf/<int:paper_id>')
def open_pdf(paper_id):
    try:
        paper = next(p for p in papers if p['id'] == paper_id)
    except StopIteration:
        return "Paper not found", 404
    pdf_url = paper.get("pdf_link")
    if not pdf_url:
        return "No PDF URL available for this paper.", 404
    return redirect(pdf_url)

@app.route('/bookmarks')
def bookmarks():
    return render_template('bookmarks.html', papers=papers)

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
