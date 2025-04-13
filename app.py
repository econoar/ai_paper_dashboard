import feedparser
import requests
import io
import re
from PyPDF2 import PdfReader
from flask import Flask, render_template, redirect, url_for, request, jsonify
from transformers import pipeline
from urllib.parse import quote
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo  # Requires Python 3.9+

# Flask & SQLAlchemy Setup
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///summaries.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy(app)

# Redis caching setup
import redis
import json
cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def cache_set(key, data, expire_seconds=600):
    cache.set(key, json.dumps(data), ex=expire_seconds)

def cache_get(key):
    data = cache.get(key)
    if data:
        return json.loads(data)
    return None

# Database model for generated summaries
class Summary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, nullable=False, unique=True)
    summary_text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Summary paper_id={self.paper_id}>"

# Global list for papers
papers = []

# Define desired topics (full names)
DESIRED_TOPICS = [
    "reinforcement learning",
    "digital twin",
    "agent coordination",
    "multi-agent systems",
    "transformers",  # We'll handle transformer variations via regex
    "explainable ai",
    "self-supervised learning",
    "federated learning"
]

# Default summarization model
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
    """
    Returns a list of full tag names using exact matching and a regex for transformer variations.
    """
    keywords = [
        "reinforcement learning",
        "digital twin",
        "agent coordination",
        "multi-agent systems",
        "explainable ai",
        "self-supervised learning",
        "federated learning"
    ]
    tags = []
    for keyword in keywords:
        if keyword in combined_text:
            tags.append(keyword)
    if re.search(r'\btransformer(s)?\b', combined_text):
        tags.append("transformers")
    return tags

def fetch_papers(selected_tag="all", max_results=100, start=0):
    global papers
    cache_key = f"papers:{selected_tag}:{start}:{max_results}"
    cached = cache_get(cache_key)
    if cached:
        papers = cached
        return

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
    cache_set(cache_key, papers, expire_seconds=600)

def extract_pdf_text(pdf_url):
    try:
        response = requests.get(pdf_url, timeout=30)
        if response.status_code == 200:
            with io.BytesIO(response.content) as f:
                reader = PdfReader(f)
                all_text = []
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    all_text.append(page_text)
                return "\n".join(all_text)
        else:
            print("Failed to download PDF; status code", response.status_code)
            return None
    except Exception as e:
        print("Error extracting PDF text:", e)
        return None

def generate_summary(text, min_length=80, max_length=300, model_name=SUMMARIZER_MODEL, custom_prompt=None):
    if custom_prompt:
        prompt_text = custom_prompt + "\n\n" + text
    else:
        prompt_text = (
            "Provide a concise 2-3 sentence summary for social media audiences. "
            "Highlight the paper's key contributions, novelty, and potential impact:\n\n" + text
        )
    if model_name != SUMMARIZER_MODEL:
        temp_summarizer = pipeline("summarization", model=model_name)
        summary_output = temp_summarizer(
            prompt_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
    else:
        summary_output = summarizer(
            prompt_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
    return summary_output[0]['summary_text'].strip()

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
            time_str = dt.strftime("%H:%M")
        except Exception:
            day_key = pub
            time_str = ""
        paper['time'] = time_str
        grouped_papers[day_key].append(paper)
    
    return render_template('index.html', grouped_papers=grouped_papers,
                           selected_tag=selected_tag, page=page, papers=papers)

@app.route('/summarize/<int:paper_id>')
def summarize_paper(paper_id):
    try:
        paper = next(p for p in papers if p['id'] == paper_id)
    except StopIteration:
        return jsonify({'error': 'Paper not found'}), 404
    pdf_url = paper.get("pdf_link")
    if not pdf_url:
        return jsonify({'error': 'No PDF URL available for this paper.'}), 404

    # Check persistent storage for an existing summary
    from app import Summary  # Ensure Summary is imported from the same file
    existing = Summary.query.filter_by(paper_id=paper_id).first()
    if existing:
        ai_summary = existing.summary_text
    else:
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
        # Save to persistent storage
        new_summary = Summary(paper_id=paper_id, summary_text=ai_summary)
        db.session.add(new_summary)
        db.session.commit()
    
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
    app.run(host="0.0.0.0", debug=True)