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

app = Flask(__name__)

# Global list to store papers
papers = []

# Define our desired topics (used for filtering and tagging)
DESIRED_TOPICS = [
    "reinforcement learning",
    "digital twin",
    "agent coordination",
    "multi-agent systems",
    "transformers",  # We'll handle transformers via regex too
    "explainable ai",
    "self-supervised learning",
    "federated learning"
]

# Default summarization model
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"

# Initialize the summarization pipeline with the default model
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
    Uses both exact matching and regex to detect keywords.
    Returns a list of abbreviated tags.
    """
    # Basic mapping for most keywords:
    tag_mapping = {
        "reinforcement learning": "RL",
        "digital twin": "DT",
        "agent coordination": "AC",
        "multi-agent systems": "MAS",
        "explainable ai": "XAI",
        "self-supervised learning": "SSL",
        "federated learning": "FL"
    }
    tags = []
    # Exact matching for the terms in tag_mapping
    for key, short in tag_mapping.items():
        if key in combined_text:
            tags.append(short)
    # Use regex for "transformer" variations: match both "transformer" and "transformers"
    if re.search(r'\btransformer(s)?\b', combined_text):
        tags.append("TR")
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
            published_str = dt_pst.strftime("%b %d, %Y %I:%M %p PST")
        except Exception:
            published_str = published

        # Concatenate title and summary in lower case for keyword detection.
        combined_text = (entry.title + " " + entry.summary).lower()
        tags = get_tags(combined_text)

        new_papers.append({
            'id': idx,
            'title': entry.title,
            'link': entry.link,
            'pdf_link': pdf_link,
            'summary': entry.summary,
            'tags': tags,  # List of abbreviated tags (empty if no keywords found)
            'published': published_str
        })
    papers = new_papers

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
    """
    Uses the summarization pipeline to produce a concise summary.
    If a custom_prompt is provided, it is prepended to the paper text.
    """
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
            dt = datetime.strptime(pub, "%b %d, %Y %I:%M %p PST")
            day_key = dt.strftime("%b %d, %Y")
            time_str = dt.strftime("%I:%M %p PST")
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
    app.run(host="0.0.0.0", debug=True)