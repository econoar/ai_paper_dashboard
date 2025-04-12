import feedparser
import requests
import io
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
    "transformers",
    "explainable ai",
    "self-supervised learning",
    "federated learning"
]

# Define the summarizer model we're using
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"

# Initialize the summarization pipeline
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

        combined_text = (entry.title + " " + entry.summary).lower()
        tags = [topic for topic in DESIRED_TOPICS if topic in combined_text]

        new_papers.append({
            'id': idx,  # Unique ID for this batch.
            'title': entry.title,
            'link': entry.link,
            'pdf_link': pdf_link,
            'summary': entry.summary,
            'tags': tags,
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

def generate_summary(text):
    """
    Uses the summarization pipeline to produce a concise summary.
    The prompt now instructs the model to generate 2-3 sentences focusing on key contributions.
    """
    improved_prompt = (
        "Provide a concise, 2-3 sentence summary highlighting the key contributions, novelty, and potential impact of the following research paper content:\n\n" 
        + text
    )
    summary_output = summarizer(
        improved_prompt,
        max_length=300,
        min_length=80,
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
    prompt = f"{truncated_text}"
    ai_summary = generate_summary(prompt)
    
    # Check if the request is AJAX
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'summary': ai_summary})
    else:
        return render_template("tweet_summary.html", paper=paper, tweet_summary=ai_summary)

@app.route('/insights/<int:paper_id>')
def insights(paper_id):
    try:
        paper = next(p for p in papers if p['id'] == paper_id)
    except StopIteration:
        return "Paper not found", 404
    if not paper.get("summary"):
        return "No summary available for generating insights.", 404
    # For demonstration, we simply echo the summary as the insight. You could replace this with a similar text-generation call.
    insight_text = "Insight: " + paper["summary"][:150] + "..."
    return render_template("insights.html", paper=paper, insight=insight_text)

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

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
