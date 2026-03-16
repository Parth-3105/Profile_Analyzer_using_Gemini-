
"""
Flask app for LinkedIn & Instagram Profile Analyzer
Uses Gemini AI with Google Search grounding for Instagram profiles
"""

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import re
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Gemini client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

client = genai.Client(api_key=api_key)


# ─── Shared Utilities ───────────────────────────────────────────────

def clean_response(text):
    """Clean markdown formatting from AI response"""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-•●○▪▸→]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    return text.strip()


def parse_analysis(text):
    """Parse cleaned analysis into structured sections"""
    sections = {
        'strengths': [],
        'weaknesses': [],
        'suggestions': [],
        'rating': None
    }

    # Extract rating
    rating_match = re.search(
        r'(?:rating[:\s]+)?(\d+(?:\.\d+)?)\s*/\s*10', text, re.IGNORECASE
    )
    if rating_match:
        sections['rating'] = float(rating_match.group(1))

    lines = text.split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.search(r'strengths?', line, re.IGNORECASE):
            current_section = 'strengths'
        elif re.search(r'weaknesses?|areas for improvement', line, re.IGNORECASE):
            current_section = 'weaknesses'
        elif re.search(r'suggestions?|recommendations?|improvements?', line, re.IGNORECASE):
            current_section = 'suggestions'
        elif current_section and len(line) > 15:
            sections[current_section].append(line)

    return sections


# ─── LinkedIn Analysis ──────────────────────────────────────────────

def analyze_linkedin(profile_text):
    """Analyze LinkedIn profile using Gemini AI"""
    prompt = f"""
You are an expert LinkedIn profile reviewer. Analyze the following profile and provide a structured response.

Profile:
{profile_text}

Provide your analysis in the following format:

**Rating: [score]/10**

**Strengths:**
- [List 3-5 specific strengths]

**Weaknesses:**
- [List 3-5 areas for improvement]

**Suggestions:**
- [List 3-5 actionable recommendations]

Be concise, professional, and specific. Avoid markdown formatting in your response."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return clean_response(response.text)


# ─── Instagram Analysis (2-step with Google Search grounding) ───────

def fetch_instagram_profile(profile_url):
    """
    Step 1: Use Gemini with Google Search grounding to fetch
    public Instagram profile data from the given URL.
    """
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    prompt = f"""
Using Google Search, find all publicly available information about the Instagram profile at this URL: {profile_url}

Collect and summarise the following details as completely as possible:
- Username / handle
- Full name
- Bio / description
- Number of posts
- Number of followers
- Number of following
- Whether the account is verified
- Content niche / category (e.g. sports, fashion, tech, memes)
- Recent post topics or themes
- Engagement style (e.g. reels-heavy, carousel posts, stories highlights)
- Any notable collaborations, brand partnerships, or achievements
- Link in bio (if any)

Return ALL the information you can find in a clear, structured format.
If some data is unavailable, note it as "Not found".
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[grounding_tool]
        )
    )
    return response.text


def analyze_instagram_profile(profile_data, profile_url):
    """
    Step 2: Run a full analysis on the fetched profile data
    and generate a structured report.
    """
    prompt = f"""
You are an expert Instagram profile and social-media branding consultant.

Below is publicly available data about an Instagram profile ({profile_url}):

--- START PROFILE DATA ---
{profile_data}
--- END PROFILE DATA ---

Provide a thorough analysis in EXACTLY this format:

**Rating: [score]/10**

**Strengths:**
- [List 3-5 specific strengths of the profile — e.g. strong bio, consistent theme, high engagement, good use of reels]

**Weaknesses:**
- [List 3-5 areas for improvement — e.g. inconsistent posting schedule, weak call-to-action, low story engagement]

**Suggestions:**
- [List 3-5 actionable recommendations — e.g. post more reels, optimise bio link, add branded hashtags]

Be specific, reference actual data from the profile, and be professional. Avoid generic advice.
Do NOT use markdown formatting characters like ** or ## in your response.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return clean_response(response.text)


# ─── Routes ─────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze LinkedIn profile"""
    data = request.get_json()
    profile_text = data.get('profile_text', '').strip()

    if not profile_text:
        return jsonify({'error': 'Please enter some profile text to analyze'}), 400

    try:
        analysis_text = analyze_linkedin(profile_text)
        structured_data = parse_analysis(analysis_text)

        return jsonify({
            'success': True,
            'analysis': structured_data,
            'raw_text': analysis_text
        })
    except Exception as e:
        return jsonify({'error': f'Error analyzing profile: {str(e)}'}), 500


@app.route('/analyze-instagram', methods=['POST'])
def analyze_instagram():
    """
    Analyze Instagram profile using two Gemini calls:
      1. Google Search grounding to fetch public profile data
      2. Analysis & report generation
    """
    data = request.get_json()
    profile_url = data.get('profile_url', '').strip()

    if not profile_url:
        return jsonify({'error': 'Please enter an Instagram profile URL'}), 400

    # Basic URL validation
    ig_regex = re.compile(
        r'(https?://)?(www\.)?instagram\.com/[A-Za-z0-9_.]+/?', re.IGNORECASE
    )
    if not ig_regex.match(profile_url):
        return jsonify({
            'error': 'Please enter a valid Instagram profile URL '
                     '(e.g. https://www.instagram.com/cristiano/)'
        }), 400

    try:
        # Step 1 — fetch profile data via Search grounding
        profile_data = fetch_instagram_profile(profile_url)

        # Step 2 — run the full analysis
        analysis_text = analyze_instagram_profile(profile_data, profile_url)
        structured_data = parse_analysis(analysis_text)

        return jsonify({
            'success': True,
            'analysis': structured_data,
            'raw_text': analysis_text,
            'profile_data': profile_data   # optional: send raw fetched data too
        })
    except Exception as e:
        return jsonify({'error': f'Error analyzing Instagram profile: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
