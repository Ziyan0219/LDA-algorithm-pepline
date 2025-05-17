import os
import csv
import re
import time
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build

# Replace with your YouTube Data API Key
API_KEY = "YOUR_API_KEY_HERE"
youtube = build('youtube', 'v3', developerKey=API_KEY)

def sanitize_filename(name: str) -> str:
    """
    Clean illegal filename characters, preserving Chinese, English, numbers, spaces,
    underscores, and hyphens
    """
    return re.sub(r'[^\w\u4e00-\u9fa5\-_ ]', '_', name).strip()

def extract_video_id(url: str) -> str:
    """
    Extract the 11-character video ID from various YouTube URL formats
    """
    if not url:
        return ''

    url = url.strip().replace('\r', '').replace('\n', '')

    # If no protocol, add one to facilitate urlparse
    if not re.match(r'^[a-zA-Z]+://', url):
        url = 'https://' + url

    parsed = urlparse(url)
    host   = (parsed.hostname or '').lower()

    # Shortened youtu.be link
    if host.endswith('youtu.be'):
        vid = parsed.path.lstrip('/')
        return vid[:11] if len(vid) >= 11 else ''

    # Standard YouTube URLs
    if host.endswith('youtube.com') or host.endswith('m.youtube.com'):
        if parsed.path == '/watch':
            qs = parse_qs(parsed.query)
            if 'v' in qs:
                return qs['v'][0][:11]
        # /embed/VIDEO_ID, /v/VIDEO_ID, /shorts/VIDEO_ID
        m = re.match(r'^/(?:embed|v|shorts)/([^/?&]{11})', parsed.path)
        if m:
            return m.group(1)

    # Fallback: search for any 11-character ID in the string
    m = re.search(r'([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else ''


def get_top_level_comments(video_id: str, max_results: int = 100) -> list:
    """
    Retrieve all top-level comments for a given video
    """
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    while request:
        response = request.execute()
        for thread in response.get("items", []):
            snip = thread["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": thread["id"],
                "author": snip["authorDisplayName"],
                "text": snip["textDisplay"].replace('\n', ' '),
                "like_count": snip["likeCount"],
                "published_at": snip["publishedAt"]
            })
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            pageToken=response.get("nextPageToken")
        ) if response.get("nextPageToken") else None
    return comments


def save_comments_to_csv(comments: list, filename: str):
    """
    Write comments to CSV with fields: comment_id, author, text, like_count, published_at
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["comment_id", "author", "text", "like_count", "published_at"])
        for c in comments:
            writer.writerow([
                c["comment_id"],
                c["author"],
                c["text"],
                c["like_count"],
                c["published_at"]
            ])

if __name__ == "__main__":
    input_csv = "combined_short_sample.csv"
    output_folder = "output_comments"
    os.makedirs(output_folder, exist_ok=True)

    # Open with utf-8-sig to strip BOM
    with open(input_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        print("Detected CSV columns:", reader.fieldnames)

        title_field = next(f for f in reader.fieldnames if "title" in f)
        link_field  = next(f for f in reader.fieldnames if "video_id" in f)

        for idx, row in enumerate(reader):
            title = row.get(title_field, "").strip()
            link  = row.get(link_field,  "").strip()
            video_id = extract_video_id(link)
            if not video_id:
                print(f"[Skip] Cannot parse video_id, original link=<${link}>")
                continue

            safe_title   = sanitize_filename(title)
            out_filename = os.path.join(output_folder, f"{safe_title}.csv")

            try:
                print(f"[{idx+1}] Processing: {title} → {video_id}")
                comments = get_top_level_comments(video_id)
                save_comments_to_csv(comments, out_filename)
                print(f"✅ Saved {len(comments)} comments to {out_filename}\n")
            except Exception as e:
                print(f"❌ Failed {title}({video_id}): {e}\n")

            time.sleep(1)
