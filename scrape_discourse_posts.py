# scrape_discourse_posts.py
import requests
import json
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Configuration for your Discourse forum
DISCOURSE_BASE_URL = os.getenv("DISCOURSE_BASE_URL", "https://discourse.onlinedegree.iitm.ac.in")
DISCOURSE_T_COOKIE = os.getenv("DISCOURSE_T_COOKIE")
DISCOURSE_FORUM_SESSION_COOKIE = os.getenv("DISCOURSE_FORUM_SESSION_COOKIE")

if not DISCOURSE_T_COOKIE or not DISCOURSE_FORUM_SESSION_COOKIE:
    print("WARNING: DISCOURSE_T_COOKIE or DISCOURSE_FORUM_SESSION_COOKIE not set in .env.")
    print("         Please extract these from your browser and add them to your .env file.")
    print("         Scraping may fail for restricted content.")

def create_session_with_browser_cookies(discourse_url, cookies_dict):
    """
    Create a requests session with cookies extracted from browser.
    
    Parameters:
        discourse_url (str): Base URL of the IIT Madras Discourse instance.
        cookies_dict (dict): Dictionary of cookie names and values from browser.
        
    Returns:
        requests.Session: Session object with authentication cookies.
    """
    session = requests.Session()
    
    # Add each cookie to the session
    # Extract domain from discourse_url
    domain = discourse_url.split('//')[1].split('/')[0]
    for name, value in cookies_dict.items():
        session.cookies.set(name, value, domain=domain)
    
    return session

# Inside scrape_discourse_posts.py, modify the get_posts_in_topic function:

def get_posts_in_topic(session, topic_id, topic_slug):
    """Fetches all posts for a given topic ID using the authenticated session."""
    posts = []
    url = f"{DISCOURSE_BASE_URL}/t/{topic_id}.json"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        topic_data = response.json()

        for post in topic_data.get('post_stream', {}).get('posts', []):
            # Attempt to get 'raw' content first
            post_content = post.get('raw')
            if not post_content:
                # If 'raw' is not available, try 'cooked' (HTML content)
                post_content = post.get('cooked')
                if post_content:
                    # If 'cooked' is found, convert HTML to plain text
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(post_content, 'html.parser')
                    post_content = soup.get_text(separator=' ', strip=True)
                else:
                    # If neither 'raw' nor 'cooked' are found, log and skip or use empty string
                    print(f"Warning: No 'raw' or 'cooked' content found for post ID {post.get('id')} in topic {topic_id}")
                    post_content = "" # Fallback to empty string

            posts.append({
                'id': post['id'],
                'topic_id': topic_id,
                'post_number': post['post_number'],
                'raw': post_content, # Store the extracted plain text
                'created_at': post['created_at'],
                'url': f"{DISCOURSE_BASE_URL}/t/{topic_slug}/{topic_id}/{post['post_number']}",
                'username': post['username']
            })
        return posts
    except requests.exceptions.RequestException as e:
        print(f"Error fetching posts for topic {topic_id}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while processing topic {topic_id}: {e}")
        return []
    
def get_topics_by_date_range(session, start_date, end_date):
    """
    Fetches topic IDs within a specific date range using Discourse search API
    and the authenticated session.
    """
    all_topic_info = {} # Using a dict to store unique topics with their slugs
    page = 1
    per_page = 50 # Default per page for search results

    # Format dates for Discourse search query
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Search query to find topics created or updated within the date range.
    search_query = f"after:{start_str} before:{end_str}"
    print(f"Searching Discourse for: '{search_query}'")

    while True:
        search_url = f"{DISCOURSE_BASE_URL}/search.json?q={search_query}&page={page}"
        try:
            response = session.get(search_url, timeout=15)
            response.raise_for_status()
            search_results = response.json()

            topics = search_results.get('topics', [])
            posts = search_results.get('posts', []) # Search also returns individual posts

            if not topics and not posts: # No more results
                break

            for topic in topics:
                all_topic_info[topic['id']] = topic['slug']
            for post in posts: # Ensure to capture topic_ids and slugs from posts in search results too
                # Need to fetch topic details to get slug if not directly in post search result
                # For simplicity, assuming topic slug is generally available or can be derived.
                # In a real scenario, you might need an extra API call for topic slug if not present.
                if 'topic_id' in post and 'topic_slug' in post: # If post search results include topic_slug
                     all_topic_info[post['topic_id']] = post['topic_slug']
                elif 'topic_id' in post: # Fallback: just use topic_id if slug not readily available
                    all_topic_info[post['topic_id']] = f"topic-{post['topic_id']}" # Dummy slug
                
            print(f"Found {len(topics)} topics and {len(posts)} posts on page {page}. Total unique topics found so far: {len(all_topic_info)}")

            # Heuristic to stop pagination if current page has fewer results than per_page
            if len(topics) + len(posts) < per_page and page > 1:
                break

            page += 1
            time.sleep(1) # Be polite
        except requests.exceptions.RequestException as e:
            print(f"Error during Discourse search page {page}: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred during search page {page}: {e}")
            break
    
    # Return as a list of (topic_id, topic_slug) tuples
    return list(all_topic_info.items())


def get_discourse_posts(start_date_str, end_date_str):
    """Gathers Discourse posts within a specified date range using browser cookies."""
    all_posts = []
    
    # Prepare cookies for the session
    browser_cookies = {
        '_t': DISCOURSE_T_COOKIE,
        '_forum_session': DISCOURSE_FORUM_SESSION_COOKIE
        # Add other relevant cookies if needed, but these two are usually sufficient
    }

    # Create authenticated session
    session = create_session_with_browser_cookies(DISCOURSE_BASE_URL, browser_cookies)

    # Verify authentication (optional, but good for debugging)
    try:
        response = session.get(f"{DISCOURSE_BASE_URL}/session/current.json", timeout=5)
        if response.status_code == 200:
            user_data = response.json()
            print(f"Authenticated to Discourse as: {user_data['current_user']['username']}")
        else:
            print(f"Authentication verification failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            print("Please check your cookies and their expiration.")
            return [] # Exit if authentication fails
    except requests.exceptions.RequestException as e:
        print(f"Authentication verification request failed: {e}")
        print("Please check your internet connection and Discourse URL.")
        return []


    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    print(f"Getting topic IDs from {start_date_str} to {end_date_str}...")
    topic_info_list = get_topics_by_date_range(session, start_date, end_date)
    print(f"Retrieved {len(topic_info_list)} unique topic IDs and slugs.")

    for i, (topic_id, topic_slug) in enumerate(topic_info_list):
        print(f"Fetching posts for topic {topic_id} (slug: {topic_slug}) ({i+1}/{len(topic_info_list)})")
        posts_in_topic = get_posts_in_topic(session, topic_id, topic_slug)
        
        # Filter posts by date again, as topic-level retrieval might include older/newer posts
        for post in posts_in_topic:
            # Parse 'created_at' and ensure it's timezone-aware or strip timezone for comparison
            # Example: "2024-04-14T10:30:00.000Z"
            # Using fromisoformat and then .replace(tzinfo=None) to make it naive for comparison
            post_date = datetime.fromisoformat(post['created_at'].replace('Z', '+00:00')).replace(tzinfo=None) 
            
            # Compare only date parts to match the "1 Jan 2025 - 14 Apr 2025" requirement
            if start_date.date() <= post_date.date() <= end_date.date():
                all_posts.append({
                    'text': post['raw'],
                    'source_url': post['url'],
                    'type': 'discourse_post',
                    'title': f"Discourse Topic: {post['topic_id']} Post {post['post_number']} by {post['username']}"
                })
        time.sleep(0.5) # Be polite

    return all_posts

if __name__ == "__main__":
    # Define your desired date range
    START_DATE = "2025-01-01"
    END_DATE = "2025-04-14"

    print("Starting Discourse post scraping with browser cookies...")
    all_discourse_posts = get_discourse_posts(START_DATE, END_DATE)
    output_path = os.path.join('data', 'discourse_posts_raw.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_discourse_posts, f, indent=2, ensure_ascii=False)
    print(f"Scraped {len(all_discourse_posts)} Discourse posts.")
    print(f"Raw Discourse posts saved to {output_path}")

