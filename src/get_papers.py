import re
import time
from typing import Dict, List, Iterator, Any
from pathlib import Path

import openreview.api
import pandas as pd
from tqdm import tqdm
import numpy as np

def _extract_paper_data_from_iterator(notes_iterator: Iterator) -> List[Dict]:
    """Extracts relevant paper fields from an OpenReview note iterator."""
    papers_list = []
    for note in notes_iterator:
        content = note.content
        papers_list.append({
            'id': note.id,
            'title': content.get('title', {}).get('value', 'N/A'),
            'abstract': content.get('abstract', {}).get('value', 'N/A'),
            'keywords': content.get('keywords', {}).get('value', []),
            'authors': content.get('authors', {}).get('value', [])
        })
    return papers_list

def get_rich_paper_details(client: openreview.api.OpenReviewClient, submissions: List[Dict]) -> List[Dict]:
    """Enriches paper data with review ratings and decisions."""
    for paper in tqdm(submissions, desc="Fetching detailed review information"):
        try:
            related_notes = client.get_notes(forum=paper['id'])
        except Exception as e:
            print(f"Warning: Could not fetch details for paper {paper['id']}. Error: {e}")
            continue

        ratings = []
        decision = 'N/A'
        for note in related_notes:
            is_decision_note = any(re.search(r'/Meta_Review|/Decision$', inv) for inv in note.invitations)
            if is_decision_note:
                decision_value = note.content.get('decision', {}).get('value')
                if decision_value: 
                    decision = decision_value

            is_review_note = any(re.search(r'/Official_Review$', inv) for inv in note.invitations)
            if is_review_note:
                rating_value = note.content.get('rating', {}).get('value')
                if isinstance(rating_value, str):
                    match = re.search(r'^\d+', rating_value)
                    if match: 
                        ratings.append(int(match.group(0)))
                elif isinstance(rating_value, (int, float)):
                    ratings.append(int(rating_value))

        paper['decision'] = decision
        paper['review_ratings'] = ratings
        paper['avg_rating'] = round(np.mean(ratings), 2) if ratings else None
        time.sleep(1.1)  # Rate limit to avoid overwhelming the API
    return submissions

def get_all_papers(client: openreview.api.OpenReviewClient, conference_id: str) -> List[Dict]:
    """Fetches all papers for a conference, trying multiple methods."""
    invitation_id = f'{conference_id}/-/Submission'
    print(f"--> Attempting with Invitation ID: '{invitation_id}'")
    try:
        notes_iterator = client.get_all_notes(invitation=invitation_id)
        papers = _extract_paper_data_from_iterator(tqdm(notes_iterator, desc="Processing papers (Invitation)"))
        if papers: 
            return papers
    except Exception as e:
        print(f"    - Invitation ID method failed: {e}")

    print(f"\n--> Falling back to Venue ID: '{conference_id}'")
    try:
        notes_iterator = client.get_all_notes(content={'venueid': conference_id})
        papers = _extract_paper_data_from_iterator(tqdm(notes_iterator, desc="Processing papers (Venue ID)"))
        if papers: 
            return papers
    except Exception as e:
        print(f"    - Venue ID method failed: {e}")
    return []

def main(config: Dict[str, Any], raw_data_dir: Path) -> Path:
    """Main function to be called as a module for fetching paper data."""
    conference_id = config['conference_id']
    fetch_reviews = config.get('fetch_reviews', False)
    limit = config.get('limit', None)

    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    submissions_data = get_all_papers(client, conference_id)
    
    if not submissions_data:
        print("\nCould not fetch any paper data.")
        return None

    if limit and limit > 0:
        print(f"\nLimiting to the first {limit} papers as requested.")
        submissions_data = submissions_data[:limit]

    if fetch_reviews:
        print("\nFetching review details as requested...")
        final_data = get_rich_paper_details(client, submissions_data)
    else:
        final_data = submissions_data

    print(f"\nProcessed {len(final_data)} papers.")
    df = pd.DataFrame(final_data)
    
    safe_conf_name = conference_id.replace('/', '_').replace('.', '')
    suffix = "_reviews" if fetch_reviews else ""
    limit_suffix = f"_limit{limit}" if limit else ""
    output_filename = f"{safe_conf_name}_papers{suffix}{limit_suffix}.jsonl"
    output_path = raw_data_dir / output_filename
    
    df.to_json(output_path, orient='records', lines=True)
    print(f"Data successfully saved to: {output_path}")
    
    return output_path