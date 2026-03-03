"""
setup_accounts.py
=================
Run this once to:
1. Merge duplicate stats entries for Owen and Kyle
2. Create login accounts for each with password "aok"

Usage:
    pip install google-cloud-firestore bcrypt
    python setup_accounts.py
"""

from google.cloud import firestore
import bcrypt
import time

PROJECT = 'project-72eca311-a412-4fda-9be'
db = firestore.Client(project=PROJECT)

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def merge_and_create(display_name, password):
    key = display_name.lower()
    print(f"\n── Processing: {display_name} ──")

    # Find all stats docs that could belong to this user
    # Could be stored under the username or old random user_id keys
    stats_ref = db.collection('player_stats')
    all_docs = stats_ref.stream()

    matching = []
    for doc in all_docs:
        d = doc.to_dict()
        # Match by username field (case-insensitive)
        if d.get('username', '').lower() == key:
            matching.append((doc.id, d))

    print(f"  Found {len(matching)} stats doc(s):")
    for doc_id, data in matching:
        print(f"    [{doc_id}] games={data.get('games_played',0)}, wins={data.get('wins',0)}, total_score={data.get('total_score',0)}")

    # Merge all into one combined entry
    merged = {
        'username': display_name,
        'games_played': 0,
        'wins': 0,
        'total_score': 0,
        'total_bids': 0,
        'total_bid_count': 0,
    }
    for _, data in matching:
        merged['games_played']    += data.get('games_played', 0)
        merged['wins']            += data.get('wins', 0)
        merged['total_score']     += data.get('total_score', 0)
        merged['total_bids']      += data.get('total_bids', 0)
        merged['total_bid_count'] += data.get('total_bid_count', 0)

    print(f"  Merged stats: games={merged['games_played']}, wins={merged['wins']}, total_score={merged['total_score']}")

    # Write merged stats to the canonical key (lowercase username)
    stats_ref.document(key).set(merged)
    print(f"  ✓ Wrote merged stats to player_stats/{key}")

    # Delete any old docs that were stored under different keys
    for doc_id, _ in matching:
        if doc_id != key:
            stats_ref.document(doc_id).delete()
            print(f"  ✓ Deleted old stats doc: {doc_id}")

    # Create account (skip if already exists)
    account_ref = db.collection('accounts').document(key)
    if account_ref.get().exists:
        print(f"  Account already exists for {display_name}, skipping account creation.")
    else:
        account_ref.set({
            'username': display_name,
            'username_lower': key,
            'password_hash': hash_password(password),
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        })
        print(f"  ✓ Created account for {display_name}")

    print(f"  Done! {display_name} is ready to log in.")


if __name__ == '__main__':
    merge_and_create('Owen', 'aok')
    merge_and_create('Kyle', 'aok')
    print("\n✓ All done! Both accounts are set up.")
