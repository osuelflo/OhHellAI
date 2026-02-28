"""
oh_hell_server.py
=================
Flask API server for Oh Hell card game.
Multi-session: each browser tab gets its own game via session_id.

Usage (local):
    python oh_hell_server.py

Usage (Railway):
    Deployed automatically via Procfile.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import random
import traceback
import os
import uuid
import time
import threading
import logging
import sys
import json

# Configure logging to stdout so Railway captures it
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Import your existing game logic
try:
    from oh_hell_game import OhHellState, ISMCTS
    from CardHelper import CardHelper
    print("✓ Successfully imported OhHellState, CardHelper, and ISMCTS")
except ImportError as e:
    print("=" * 60)
    print("ERROR: Could not import game modules")
    print("=" * 60)
    print(f"\nImport error: {e}")
    print("\nMake sure these files are in the same directory:")
    print("  - oh_hell_game.py")
    print("  - CardHelper.py")
    print("  - oh_hell_server.py")
    print("\nCurrent working directory:", os.getcwd())
    print("=" * 60)
    exit(1)

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['DEBUG'] = False

# ─── Preload tables once at startup ───────────────────────────────────────────
# This avoids intermittent read_csv failures caused by working directory issues
# under gunicorn. All game states share these read-only tables.
import pandas as pd
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))

def _load_csv(relative_path):
    full_path = os.path.join(_script_dir, relative_path)
    return pd.read_csv(full_path)

# ─── Preload tables once at startup ───────────────────────────────────────────
import pandas as pd
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))

def _load_csv(relative_path):
    full_path = os.path.join(_script_dir, relative_path)
    return pd.read_csv(full_path)

# Tables stored here — empty until background thread finishes loading
SIDE_ONE_TRICK_PROBS = {}
TRUMP_ONE_TRICK_PROBS = {}
_prob_table_cache = {}
_tables_ready = False

def _load_all_tables():
    global SIDE_ONE_TRICK_PROBS, TRUMP_ONE_TRICK_PROBS, _prob_table_cache, _tables_ready
    print("Preloading probability tables in background...")
    try:
        _side_one_df = _load_csv("probabilityData/sideOneTrickProbs.csv")
        _trump_one_df = _load_csv("probabilityData/trumpOneTrickProbs.csv")
        SIDE_ONE_TRICK_PROBS = {
            (int(r['num_players']), int(r['player_order']), int(r['rank'])): float(r['probability'])
            for _, r in _side_one_df.iterrows()
        }
        TRUMP_ONE_TRICK_PROBS = {
            (int(r['num_players']), int(r['player_order']), int(r['rank'])): float(r['probability'])
            for _, r in _trump_one_df.iterrows()
        }
        del _side_one_df, _trump_one_df
        print("✓ One-trick probability tables loaded")
    except Exception as e:
        print(f"WARNING: Could not load one-trick prob tables: {e}")

    for _t in range(1, 11):
        for _p in [3, 4]:
            try:
                _df  = _load_csv(f"probabilityData/{_t}Tricks{_p}PSideSuit.csv")
                _df2 = _load_csv(f"probabilityData/{_t}Tricks{_p}PTrumpSuit.csv")
                _side_dict = {
                    (int(r['numMySuit']), int(r['numAtLeastOppSuit'])): float(r['probability'])
                    for _, r in _df.iterrows()
                }
                _trump_dict = {
                    (int(r['numMySuit']), int(r['numAtLeastOppSuit'])): float(r['probability'])
                    for _, r in _df2.iterrows()
                }
                del _df, _df2
                _prob_table_cache[(_t, _p)] = (_side_dict, _trump_dict)
                print(f"✓ Loaded prob tables: {_t} tricks, {_p} players")
            except Exception as e:
                print(f"WARNING: Missing prob tables {_t}T {_p}P: {e}")
                _prob_table_cache[(_t, _p)] = ({}, {})

    _tables_ready = True
    print("✓ All probability tables ready")

# Start loading in background so server can respond to health checks immediately
threading.Thread(target=_load_all_tables, daemon=True).start()

def get_prob_tables_cached(tricks_in_round, num_players):
    return _prob_table_cache.get((tricks_in_round, num_players), ({}, {}))

# Patch OhHellState lookup methods to use plain dict instead of DataFrame filtering
# This eliminates pandas memory spikes during gameplay
def _get_side_prob(self, numMySuits, numOppSuits):
    return self.sideSuitProbs.get((numMySuits, numOppSuits), 0)

def _get_trump_prob(self, numMySuits, numOppSuits):
    return self.trumpSuitProbs.get((numMySuits, numOppSuits), 0)

def _get_trump_one_trick_prob(self, numPlayers, order, rank):
    return self.trumpOneTrickProbs.get((numPlayers, order, rank), 0)

def _get_side_one_trick_prob(self, numPlayers, order, rank):
    return self.sideOneTrickProbs.get((numPlayers, order, rank), 0)

OhHellState.getSideProb = _get_side_prob
OhHellState.getTrumpProb = _get_trump_prob
OhHellState.getTrumpOneTrickProb = _get_trump_one_trick_prob
OhHellState.getSideOneTrickProb = _get_side_one_trick_prob

# ─── Session store ────────────────────────────────────────────────────────────
# { session_id: { ...game data..., 'last_active': timestamp } }
game_instances = {}
SESSION_TTL = 60 * 60 * 2  # 2 hours idle before cleanup

def cleanup_old_sessions():
    """Background thread: remove sessions idle for SESSION_TTL seconds."""
    while True:
        time.sleep(300)  # check every 5 minutes
        now = time.time()
        expired = [sid for sid, g in list(game_instances.items())
                   if now - g.get('last_active', now) > SESSION_TTL]
        for sid in expired:
            del game_instances[sid]
            print(f"[cleanup] Removed expired session {sid}")

threading.Thread(target=cleanup_old_sessions, daemon=True).start()

def touch(game):
    game['last_active'] = time.time()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_session_id():
    """Read session_id from request JSON or query param."""
    if request.method == 'POST' and request.is_json:
        return request.json.get('session_id')
    return request.args.get('session_id')

def create_game_state(num_players=4, tricks_in_round=10, human_player=0):
    dealer = random.randint(0, num_players - 1)
    is_ai = [i != human_player for i in range(num_players)]
    use_heuristic = [True] * num_players
    bid_style = ["normal"] * num_players
    enter_cards = [False] * num_players

    state = OhHellState(
        n=num_players,
        numTricks=tricks_in_round,
        dealer=dealer,
        main=True,
        isAI=is_ai,
        useHeuristic=use_heuristic,
        bidStyle=bid_style,
        enterCards=enter_cards,
        start=True
    )

    # Overwrite tables with preloaded versions to avoid any disk-read races
    state.sideOneTrickProbs = SIDE_ONE_TRICK_PROBS
    state.trumpOneTrickProbs = TRUMP_ONE_TRICK_PROBS
    state.sideSuitProbs, state.trumpSuitProbs = get_prob_tables_cached(tricks_in_round, num_players)

    # For 1-trick rounds, probTables is not initialized in __init__ but ISMCTS needs it
    if not hasattr(state, 'probTables'):
        state.probTables = state.initializeProbTables()

    return state
    return state

def state_to_json(state, human_player=0):
    players_data = []
    for i in range(state.numberOfPlayers):
        player_data = {
            'bid': state.bids[i] if not state.haventBid[i] else None,
            'tricksTaken': state.tricksTaken[i],
            'numCards': CardHelper.get_num_cards(state.playerHands[i])
        }
        if i == human_player:
            player_data['hand'] = CardHelper.to_list(state.playerHands[i])
        else:
            player_data['hand'] = []
        players_data.append(player_data)

    all_bids_placed = not any(state.haventBid)
    phase = 'playing' if all_bids_placed else 'bidding'

    trick_data = [{'player': p, 'card': c} for p, c in state.currentTrick]

    legal_moves = []
    if state.playerToMove == human_player and phase == 'playing':
        legal_moves = CardHelper.to_list(state.GetMoves())

    result = {
        'players': players_data,
        'currentPlayer': state.playerToMove,
        'trumpSuit': state.trumpSuit,
        'trick': trick_data,
        'tricksInRound': state.tricksInRound,
        'phase': phase,
        'legalMoves': legal_moves
    }

    if hasattr(state, '_last_completed_trick'):
        result['lastCompletedTrick'] = state._last_completed_trick
        delattr(state, '_last_completed_trick')

    if hasattr(state, '_last_trick_winner'):
        result['lastTrickWinner'] = state._last_trick_winner
        delattr(state, '_last_trick_winner')

    if hasattr(state, '_last_winning_card'):
        result['lastWinningCard'] = state._last_winning_card
        delattr(state, '_last_winning_card')

    return result

def run_bidding_phase(state, human_player=0, game_id=None, round_num=None):
    messages = []
    while any(state.haventBid):
        current = state.playerToMove
        if not state.haventBid[current]:
            state.playerToMove = state.GetNextPlayer(state.playerToMove)
            continue
        if current == human_player:
            break

        bids_in_order = []
        position = 0
        p = state.GetNextPlayer(state.dealer)
        for i in range(state.numberOfPlayers):
            if not state.haventBid[p]:
                bids_in_order.append(state.bids[p])
            else:
                bids_in_order.append(0)
            if p == current:
                position = i
            p = state.GetNextPlayer(p)

        state.Bid(bids_in_order, position, current)
        if game_id:
            _log_bid_async(game_id, state, current, state.bids[current], round_num)
        messages.append(f"Player {current + 1} bid {state.bids[current]}")
        state.playerToMove = state.GetNextPlayer(state.playerToMove)

    return messages

def run_playing_phase(state, human_player=0, iterations=2500, game_id=None, round_num=None):
    tricks = state.tricksInRound
    messages = []
    if state.GetMoves() == 0:
        return messages

    current = state.playerToMove
    if current == human_player:
        return messages

    if state.currentTrick != []:
        (leader, leadCard) = state.currentTrick[0]
        hand = state.GetMoves()
        cardsInSuit = CardHelper.get_cards_in_suit(
            CardHelper.get_card_suit(leadCard, isHand=False), hand)
    else:
        cardsInSuit = 0

    if CardHelper.get_num_cards(state.GetMoves()) == 1:
        m = CardHelper.hand_to_card(state.GetMoves())
    elif state.GetTricksNeeded(current) < 0:
        m = CardHelper.get_highest_card(state.GetMoves())
    elif state.currentTrick != [] and state.GetTricksNeeded(current) == 0 and cardsInSuit != 0:
        m = CardHelper.get_highest_losing_card(hand, state.currentTrick, state.trumpSuit)
        if m < 0:
            m = ISMCTS(rootstate=state, itermax=iterations, randomRollout=True,
                       mainPlayer=current, verbose=False)
    else:
        m = ISMCTS(rootstate=state, itermax=iterations, randomRollout=True,
                   mainPlayer=current, verbose=False)

    card_str = CardHelper.to_str(m)
    trick_before = list(state.currentTrick)

    # Log AI play to BigQuery before DoMove (hand still intact)
    if game_id:
        _log_play_async(game_id, state, current, m, list(state.currentTrick), round_num)

    trick_before.append((current, m))
    state.DoMove(m)
    messages.append(f"Player {current + 1} played {card_str}")

    if len(state.currentTrick) == 0 and state.GetMoves() != 0:
        state._last_completed_trick = [{'player': p, 'card': c} for p, c in trick_before]
        winner = state.playerToMove
        state._last_trick_winner = winner
        winning_play = next((c for p, c in trick_before if p == winner), None)
        state._last_winning_card = winning_play
        messages.append(f"Player {winner + 1} won the trick!")

    if state.GetMoves() == 0:
        print("=== ROUND COMPLETE ===")

    return messages

def add_round_over(response, game, state):
    """Mutate response dict with round/game-over info if applicable."""
    if state.GetMoves() == 0:
        round_scores = []
        for p in range(state.numberOfPlayers):
            score = state.GetActualScore(p)
            round_scores.append(score)
            game['cumulative_scores'][p] += score

        response['roundOver'] = True
        response['roundScores'] = round_scores
        response['cumulativeScores'] = game['cumulative_scores']

        game['current_round'] += 1
        if game['current_round'] >= len(game['tricks_sequence']):
            game['current_round'] -= 1
            response['gameOver'] = True
            winner = max(range(len(game['cumulative_scores'])),
                         key=lambda i: game['cumulative_scores'][i])
            response['winner'] = winner
            _log_game_result_async(game['game_id'], game['num_players'], game['cumulative_scores'], winner)
        else:
            response['nextRound'] = game['current_round'] + 1

def add_progress(response, game):
    response['gameProgress'] = {
        'currentRound': game['current_round'] + 1,
        'totalRounds': len(game['tricks_sequence']),
        'cumulativeScores': game['cumulative_scores']
    }

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the HTML game interface."""
    try:
        with open('oh_hell_game.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error</h1><p>oh_hell_game.html not found.</p>", 404

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'tables_ready': _tables_ready})

@app.route('/start', methods=['POST'])
def start_game():
    try:
        # Wait up to 30s for tables to finish loading if needed
        if not _tables_ready:
            for _ in range(60):
                if _tables_ready:
                    break
                time.sleep(0.5)
            if not _tables_ready:
                return jsonify({'error': 'Server is still loading, please try again in a moment.'}), 503
        data = request.json
        num_players = data.get('numPlayers', 4)
        human_player = data.get('humanPlayer', 0)

        # Create a fresh session ID for this game
        session_id = str(uuid.uuid4())

        tricks_sequence = list(range(10, 0, -1)) + list(range(2, 11))

        game_instances[session_id] = {
            'current_state': None,
            'num_players': num_players,
            'human_player': human_player,
            'tricks_sequence': tricks_sequence,
            'current_round': 0,
            'cumulative_scores': [0] * num_players,
            'dealer': random.randint(0, num_players - 1),
            'last_active': time.time(),
            'game_id': str(uuid.uuid4()),  # unique ID for analytics
        }

        game = game_instances[session_id]
        tricks_in_round = game['tricks_sequence'][0]
        state = create_game_state(num_players, tricks_in_round, human_player)
        state.dealer = game['dealer']
        game['current_state'] = state

        state.playerToMove = state.GetNextPlayer(state.dealer)
        messages = run_bidding_phase(state, human_player, game['game_id'], 1)

        response = state_to_json(state, human_player)
        response['session_id'] = session_id  # ← send back to client
        response['gameProgress'] = {
            'currentRound': 1,
            'totalRounds': len(tricks_sequence),
            'cumulativeScores': game['cumulative_scores']
        }
        if messages:
            response['messages'] = messages

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/bid', methods=['POST'])
def submit_bid():
    try:
        data = request.json
        session_id = data.get('session_id')
        player = data.get('player', 0)
        bid = data.get('bid')

        if not session_id or session_id not in game_instances:
            return jsonify({'error': 'Session not found. Please start a new game.'}), 404

        game = game_instances[session_id]
        touch(game)
        state = game['current_state']

        if player != state.playerToMove:
            return jsonify({'error': f'Not your turn to bid.'}), 400
        if not state.haventBid[player]:
            return jsonify({'error': 'You have already bid'}), 400

        state.bids[player] = bid
        state.haventBid[player] = False
        state.playerToMove = state.GetNextPlayer(state.playerToMove)

        # Log human bid to BigQuery
        _log_bid_async(game['game_id'], state, player, bid, game['current_round'] + 1)

        messages = [f"You bid {bid}"]
        messages.extend(run_bidding_phase(state, player, game['game_id'], game['current_round'] + 1))

        all_bids_placed = not any(state.haventBid)
        if all_bids_placed:
            if state.tricksInRound != 1:
                state.adjustProbsBids()
            state.playerToMove = state.GetNextPlayer(state.dealer)
            messages.append("Bidding complete! Let's play!")
            messages.extend(run_playing_phase(state, player, game_id=game['game_id'], round_num=game['current_round'] + 1))

        response = state_to_json(state, player)
        response['session_id'] = session_id
        add_progress(response, game)
        if messages:
            response['messages'] = messages

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/play', methods=['POST'])
def play_card():
    try:
        data = request.json
        session_id = data.get('session_id')
        player = data.get('player', 0)
        card = data.get('card')

        if not session_id or session_id not in game_instances:
            return jsonify({'error': 'Session not found. Please start a new game.'}), 404

        game = game_instances[session_id]
        touch(game)
        state = game['current_state']

        if player != state.playerToMove:
            return jsonify({'error': 'Not your turn.'}), 400
        if not CardHelper.has_card(state.GetMoves(), card):
            return jsonify({'error': 'Illegal move.'}), 400

        # Capture trick before the move in case human completes it
        trick_before = list(state.currentTrick)
        trick_before.append((player, card))

        # Log human play to BigQuery
        _log_play_async(game['game_id'], state, player, card, list(state.currentTrick), game['current_round'] + 1)

        state.DoMove(card)
        messages = [f"You played {CardHelper.to_str(card)}"]

        if len(state.currentTrick) == 0 and state.GetMoves() != 0:
            winner = state.playerToMove
            state._last_completed_trick = [{'player': p, 'card': c} for p, c in trick_before]
            state._last_trick_winner = winner
            winning_play = next((c for p, c in trick_before if p == winner), None)
            state._last_winning_card = winning_play
            messages.append(f"Player {winner + 1} won the trick!")

        response = state_to_json(state, player)
        response['session_id'] = session_id
        if messages:
            response['message'] = ' | '.join(messages)

        add_round_over(response, game, state)
        add_progress(response, game)

        return jsonify(response)

    except Exception as e:
        logger.exception("ERROR in play_card")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/advance_ai', methods=['POST'])
def advance_ai():
    try:
        data = request.json
        session_id = data.get('session_id')

        if not session_id or session_id not in game_instances:
            return jsonify({'error': 'Session not found. Please start a new game.'}), 404

        game = game_instances[session_id]
        touch(game)
        state = game['current_state']
        player = game['human_player']

        messages = run_playing_phase(state, player, game_id=game['game_id'], round_num=game['current_round'] + 1)

        response = state_to_json(state, player)
        response['session_id'] = session_id
        if messages:
            response['message'] = ' | '.join(messages)

        add_round_over(response, game, state)
        add_progress(response, game)

        return jsonify(response)

    except Exception as e:
        logger.exception("ERROR in advance_ai")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/next_round', methods=['POST'])
def next_round():
    try:
        data = request.json
        session_id = data.get('session_id')

        if not session_id or session_id not in game_instances:
            return jsonify({'error': 'Session not found. Please start a new game.'}), 404

        game = game_instances[session_id]
        touch(game)

        if game['current_round'] >= len(game['tricks_sequence']):
            return jsonify({'error': 'Game is over'}), 400

        game['dealer'] = (game['dealer'] + 1) % game['num_players']
        tricks_in_round = game['tricks_sequence'][game['current_round']]
        state = create_game_state(game['num_players'], tricks_in_round, game['human_player'])
        state.dealer = game['dealer']
        game['current_state'] = state

        state.playerToMove = state.GetNextPlayer(state.dealer)
        messages = run_bidding_phase(state, game['human_player'], game['game_id'], game['current_round'] + 1)

        response = state_to_json(state, game['human_player'])
        response['session_id'] = session_id
        add_progress(response, game)
        if messages:
            response['messages'] = messages

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/state', methods=['GET'])
def get_state():
    session_id = request.args.get('session_id')
    if not session_id or session_id not in game_instances:
        return jsonify({'error': 'Session not found.'}), 404

    game = game_instances[session_id]
    touch(game)
    state = game['current_state']

    response = state_to_json(state, game['human_player'])
    response['session_id'] = session_id
    add_progress(response, game)

    return jsonify(response)

# ─── Stats storage (Firestore + local fallback) ────────────────────────────────

try:
    from google.cloud import firestore as _firestore
    _db = _firestore.Client(project='project-72eca311-a412-4fda-9be')
    _firestore_available = True
    print("✓ Firestore connected")
except Exception as e:
    print(f"WARNING: Firestore not available, using local JSON fallback: {e}")
    _db = None
    _firestore_available = False

# Local JSON fallback for development
STATS_FILE = os.path.join(_script_dir, 'player_stats.json')
_stats_lock = threading.Lock()

def load_stats_local():
    try:
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_stats_local(stats):
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)

@app.route('/record_game', methods=['POST'])
def record_game():
    """Called when a game completes. Records result for the player."""
    try:
        data = request.json
        username = data.get('username', '').strip() or 'Anonymous'
        user_id = data.get('user_id', '').strip()
        score = data.get('score', 0)
        won = data.get('won', False)

        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400

        if _firestore_available:
            ref = _db.collection('player_stats').document(user_id)
            doc = ref.get()
            if doc.exists:
                entry = doc.to_dict()
            else:
                entry = {'username': username, 'games_played': 0, 'wins': 0, 'total_score': 0}
            entry['username'] = username
            entry['games_played'] += 1
            entry['total_score'] += score
            if won:
                entry['wins'] += 1
            ref.set(entry)
        else:
            with _stats_lock:
                stats = load_stats_local()
                if user_id not in stats:
                    stats[user_id] = {'username': username, 'games_played': 0, 'wins': 0, 'total_score': 0}
                entry = stats[user_id]
                entry['username'] = username
                entry['games_played'] += 1
                entry['total_score'] += score
                if won:
                    entry['wins'] += 1
                save_stats_local(stats)

        return jsonify({'ok': True})
    except Exception as e:
        logger.exception("ERROR in record_game")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Returns stats for a given user_id."""
    try:
        user_id = request.args.get('user_id', '').strip()
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400

        if _firestore_available:
            ref = _db.collection('player_stats').document(user_id)
            doc = ref.get()
            if not doc.exists:
                return jsonify({'games_played': 0, 'wins': 0, 'avg_score': 0, 'total_score': 0})
            entry = doc.to_dict()
        else:
            with _stats_lock:
                stats = load_stats_local()
            entry = stats.get(user_id)
            if not entry:
                return jsonify({'games_played': 0, 'wins': 0, 'avg_score': 0, 'total_score': 0})

        avg = round(entry['total_score'] / entry['games_played'], 1) if entry['games_played'] > 0 else 0
        return jsonify({
            'username': entry.get('username', ''),
            'games_played': entry['games_played'],
            'wins': entry['wins'],
            'total_score': entry['total_score'],
            'avg_score': avg,
        })
    except Exception as e:
        logger.exception("ERROR in get_stats")
        return jsonify({'error': str(e)}), 500

@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    """Returns top players sorted by average score (min 1 game played)."""
    try:
        if _firestore_available:
            docs = _db.collection('player_stats').stream()
            players = [doc.to_dict() | {'user_id': doc.id} for doc in docs]
        else:
            with _stats_lock:
                stats = load_stats_local()
            players = [v | {'user_id': k} for k, v in stats.items()]

        leaderboard = []
        for p in players:
            if p.get('games_played', 0) < 1:
                continue
            avg = round(p['total_score'] / p['games_played'], 1)
            leaderboard.append({
                'username': p.get('username', 'Anonymous'),
                'user_id_snippet': p['user_id'][-6:],
                'games_played': p['games_played'],
                'wins': p['wins'],
                'avg_score': avg,
            })

        leaderboard.sort(key=lambda x: x['avg_score'], reverse=True)
        return jsonify(leaderboard[:50])
    except Exception as e:
        logger.exception("ERROR in get_leaderboard")
        return jsonify({'error': str(e)}), 500

# ─── BigQuery analytics logging ────────────────────────────────────────────────

try:
    from google.cloud import bigquery as _bigquery
    _bq = _bigquery.Client(project='project-72eca311-a412-4fda-9be')
    _bq_plays_table   = 'project-72eca311-a412-4fda-9be.ohhell_analytics.plays'
    _bq_bids_table    = 'project-72eca311-a412-4fda-9be.ohhell_analytics.bids'
    _bq_results_table = 'project-72eca311-a412-4fda-9be.ohhell_analytics.game_results'
    _bigquery_available = True
    print("✓ BigQuery connected")
except Exception as e:
    print(f"WARNING: BigQuery not available: {e}")
    _bq = None
    _bigquery_available = False

def _hand_to_str_list(hand_int):
    """Convert integer hand bitmask to list of card strings e.g. ['5D', 'AH']"""
    return CardHelper.to_list(hand_int) if hand_int else []

def _cards_to_str(card_list):
    """Convert list of card ints to comma-separated string of card names."""
    return ','.join(CardHelper.to_str(c) for c in card_list)

def _trick_to_str(trick):
    """Convert currentTrick list of (player, card) to string e.g. '0:5D,1:AH'"""
    return ','.join(f"{p}:{CardHelper.to_str(c)}" for p, c in trick)

def _log_play_async(game_id, state, player, card, trick_before, round_num):
    """Log a single card play to BigQuery in a background thread."""
    if not _bigquery_available:
        return
    def _do_log():
        try:
            tricks_taken = state.tricksTaken[player] if hasattr(state, 'tricksTaken') and state.tricksTaken else 0
            bid = state.bids[player] if state.bids and len(state.bids) > player else -1
            tricks_left = CardHelper.get_num_cards(state.playerHands[player])
            hand_str = _cards_to_str(CardHelper.to_list(state.playerHands[player]))
            trick_str = _trick_to_str(trick_before)
            row = [{
                'game_id': game_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                'num_players': state.numberOfPlayers,
                'round': round_num,
                'tricks_in_round': state.tricksInRound,
                'trick_num': state.tricksInRound - tricks_left,
                'player': player,
                'card': CardHelper.to_str(card),
                'current_trick': trick_str,
                'tricks_left': tricks_left,
                'player_tricks_won': tricks_taken,
                'player_bid': bid,
                'trump_suit': state.trumpSuit if state.trumpSuit is not None else -1,
                'player_hand': hand_str,
            }]
            errors = _bq.insert_rows_json(_bq_plays_table, row)
            if errors:
                print(f"BigQuery plays insert errors: {errors}")
        except Exception as e:
            print(f"BigQuery play log error: {e}")
    threading.Thread(target=_do_log, daemon=True).start()

def _log_bid_async(game_id, state, player, bid, round_num):
    """Log a single bid to BigQuery in a background thread."""
    if not _bigquery_available:
        return
    def _do_log():
        try:
            # Build previous bids in bidding order
            prev_bids = []
            p = state.GetNextPlayer(state.dealer)
            for _ in range(state.numberOfPlayers):
                if p == player:
                    break
                if not state.haventBid[p]:
                    prev_bids.append(f"{p}:{state.bids[p]}")
                p = state.GetNextPlayer(p)
            hand_str = _cards_to_str(CardHelper.to_list(state.playerHands[player]))
            expected_bid = round(state.tricksInRound / state.numberOfPlayers, 2)
            row = [{
                'game_id': game_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                'num_players': state.numberOfPlayers,
                'round': round_num,
                'tricks_in_round': state.tricksInRound,
                'player': player,
                'bid': bid,
                'previous_bids': ','.join(prev_bids),
                'player_hand': hand_str,
                'expected_bid': expected_bid,
            }]
            errors = _bq.insert_rows_json(_bq_bids_table, row)
            if errors:
                print(f"BigQuery bids insert errors: {errors}")
        except Exception as e:
            print(f"BigQuery bid log error: {e}")
    threading.Thread(target=_do_log, daemon=True).start()

def _log_game_result_async(game_id, num_players, cumulative_scores, winner):
    """Log final game result to BigQuery in a background thread."""
    if not _bigquery_available:
        return
    def _do_log():
        try:
            # Always log 4 player score columns; N/A if player doesn't exist
            scores = [cumulative_scores[i] if i < num_players else None for i in range(4)]
            row = [{
                'game_id': game_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                'num_players': num_players,
                'player_0_score': scores[0],
                'player_1_score': scores[1],
                'player_2_score': scores[2],
                'player_3_score': scores[3],
                'winning_player': winner,
            }]
            errors = _bq.insert_rows_json(_bq_results_table, row)
            if errors:
                print(f"BigQuery game_results insert errors: {errors}")
        except Exception as e:
            print(f"BigQuery game result log error: {e}")
    threading.Thread(target=_do_log, daemon=True).start()

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"✓ Oh Hell server starting on port {port}")
    app.run(debug=False, port=port, host='0.0.0.0')
