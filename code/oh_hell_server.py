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

print("Preloading probability tables...")
try:
    _side_one_df = _load_csv("probabilityData/sideOneTrickProbs.csv")
    _trump_one_df = _load_csv("probabilityData/trumpOneTrickProbs.csv")
    # Convert to dict keyed by (num_players, player_order, rank) for O(1) lookup
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
    SIDE_ONE_TRICK_PROBS = {}
    TRUMP_ONE_TRICK_PROBS = {}

# Eagerly preload all round/player combinations used in a full game
_prob_table_cache = {}
_all_trick_counts = list(range(1, 11))
_all_player_counts = [3, 4]

for _t in _all_trick_counts:
    for _p in _all_player_counts:
        try:
            _df  = _load_csv(f"probabilityData/{_t}Tricks{_p}PSideSuit.csv")
            _df2 = _load_csv(f"probabilityData/{_t}Tricks{_p}PTrumpSuit.csv")
            # Convert to dict keyed by (numMySuit, numAtLeastOppSuit)
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

def get_prob_tables_cached(tricks_in_round, num_players):
    return _prob_table_cache.get((tricks_in_round, num_players), ({}, {}))

print("✓ All probability tables preloaded as dicts — pandas no longer needed at runtime")

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

def run_bidding_phase(state, human_player=0):
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
        messages.append(f"Player {current + 1} bid {state.bids[current]}")
        state.playerToMove = state.GetNextPlayer(state.playerToMove)

    return messages

def run_playing_phase(state, human_player=0, iterations=2500):
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

@app.route('/start', methods=['POST'])
def start_game():
    try:
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
            'last_active': time.time()
        }

        game = game_instances[session_id]
        tricks_in_round = game['tricks_sequence'][0]
        state = create_game_state(num_players, tricks_in_round, human_player)
        state.dealer = game['dealer']
        game['current_state'] = state

        state.playerToMove = state.GetNextPlayer(state.dealer)
        messages = run_bidding_phase(state, human_player)

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

        messages = [f"You bid {bid}"]
        messages.extend(run_bidding_phase(state, player))

        all_bids_placed = not any(state.haventBid)
        if all_bids_placed:
            if state.tricksInRound != 1:
                state.adjustProbsBids()
            state.playerToMove = state.GetNextPlayer(state.dealer)
            messages.append("Bidding complete! Let's play!")
            messages.extend(run_playing_phase(state, player))

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

        messages = run_playing_phase(state, player)

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
        messages = run_bidding_phase(state, game['human_player'])

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

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"✓ Oh Hell server starting on port {port}")
    app.run(debug=False, port=port, host='0.0.0.0')
