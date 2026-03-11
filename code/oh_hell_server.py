"""
oh_hell_server.py  —  Flask API server for Oh Hell card game.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import random, traceback, os, uuid, time, threading, logging, sys, json, pickle, base64
from concurrent.futures import ThreadPoolExecutor

_ai_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix='ai_worker')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    from oh_hell_game import OhHellState, ISMCTS
    from CardHelper import CardHelper
    print("✓ Imported OhHellState, CardHelper, ISMCTS")
except ImportError as e:
    print(f"ERROR: Could not import game modules: {e}"); exit(1)

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": [
    "https://ohhellai.com", "https://www.ohhellai.com",
    "http://localhost:5001", "http://127.0.0.1:5001",
]}})
app.config['DEBUG'] = False

# ── Probability table preload ─────────────────────────────────────────────────
import pandas as pd
_script_dir = os.path.dirname(os.path.abspath(__file__))

def _load_csv(rel):
    return pd.read_csv(os.path.join(_script_dir, rel))

SIDE_ONE_TRICK_PROBS = {}
TRUMP_ONE_TRICK_PROBS = {}
_prob_table_cache = {}
_tables_ready = False

def _load_all_tables():
    global SIDE_ONE_TRICK_PROBS, TRUMP_ONE_TRICK_PROBS, _prob_table_cache, _tables_ready
    print("Preloading probability tables...")
    try:
        s = _load_csv("probabilityData/sideOneTrickProbs.csv")
        t = _load_csv("probabilityData/trumpOneTrickProbs.csv")
        SIDE_ONE_TRICK_PROBS  = {(int(r['num_players']),int(r['player_order']),int(r['rank'])):float(r['probability']) for _,r in s.iterrows()}
        TRUMP_ONE_TRICK_PROBS = {(int(r['num_players']),int(r['player_order']),int(r['rank'])):float(r['probability']) for _,r in t.iterrows()}
        del s, t
    except Exception as e:
        print(f"WARNING one-trick probs: {e}")
    for _t in range(1,11):
        for _p in [3,4]:
            try:
                d1 = _load_csv(f"probabilityData/{_t}Tricks{_p}PSideSuit.csv")
                d2 = _load_csv(f"probabilityData/{_t}Tricks{_p}PTrumpSuit.csv")
                _prob_table_cache[(_t,_p)] = (
                    {(int(r['numMySuit']),int(r['numAtLeastOppSuit'])):float(r['probability']) for _,r in d1.iterrows()},
                    {(int(r['numMySuit']),int(r['numAtLeastOppSuit'])):float(r['probability']) for _,r in d2.iterrows()},
                )
                del d1, d2
            except Exception as e:
                print(f"WARNING {_t}T {_p}P: {e}")
                _prob_table_cache[(_t,_p)] = ({},{})
    _tables_ready = True
    print("✓ All probability tables ready")

threading.Thread(target=_load_all_tables, daemon=True).start()

def get_prob_tables_cached(t, p): return _prob_table_cache.get((t,p),({},{}))

OhHellState.getSideProb         = lambda self,a,b: self.sideSuitProbs.get((a,b),0)
OhHellState.getTrumpProb        = lambda self,a,b: self.trumpSuitProbs.get((a,b),0)
OhHellState.getTrumpOneTrickProb= lambda self,n,o,r: self.trumpOneTrickProbs.get((n,o,r),0)
OhHellState.getSideOneTrickProb = lambda self,n,o,r: self.sideOneTrickProbs.get((n,o,r),0)

# ── Session store (Firestore-backed, defined after Firestore init below) ──────
SESSION_TTL = 7200
_sessions_collection = 'game_sessions'

# ── Game helpers ──────────────────────────────────────────────────────────────
def create_game_state(num_players=4, tricks_in_round=10, human_player=0):
    is_ai = [i != human_player for i in range(num_players)]
    state = OhHellState(n=num_players, numTricks=tricks_in_round,
                        dealer=random.randint(0,num_players-1), main=True,
                        isAI=is_ai, useHeuristic=[True]*num_players,
                        bidStyle=["normal"]*num_players, enterCards=[False]*num_players, start=True)
    state.sideOneTrickProbs  = SIDE_ONE_TRICK_PROBS
    state.trumpOneTrickProbs = TRUMP_ONE_TRICK_PROBS
    state.sideSuitProbs, state.trumpSuitProbs = get_prob_tables_cached(tricks_in_round, num_players)
    if not hasattr(state,'probTables'): state.probTables = state.initializeProbTables()
    return state

def state_to_json(state, human_player=0):
    players_data = []
    for i in range(state.numberOfPlayers):
        pd_ = {'bid': state.bids[i] if not state.haventBid[i] else None,
               'tricksTaken': state.tricksTaken[i],
               'numCards': CardHelper.get_num_cards(state.playerHands[i])}
        pd_['hand'] = CardHelper.to_list(state.playerHands[i]) if i==human_player else []
        players_data.append(pd_)
    phase = 'playing' if not any(state.haventBid) else 'bidding'
    legal_moves = CardHelper.to_list(state.GetMoves()) if state.playerToMove==human_player and phase=='playing' else []
    result = {'players': players_data, 'currentPlayer': state.playerToMove,
              'trumpSuit': state.trumpSuit, 'trick': [{'player':p,'card':c} for p,c in state.currentTrick],
              'tricksInRound': state.tricksInRound, 'phase': phase, 'legalMoves': legal_moves}
    for attr in ('_last_completed_trick','_last_trick_winner','_last_winning_card'):
        key = attr.lstrip('_')
        key = {'last_completed_trick':'lastCompletedTrick','last_trick_winner':'lastTrickWinner','last_winning_card':'lastWinningCard'}[key]
        if hasattr(state, attr): result[key] = getattr(state,attr); delattr(state,attr)
    return result

def run_bidding_phase(state, human_player=0, game_id=None, round_num=None, username='Anonymous'):
    messages = []
    while any(state.haventBid):
        current = state.playerToMove
        if not state.haventBid[current]:
            state.playerToMove = state.GetNextPlayer(state.playerToMove); continue
        if current == human_player: break
        bids_in_order, position = [], 0
        p = state.GetNextPlayer(state.dealer)
        for i in range(state.numberOfPlayers):
            bids_in_order.append(state.bids[p] if not state.haventBid[p] else 0)
            if p == current: position = i
            p = state.GetNextPlayer(p)
        state.Bid(bids_in_order, position, current)
        if game_id: _log_bid_async(game_id, state, current, state.bids[current], round_num, username)
        messages.append(f"Player {current+1} bid {state.bids[current]}")
        state.playerToMove = state.GetNextPlayer(state.playerToMove)
    return messages

def run_playing_phase(state, human_player=0, iterations=2500, game_id=None, round_num=None, username='Anonymous'):
    messages = []
    if state.GetMoves()==0 or state.playerToMove==human_player: return messages
    current = state.playerToMove
    cardsInSuit = 0
    if state.currentTrick:
        _,leadCard = state.currentTrick[0]
        hand = state.GetMoves()
        cardsInSuit = CardHelper.get_cards_in_suit(CardHelper.get_card_suit(leadCard,isHand=False), hand)
    if CardHelper.get_num_cards(state.GetMoves())==1:
        m = CardHelper.hand_to_card(state.GetMoves())
    elif state.GetTricksNeeded(current)<0:
        m = CardHelper.get_highest_card(state.GetMoves())
    elif state.currentTrick and state.GetTricksNeeded(current)==0 and cardsInSuit!=0:
        m = CardHelper.get_highest_losing_card(hand, state.currentTrick, state.trumpSuit)
        if m<0: m = ISMCTS(rootstate=state,itermax=iterations,randomRollout=True,mainPlayer=current,verbose=False)
    else:
        m = ISMCTS(rootstate=state,itermax=iterations,randomRollout=True,mainPlayer=current,verbose=False)
    trick_before = list(state.currentTrick)
    if game_id: _log_play_async(game_id, state, current, m, list(state.currentTrick), round_num, username)
    trick_before.append((current,m))
    state.DoMove(m)
    messages.append(f"Player {current+1} played {CardHelper.to_str(m)}")
    if len(state.currentTrick)==0 and state.GetMoves()!=0:
        state._last_completed_trick = [{'player':p,'card':c} for p,c in trick_before]
        winner = state.playerToMove
        state._last_trick_winner = winner
        state._last_winning_card = next((c for p,c in trick_before if p==winner), None)
        messages.append(f"Player {winner+1} won the trick!")
    if state.GetMoves()==0: print("=== ROUND COMPLETE ===")
    return messages

def add_round_over(response, game, state):
    if state.GetMoves()==0:
        round_scores = []
        for p in range(state.numberOfPlayers):
            s = state.GetActualScore(p); round_scores.append(s); game['cumulative_scores'][p] += s
        response.update({'roundOver':True,'roundScores':round_scores,'cumulativeScores':game['cumulative_scores']})
        game['current_round'] += 1
        if game['current_round'] >= len(game['tricks_sequence']):
            game['current_round'] -= 1
            winner = max(range(len(game['cumulative_scores'])), key=lambda i: game['cumulative_scores'][i])
            response.update({'gameOver':True,'winner':winner})
            _log_game_result_async(game['game_id'], game['num_players'], game['cumulative_scores'], winner, game.get('username','Anonymous'))
        else:
            response['nextRound'] = game['current_round']+1

def add_progress(response, game):
    response['gameProgress'] = {'currentRound':game['current_round']+1,
                                'totalRounds':len(game['tricks_sequence']),
                                'cumulativeScores':game['cumulative_scores']}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    try:
        with open('oh_hell_game.html','r') as f: return f.read()
    except FileNotFoundError: return "<h1>Error</h1><p>oh_hell_game.html not found.</p>", 404

@app.route('/health')
def health(): return jsonify({'status':'ok','tables_ready':_tables_ready})

@app.route('/start', methods=['POST'])
def start_game():
    try:
        if not _tables_ready:
            for _ in range(60):
                if _tables_ready: break
                time.sleep(0.5)
            if not _tables_ready: return jsonify({'error':'Server still loading'}), 503
        data = request.json
        num_players  = data.get('numPlayers', 4)
        human_player = data.get('humanPlayer', 0)
        username     = data.get('username', 'Anonymous').strip() or 'Anonymous'
        session_id   = str(uuid.uuid4())
        tricks_seq   = list(range(10,0,-1)) + list(range(2,11))
        game = {
            'current_state': None, 'num_players': num_players, 'human_player': human_player,
            'username': username, 'tricks_sequence': tricks_seq, 'current_round': 0,
            'cumulative_scores': [0]*num_players, 'dealer': random.randint(0,num_players-1),
            'last_active': time.time(), 'game_id': str(uuid.uuid4()),
        }
        state = create_game_state(num_players, game['tricks_sequence'][0], human_player)
        state.dealer = game['dealer']; game['current_state'] = state
        state.playerToMove = state.GetNextPlayer(state.dealer)
        messages = run_bidding_phase(state, human_player, game['game_id'], 1, username)
        response = state_to_json(state, human_player)
        save_game(session_id, game)
        response['session_id'] = session_id
        response['gameProgress'] = {'currentRound':1,'totalRounds':len(tricks_seq),'cumulativeScores':game['cumulative_scores']}
        if messages: response['messages'] = messages
        return jsonify(response)
    except Exception as e:
        traceback.print_exc(); return jsonify({'error':str(e)}), 500

@app.route('/bid', methods=['POST'])
def submit_bid():
    try:
        data = request.json
        session_id = data.get('session_id'); player = data.get('player',0); bid = data.get('bid')
        if not session_id or not (game := load_game(session_id)):
            return jsonify({'error':'Session not found. Please start a new game.'}), 404
        state = game['current_state']; username = game.get('username','Anonymous')
        if player != state.playerToMove: return jsonify({'error':'Not your turn to bid.'}), 400
        if not state.haventBid[player]:  return jsonify({'error':'You have already bid'}), 400
        state.bids[player]=bid; state.haventBid[player]=False
        state.playerToMove = state.GetNextPlayer(state.playerToMove)
        _log_bid_async(game['game_id'], state, player, bid, game['current_round']+1, username)
        messages = [f"You bid {bid}"]
        messages.extend(run_bidding_phase(state, player, game['game_id'], game['current_round']+1, username))
        if not any(state.haventBid):
            if state.tricksInRound!=1: state.adjustProbsBids()
            state.playerToMove = state.GetNextPlayer(state.dealer)
            messages.append("Bidding complete! Let's play!")
            messages.extend(run_playing_phase(state, player, game_id=game['game_id'], round_num=game['current_round']+1, username=username))
        response = state_to_json(state, player); response['session_id']=session_id
        save_game(session_id, game)
        add_progress(response, game)
        if messages: response['messages'] = messages
        return jsonify(response)
    except Exception as e:
        traceback.print_exc(); return jsonify({'error':str(e)}), 500

@app.route('/play', methods=['POST'])
def play_card():
    try:
        data = request.json
        session_id = data.get('session_id'); player = data.get('player',0); card = data.get('card')
        if not session_id or not (game := load_game(session_id)):
            return jsonify({'error':'Session not found. Please start a new game.'}), 404
        state = game['current_state']; username = game.get('username','Anonymous')
        if player != state.playerToMove: return jsonify({'error':'Not your turn.'}), 400
        if not CardHelper.has_card(state.GetMoves(), card): return jsonify({'error':'Illegal move.'}), 400
        trick_before = list(state.currentTrick); trick_before.append((player,card))
        _log_play_async(game['game_id'], state, player, card, list(state.currentTrick), game['current_round']+1, username)
        state.DoMove(card)
        messages = [f"You played {CardHelper.to_str(card)}"]
        if len(state.currentTrick)==0 and state.GetMoves()!=0:
            winner = state.playerToMove
            state._last_completed_trick = [{'player':p,'card':c} for p,c in trick_before]
            state._last_trick_winner = winner
            state._last_winning_card = next((c for p,c in trick_before if p==winner), None)
            messages.append(f"Player {winner+1} won the trick!")
        response = state_to_json(state, player); response['session_id']=session_id
        save_game(session_id, game)
        if messages: response['message']=' | '.join(messages)
        add_round_over(response, game, state); add_progress(response, game)
        return jsonify(response)
    except Exception as e:
        logger.exception("ERROR in play_card"); return jsonify({'error':str(e)}), 500

def _do_ai_turn(session_id):
    """Run one AI move in the thread pool, save result back to Firestore."""
    try:
        game = load_game(session_id)
        if not game: return
        state = game['current_state']; player = game['human_player']
        username = game.get('username','Anonymous')
        messages = run_playing_phase(state, player, game_id=game['game_id'], round_num=game['current_round']+1, username=username)
        response = state_to_json(state, player); response['session_id']=session_id
        add_round_over(response, game, state); add_progress(response, game)
        if messages: response['message']=' | '.join(messages)
        game['_ai_result'] = response
        game['_ai_thinking'] = False
        save_game(session_id, game)
    except Exception as e:
        logger.exception("ERROR in _do_ai_turn")
        try:
            game = load_game(session_id)
            if game:
                game['_ai_thinking'] = False
                game['_ai_error'] = str(e)
                save_game(session_id, game)
        except Exception: pass

@app.route('/advance_ai', methods=['POST'])
def advance_ai():
    try:
        data = request.json; session_id = data.get('session_id')
        if not session_id or not (game := load_game(session_id)):
            return jsonify({'error':'Session not found. Please start a new game.'}), 404
        # If already thinking, just tell client to keep polling
        if game.get('_ai_thinking'):
            return jsonify({'status':'thinking','session_id':session_id})
        # Clear any previous result, mark as thinking, submit to thread pool
        game['_ai_thinking'] = True
        game['_ai_result'] = None
        game.pop('_ai_error', None)
        save_game(session_id, game)
        _ai_executor.submit(_do_ai_turn, session_id)
        return jsonify({'status':'thinking','session_id':session_id})
    except Exception as e:
        logger.exception("ERROR in advance_ai"); return jsonify({'error':str(e)}), 500

@app.route('/ai_result', methods=['POST'])
def ai_result():
    try:
        data = request.json; session_id = data.get('session_id')
        if not session_id or not (game := load_game(session_id)):
            return jsonify({'error':'Session not found. Please start a new game.'}), 404
        if game.get('_ai_error'):
            err = game['_ai_error']
            game.pop('_ai_error', None); game['_ai_thinking'] = False
            save_game(session_id, game)
            return jsonify({'error': err}), 500
        if game.get('_ai_thinking') or not game.get('_ai_result'):
            return jsonify({'status':'thinking','session_id':session_id})
        # Result is ready — return it and clear
        response = game['_ai_result']
        game['_ai_result'] = None; game['_ai_thinking'] = False
        save_game(session_id, game)
        return jsonify(response)
    except Exception as e:
        logger.exception("ERROR in ai_result"); return jsonify({'error':str(e)}), 500

@app.route('/next_round', methods=['POST'])
def next_round():
    try:
        data = request.json; session_id = data.get('session_id')
        if not session_id or not (game := load_game(session_id)):
            return jsonify({'error':'Session not found. Please start a new game.'}), 404
        username = game.get('username','Anonymous')
        if game['current_round'] >= len(game['tricks_sequence']): return jsonify({'error':'Game is over'}), 400
        game['dealer'] = (game['dealer']+1) % game['num_players']
        state = create_game_state(game['num_players'], game['tricks_sequence'][game['current_round']], game['human_player'])
        state.dealer = game['dealer']; game['current_state'] = state
        state.playerToMove = state.GetNextPlayer(state.dealer)
        messages = run_bidding_phase(state, game['human_player'], game['game_id'], game['current_round']+1, username)
        response = state_to_json(state, game['human_player']); response['session_id']=session_id
        save_game(session_id, game)
        add_progress(response, game)
        if messages: response['messages'] = messages
        return jsonify(response)
    except Exception as e:
        traceback.print_exc(); return jsonify({'error':str(e)}), 500

@app.route('/state', methods=['GET'])
def get_state():
    session_id = request.args.get('session_id')
    if not session_id or not (game := load_game(session_id)): return jsonify({'error':'Session not found.'}), 404
    state = game['current_state']
    response = state_to_json(state, game['human_player']); response['session_id']=session_id
    add_progress(response, game); return jsonify(response)

# ── Auth + Stats ──────────────────────────────────────────────────────────────
try:
    from google.cloud import firestore as _firestore
    _db = _firestore.Client(project='project-72eca311-a412-4fda-9be')
    _firestore_available = True; print("✓ Firestore connected")
except Exception as e:
    print(f"WARNING Firestore: {e}"); _db = None; _firestore_available = False

try:
    import bcrypt; _bcrypt_available = True; print("✓ bcrypt available")
except ImportError:
    _bcrypt_available = False; print("WARNING: bcrypt not available")

def _hash_pw(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode() if _bcrypt_available else pw

def _check_pw(pw, hashed):
    return bcrypt.checkpw(pw.encode(), hashed.encode()) if _bcrypt_available else pw==hashed

STATS_FILE    = os.path.join(_script_dir, 'player_stats.json')
ACCOUNTS_FILE = os.path.join(_script_dir, 'accounts.json')
_stats_lock   = threading.Lock()

def _load_json(path):
    try:
        with open(path,'r') as f: return json.load(f)
    except: return {}

def _save_json(path, data):
    with open(path,'w') as f: json.dump(data, f)

# ── Session store helpers ─────────────────────────────────────────────────────
def _state_to_b64(state):
    """Pickle the OhHellState object and base64-encode it for Firestore storage."""
    return base64.b64encode(pickle.dumps(state)).decode('utf-8')

def _b64_to_state(b64):
    """Decode and unpickle an OhHellState object from Firestore."""
    return pickle.loads(base64.b64decode(b64.encode('utf-8')))

def load_game(session_id):
    """Load a game session from Firestore (or local JSON fallback).
    Returns the game dict with 'current_state' as a live OhHellState, or None."""
    if _firestore_available:
        doc = _db.collection(_sessions_collection).document(session_id).get()
        if not doc.exists: return None
        data = doc.to_dict()
        data['current_state'] = _b64_to_state(data['state_b64'])
        return data
    else:
        sessions = _load_json('game_sessions.json')
        data = sessions.get(session_id)
        if not data: return None
        data['current_state'] = _b64_to_state(data['state_b64'])
        return data

def save_game(session_id, game):
    """Serialize and save a game session to Firestore (or local JSON fallback)."""
    data = {k: v for k, v in game.items() if k != 'current_state'}
    data['state_b64'] = _state_to_b64(game['current_state'])
    data['last_active'] = time.time()
    if _firestore_available:
        _db.collection(_sessions_collection).document(session_id).set(data)
    else:
        with _stats_lock:
            sessions = _load_json('game_sessions.json')
            sessions[session_id] = data
            _save_json('game_sessions.json', sessions)

def _cleanup_old_sessions():
    """Background thread: delete sessions older than SESSION_TTL."""
    while True:
        time.sleep(600)
        try:
            cutoff = time.time() - SESSION_TTL
            if _firestore_available:
                old = _db.collection(_sessions_collection).where('last_active', '<', cutoff).stream()
                for doc in old: doc.reference.delete()
            else:
                with _stats_lock:
                    sessions = _load_json('game_sessions.json')
                    to_del = [s for s,g in sessions.items() if g.get('last_active',0) < cutoff]
                    for s in to_del: del sessions[s]
                    if to_del: _save_json('game_sessions.json', sessions)
        except Exception as e:
            logger.warning(f"Session cleanup error: {e}")

threading.Thread(target=_cleanup_old_sessions, daemon=True).start()

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        username = data.get('username','').strip()
        password = data.get('password','').strip()
        if not username or not password:
            return jsonify({'error':'Username and password are required'}), 400
        if len(username)<2 or len(username)>20:
            return jsonify({'error':'Username must be 2–20 characters'}), 400
        if len(password)<3:
            return jsonify({'error':'Password must be at least 3 characters'}), 400
        key = username.lower()
        if _firestore_available:
            if _db.collection('accounts').document(key).get().exists:
                return jsonify({'error':'Username already taken'}), 409
            _db.collection('accounts').document(key).set({
                'username': username, 'username_lower': key,
                'password_hash': _hash_pw(password),
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            })
            _db.collection('player_stats').document(key).set({
                'username': username, 'games_played':0, 'wins':0,
                'total_score':0, 'total_bid_delta':0, 'total_bid_count':0,
            })
        else:
            with _stats_lock:
                accounts = _load_json(ACCOUNTS_FILE)
                if key in accounts: return jsonify({'error':'Username already taken'}), 409
                accounts[key] = {'username': username, 'password_hash': _hash_pw(password)}
                _save_json(ACCOUNTS_FILE, accounts)
                stats = _load_json(STATS_FILE)
                stats[key] = {'username':username,'games_played':0,'wins':0,'total_score':0,'total_bid_delta':0,'total_bid_count':0}
                _save_json(STATS_FILE, stats)
        return jsonify({'ok':True,'username':username})
    except Exception as e:
        logger.exception("ERROR in register"); return jsonify({'error':str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username','').strip()
        password = data.get('password','').strip()
        if not username or not password:
            return jsonify({'error':'Username and password are required'}), 400
        key = username.lower()
        if _firestore_available:
            doc = _db.collection('accounts').document(key).get()
            if not doc.exists: return jsonify({'error':'Invalid username or password'}), 401
            account = doc.to_dict()
            if not _check_pw(password, account['password_hash']): return jsonify({'error':'Invalid username or password'}), 401
            display_name = account['username']
        else:
            with _stats_lock: accounts = _load_json(ACCOUNTS_FILE)
            account = accounts.get(key)
            if not account or not _check_pw(password, account['password_hash']):
                return jsonify({'error':'Invalid username or password'}), 401
            display_name = account['username']
        return jsonify({'ok':True,'username':display_name})
    except Exception as e:
        logger.exception("ERROR in login"); return jsonify({'error':str(e)}), 500

@app.route('/record_bid_delta', methods=['POST'])
def record_bid_delta():
    try:
        data = request.json
        username   = data.get('username', '').strip() or 'Anonymous'
        bid_delta  = data.get('bid_delta', 0)
        key = username.lower()
        if _firestore_available:
            ref = _db.collection('player_stats').document(key)
            doc = ref.get()
            entry = doc.to_dict() if doc.exists else {
                'username':username,'games_played':0,'wins':0,
                'total_score':0,'total_bid_delta':0,'total_bid_count':0
            }
            entry['total_bid_delta'] = entry.get('total_bid_delta', 0) + bid_delta
            entry['total_bid_count'] = entry.get('total_bid_count', 0) + 1
            ref.set(entry)
        else:
            with _stats_lock:
                stats = _load_json(STATS_FILE)
                if key not in stats:
                    stats[key] = {'username':username,'games_played':0,'wins':0,'total_score':0,'total_bid_delta':0,'total_bid_count':0}
                stats[key]['total_bid_delta'] = stats[key].get('total_bid_delta', 0) + bid_delta
                stats[key]['total_bid_count'] = stats[key].get('total_bid_count', 0) + 1
                _save_json(STATS_FILE, stats)
        return jsonify({'ok': True})
    except Exception as e:
        logger.exception("ERROR in record_bid_delta"); return jsonify({'error': str(e)}), 500

@app.route('/record_game', methods=['POST'])
def record_game():
    try:
        data = request.json
        username   = data.get('username','').strip() or 'Anonymous'
        score      = data.get('score', 0)
        won        = data.get('won', False)
        total_bid_delta = 0  # tracked live via /record_bid_delta
        bid_count  = 0
        key = username.lower()
        empty = {'username':username,'games_played':0,'wins':0,'total_score':0,'total_bid_delta':0,'total_bid_count':0}
        if _firestore_available:
            ref = _db.collection('player_stats').document(key)
            doc = ref.get(); entry = doc.to_dict() if doc.exists else empty.copy()
            entry['username'] = username; entry['games_played']+=1; entry['total_score']+=score
            entry['total_bid_delta'] = entry.get('total_bid_delta',0)+total_bid_delta
            entry['total_bid_count'] = entry.get('total_bid_count',0)+bid_count
            if won: entry['wins']+=1
            ref.set(entry)
        else:
            with _stats_lock:
                stats = _load_json(STATS_FILE)
                if key not in stats: stats[key] = empty.copy()
                e = stats[key]; e['username']=username; e['games_played']+=1; e['total_score']+=score
                e['total_bid_delta']=e.get('total_bid_delta',0)+total_bid_delta; e['total_bid_count']=e.get('total_bid_count',0)+bid_count
                if won: e['wins']+=1
                _save_json(STATS_FILE, stats)
        return jsonify({'ok':True})
    except Exception as e:
        logger.exception("ERROR in record_game"); return jsonify({'error':str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        username = request.args.get('username','').strip()
        if not username: return jsonify({'error':'Missing username'}), 400
        key = username.lower()
        if _firestore_available:
            doc = _db.collection('player_stats').document(key).get()
            if not doc.exists: return jsonify({'games_played':0,'wins':0,'avg_score':0,'total_score':0,'avg_bid_delta':0})
            entry = doc.to_dict()
        else:
            with _stats_lock: stats = _load_json(STATS_FILE)
            entry = stats.get(key)
            if not entry: return jsonify({'games_played':0,'wins':0,'avg_score':0,'total_score':0,'avg_bid_delta':0})
        gp = entry['games_played']
        avg_score = round(entry['total_score']/gp, 1) if gp>0 else 0
        bc = entry.get('total_bid_count',0)
        avg_bid_delta = round(entry.get('total_bid_delta',0)/bc, 2) if bc>0 else 0
        return jsonify({'username':entry.get('username',username),'games_played':gp,'wins':entry['wins'],
                        'total_score':entry['total_score'],'avg_score':avg_score,'avg_bid_delta':avg_bid_delta})
    except Exception as e:
        logger.exception("ERROR in get_stats"); return jsonify({'error':str(e)}), 500

@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    try:
        if _firestore_available:
            players = [d.to_dict() for d in _db.collection('player_stats').stream()]
        else:
            with _stats_lock: stats = _load_json(STATS_FILE)
            players = list(stats.values())
        board = []
        for p in players:
            gp = p.get('games_played',0)
            if gp<1: continue
            bc = p.get('total_bid_count', 0)
            avg_bid_delta = round(p.get('total_bid_delta',0)/bc, 2) if bc>0 else None
            board.append({'username':p.get('username','Anonymous'),'games_played':gp,
                          'wins':p['wins'],'avg_score':round(p['total_score']/gp,1),
                          'avg_bid_delta':avg_bid_delta})
        board.sort(key=lambda x: x['avg_score'], reverse=True)
        return jsonify(board[:50])
    except Exception as e:
        logger.exception("ERROR in get_leaderboard"); return jsonify({'error':str(e)}), 500

# ── BigQuery analytics ────────────────────────────────────────────────────────
try:
    from google.cloud import bigquery as _bigquery
    _bq = _bigquery.Client(project='project-72eca311-a412-4fda-9be')
    _bq_plays   = 'project-72eca311-a412-4fda-9be.ohhell_analytics.plays'
    _bq_bids    = 'project-72eca311-a412-4fda-9be.ohhell_analytics.bids'
    _bq_results = 'project-72eca311-a412-4fda-9be.ohhell_analytics.game_results'
    _bigquery_available = True; print("✓ BigQuery connected")
except Exception as e:
    print(f"WARNING BigQuery: {e}"); _bq=None; _bigquery_available=False

def _c2s(cards): return ','.join(CardHelper.to_str(c) for c in cards)
def _t2s(trick): return ','.join(f"{p}:{CardHelper.to_str(c)}" for p,c in trick)

def _log_play_async(game_id, state, player, card, trick_before, round_num, username='Anonymous'):
    if not _bigquery_available: return
    def _do():
        try:
            tricks_taken = state.tricksTaken[player] if hasattr(state,'tricksTaken') and state.tricksTaken else 0
            bid = state.bids[player] if state.bids and len(state.bids)>player else -1
            tricks_left = CardHelper.get_num_cards(state.playerHands[player])
            row = [{'game_id':game_id,'timestamp':time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),
                    'num_players':state.numberOfPlayers,'round':round_num,'tricks_in_round':state.tricksInRound,
                    'trick_num':state.tricksInRound-tricks_left,'player':player,'card':CardHelper.to_str(card),
                    'current_trick':_t2s(trick_before),'tricks_left':tricks_left,'player_tricks_won':tricks_taken,
                    'player_bid':bid,'trump_suit':state.trumpSuit if state.trumpSuit is not None else -1,
                    'player_hand':_c2s(CardHelper.to_list(state.playerHands[player])),'human_player_name':username}]
            errs = _bq.insert_rows_json(_bq_plays, row)
            if errs: print(f"BQ plays errors: {errs}")
        except Exception as e: print(f"BQ play error: {e}")
    threading.Thread(target=_do, daemon=True).start()

def _log_bid_async(game_id, state, player, bid, round_num, username='Anonymous'):
    if not _bigquery_available: return
    def _do():
        try:
            prev_bids=[]; p=state.GetNextPlayer(state.dealer)
            for _ in range(state.numberOfPlayers):
                if p==player: break
                if not state.haventBid[p]: prev_bids.append(f"{p}:{state.bids[p]}")
                p=state.GetNextPlayer(p)
            row = [{'game_id':game_id,'timestamp':time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),
                    'num_players':state.numberOfPlayers,'round':round_num,'tricks_in_round':state.tricksInRound,
                    'player':player,'bid':bid,'previous_bids':','.join(prev_bids),
                    'player_hand':_c2s(CardHelper.to_list(state.playerHands[player])),
                    'expected_bid':round(state.tricksInRound/state.numberOfPlayers,2),'human_player_name':username}]
            errs = _bq.insert_rows_json(_bq_bids, row)
            if errs: print(f"BQ bids errors: {errs}")
        except Exception as e: print(f"BQ bid error: {e}")
    threading.Thread(target=_do, daemon=True).start()

def _log_game_result_async(game_id, num_players, cumulative_scores, winner, username='Anonymous'):
    if not _bigquery_available: return
    def _do():
        try:
            scores = [cumulative_scores[i] if i<num_players else None for i in range(4)]
            row = [{'game_id':game_id,'timestamp':time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),
                    'num_players':num_players,'player_0_score':scores[0],'player_1_score':scores[1],
                    'player_2_score':scores[2],'player_3_score':scores[3],'winning_player':winner,
                    'human_player_name':username}]
            errs = _bq.insert_rows_json(_bq_results, row)
            if errs: print(f"BQ results errors: {errs}")
        except Exception as e: print(f"BQ result error: {e}")
    threading.Thread(target=_do, daemon=True).start()

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"✓ Oh Hell server starting on port {port}")
    app.run(debug=False, port=port, host='0.0.0.0')
