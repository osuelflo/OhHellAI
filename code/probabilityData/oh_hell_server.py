"""
oh_hell_server.py
=================
Flask API server for Oh Hell card game.
Integrates with your existing OhHellState using ISMCTS AI.

Usage:
    python oh_hell_server.py
    
Then open oh_hell_game.html in your browser.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import traceback
import sys
import os

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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Enable debug mode
app.config['DEBUG'] = True

# Global game state (in production, use session management or database)
game_instances = {}

@app.route('/')
def index():
    """Serve the HTML game interface"""
    try:
        with open('oh_hell_game.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <body>
                <h1>Oh Hell Card Game</h1>
                <p>Error: oh_hell_game.html not found in the same directory as oh_hell_server.py</p>
                <p>Please make sure both files are in the same folder.</p>
            </body>
        </html>
        """, 404

def create_game_state(num_players=4, tricks_in_round=10, human_player=0):
    """Create a new OhHellState instance"""
    dealer = random.randint(0, num_players - 1)
    
    # Configure: human is player 0, rest are AI
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
    
    return state

def state_to_json(state, human_player=0):
    """Convert OhHellState to JSON for frontend"""
    players_data = []
    for i in range(state.numberOfPlayers):
        player_data = {
            'bid': state.bids[i] if state.bids[i] is not None else None,
            'tricksTaken': state.tricksTaken[i],
        }
        
        # Only send hand to human player
        if i == human_player:
            player_data['hand'] = CardHelper.to_list(state.playerHands[i])
        else:
            player_data['hand'] = []  # Hidden
            
        players_data.append(player_data)
    
    # Determine phase
    all_bids_placed = all(b is not None and b >= 0 for b in state.bids)
    phase = 'playing' if all_bids_placed else 'bidding'
    
    # Current trick
    trick_data = [
        {'player': player, 'card': card}
        for player, card in state.currentTrick
    ]
    
    # Legal moves for human player
    legal_moves = []
    if state.playerToMove == human_player and phase == 'playing':
        legal_moves = CardHelper.to_list(state.GetMoves())
    
    return {
        'players': players_data,
        'currentPlayer': state.playerToMove,
        'trumpSuit': state.trumpSuit,
        'trick': trick_data,
        'tricksInRound': state.tricksInRound,
        'phase': phase,
        'legalMoves': legal_moves
    }

def run_ai_turns(state, human_player=0, iterations=1000):
    """Run AI turns until it's the human player's turn or game ends"""
    messages = []
    
    while state.GetMoves() != 0 and state.playerToMove != human_player:
        current = state.playerToMove
        
        # Check if AI needs to bid
        if state.bids[current] is None or state.bids[current] < 0:
            # Use the existing Bid method from OhHellState
            bids_in_order = [0] * state.numberOfPlayers
            for idx, p_idx in enumerate(state.players):
                if state.bids[p_idx] is not None:
                    bids_in_order[idx] = state.bids[p_idx]
            
            # Find position of current player
            position = 0
            p = state.GetNextPlayer(state.dealer)
            while p != current:
                position += 1
                p = state.GetNextPlayer(p)
            
            state.Bid(bids_in_order, position, current)
            messages.append(f"Player {current + 1} bid {state.bids[current]}")
        else:
            # AI plays a card using ISMCTS
            if state.tricksInRound == 1 or CardHelper.get_num_cards(state.GetMoves()) == 1:
                # Single card or single trick - just play it
                m = CardHelper.hand_to_card(state.GetMoves()) if CardHelper.get_num_cards(state.GetMoves()) == 1 else CardHelper.get_highest_card(state.GetMoves())
            else:
                # Use ISMCTS
                m = ISMCTS(rootstate=state, itermax=iterations, randomRollout=True, mainPlayer=current, verbose=False)
            
            state.DoMove(m)
            messages.append(f"Player {current + 1} played {CardHelper.to_str(m)}")
    
    return messages

@app.route('/start', methods=['POST'])
def start_game():
    """Start a new game"""
    try:
        print("\n" + "=" * 60)
        print("Starting new game...")
        data = request.json
        print(f"Received request data: {data}")
        
        num_players = data.get('numPlayers', 4)
        human_player = data.get('humanPlayer', 0)
        
        print(f"Number of players: {num_players}")
        print(f"Human player: {human_player}")
        
        # Start a full game: 10 tricks down to 1, then 2 to 10
        tricks_sequence = list(range(10, 0, -1)) + list(range(2, 11))
        print(f"Tricks sequence: {tricks_sequence}")
        
        # Create game instance with metadata
        game_id = 'default'  # For simplicity, using single game instance
        
        game_instances[game_id] = {
            'current_state': None,
            'num_players': num_players,
            'human_player': human_player,
            'tricks_sequence': tricks_sequence,
            'current_round': 0,
            'cumulative_scores': [0] * num_players,
            'dealer': random.randint(0, num_players - 1)
        }
        
        print("Game metadata created")
        
        # Start first round
        game = game_instances[game_id]
        tricks_in_round = game['tricks_sequence'][0]
        
        print(f"Creating OhHellState with {num_players} players, {tricks_in_round} tricks...")
        state = create_game_state(num_players, tricks_in_round, human_player)
        state.dealer = game['dealer']
        game['current_state'] = state
        
        print("OhHellState created successfully")
        print(f"Trump suit: {state.trumpSuit}")
        print(f"Current player: {state.playerToMove}")
        
        # Run AI turns if human isn't first
        print("Running AI turns...")
        messages = run_ai_turns(state, human_player)
        print(f"AI turns complete. Messages: {messages}")
        
        print("Converting state to JSON...")
        response = state_to_json(state, human_player)
        response['gameProgress'] = {
            'currentRound': 1,
            'totalRounds': len(tricks_sequence),
            'cumulativeScores': game['cumulative_scores']
        }
        
        if messages:
            response['messages'] = messages
        
        print("Game started successfully!")
        print("=" * 60 + "\n")
        
        return jsonify(response)
    
    except Exception as e:
        print("\n" + "!" * 60)
        print("ERROR in start_game:")
        print("!" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("!" * 60 + "\n")
        
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }), 500

@app.route('/bid', methods=['POST'])
def submit_bid():
    """Submit a bid for a player"""
    data = request.json
    player = data.get('player', 0)
    bid = data.get('bid')
    
    game_id = 'default'
    if game_id not in game_instances:
        return jsonify({'error': 'Game not found'}), 404
    
    game = game_instances[game_id]
    state = game['current_state']
    
    # Validate bid
    if player != state.playerToMove:
        return jsonify({'error': 'Not your turn'}), 400
    
    # Place bid
    state.bids[player] = bid
    state.playerToMove = state.GetNextPlayer(state.playerToMove)
    
    # Run AI turns
    messages = run_ai_turns(state, player)
    
    response = state_to_json(state, player)
    response['gameProgress'] = {
        'currentRound': game['current_round'] + 1,
        'totalRounds': len(game['tricks_sequence']),
        'cumulativeScores': game['cumulative_scores']
    }
    
    if messages:
        response['messages'] = messages
    
    return jsonify(response)

@app.route('/play', methods=['POST'])
def play_card():
    """Play a card"""
    data = request.json
    player = data.get('player', 0)
    card = data.get('card')
    
    game_id = 'default'
    if game_id not in game_instances:
        return jsonify({'error': 'Game not found'}), 404
    
    game = game_instances[game_id]
    state = game['current_state']
    
    # Validate move
    if player != state.playerToMove:
        return jsonify({'error': 'Not your turn'}), 400
    
    legal_moves = state.GetMoves()
    if not CardHelper.has_card(legal_moves, card):
        return jsonify({'error': 'Illegal move'}), 400
    
    # Play the card
    state.DoMove(card)
    
    # Check if trick is complete
    message = None
    if not state.currentTrick:  # Trick just finished
        winner = state.playerToMove
        message = f"Player {winner + 1} won the trick!"
    
    # Run AI turns
    messages = run_ai_turns(state, player)
    if message:
        messages.insert(0, message)
    
    response = state_to_json(state, player)
    if messages:
        response['message'] = ' | '.join(messages)
    
    # Check if round is over
    if state.GetMoves() == 0:
        # Calculate round scores
        round_scores = []
        for p in range(state.numberOfPlayers):
            score = state.GetActualScore(p)
            round_scores.append(score)
            game['cumulative_scores'][p] += score
        
        response['roundOver'] = True
        response['roundScores'] = round_scores
        response['cumulativeScores'] = game['cumulative_scores']
        
        # Check if game is completely over
        game['current_round'] += 1
        if game['current_round'] >= len(game['tricks_sequence']):
            response['gameOver'] = True
            winner = max(range(len(game['cumulative_scores'])), key=lambda i: game['cumulative_scores'][i])
            response['winner'] = winner
        else:
            # Set up next round
            response['nextRound'] = game['current_round'] + 1
    
    response['gameProgress'] = {
        'currentRound': game['current_round'] + 1,
        'totalRounds': len(game['tricks_sequence']),
        'cumulativeScores': game['cumulative_scores']
    }
    
    return jsonify(response)

@app.route('/next_round', methods=['POST'])
def next_round():
    """Start the next round of the game"""
    game_id = 'default'
    if game_id not in game_instances:
        return jsonify({'error': 'Game not found'}), 404
    
    game = game_instances[game_id]
    
    if game['current_round'] >= len(game['tricks_sequence']):
        return jsonify({'error': 'Game is over'}), 400
    
    # Update dealer
    game['dealer'] = (game['dealer'] + 1) % game['num_players']
    
    # Create new round
    tricks_in_round = game['tricks_sequence'][game['current_round']]
    state = create_game_state(game['num_players'], tricks_in_round, game['human_player'])
    state.dealer = game['dealer']
    game['current_state'] = state
    
    # Run AI turns if human isn't first
    messages = run_ai_turns(state, game['human_player'])
    
    response = state_to_json(state, game['human_player'])
    response['gameProgress'] = {
        'currentRound': game['current_round'] + 1,
        'totalRounds': len(game['tricks_sequence']),
        'cumulativeScores': game['cumulative_scores']
    }
    
    if messages:
        response['messages'] = messages
    
    return jsonify(response)

@app.route('/state', methods=['GET'])
def get_state():
    """Get current game state"""
    game_id = 'default'
    if game_id not in game_instances:
        return jsonify({'error': 'Game not found'}), 404
    
    game = game_instances[game_id]
    state = game['current_state']
    
    response = state_to_json(state, game['human_player'])
    response['gameProgress'] = {
        'currentRound': game['current_round'] + 1,
        'totalRounds': len(game['tricks_sequence']),
        'cumulativeScores': game['cumulative_scores']
    }
    
    return jsonify(response)

if __name__ == '__main__':
    print("=" * 60)
    print("Oh Hell Card Game Server")
    print("=" * 60)
    print("\n✓ Server starting on http://localhost:5001")
    print("\n🎮 TO PLAY THE GAME:")
    print("   Open your browser and go to:")
    print("   → http://localhost:5001")
    print("\n   (Don't open the HTML file directly - use the URL above!)")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Install Flask and flask-cors if not present
    try:
        import flask_cors
    except ImportError:
        print("\nInstalling required package: flask-cors...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'flask-cors', '--break-system-packages'])
        print("Installation complete!\n")
    
    app.run(debug=True, port=5001, host='0.0.0.0')
