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
            'bid': state.bids[i] if not state.haventBid[i] else None,
            'tricksTaken': state.tricksTaken[i],
            'numCards': CardHelper.get_num_cards(state.playerHands[i])  # For face-down cards
        }
        
        # Only send hand to human player
        if i == human_player:
            player_data['hand'] = CardHelper.to_list(state.playerHands[i])
        else:
            player_data['hand'] = []  # Hidden
            
        players_data.append(player_data)
    
    # Determine phase using haventBid
    all_bids_placed = not any(state.haventBid)
    phase = 'playing' if all_bids_placed else 'bidding'
    
    # Current trick
    trick_data = [
        {'player': player, 'card': card}
        for player, card in state.currentTrick
    ]
    
    # Legal moves for human player
    legal_moves = []
    if state.playerToMove == human_player:
        if phase == 'playing':
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
    
    # Store last completed trick if available
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
    """
    Handle the bidding phase - players bid in turn order.
    Returns messages about bids placed.
    Stops when it's the human player's turn to bid (and hasn't bid yet).
    """
    messages = []
    
    print("\n=== BIDDING PHASE ===")
    print(f"Bids at start: {state.bids}")
    print(f"Haven't bid: {state.haventBid}")
    print(f"Current player to bid: {state.playerToMove}")
    
    while any(state.haventBid):
        current = state.playerToMove
        
        # Check if this player has already bid using haventBid
        if not state.haventBid[current]:
            print(f"Player {current} has already bid {state.bids[current]}")
            # Already bid, move to next player
            state.playerToMove = state.GetNextPlayer(state.playerToMove)
            continue
        
        # If it's human's turn to bid, stop and wait
        if current == human_player:
            print(f"Waiting for human player {human_player} to bid")
            break
        
        # AI player bids
        print(f"AI Player {current} is bidding...")
        
        # Build bids_in_order array for Bid method
        bids_in_order = []
        position = 0
        p = state.GetNextPlayer(state.dealer)
        
        for i in range(state.numberOfPlayers):
            if not state.haventBid[p]:
                bids_in_order.append(state.bids[p])
            else:
                bids_in_order.append(0)  # Placeholder for not yet bid
            
            if p == current:
                position = i
            
            p = state.GetNextPlayer(p)
        
        # AI makes bid (this will set haventBid[current] = False)
        state.Bid(bids_in_order, position, current)
        messages.append(f"Player {current + 1} bid {state.bids[current]}")
        print(f"Player {current} bid {state.bids[current]}")
        
        # Move to next player
        state.playerToMove = state.GetNextPlayer(state.playerToMove)
    
    print(f"Final bids: {state.bids}")
    print(f"Haven't bid: {state.haventBid}")
    print("=== END BIDDING PHASE ===\n")
    
    return messages


def run_playing_phase(state, human_player=0, iterations=2500):
    """
    Handle the playing phase - plays ONE AI card and returns.
    Returns messages about the card played.
    Stops immediately after playing one card OR when it's the human player's turn.
    """
    messages = []
    
    print("\n=== PLAYING PHASE - SINGLE CARD ===")
    
    # Only play ONE card
    if state.GetMoves() != 0:
        current = state.playerToMove
        
        # If it's human's turn, stop and wait
        if current == human_player:
            print(f"Waiting for human player {human_player} to play")
            return messages
        
        # AI player plays ONE card
        print(f"AI Player {current} playing... (needs {state.GetTricksNeeded(current)} more tricks)")
        
        # Determine which card to play
        if state.currentTrick != []:
            (leader, leadCard) = state.currentTrick[0]
            hand = state.GetMoves()
            cardsInSuit = CardHelper.get_cards_in_suit(CardHelper.get_card_suit(leadCard, isHand=False), hand)
        else:
            cardsInSuit = 0
        if CardHelper.get_num_cards(state.GetMoves()) == 1:
            m = CardHelper.hand_to_card(state.GetMoves())
        elif state.GetTricksNeeded(current) < 0:
            # If we can't win, play the highest rank card you can
            m = CardHelper.get_highest_card(state.GetMoves())
        elif state.currentTrick != [] and state.GetTricksNeeded(current) == 0 and cardsInSuit != 0:
            m = CardHelper.get_highest_losing_card(hand, state.currentTrick, state.trumpSuit)
            if m < 0:
                print(f"  Running ISMCTS with {iterations} iterations...")
                m = ISMCTS(rootstate=state, itermax=iterations, randomRollout=True, mainPlayer=current, verbose=False)
        else:
            # Use ISMCTS
            print(f"  Running ISMCTS with {iterations} iterations...")
            m = ISMCTS(rootstate=state, itermax=iterations, randomRollout=True, mainPlayer=current, verbose=False)
        
        card_str = CardHelper.to_str(m)
        print(f"  Played {card_str}")
        state.printHand(current)
        print(state.currentTrick)
        # Store current trick before playing (in case this completes it)
        trick_before = list(state.currentTrick)
        trick_before.append((current, m))
        
        state.DoMove(m)
        messages.append(f"Player {current + 1} played {card_str}")
        
        # Check if trick just finished
        if len(state.currentTrick) == 0 and state.GetMoves() != 0:
            # Trick finished, store it so frontend can display it
            state._last_completed_trick = [{'player': p, 'card': c} for p, c in trick_before]
            winner = state.playerToMove
            state._last_trick_winner = winner
            # The winning card is the one played by the winner in trick_before
            winning_play = next((c for p, c in trick_before if p == winner), None)
            state._last_winning_card = winning_play
            messages.append(f"Player {winner + 1} won the trick!")
            print(f"  Player {winner} won the trick!")
    
    if state.GetMoves() == 0:
        print("=== ROUND COMPLETE ===\n")
    
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
        game_id = 'default'
        
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
        print(f"Dealer: {state.dealer}")
        
        # Set up bidding order - first player after dealer bids first
        state.playerToMove = state.GetNextPlayer(state.dealer)
        print(f"First bidder: Player {state.playerToMove}")
        
        # Run AI bids until it's human's turn (if human isn't first)
        print("Running bidding phase...")
        messages = run_bidding_phase(state, human_player)
        print(f"Bidding phase complete. Messages: {messages}")
        
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
    try:
        data = request.json
        player = data.get('player', 0)
        bid = data.get('bid')
        
        print(f"\n>>> Player {player} submitting bid: {bid}")
        
        game_id = 'default'
        if game_id not in game_instances:
            return jsonify({'error': 'Game not found'}), 404
        
        game = game_instances[game_id]
        state = game['current_state']
        
        # Validate it's this player's turn to bid
        if player != state.playerToMove:
            return jsonify({'error': f'Not your turn to bid. Waiting for Player {state.playerToMove + 1}'}), 400
        
        # Validate this player hasn't already bid
        if not state.haventBid[player]:
            return jsonify({'error': 'You have already bid'}), 400
        
        # Place the bid
        state.bids[player] = bid
        state.haventBid[player] = False
        print(f"Bid placed. Current bids: {state.bids}")
        print(f"Haven't bid: {state.haventBid}")
        
        # Move to next player
        state.playerToMove = state.GetNextPlayer(state.playerToMove)
        
        # Continue bidding phase with AI players
        messages = [f"You bid {bid}"]
        ai_bid_messages = run_bidding_phase(state, player)
        messages.extend(ai_bid_messages)
        
        # Check if bidding is complete
        all_bids_placed = not any(state.haventBid)
        if all_bids_placed:
            print(">>> All bids placed! Adjusting probability tables and starting play phase")
            
            # Adjust probability tables now that all bids are in
            if state.tricksInRound != 1:
                state.adjustProbsBids()
            
            # Start playing phase - first player after dealer leads
            state.playerToMove = state.GetNextPlayer(state.dealer)
            messages.append("Bidding complete! Let's play!")
            
            # Run AI plays if human isn't first to play
            ai_play_messages = run_playing_phase(state, player)
            messages.extend(ai_play_messages)
        
        response = state_to_json(state, player)
        response['gameProgress'] = {
            'currentRound': game['current_round'] + 1,
            'totalRounds': len(game['tricks_sequence']),
            'cumulativeScores': game['cumulative_scores']
        }
        
        if messages:
            response['messages'] = messages
        
        return jsonify(response)
    
    except Exception as e:
        print(f"ERROR in submit_bid: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/play', methods=['POST'])
def play_card():
    """Play a card"""
    try:
        data = request.json
        player = data.get('player', 0)
        card = data.get('card')
        
        print(f"\n>>> Player {player} playing card: {CardHelper.to_str(card)}")
        
        game_id = 'default'
        if game_id not in game_instances:
            return jsonify({'error': 'Game not found'}), 404
        
        game = game_instances[game_id]
        state = game['current_state']
        
        # Validate move
        if player != state.playerToMove:
            return jsonify({'error': f'Not your turn. Waiting for Player {state.playerToMove + 1}'}), 400
        
        legal_moves = state.GetMoves()
        if not CardHelper.has_card(legal_moves, card):
            return jsonify({'error': 'Illegal move - you must follow suit if possible'}), 400
        
        # Play the card
        state.DoMove(card)
        messages = [f"You played {CardHelper.to_str(card)}"]
        
        # Check if trick just finished
        if len(state.currentTrick) == 0 and state.GetMoves() != 0:
            winner = state.playerToMove
            state._last_trick_winner = winner
            state._last_winning_card = card
            # Build completed trick for display (reconstruct from what we know)
            # The trick_before is not tracked here, but we can set lastCompletedTrick in state_to_json
            messages.append(f"Player {winner + 1} won the trick!")
        
        # DO NOT run AI playing phase here - frontend will poll for it
        
        response = state_to_json(state, player)
        if messages:
            response['message'] = ' | '.join(messages)
        
        # Check if round is over
        if state.GetMoves() == 0:
            print(">>> Round complete!")
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
                game['current_round'] -= 1
                response['gameOver'] = True
                winner = max(range(len(game['cumulative_scores'])), key=lambda i: game['cumulative_scores'][i])
                response['winner'] = winner
                print(f">>> Game over! Winner: Player {winner + 1}")
            else:
                response['nextRound'] = game['current_round'] + 1
        
        response['gameProgress'] = {
            'currentRound': game['current_round'] + 1,
            'totalRounds': len(game['tricks_sequence']),
            'cumulativeScores': game['cumulative_scores']
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"ERROR in play_card: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/advance_ai', methods=['POST'])
def advance_ai():
    """Advance AI by one card play"""
    try:
        game_id = 'default'
        if game_id not in game_instances:
            return jsonify({'error': 'Game not found'}), 404
        
        game = game_instances[game_id]
        state = game['current_state']
        player = 0  # human player
        
        # Play one AI card
        messages = run_playing_phase(state, player)
        
        response = state_to_json(state, player)
        if messages:
            response['message'] = ' | '.join(messages)
        
        # Check if round is over
        if state.GetMoves() == 0:
            print(">>> Round complete!")
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
                winner = max(range(len(game['cumulative_scores'])), key=lambda i: game['cumulative_scores'][i])
                response['winner'] = winner
            else:
                response['nextRound'] = game['current_round'] + 1
        
        response['gameProgress'] = {
            'currentRound': game['current_round'] + 1,
            'totalRounds': len(game['tricks_sequence']),
            'cumulativeScores': game['cumulative_scores']
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"ERROR in advance_ai: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    except Exception as e:
        print(f"ERROR in play_card: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/next_round', methods=['POST'])
def next_round():
    """Start the next round of the game"""
    try:
        game_id = 'default'
        if game_id not in game_instances:
            return jsonify({'error': 'Game not found'}), 404
        
        game = game_instances[game_id]
        
        if game['current_round'] >= len(game['tricks_sequence']):
            return jsonify({'error': 'Game is over'}), 400
        
        print(f"\n>>> Starting round {game['current_round'] + 1}")
        
        # Update dealer
        game['dealer'] = (game['dealer'] + 1) % game['num_players']
        
        # Create new round
        tricks_in_round = game['tricks_sequence'][game['current_round']]
        state = create_game_state(game['num_players'], tricks_in_round, game['human_player'])
        state.dealer = game['dealer']
        game['current_state'] = state
        
        print(f"New round: {tricks_in_round} tricks, dealer: {state.dealer}, trump: {state.trumpSuit}")
        
        # Set up bidding - first player after dealer
        state.playerToMove = state.GetNextPlayer(state.dealer)
        
        # Run AI bids until it's human's turn
        messages = run_bidding_phase(state, game['human_player'])
        
        response = state_to_json(state, game['human_player'])
        response['gameProgress'] = {
            'currentRound': game['current_round'] + 1,
            'totalRounds': len(game['tricks_sequence']),
            'cumulativeScores': game['cumulative_scores']
        }
        
        if messages:
            response['messages'] = messages
        
        return jsonify(response)
    
    except Exception as e:
        print(f"ERROR in next_round: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

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
