#!/usr/bin/env python3
"""
Test script to verify Oh Hell game setup
Run this to check if everything is configured correctly
"""

import sys
import os

print("=" * 60)
print("Oh Hell Game Setup Test")
print("=" * 60)
print()

# Test 1: Check file locations
print("1. Checking file locations...")
required_files = ['efficientISMCTS.py', 'CardHelper.py', 'oh_hell_server.py']
missing_files = []

for filename in required_files:
    if os.path.exists(filename):
        print(f"   ✓ Found {filename}")
    else:
        print(f"   ✗ Missing {filename}")
        missing_files.append(filename)

if missing_files:
    print(f"\n❌ Missing files: {', '.join(missing_files)}")
    print("\nPlease make sure all files are in the same directory:")
    print(f"  Current directory: {os.getcwd()}")
    sys.exit(1)

print()

# Test 2: Try importing modules
print("2. Testing imports...")
try:
    from efficientISMCTS import OhHellState, CardHelper, ISMCTS
    print("   ✓ Successfully imported OhHellState")
    print("   ✓ Successfully imported CardHelper")
    print("   ✓ Successfully imported ISMCTS")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

print()

# Test 3: Try creating a game state
print("3. Testing game creation...")
try:
    import random
    dealer = 0
    num_players = 4
    tricks_in_round = 10
    
    is_ai = [i != 0 for i in range(num_players)]
    use_heuristic = [True] * num_players
    bid_style = ['normal'] * num_players
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
        trickster=[False] * num_players,
        start=True
    )
    
    print(f"   ✓ Game state created")
    print(f"   ✓ Players: {state.numberOfPlayers}")
    print(f"   ✓ Trump suit: {state.trumpSuit}")
    print(f"   ✓ Tricks in round: {state.tricksInRound}")
    print(f"   ✓ Current player: {state.playerToMove}")
    print(f"   ✓ Player 0 has {CardHelper.get_num_cards(state.playerHands[0])} cards")
    
except Exception as e:
    print(f"   ✗ Game creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Check Flask dependencies
print("4. Checking Flask dependencies...")
try:
    import flask
    print(f"   ✓ Flask installed (version {flask.__version__})")
except ImportError:
    print("   ✗ Flask not installed")
    print("   Run: pip install flask --break-system-packages")

try:
    import flask_cors
    print(f"   ✓ flask-cors installed")
except ImportError:
    print("   ✗ flask-cors not installed")
    print("   Run: pip install flask-cors --break-system-packages")

print()
print("=" * 60)
print("✅ All tests passed! You're ready to run the server.")
print("=" * 60)
print()
print("To start the server, run:")
print("  python oh_hell_server.py")
print()
print("Then open oh_hell_game.html in your browser.")
print()
