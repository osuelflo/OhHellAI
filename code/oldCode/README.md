# Oh Hell Card Game - Setup Guide

## 🎴 What You've Got

A beautiful vintage-style Oh Hell card game with:
- **Elegant web interface** (oh_hell_game.html) - sophisticated card club aesthetic
- **Python Flask backend** (oh_hell_server.py) - integrates with your existing game logic
- **AI opponents** powered by your CSharpBot

## 📋 Prerequisites

You need:
- Python 3.7+
- Flask (`pip install flask`)
- flask-cors (`pip install flask-cors`)
- Your existing files: `efficientISMCTS.py`, `CardHelper.py`, `CSharpBot.py`

## 🚀 Quick Start

### Step 1: Organize Your Files

Put all these files in the same directory:
```
your_project_folder/
├── oh_hell_server.py          (the Flask backend)
├── oh_hell_game.html           (the web interface)
├── efficientISMCTS.py          (your existing game logic)
├── CardHelper.py               (your existing card utilities)
├── CSharpBot.py                (the bot we just created)
└── probabilityData/            (your existing probability tables)
```

### Step 2: Install Dependencies

```bash
pip install flask flask-cors --break-system-packages
```

### Step 3: Start the Server

```bash
python oh_hell_server.py
```

You should see:
```
============================================================
Oh Hell Card Game Server
============================================================

Starting server on http://localhost:5000
Open oh_hell_game.html in your browser to play!
```

### Step 4: Open the Game

Open `oh_hell_game.html` in your web browser (Chrome, Firefox, Safari, etc.)

That's it! 🎉

## 🎮 How to Play

1. **Game starts automatically** - cards are dealt, trump is revealed
2. **Bidding phase** - Click a number to bid how many tricks you'll take
   - ⚠️ Last player can't make the bids equal the total tricks
3. **Playing phase** - Click cards in your hand to play them
   - You must follow suit if you can
   - Green glow = your turn
4. **Watch the AI** - AI opponents play automatically
5. **New Game** - Click "New Game" button to start over

## 🎨 Design Features

The interface has a vintage card club aesthetic:
- **Felt green table** with elegant oval shape
- **Cormorant Garamond** serif font for titles (classic, refined)
- **Gold and forest green** color scheme
- **Smooth animations** - cards flip, pulse effects, hover states
- **Real-time updates** - watch AI players take their turns

## 🔧 Configuration

### Change Number of Players or Tricks

In `oh_hell_game.html`, find the `startGame()` function and modify:

```javascript
const response = await fetch(`${API_URL}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        numPlayers: 4,        // Change to 3-5
        tricksInRound: 5,     // Change to 1-10
        humanPlayer: 0        // Your player index
    })
});
```

### Use Different Bot Settings

In `oh_hell_server.py`, modify the `create_game_state()` function:

```python
bid_style = ["normal"] * num_players     # or "aggressive", "passive"
use_heuristic = [True] * num_players     # Enable/disable particle filtering
```

### Change API URL (if deploying elsewhere)

In `oh_hell_game.html`, update:

```javascript
const API_URL = 'http://localhost:5000';  // Change to your server URL
```

## 🐛 Troubleshooting

### "Failed to connect to server"
- Make sure `python oh_hell_server.py` is running
- Check that it says "Running on http://localhost:5000"
- Try refreshing the browser

### "Could not import OhHellState"
- Ensure all Python files are in the same directory
- Check that `efficientISMCTS.py` and `CardHelper.py` have no syntax errors

### Cards won't play
- Check browser console (F12) for error messages
- Verify you're following suit rules
- Make sure it's your turn (green glow around your name)

### AI isn't playing
- Check the server terminal for error messages
- Verify `CSharpBot.py` is working correctly
- Try restarting the server

## 🎯 Next Steps

Want to enhance the game? Here are some ideas:

1. **Add scoring across rounds** - Track cumulative scores
2. **Add animations** - Card dealing, trick collection
3. **Add sound effects** - Card shuffling, trick wins
4. **Add multiplayer** - Real humans can play together
5. **Add game history** - Review past tricks and bids
6. **Add statistics** - Win rates, average scores, etc.
7. **Mobile responsive** - Optimize for phones/tablets

## 📝 API Endpoints

The backend provides these endpoints:

- `POST /start` - Start a new game
  - Body: `{numPlayers, tricksInRound, humanPlayer}`
  - Returns: Initial game state

- `POST /bid` - Submit a bid
  - Body: `{player, bid}`
  - Returns: Updated game state

- `POST /play` - Play a card
  - Body: `{player, card}`
  - Returns: Updated game state

- `GET /state` - Get current game state
  - Returns: Current game state

## 📄 License

This is a port/integration of your existing Oh Hell game logic with a new web interface.
The game logic remains yours - the interface is provided as-is for your use.

Enjoy your elegant card game! 🎴✨
