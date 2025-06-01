# ChessAI Project

A chess AI project that combines traditional chess algorithms with modern machine learning techniques. The project includes both an SVM-based and DQN-based chess engine, with integration to play on Lichess.org.

## Features

- SVM-based move evaluation
- Deep Q-Learning Network (DQN) for chess
- Lichess.org integration for online play
- Web interface for game visualization

## Setup

1. Clone the repository:
```bash
git clone https://github.com/cameron-j1/ChessAI.git
cd ChessAI
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Lichess Bot:
   - Create a Lichess account at https://lichess.org
   - Create a bot account by upgrading your account to a BOT account
   - Generate an API token at https://lichess.org/account/oauth/token
   - Create a file named `.env` in the project root with:
     ```
     LICHESS_API_TOKEN=your_token_here
     ```

5. Start the Lichess bot:
```bash
python src/lichess_bot.py
```

## Playing Against the AI

1. Visit our bot profile at https://lichess.org/@/AI_in_robo_engine
2. Click "Challenge Bot" to start a game
3. Choose your time control and color
4. Play your game!

## Development

The project consists of several components:

- `src/lichess_bot.py`: Lichess.org integration
- `SVM_For_Move_Choice/`: SVM-based move evaluation
- `scripts/`: Web interface scripts
- `styles/`: CSS styling
- `index.html`: Main web interface

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- University of Technology Sydney
- Lichess.org for their excellent API
- The python-chess and berserk libraries 