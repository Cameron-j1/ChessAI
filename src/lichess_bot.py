import berserk
import chess
import threading
import time
from chess_svm_model import ChessMoveEvaluator
from config import LICHESS_API_TOKEN

class LichessBot:
    def __init__(self, api_token):
        self.session = berserk.TokenSession(api_token)
        self.client = berserk.Client(self.session)
        self.evaluator = ChessMoveEvaluator()
        self.current_games = {}
        self.is_running = False

    def start(self):
        """Start the bot"""
        self.is_running = True
        # Start game stream in a separate thread
        stream_thread = threading.Thread(target=self._handle_game_stream)
        stream_thread.start()
        # Start event stream in a separate thread
        event_thread = threading.Thread(target=self._handle_event_stream)
        event_thread.start()

    def stop(self):
        """Stop the bot"""
        self.is_running = False

    def _handle_game_stream(self):
        """Handle the game stream from Lichess"""
        while self.is_running:
            try:
                for game in self.client.bots.stream_incoming_events():
                    if game['type'] == 'challenge':
                        self._handle_challenge(game['challenge'])
                    elif game['type'] == 'gameStart':
                        self._handle_game_start(game['game']['id'])
            except Exception as e:
                print(f"Error in game stream: {e}")
                time.sleep(5)  # Wait before reconnecting

    def _handle_event_stream(self):
        """Handle the event stream from Lichess"""
        while self.is_running:
            try:
                for event in self.client.bots.stream_game_state(self.current_games.keys()):
                    self._handle_game_state(event)
            except Exception as e:
                print(f"Error in event stream: {e}")
                time.sleep(5)  # Wait before reconnecting

    def _handle_challenge(self, challenge):
        """Accept or decline challenges"""
        try:
            # Accept standard chess challenges
            if challenge['variant']['key'] == 'standard':
                self.client.bots.accept_challenge(challenge['id'])
            else:
                self.client.bots.decline_challenge(challenge['id'])
        except Exception as e:
            print(f"Error handling challenge: {e}")

    def _handle_game_start(self, game_id):
        """Initialize a new game"""
        try:
            self.current_games[game_id] = chess.Board()
            # Start game in a separate thread
            game_thread = threading.Thread(target=self._play_game, args=(game_id,))
            game_thread.start()
        except Exception as e:
            print(f"Error handling game start: {e}")

    def _handle_game_state(self, game_state):
        """Update game state based on opponent's moves"""
        try:
            game_id = game_state['id']
            if game_id in self.current_games:
                board = self.current_games[game_id]
                # Apply moves to our board
                for move in game_state['moves'].split():
                    if move and not board.is_game_over():
                        board.push_uci(move)
        except Exception as e:
            print(f"Error handling game state: {e}")

    def _play_game(self, game_id):
        """Main game loop for a specific game"""
        try:
            board = self.current_games[game_id]
            while not board.is_game_over() and game_id in self.current_games:
                # Check if it's our turn
                if board.turn == chess.WHITE and self.client.games.get_ongoing()[0]['color'] == 'white' or \
                   board.turn == chess.BLACK and self.client.games.get_ongoing()[0]['color'] == 'black':
                    # Get best move from our AI
                    best_move, confidence = self.evaluator.find_best_move(board)
                    if best_move:
                        # Make the move on Lichess
                        self.client.bots.make_move(game_id, best_move.uci())
                        # Update our local board
                        board.push(best_move)
                time.sleep(0.1)  # Prevent busy waiting
        except Exception as e:
            print(f"Error in game loop: {e}")
        finally:
            # Clean up when game is over
            if game_id in self.current_games:
                del self.current_games[game_id]

if __name__ == "__main__":
    bot = LichessBot(LICHESS_API_TOKEN)
    bot.start()

    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.stop() 