#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import chess
import chess.engine
from chess_model_class import ChessModel, ChessModelSpatial
from other_functions import board_to_matrix, decode_move_spatial
from evaluate import ChessEvaluator
import sys
import os


# This node is designed to work with the RS2 ROS2 software package our group has created.
# It subscribes to /fen_string topic to receive the current chess position
# and publishes chess moves to /UCI_moves topic.
# The node can use either Stockfish or a trained neural network model
# to generate chess moves.
#
# Topics:
# Subscribed:
#   /fen_string (std_msgs/String) - Current chess position in FEN notation
# Published: 
#   /UCI_moves (std_msgs/String) - Generated chess moves in UCI format
#
# Parameters:
#   use_stockfish (bool) - Whether to use Stockfish engine (True) or neural network (False)
#   stockfish_elo (int) - ELO rating for Stockfish engine
#   stockfish_time_limit (float) - Time limit per move in seconds
#   model_type (str) - Type of neural network model ('categorical' or 'spatial')
#   model_path (str) - Path to neural network model weights
#   mapping_path (str) - Path to move mappings for categorical models
class ChessModelNode(Node):
    def __init__(self):
        super().__init__('chess_model_node')
        
        # Configuration - Set this boolean to choose your engine
        self.use_stockfish = True  # Set to False to use ChessEvaluator model
        
        # Model configuration (only used if use_stockfish = False)
        self.model_type = 'spatial'  # 'categorical' or 'spatial'
        self.model_path = 'models/spatial_model_checkpoint_epoch_10.pth'
        self.mapping_path = None  # Only needed for categorical models
        # For categorical models, uncomment and modify these:
        # self.model_type = 'categorical'
        # self.model_path = 'models/trainv3_model_architecture_v2_epochs250.pth'
        # self.mapping_path = "models/move_to_int_architecture_v2"
        
        # Stockfish configuration (only used if use_stockfish = True)
        self.stockfish_path = '/usr/games/stockfish'
        self.stockfish_elo = 1500
        self.stockfish_time_limit = 0.1  # seconds per move
        
        # Initialize the chess engine
        self.engine = None
        self.model_evaluator = None
        
        self.initialize_engine()
        
        # Create subscription to FEN topic
        self.fen_subscription = self.create_subscription(
            String,
            '/fen_string',
            self.fen_callback,
            10
        )
        
        # Create publisher for UCI moves
        self.uci_publisher = self.create_publisher(
            String,
            '/UCI_moves',
            10
        )
        
        self.get_logger().info(f'Chess Model Node started using {"Stockfish" if self.use_stockfish else "ChessEvaluator"} engine')
        
    def initialize_engine(self):
        """Initialize either Stockfish or ChessEvaluator based on configuration"""
        try:
            if self.use_stockfish:
                # Initialize Stockfish
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                self.engine.configure({"UCI_LimitStrength": self.stockfish_elo})
                self.get_logger().info(f'Stockfish engine initialized with ELO {self.stockfish_elo}')
            else:
                # Initialize ChessEvaluator model
                if self.model_type == 'categorical':
                    if self.mapping_path is None:
                        raise ValueError("mapping_path is required for categorical models")
                    self.model_evaluator = ChessEvaluator(
                        self.model_path, 
                        self.mapping_path, 
                        model_type='categorical'
                    )
                elif self.model_type == 'spatial':
                    self.model_evaluator = ChessEvaluator(
                        self.model_path, 
                        model_type='spatial'
                    )
                else:
                    raise ValueError("model_type must be either 'categorical' or 'spatial'")
                
                self.get_logger().info(f'ChessEvaluator model ({self.model_type}) initialized from {self.model_path}')
                
        except Exception as e:
            self.get_logger().error(f'Failed to initialize engine: {str(e)}')
            sys.exit(1)
    
    def fen_callback(self, msg):
        """Callback function for FEN string messages"""
        fen_string = msg.data.strip()
        self.get_logger().info(f'Received FEN: {fen_string}')
        
        try:
            # Create chess board from FEN
            board = chess.Board(fen_string)
            
            # Validate the board position
            if not board.is_valid():
                self.get_logger().error('Invalid board position from FEN')
                return
            
            # Check if the game is already over
            if board.is_game_over():
                self.get_logger().warn('Game is already over, no move to generate')
                return
            
            # Generate move based on selected engine
            uci_move = self.generate_move(board)
            
            if uci_move:
                # Publish the UCI move
                move_msg = String()
                move_msg.data = uci_move
                self.uci_publisher.publish(move_msg)
                self.get_logger().info(f'Published UCI move: {uci_move}')
            else:
                self.get_logger().error('Failed to generate a valid move')
                
        except Exception as e:
            self.get_logger().error(f'Error processing FEN: {str(e)}')
    
    def generate_move(self, board):
        """Generate a move using the selected engine"""
        try:
            if self.use_stockfish:
                # Use Stockfish to generate move
                result = self.engine.play(board, chess.engine.Limit(time=self.stockfish_time_limit))
                return result.move.uci()
            else:
                # Use ChessEvaluator model to generate move
                # Calculate move number and determine if it's white's turn
                move_num = (board.fullmove_number)
                is_white = board.turn
                
                uci_move = self.model_evaluator.get_model_move(board, move_num, is_white)
                return uci_move
                
        except Exception as e:
            self.get_logger().error(f'Error generating move: {str(e)}')
            return None
    
    def destroy_node(self):
        """Clean up resources when node is destroyed"""
        if self.engine:
            try:
                self.engine.quit()
                self.get_logger().info('Stockfish engine closed')
            except:
                pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    chess_model_node = ChessModelNode()
    
    try:
        rclpy.spin(chess_model_node)
    except KeyboardInterrupt:
        chess_model_node.get_logger().info('Node interrupted by user')
    finally:
        chess_model_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 