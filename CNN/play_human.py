import chess
import chess.svg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cairosvg
import io
from evaluate import ChessEvaluator
import os

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess AI Game")
        
        # Set window minimum size
        self.root.minsize(800, 900)
        
        # Initialize the board
        self.board = chess.Board()
        
        # Initialize the AI - modify to fit whatever AI you want to test
        model_path = 'models/attention_model_checkpoint_epoch_10.pth'
        # mapping_path = 'models/move_to_int_architecture_v2' #categorical models need a mapping path
        mapping_path = None #spatial and attention models don't need a mapping path
        self.ai = ChessEvaluator(model_path, mapping_path, model_type='attention') #'spatial', 'attention', 'categorical'
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Create board display with larger size
        self.canvas = tk.Canvas(self.main_frame, width=700, height=700)
        self.canvas.grid(row=0, column=0, columnspan=2, padx=20, pady=20)
        
        # Create status label with larger font
        self.status_var = tk.StringVar(value="Welcome! You play as White.")
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var, font=('TkDefaultFont', 12))
        self.status_label.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Create move entry with larger size
        self.move_frame = ttk.Frame(self.main_frame)
        self.move_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Label(self.move_frame, text="Your move (e.g. e2e4):", font=('TkDefaultFont', 11)).grid(row=0, column=0, padx=5)
        self.move_entry = ttk.Entry(self.move_frame, width=15, font=('TkDefaultFont', 11))
        self.move_entry.grid(row=0, column=1, padx=5)
        
        # Create buttons with larger size
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        style = ttk.Style()
        style.configure('Large.TButton', font=('TkDefaultFont', 11))
        
        ttk.Button(self.button_frame, text="Make Move", command=self.make_move, style='Large.TButton').grid(row=0, column=0, padx=10)
        ttk.Button(self.button_frame, text="New Game", command=self.new_game, style='Large.TButton').grid(row=0, column=1, padx=10)
        
        # Bind Enter key to make_move
        self.move_entry.bind('<Return>', lambda e: self.make_move())
        
        # Initialize the display
        self.update_display()
        
    def update_display(self):
        # Generate SVG of current board position with larger size
        svg_data = chess.svg.board(
            board=self.board,
            size=700,  # Increased size
            lastmove=self.board.peek() if self.board.move_stack else None,
            check=self.board.king(self.board.turn) if self.board.is_check() else None
        )
        
        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
        
        # Convert PNG to PhotoImage
        image = Image.open(io.BytesIO(png_data))
        photo = ImageTk.PhotoImage(image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # Keep a reference to prevent garbage collection
        
        # Update status
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn else "White"
                self.status_var.set(f"Checkmate! {winner} wins!")
            else:
                self.status_var.set("Game Over! It's a draw!")
        else:
            turn = "White" if self.board.turn else "Black"
            self.status_var.set(f"{turn} to move")
            
            if self.board.is_check():
                self.status_var.set(f"{turn} to move - CHECK!")
    
    def make_move(self):
        if self.board.is_game_over():
            self.status_var.set("Game is over! Start a new game.")
            return
            
        # Get player's move
        move_uci = self.move_entry.get().strip().lower()
        self.move_entry.delete(0, tk.END)
        
        # Validate move
        try:
            move = chess.Move.from_uci(move_uci)
            if move not in self.board.legal_moves:
                self.status_var.set("Illegal move! Try again.")
                return
        except ValueError:
            self.status_var.set("Invalid move format! Use format like 'e2e4'")
            return
            
        # Make player's move
        self.board.push(move)
        self.update_display()
        
        if not self.board.is_game_over():
            # AI's turn
            self.status_var.set("AI is thinking...")
            self.root.update()
            
            # Get AI's move
            ai_move = self.ai.get_model_move(self.board, len(self.board.move_stack) // 2 + 1, False)
            if ai_move:
                self.board.push_uci(ai_move)
                self.update_display()
            else:
                self.status_var.set("AI couldn't find a move!")
    
    def new_game(self):
        self.board = chess.Board()
        self.status_var.set("New game started! You play as White.")
        self.update_display()

def main():
    # Create the main window
    root = tk.Tk()
    root.title("Chess AI Game")
    
    # Create the GUI
    gui = ChessGUI(root)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main() 