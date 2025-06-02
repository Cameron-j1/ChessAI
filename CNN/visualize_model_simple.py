from graphviz import Digraph

def create_simple_visualization():
    dot = Digraph(comment='Chess Model Architecture')
    dot.attr(rankdir='TB')
    
    # Style configurations
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    # Input
    dot.node('input', 'Input\n(13 channels)')
    
    # Initial conv layer
    dot.node('init_conv', 'Initial Conv Layer\n384 channels')
    dot.edge('input', 'init_conv')
    
    # Create a subgraph for the main tower
    with dot.subgraph(name='cluster_main_tower') as main_tower:
        main_tower.attr(label='Main Tower (8 blocks)')
        main_tower.node('block_start', 'Start')
        main_tower.node('res_block', 'Residual Block\n384 channels')
        main_tower.node('attn_block', 'Attention Block\n8 heads')
        main_tower.node('block_end', 'End')
        
        main_tower.edge('block_start', 'res_block')
        main_tower.edge('res_block', 'attn_block')
        main_tower.edge('attn_block', 'block_end')
    
    # Connect main parts
    dot.edge('init_conv', 'block_start')
    
    # Prediction heads
    dot.node('from_head', 'From Square Head\n384→128→32→1')
    dot.node('to_head', 'To Square Head\n384→128→32→1')
    
    dot.edge('block_end', 'from_head')
    dot.edge('block_end', 'to_head')
    
    # Output
    dot.node('output', 'Output\n(2, 8, 8)')
    dot.edge('from_head', 'output')
    dot.edge('to_head', 'output')
    
    # Save the visualization
    dot.render('chess_model_simple', format='png', cleanup=True)
    print("Simplified visualization has been saved as 'chess_model_simple.png'")

if __name__ == '__main__':
    create_simple_visualization() 