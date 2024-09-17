from manim import *

class NeuralNetwork(Scene):
    def construct(self):
        self.camera.color = WHITE
        # Function to create a neuron at a specific location
        def create_neuron(x, y):
            neuron = Circle(
                radius=0.3, 
                color=WHITE,
                fill_opacity = 1,
                sheen_factor = -.5,
                stroke_color = BLACK
                )
            neuron.move_to(np.array([x, y, 0]))
            return neuron

        # Function to draw layers of neurons
        def create_layer(num_neurons, x_offset, y_start, y_end):
            layer = VGroup()
            y_positions = np.linspace(y_start, y_end, num_neurons)
            for y in y_positions:
                neuron = create_neuron(x_offset, y)
                layer.add(neuron)
            return layer
        
        # Create layers: input layer, hidden layer, output layer
        input_layer = create_layer(3, -4, 2, -2)  # 3 neurons
        hidden_layer = create_layer(5, 0, 2.5, -2.5)  # 5 neurons
        output_layer = create_layer(2, 4, 1, -1)  # 2 neurons
        
        
        
        # Create connections (edges) between layers
        def connect_layers(layer1, layer2):
            lines = VGroup()
            for neuron1 in layer1:
                for neuron2 in layer2:
                    line = Line(neuron1.get_center(), neuron2.get_center(), color=BLACK)
                    lines.add(line)
            return lines
        
        # Draw connections between input->hidden and hidden->output
        input_hidden_connections = connect_layers(input_layer, hidden_layer)
        hidden_output_connections = connect_layers(hidden_layer, output_layer)
        
        # Show connections
        self.play(Create(input_hidden_connections), Create(hidden_output_connections))

        # Add neurons to the scene
        self.play(DrawBorderThenFill(input_layer, stroke_width=4), DrawBorderThenFill(hidden_layer, stroke_width=4), DrawBorderThenFill(output_layer, stroke_width=4))
        
        # Highlight activation animation for neurons
        def activate_neuron(neuron, color=BLUE_E, scale_factor=1.2):
            return neuron.animate.set_fill(color, opacity=1).scale(scale_factor)
        
        # Sequential activation animation from input to output
        self.play(activate_neuron(input_layer[0]), activate_neuron(hidden_layer[2]), activate_neuron(output_layer[1]))
        self.play(activate_neuron(input_layer[1]), activate_neuron(hidden_layer[1]), activate_neuron(output_layer[0]))
        self.play(activate_neuron(input_layer[2]), activate_neuron(hidden_layer[4]), activate_neuron(output_layer[1]))
        
        # Wait before ending the animation
        self.wait(2)

# To render the scene, you would normally run this in your terminal:
# manim -pql your_script_name.py NeuralNetwork
