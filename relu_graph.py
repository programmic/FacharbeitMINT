from manim import *

class ReLUActivation(Scene):
    def construct(self):
        # Create the axes with a larger animation duration
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-1, 4, 1],
            axis_config={"color": BLUE},
            tips=False
        )
        labels = axes.get_axis_labels(x_label="x", y_label="ReLU(x)")
        
        # Add title text at the top
        title = Text("ReLU Activation Function", font_size=36).to_corner(UL)
        self.play(Write(title))
        
        # Animate the creation of the axes
        self.play(Create(axes), run_time=3)
        self.play(Write(labels))

        # Define the ReLU function for plotting
        relu_graph = axes.plot(lambda x: max(0, x), color=YELLOW)
        
        # Create dots at key points
        zero_dot = Dot(point=axes.coords_to_point(0, 0), color=RED)
        pos_dot = Dot(point=axes.coords_to_point(2, 2), color=PINK)

        # Animate the drawing of the ReLU graph
        self.play(Create(relu_graph), run_time=4)

        # Add text explanations for ReLU(x) = 0 when x < 0
        text_left = Text("ReLU(x) = 0 for x < 0", font_size=24).next_to(axes, LEFT)
        self.play(FadeIn(text_left), FadeIn(zero_dot))

        # Wait and then remove the text
        self.wait(2)
        self.play(FadeOut(text_left))

        # Add explanation for ReLU(x) = x for x >= 0
        text_right = Text("ReLU(x) = x for x >= 0", font_size=24).next_to(axes, RIGHT)
        self.play(FadeIn(text_right), FadeIn(pos_dot))

        # Animate a moving dot on the ReLU graph to show it "growing" for x > 0
        moving_dot = Dot(axes.coords_to_point(-3, 0), color=WHITE)
        moving_dot_text = MathTex("ReLU(x)").next_to(moving_dot, UP)

        # Animate the dot moving from the left side (-3, 0) to right side (3, 3)
        self.play(FadeIn(moving_dot, moving_dot_text))
        self.play(
            moving_dot.animate.move_to(axes.coords_to_point(3, 3)),
            moving_dot_text.animate.next_to(moving_dot, UP),
            run_time=5
        )

        # Pause at the final position
        self.wait(2)

        # Remove all text and graph to end the animation
        self.play(FadeOut(text_right, moving_dot, moving_dot_text, zero_dot, pos_dot, relu_graph, labels, axes, title))

        # Final wait time for the clean scene
        self.wait(2)
