# graduation-thesis_Takezaki
Graduation thesis on an AR-based task support system

I developed two AR-based support systems for jigsaw puzzle assembly: adaptive support and fixed-sequence support. The purpose of these systems was to compare how different support methods affect task efficiency and usability in non-routine tasks, where the procedure is not strictly predetermined.

In the adaptive support system, the user can freely choose any puzzle piece based on their own judgment. When the user selects a piece, the system identifies it and highlights its correct position on the board in red. This approach allows users to decide the order of assembly by themselves while receiving AR guidance for placing each selected piece.

In the fixed-sequence support system, the system determines the next puzzle piece to be placed according to a predefined order. The target piece is highlighted on the right side of the display, and its correct position is shown on the left side of the display. This method reduces the user’s cognitive burden of deciding which piece to choose next.

Both systems used camera images and image processing techniques to detect puzzle pieces and provide AR instructions. Specifically, I used OpenCV to detect puzzle piece contours and MediaPipe Hands to track the user’s fingertip. The system identified each piece using aspect ratio filtering and shape matching based on Hu moments, and then displayed the appropriate placement position as visual guidance. 

Through experiments using 8-piece and 16-piece jigsaw puzzles, I compared four conditions: no support, adaptive support, fixed-sequence support with an optimal order, and fixed-sequence support with a random order. The results showed that AR-based support significantly reduced assembly time compared with no support. In particular, fixed-sequence support with an optimal order achieved the shortest average completion time, while adaptive support was effective when users were allowed to decide the assembly order themselves.
