# import cv2

# class GameManager:
#     """
#     A class to manage player names, HP values, and display HP bars on a video feed.
#     This example handles up to two players, each with a bar that changes color
#     from green to red as HP decreases.
#     """

#     def __init__(self, max_hp=1000):
#         """
#         Args:
#             max_hp (int): The maximum HP each player can have.
#         """
#         self.players = {}     # Dictionary to store {player_name: current_hp}
#         self.max_hp = max_hp  # Maximum HP for any player

#         # Bar display settings
#         self.bar_width = 250
#         self.bar_height = 25

#     def add_player(self, player_name, initial_hp=None):
#         """
#         Add a new player to the game.

#         Args:
#             player_name (str): Unique identifier for the player.
#             initial_hp (int, optional): Starting HP. Defaults to self.max_hp.
#         """
#         if initial_hp is None:
#             initial_hp = self.max_hp
#         self.players[player_name] = initial_hp

#     def reduce_hp(self, player_name, amount):
#         """
#         Reduce HP of a specific player by a certain amount.

#         Args:
#             player_name (str): The player's name.
#             amount (int): The amount of HP to subtract.
#         """
#         if player_name in self.players:
#             new_hp = self.players[player_name] - amount
#             self.players[player_name] = max(0, min(new_hp, self.max_hp))

#     def set_hp(self, player_name, hp):
#         """
#         Directly set the HP of a specific player, clamped between 0 and max_hp.

#         Args:
#             player_name (str): The player's name.
#             hp (int): The new HP value.
#         """
#         if player_name in self.players:
#             self.players[player_name] = max(0, min(hp, self.max_hp))

#     def _get_bar_color(self, hp_percentage):
#         """
#         Return a color that transitions from green to red as HP goes from 100% to 0%.

#         Args:
#             hp_percentage (float): The player's HP as a fraction of max HP (0.0 to 1.0).

#         Returns:
#             (b, g, r) tuple for OpenCV drawing.
#         """
#         # Start color: green = (0, 255, 0)
#         # End color:   red   = (0, 0, 255)
#         # Simple linear interpolation:
#         r = int(255 * (1 - hp_percentage))   # goes from 0   to 255
#         g = int(255 * hp_percentage)         # goes from 255 to 0
#         b = 0
#         return (b, g, r)

#     def display_hp_bars(self, image):
#         """
#         Draw HP bars for up to two players on the provided image.

#         Args:
#             image (numpy.ndarray): The frame (BGR) on which HP bars will be drawn.
#         """
#         # Only display bars for the first two players
#         sorted_players = list(self.players.items())[:2]

#         for i, (player_name, hp) in enumerate(sorted_players):
#             hp_percentage = hp / self.max_hp

#             # Decide bar position: left for player 0, right for player 1
#             bar_x_offset = 10 if i == 0 else image.shape[1] - self.bar_width - 10
#             bar_y_offset = 30

#             # Draw a background rectangle (dark gray)
#             cv2.rectangle(
#                 image,
#                 (bar_x_offset, bar_y_offset),
#                 (bar_x_offset + self.bar_width, bar_y_offset + self.bar_height),
#                 (50, 50, 50),
#                 -1
#             )

#             # Calculate width of the filled portion based on HP percentage
#             current_bar_width = int(hp_percentage * self.bar_width)
#             bar_color = self._get_bar_color(hp_percentage)

#             # Draw the HP portion of the bar
#             cv2.rectangle(
#                 image,
#                 (bar_x_offset, bar_y_offset),
#                 (bar_x_offset + current_bar_width, bar_y_offset + self.bar_height),
#                 bar_color,
#                 -1
#             )

#             # Prepare text (with outline for a more "animated" look)
#             text = f"{player_name}: {hp} HP"
#             text_x = bar_x_offset
#             text_y = bar_y_offset - 5
#             font_scale = 0.6
#             thickness = 2
#             font = cv2.FONT_HERSHEY_SIMPLEX

#             # Draw outline in black (offset thickness)
#             cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
#             # Draw main text in white
#             cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
