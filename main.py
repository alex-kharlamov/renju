from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.stacklayout import StackLayout
from kivy.config import Config
from ai import Ai
from random import randint

from ai import Model, Agent, TicTacToeModel, TicTacToeQPlayer

class TicTacToe(App):

    title = 'Tic Tac Toe'
    board = []
    choices = ["X","O"]

    # On application build handler
    def build(self):
        Config.set('graphics', 'width', '600')
        Config.set('graphics', 'height', '600')
        self.layout = StackLayout()
        for x in range(9):
            bt = Button(text='', font_size=200, width=200, height=200, size_hint=(None, None), id=str(x))
            bt.bind(on_release=self.btn_pressed)
            self.board.append(bt)
            self.layout.add_widget(bt)
        return self.layout

    # On application start handler
    def on_start(self):
        self.init_players();
        greeting = "Hello Player! You are playing with \"" + self.player + "\""
        self.popup_message(greeting)


    # On button pressed handler
    def btn_pressed(self, button):
        if len(button.text.strip()) < 1: # Continue only if the button has no mark on it...
            button.text = self.player
            mapping = [6,7,8,3,4,5,0,1, 2]
            self.bot.make_move(self.board, int(button.id))
            self.check_winner()
        #6 7 8              0 1 2
        #3 4 5              3 4 5
        #0 1 2              6 7 8
    # Initializes players
    def init_players(self):
        rand_choice = randint(0,1);
        self.bot = Ai(self.choices[rand_choice]);
        self.player = self.choices[0] if rand_choice == 1 else self.choices[1]

    # Checks winner after every move...

    def check_winner(self):
        pass # Just void..

    def popup_message(self, msg):
        popup = Popup(title="Welcome!", content=Label(text=msg), size=(300, 100), size_hint=(None, None))
        popup.open()




if __name__ == '__main__':
    TicTacToe().run()
