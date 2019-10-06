from random import randrange, choice
from itertools import chain
import curses

class Action(object):

    actions = ['up', 'left', 'down', 'right', 'restart', 'exit']
    letter_codes = [ord(ch) for ch in 'WASDRQwasdrq']
    actions_dict = dict(zip(letter_codes, actions * 2))

    def __init__(self, stdscr):
        self.stdscr = stdscr

    def get(self):
        ch = 'N'
        while ch not in self.actions_dict:
            ch = self.stdscr.getch()
        return self.actions_dict[ch]


class Grid(object):
    def __init__(self, size):
        self.size = size
        self.cells = None
        self.reset()
        self.score = 0
    def reset(self):
        self.cells = [[0 for i in range(self.size)] for j in range(self.size)]
    def add_random_item(self):
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.cells[i][j] == 0]
        (i, j) = choice(empty_cells)
        self.cells[i][j] = 4 if randrange(100) >= 90 else 2
    def tranpose(self):
        self.cells = [list(row) for row in zip(*self.cells)]
    def invert(self):
        self.cells = [row[::-1] for row in self.cells]
    def move_row_left(self, row):
        def tighten(row):
            new_row = [i for i in row if i != 0]
            new_row += [0 for i in range(len(row) - len(new_row))]
            return new_row
        def merge(row):
            pair = False
            new_row = []
            for i in range(len(row)):
                if pair:
                    new_row.append(2 * row[i])
                    self.score += 2 * row[i]
                    pair = False
                else:
                    if i + 1 < len(row) and row[i] == row[i + 1]:
                        pair = True
                        new_row.append(0)
                    else:
                        new_row.append(row[i])
            return new_row
        return tighten(merge(tighten(row)))
    def move_left(self):
        self.cells = [self.move_row_left(row) for row in self.cells]
    def move_right(self):
        self.invert()
        self.move_left()
        self.invert()
    def move_up(self):
        self.tranpose()
        self.move_left()
        self.tranpose()
    def move_down(self):
        self.tranpose()
        self.move_right()
        self.tranpose()
    @staticmethod
    def row_can_move_left(row):
        def change(i):
            if row[i] == 0 and row[i + 1] != 0:
                return True
            if row[i] != 0 and row[i] == row[i + 1]:
                return True
            return False
        return any(change(i) for i in range(len(row) - 1))
    def can_move_left(self):
        return any(self.row_can_move_left(row) for row in self.cells)
    def can_move_right(self):
        self.invert()
        if_can = self.can_move_left()
        self.invert()
        return if_can
    def can_move_up(self):
        self.tranpose()
        if_can = self.can_move_left()
        self.tranpose()
        return if_can
    def can_move_down(self):
        self.tranpose()
        if_can = self.can_move_right()
        self.tranpose()
        return if_can


class Screen(object):

    help_string1 = "(W)up (S)down (A)left (D)right"
    help_string2 = "     (R)Restart (Q)Exit"
    over_string = "           GMAE OVER"
    win_string = "         YOU WIN!"

    def __init__(self, screen = None, grid = None, score = 0, over = False, win = False):
        self.screen = screen
        self.score = score
        self.over = over
        self.win = win
        self.grid = grid
        self.counter = 0
    def cast(self, string):
        self.screen.addstr(string + '\n')
    def draw_row(self, row):
        self.cast(''.join('|{:^5}'.format(num) if num > 0 else '|     'for num in row) + '|')
    def draw(self, score, best_score):
        self.screen.clear()
        self.cast('SCORE:' + str(score) + '  ' + 'BEST:' + str(best_score))
        for row in self.grid.cells:
            self.cast('+-----' * self.grid.size + '+')
            self.draw_row(row)
        self.cast('+-----' * self.grid.size + '+')
        if self.win:
            self.cast(self.win_string)
        elif self.over:
            self.cast(self.over_string)
        self.cast(self.help_string1)
        self.cast(self.help_string2)


class GameManager(object):
    def __init__(self, size = 4, win_num = 2048):
        self.size = size
        self.win_num = win_num
        self.best_score = 0
        self.reset()

    def reset(self):
        self.state = 'init'
        self.win = False
        self.over = False
        self.grid = Grid(self.size)
        self.grid.reset()
        self.score = self.grid.score
        self.grid.add_random_item()
        self.grid.add_random_item()
    @property
    def screen(self):
        return Screen(screen = self.stdscr, score = self.score, grid = self.grid, win = self.win, over = self.over)
    def can_move(self, dire):
        return getattr(self.grid, 'can_move_' + dire)()
    def move(self, dire):
        if self.can_move(dire):
            getattr(self.grid, 'move_' + dire)()
            self.grid.add_random_item()
            return True
        else:
            return False
    @property
    def is_win(self):
        self.win = max(chain(*self.grid.cells)) >= self.win_num
        return self.win
    @property
    def is_over(self):
        self.over = not any(self.can_move(move) if move != 'restart'and move != 'exit' else ''for move in self.action.actions)
        return self.over
    def state_init(self):
        self.reset()
        return 'game'
    def state_game(self):
        self.score = self.grid.score
        if self.score > self.best_score:
            self.best_score = self.score
        self.screen.draw(self.score, self.best_score)
        action = self.action.get()
        if action == 'restart':
            return 'init'
        if action == 'exit':
            return 'exit'
        if self.move(action):
            if self.is_win:
                return 'win'
            if self.is_over:
                return 'over'
        return 'game'
    def restart_or_exit(self):
        self.screen.draw(self.score, self.best_score)
        return 'init' if self.action.get == 'restart' else 'exit'
    def state_win(self):
        return self.restart_or_exit()
    def state_over(self):
        return self.restart_or_exit()
    def __call__(self, stdscr):
        self.stdscr = stdscr
        self.action = Action(stdscr)
        while self.state != 'exit':
            self.state = getattr(self, 'state_' + self.state)()

if __name__ == '__main__':
    curses.wrapper(GameManager())