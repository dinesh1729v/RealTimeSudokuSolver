def is_sudoku_filled(mat):
    for i in range(9):
        for j in range(9):
            if mat[i] == 0:
                return False
    return True


def cross(A, B):
    return [a + b for a in A for b in B]


digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
squares = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])


units = {}
for s in squares:
    for u in unitlist:
        if s in u:
            if s not in units:
                units[s] = []
            units[s].append(u)

peers = {}
for s in squares:
    unit_set = set()
    for unit in units[s]:
        for square in unit:
            if square != s:
                unit_set.add(square)
    peers[s] = unit_set


def grid_values(grid):
    grid1_chars = []
    for c in grid:
        if c in digits or c in '0.':
            grid1_chars.append(c)

    grid1_values = {}

    for k, v in zip(squares, grid1_chars):
        grid1_values[k] = v

    return grid1_values


def eliminate(values, s, d):
    if d not in values[s]:
        return values
    values[s] = values[s].replace(d, '')

    if len(values[s]) == 0:
        return False
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False
        elif len(dplaces) == 1:
            if not assign(values, dplaces[0], d):
                return False
    return values


def assign(values, s, d):
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False


def parse_grid(grid):
    values = dict((s, digits) for s in squares)
    for s, d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False
    return values


def search(values):
    if values is False:
        return False
    if all(len(values[s]) == 1 for s in squares):
        return values
    n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    return some(search(assign(values.copy(), s, d)) for d in values[s])


def some(seq):
    for e in seq:
        if e: return e
    return False


def solve(grid):
    grid = [item for sublist in grid for item in sublist]
    solved_grid = search(parse_grid(grid))
    if(solved_grid != False):
        sudoku_grid = []
        for i in range(9):
            sudoku_grid.append(['0', '0', '0', '0', '0', '0', '0', '0', '0'])
        for s in squares:
            sudoku_grid[ord(s[0])-ord('A')][ord(s[1])-ord('1')]=solved_grid[s]
        return sudoku_grid
    return False


#
# grid = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"
# print(solve(grid))
