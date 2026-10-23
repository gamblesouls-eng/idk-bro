"""SlotGrid geometry: linear grids and two-column panels (party screen)."""
from poke_hunter.config import SlotGrid


def test_linear_grid_slots():
    g = SlotGrid(region=(10, 20, 100, 40), step=(0, 50), count=3)
    assert g.slot(0) == (10, 20, 100, 40)
    assert g.slot(1) == (10, 70, 100, 40)
    assert g.slot(2) == (10, 120, 100, 40)


def test_two_column_grid_slots():
    # measured live: party cards, 3 rows per column, right column offset
    g = SlotGrid(region=(268, 122, 616, 190), step=(0, 229), count=6,
                 per_col_rows=3, col_offset=(692, 56))
    assert g.slot(0) == (268, 122, 616, 190)
    assert g.slot(1) == (268, 351, 616, 190)
    assert g.slot(2) == (268, 580, 616, 190)
    # slot 3 jumps to the top of the second column
    assert g.slot(3) == (960, 178, 616, 190)
    assert g.slot(4) == (960, 407, 616, 190)
    assert g.slot(5) == (960, 636, 616, 190)


def test_two_column_single_row_per_col_is_linear_equivalent():
    # with one row per column, col_offset fully replaces the step: setting it
    # to the previous step vector reproduces the linear behaviour exactly
    g = SlotGrid(region=(0, 0, 10, 10), step=(50, 20), count=4,
                 per_col_rows=1, col_offset=(50, 20))
    assert g.slot(1) == (50, 20, 10, 10)
    assert g.slot(2) == (100, 40, 10, 10)
    assert g.slot(3) == (150, 60, 10, 10)
