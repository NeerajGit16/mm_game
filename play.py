"""Play a session in the browser.

    uv run play.py --preset 60            # a new seed
    uv run play.py --preset 60 --seed 7   # a given seed (flagged if you have played it)
    uv run play.py --reveal 60m-7-0003    # reopen a saved session's reveal

Records go to records/ unless --records says otherwise.
"""

from mm_game.server import main

if __name__ == "__main__":
    main()
