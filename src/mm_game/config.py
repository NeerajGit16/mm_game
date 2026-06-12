from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    tick_size: float = 0.005
    tick_value: float = 25.0
    lot_size: int = 1
    session_length: int = 60 * 60 * 2
    dt: float = 0.001

    def __post_init__(self) -> None:
        assert self.tick_size > 0, "tick_size must be positive"
        assert self.tick_value > 0, "tick_value must be positive"
        assert self.lot_size >= 1, "lot_size must be at least 1"
        assert self.session_length > 0, "session_length must be positive"
        assert 0 < self.dt <= self.session_length, "dt must be in (0, session_length]"


DEFAULT_CONFIG = GameConfig()

