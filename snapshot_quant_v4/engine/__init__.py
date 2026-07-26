"""Engine: validation, gating, scoring, regime, state machine, execution.

Import direction inside this package is strictly one way::

    stats / validator / quality
        -> confidence -> regime -> composite -> threshold
        -> state_machine -> execution -> quant_engine

No module in this package imports the adapter or the renderer. The engine
consumes :class:`~snapshot_quant_v4.utils.types.RawSnapshot` and produces
:class:`~snapshot_quant_v4.utils.types.EngineOutput`, which is the entire
contract with the outside world.
"""

from __future__ import annotations
