import numpy as np

from voiceagents.audio.vad import EnergyVAD, VadConfig

LOUD = np.ones(160, dtype=np.float32)
QUIET = np.zeros(160, dtype=np.float32)


def test_triggers_after_sustained_loud_blocks():
    vad = EnergyVAD(VadConfig(threshold=0.02, sustain_blocks=3))

    assert vad.process_block(LOUD) is False
    assert vad.process_block(LOUD) is False
    assert vad.process_block(LOUD) is True


def test_does_not_trigger_again_on_subsequent_loud_blocks():
    vad = EnergyVAD(VadConfig(threshold=0.02, sustain_blocks=2))
    vad.process_block(LOUD)
    assert vad.process_block(LOUD) is True

    assert vad.process_block(LOUD) is False


def test_quiet_block_resets_the_counter():
    vad = EnergyVAD(VadConfig(threshold=0.02, sustain_blocks=3))
    vad.process_block(LOUD)
    vad.process_block(LOUD)

    vad.process_block(QUIET)

    assert vad.process_block(LOUD) is False
    assert vad.process_block(LOUD) is False
    assert vad.process_block(LOUD) is True


def test_reset_clears_state():
    vad = EnergyVAD(VadConfig(threshold=0.02, sustain_blocks=2))
    vad.process_block(LOUD)

    vad.reset()

    assert vad.process_block(LOUD) is False
    assert vad.process_block(LOUD) is True


def test_sustain_blocks_of_one_triggers_immediately():
    vad = EnergyVAD(VadConfig(threshold=0.02, sustain_blocks=1))

    assert vad.process_block(LOUD) is True
