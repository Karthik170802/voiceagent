from kartos.config import Config
from kartos.turn import (
    EndOfTurnDetector,
    LearnedEndOfTurnDetector,
    make_turn_detector,
)


def test_heuristic_turn_detector():
    cfg = Config()
    detector = EndOfTurnDetector(cfg)

    # Finished sentences
    assert detector.required_silence_ms("What time is it?") == cfg.complete_silence_ms
    assert detector.required_silence_ms("Tell me a joke.") == cfg.complete_silence_ms

    # Incomplete sentences
    assert detector.required_silence_ms("so what I wanted to say is") == cfg.incomplete_silence_ms
    assert detector.required_silence_ms("my phone number is 9") == cfg.incomplete_silence_ms
    assert detector.required_silence_ms("well,") == cfg.incomplete_silence_ms

    # Neutral/default
    assert detector.required_silence_ms("something neutral") == cfg.base_silence_ms


def test_learned_turn_detector_fallback_when_model_missing():
    cfg = Config(turn_detector="model", turn_detector_model_path="nonexistent_model.onnx")
    detector = make_turn_detector(cfg)

    assert isinstance(detector, LearnedEndOfTurnDetector)
    # Model is missing; must gracefully fall back to heuristic
    assert detector.required_silence_ms("What time is it?") == cfg.complete_silence_ms
    assert detector.required_silence_ms("so what I wanted to say is") == cfg.incomplete_silence_ms


def test_learned_turn_detector_with_predictor():
    cfg = Config(turn_detector="model")

    def mock_predictor(text: str) -> float:
        if "finished" in text:
            return 0.95
        elif "trailing" in text:
            return 0.10
        return 0.40

    detector = LearnedEndOfTurnDetector(cfg, predictor=mock_predictor)

    # High completion probability
    assert detector.required_silence_ms("I am finished now") == cfg.complete_silence_ms

    # Low completion probability (incomplete utterance)
    assert detector.required_silence_ms("this is trailing and") == cfg.incomplete_silence_ms

    # Neutral probability
    assert detector.required_silence_ms("regular words here") == cfg.base_silence_ms
