#!/usr/bin/env python3
"""
Test script to verify the new scoring and marking logic.
"""
import numpy as np
from backend.core.models import Transcript, Segment, DetectionParams
from backend.detector.ucc_detector import UCCDetector

def create_test_transcript():
    """Create a simple test transcript."""
    segments = [
        Segment(id=i, start_ms=i*1000, end_ms=(i+1)*1000, text=f"Segment {i}")
        for i in range(20)
    ]
    return Transcript(video_id="test", language="ja", segments=segments)

def create_test_embeddings(n_segments, dim=768):
    """Create dummy embeddings for testing."""
    # Create embeddings with a pattern that should trigger UCC detection
    embeddings = np.random.randn(n_segments, dim).astype(np.float32)

    # Make segment 10-14 significantly different to trigger detection
    embeddings[10:15] = embeddings[10:15] + np.random.randn(5, dim) * 3

    return embeddings

def test_score_assignment():
    """Test that scores are assigned to the correct indices."""
    print("=" * 60)
    print("TEST: Score Assignment")
    print("=" * 60)

    transcript = create_test_transcript()
    embeddings = create_test_embeddings(len(transcript.segments))

    params = DetectionParams(k=5, lambda_param=1.0, mark_length=3)
    detector = UCCDetector(params)

    result = detector.detect(transcript, embeddings)

    print(f"\nParameters:")
    print(f"  k (window size): {params.k}")
    print(f"  mark_length: {params.mark_length}")

    print(f"\nTotal segments: {len(result.segments)}")
    print(f"Marked segments: {len([s for s in result.segments if s.is_marked])}")

    # Check score assignment
    print(f"\nScore assignment verification:")
    print(f"{'Idx':<5} {'UCC Score':<12} {'ΔD':<12} {'Marked':<8}")
    print("-" * 45)
    for i, seg in enumerate(result.segments):
        if i < params.k or seg.scores.ucc_score > 0.5 or seg.is_marked:
            marked_str = "YES" if seg.is_marked else "no"
            print(f"{i:<5} {seg.scores.ucc_score:<12.4f} {seg.scores.context_debt_delta:<12.4f} {marked_str:<8}")

    # Verify that scores before k-1 are properly handled
    print(f"\nFirst {params.k-1} segments (should have default scores):")
    for i in range(min(params.k-1, len(result.segments))):
        seg = result.segments[i]
        print(f"  Segment {i}: UCC={seg.scores.ucc_score:.4f}, ΔD={seg.scores.context_debt_delta:.4f}")

    return result

def test_marking_range():
    """Test that marking range is [t+k-L, t+k-1]."""
    print("\n" + "=" * 60)
    print("TEST: Marking Range")
    print("=" * 60)

    transcript = create_test_transcript()
    embeddings = create_test_embeddings(len(transcript.segments))

    params = DetectionParams(k=5, lambda_param=0.5, mark_length=3, tau_topic=None)  # Disable topic filter
    detector = UCCDetector(params)

    result = detector.detect(transcript, embeddings)

    marked_indices = [i for i, s in enumerate(result.segments) if s.is_marked]
    high_score_indices = [i for i, s in enumerate(result.segments)
                          if s.scores.context_debt_delta > result.metadata['threshold']]

    print(f"\nParameters:")
    print(f"  k = {params.k}")
    print(f"  L (mark_length) = {params.mark_length}")
    print(f"  Threshold (μ + λσ) = {result.metadata['threshold']:.4f}")

    print(f"\nCandidate positions (t where ΔD > threshold):")
    print(f"  {high_score_indices}")

    print(f"\nMarked segment indices:")
    print(f"  {marked_indices}")

    print(f"\nExpected marking logic (NEW):")
    print(f"  For each candidate segment i, mark [i-L+1, i] = [i-{params.mark_length-1}, i]")
    print(f"  (The last {params.mark_length} segments ending at i)")

    # Verify specific cases
    if high_score_indices:
        print(f"\nDetailed verification:")
        for i in high_score_indices:
            expected_start = i - params.mark_length + 1
            expected_end = i
            expected_range = list(range(max(0, expected_start), min(len(result.segments), expected_end + 1)))
            actual_in_range = [j for j in expected_range if j in marked_indices]

            check_mark = "✓" if set(expected_range) <= set(marked_indices) else "✗"
            print(f"  Candidate i={i}: mark [{expected_start}, {expected_end}] → {expected_range} {check_mark}")

def test_reason_labeling():
    """Test that reason labels use correct windows."""
    print("\n" + "=" * 60)
    print("TEST: Reason Label Windows")
    print("=" * 60)

    from backend.reasons.reason_labeler import ReasonLabeler

    transcript = create_test_transcript()
    embeddings = create_test_embeddings(len(transcript.segments))

    params = DetectionParams(k=5, lambda_param=0.5, mark_length=3, tau_topic=None)
    detector = UCCDetector(params)

    result = detector.detect(transcript, embeddings)

    # Add reasons
    labeler = ReasonLabeler(params, language="en")
    result = labeler.add_reasons(result, transcript)

    print(f"\nMarked segments with their reason computation:")
    marked_segments = [s for s in result.segments if s.is_marked]

    for seg in marked_segments[:3]:  # Show first 3
        i = seg.id
        # Find which j caused this marking
        candidates = []
        for j in range(max(0, i), min(len(result.segments), i + params.mark_length)):
            if j - params.mark_length + 1 <= i <= j:
                candidates.append(j)

        if candidates:
            j_max = max(candidates, key=lambda x: result.segments[x].scores.context_debt_delta)
            t = j_max - params.k + 1
            print(f"\n  Segment {i}:")
            print(f"    Caused by detection at segment j={j_max} (ΔD={result.segments[j_max].scores.context_debt_delta:.2f})")
            print(f"    Window position t={t}")
            print(f"    W_minus(t) = [{t-params.k}, {t-1}]")
            print(f"    W_plus(t)  = [{t}, {t+params.k-1}]")
            print(f"    Reasons: {seg.reasons}")

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING NEW UCC DETECTION LOGIC")
    print("=" * 60)
    print("\nChanges:")
    print("1. Scores computed at t are assigned to segment t+k-1")
    print("2. Marking range changed from [t, t+2] to [t+k-L, t+k-1]")
    print("3. Reason labels now use correct window based on detection point")
    print()

    result1 = test_score_assignment()
    result2 = test_marking_range()
    test_reason_labeling()

    print("\n" + "=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)
    print("\nPlease verify:")
    print("✓ Scores are assigned to segment indices t+k-1 (not t)")
    print("✓ Marked segments are in range [i-L+1, i] for detected segment i")
    print("✓ First k-1 segments have default scores (not assigned from computation)")
    print("✓ Reason labels computed from correct window of detection point")
    print()

if __name__ == "__main__":
    main()
