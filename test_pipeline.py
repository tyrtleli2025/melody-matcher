from src.melody_matcher.io.midi_reader import extract_notes_from_midi
from src.melody_matcher.preprocessing.segmenter import create_segments
from src.melody_matcher.features.interval_encoder import encode_intervals

def run_demo():
    print("🎵 Starting the Melody Matcher Pipeline...\n")

    # 1. Read the raw MIDI file
    midi_file = "data/raw/dummy.mid"
    notes = extract_notes_from_midi(midi_file)
    print(f"✅ Extracted {len(notes)} notes from {midi_file}")

    # 2. Chop it into overlapping segments (let's say 5 notes each)
    segments = create_segments(notes, segment_length=5)
    print(f"✅ Chopped into {len(segments)} overlapping segments\n")

    # 3. Convert each segment into relative intervals
    print("🔍 Inspecting the first 3 segments:")
    for i, segment in enumerate(segments[:3]):
        # Extract the actual text names just so we can see them
        note_names = [n.nameWithOctave for n in segment]
        
        # The magic math step!
        intervals = encode_intervals(segment)
        
        print(f"   Segment {i+1}: {note_names}")
        print(f"   Signature: {intervals}\n")

if __name__ == "__main__":
    run_demo()