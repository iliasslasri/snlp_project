import os
import argparse
import random
import torchaudio
import torch
import numpy as np
import pyroomacoustics as pra
from tqdm import tqdm


def generate_rir(sample_rate):
    """
    Simulate a Room Impulse Response (RIR) using pyroomacoustics.
    Matches the parameters used in the on-the-fly augmentation.
    """
    # Random room dimensions (meters)
    room_x = random.uniform(3.0, 8.0)
    room_y = random.uniform(3.0, 6.0)
    room_z = random.uniform(2.5, 4.0)

    # Random RT60 reverberation time (seconds)
    rt60 = random.uniform(0.2, 0.8)

    # Compute absorption and max_order from RT60 using Sabine's formula
    try:
        e_absorption, max_order = pra.inverse_sabine(rt60, [room_x, room_y, room_z])
    except ValueError:
        # Fallback if sabine inversion fails for the generated dimensions
        e_absorption = 0.2
        max_order = 10

    room = pra.ShoeBox(
        [room_x, room_y, room_z],
        fs=sample_rate,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )

    # Random source position (inside the room with margin)
    margin = 0.3
    src_pos = [
        random.uniform(margin, room_x - margin),
        random.uniform(margin, room_y - margin),
        random.uniform(margin, room_z - margin),
    ]

    # Random microphone position (inside the room with margin)
    mic_pos = [
        random.uniform(margin, room_x - margin),
        random.uniform(margin, room_y - margin),
        random.uniform(margin, room_z - margin),
    ]

    room.add_source(src_pos)
    room.add_microphone(mic_pos)
    
    # Compute the RIR
    room.compute_rir()
    
    rir = room.rir[0][0] # Mic 0, Source 0
    return rir


def main():
    parser = argparse.ArgumentParser(description="Simulate RIRs offline using pyroomacoustics")
    parser.add_argument("--n", type=int, default=200, help="Number of RIRs to generate")
    parser.add_argument("--out_dir", type=str, default="data/rirs", help="Output directory")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Generating {args.n} RIRs in {args.out_dir} at {args.sr} Hz...")

    for i in tqdm(range(args.n)):
        rir_array = generate_rir(args.sr)
        
        # Convert to tensor shaped [1, time]
        rir_tensor = torch.from_numpy(rir_array.astype(np.float32)).unsqueeze(0)
        
        out_path = os.path.join(args.out_dir, f"rir_{i:04d}.wav")
        torchaudio.save(out_path, rir_tensor, args.sr)

    print("Done!")


if __name__ == "__main__":
    main()
