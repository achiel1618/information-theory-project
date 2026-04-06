import numpy as np
from AudioCD import AudioCD


def make_cd():
    return AudioCD(44100, 1, 8)


def test_c2_parity_middle():
    """
    Test that C2 parity is correctly inserted in the middle of a frame.
    """
    cd = make_cd()

    frame = np.arange(24, dtype='B')

    out, _ = cd.CIRC_enc_C2(frame, 1)
    out_frame = out.reshape((1, 28))[0]

    ref = np.array(list(cd.rsc2.encode(frame)), dtype='B')
    parity = ref[24:28]

    assert np.array_equal(out_frame[:12], frame[:12])
    assert np.array_equal(out_frame[12:16], parity)
    assert np.array_equal(out_frame[16:], frame[12:])

    print("✓ C2 parity correctly inserted in middle")


def test_c1_parity_end():
    """
    Test that C1 parity is correctly appended at the end of a frame.
    """
    cd = make_cd()

    frame = np.arange(28, dtype='B')

    out, _ = cd.CIRC_enc_C1(frame, 1)
    out_frame = out.reshape((1, 32))[0]

    ref = np.array(list(cd.rsc1.encode(frame)), dtype='B')

    assert np.array_equal(out_frame, ref)

    print("✓ C1 parity correctly appended at end")


def test_delay_unequal():
    """
    Test that unequal delay stage shifts data correctly.
    """
    cd = make_cd()

    n_frames = 5
    data = np.zeros((n_frames, 28), dtype='B')

    data[0, 3] = 123

    out, nf = cd.CIRC_enc_delay_unequal(data.reshape(-1), n_frames)
    out_data = out.reshape((nf, 28))

    assert out_data[12, 3] == 123
    assert np.count_nonzero(out_data[:, 3]) == 1

    print("✓ unequal delay shifts column correctly")


def test_delay_inv():
    """
    Test that delay + inversion stage works correctly.
    """
    cd = make_cd()

    n_frames = 2
    data = np.zeros((n_frames, 32), dtype='B')

    data[0, 0] = 10
    data[0, 1] = 20
    data[0, 12] = 5
    data[0, 29] = 7

    out, nf = cd.CIRC_enc_delay_inv(data.reshape(-1), n_frames)
    out_data = out.reshape((nf, 32))

    assert out_data[1, 0] == 10
    assert out_data[0, 1] == 20
    assert out_data[1, 12] == (5 ^ 0xFF)
    assert out_data[0, 29] == (7 ^ 0xFF)

    print("✓ delay + inversion stage works correctly")


def main():
    print("Running encoder tests...\n")

    test_c2_parity_middle()
    test_c1_parity_end()
    test_delay_unequal()
    test_delay_inv()

    print("\nAll encoder tests passed successfully.")


if __name__ == "__main__":
    main()